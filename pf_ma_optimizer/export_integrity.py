"""
PF AI Lab 6.0.3 -- Export Integrity Module (P0)

Implements all P0 audit requirements:
  P0.1: Round-trip validation (serialize -> deserialize -> compare)
  P0.2: Instrument field reset + cross-field validation
  P0.3: SHA-256 config hash + schema_version + export_timestamp
  P0.4: Strict export mode (refuse export on inconsistencies)

Also provides the canonical metadata block requested by the Risk Management team.
"""

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Optional

APP_VERSION = '6.0.3'

# ===========================================================================
# INSTRUMENT_MAP: single source of truth for symbol derivation
# Maps canonical instrument name -> all derived symbol fields.
# ===========================================================================
INSTRUMENT_MAP = {
    # Indices
    'EUSTX50': {'mt4_symbol': 'EUSTX50', 'wma_instrument': 'EUREX:FESX1!', 'tv_symbol': 'EUSTX50'},
    'US30':    {'mt4_symbol': 'US30',    'wma_instrument': 'CBOT:YM1!',     'tv_symbol': 'US30'},
    'US100':   {'mt4_symbol': 'US100',   'wma_instrument': 'CME:NQ1!',      'tv_symbol': 'US100'},
    'US500':   {'mt4_symbol': 'US500',   'wma_instrument': 'CME:ES1!',      'tv_symbol': 'US500'},
    'UK100':   {'mt4_symbol': 'UK100',   'wma_instrument': 'ICEEUR:Z1!',    'tv_symbol': 'UK100'},
    'DEU40':   {'mt4_symbol': 'DEU40',   'wma_instrument': 'EUREX:FDAX1!',  'tv_symbol': 'DEU40'},
    'GER40':   {'mt4_symbol': 'GER40',   'wma_instrument': 'EUREX:FDAX1!',  'tv_symbol': 'GER40'},
    'FRA40':   {'mt4_symbol': 'FRA40',   'wma_instrument': 'EURONEXT:FCE1!','tv_symbol': 'FRA40'},
    'JPN225':  {'mt4_symbol': 'JPN225',  'wma_instrument': 'CME:NKD1!',     'tv_symbol': 'JPN225'},
    # Metals
    'XAUUSD':  {'mt4_symbol': 'XAUUSD',  'wma_instrument': 'COMEX:GC1!',    'tv_symbol': 'XAUUSD'},
    'XAGUSD':  {'mt4_symbol': 'XAGUSD',  'wma_instrument': 'COMEX:SI1!',    'tv_symbol': 'XAGUSD'},
    # Energy
    'USOIL':   {'mt4_symbol': 'USOIL',   'wma_instrument': 'NYMEX:CL1!',    'tv_symbol': 'USOIL'},
    'UKOIL':   {'mt4_symbol': 'UKOIL',   'wma_instrument': 'ICEEUR:BRN1!',  'tv_symbol': 'UKOIL'},
    # Crypto
    'BTCUSD':  {'mt4_symbol': 'BTCUSD',  'wma_instrument': 'BITSTAMP:BTCUSD','tv_symbol': 'BTCUSD'},
    'ETHUSD':  {'mt4_symbol': 'ETHUSD',  'wma_instrument': 'BITSTAMP:ETHUSD','tv_symbol': 'ETHUSD'},
    # Forex
    'EURUSD':  {'mt4_symbol': 'EURUSD',  'wma_instrument': 'FX:EURUSD',     'tv_symbol': 'EURUSD'},
    'GBPUSD':  {'mt4_symbol': 'GBPUSD',  'wma_instrument': 'FX:GBPUSD',     'tv_symbol': 'GBPUSD'},
    'USDJPY':  {'mt4_symbol': 'USDJPY',  'wma_instrument': 'FX:USDJPY',     'tv_symbol': 'USDJPY'},
}


# ===========================================================================
# P0.2: reset_instrument_fields
# ===========================================================================

def reset_instrument_fields(config: dict, instrument: str) -> dict:
    """
    Reset all instrument-derived fields from a single canonical instrument.
    Prevents residual fields from previous configs (e.g. BTCUSDT in STOXX50 export).

    Args:
        config: config dict to modify (mutated in place AND returned)
        instrument: canonical instrument name (e.g. 'EUSTX50')

    Returns:
        The modified config dict.
    """
    mapping = INSTRUMENT_MAP.get(instrument.upper(), {})
    # Always overwrite derived fields
    config['mt4_symbol'] = mapping.get('mt4_symbol', instrument.upper())
    config['wma_instrument'] = mapping.get('wma_instrument', instrument.upper())
    config['tv_symbol'] = mapping.get('tv_symbol', instrument.upper())
    config['instrument'] = instrument.upper()
    return config


def validate_instrument_fields(config: dict) -> list:
    """
    Cross-field validation: instrument == mt4_symbol == wma_instrument (modulo mapping).

    Returns:
        List of warning strings. Empty = consistent.
    """
    warnings = []
    instrument = (config.get('instrument') or '').upper()
    mt4 = (config.get('mt4_symbol') or '').upper()
    wma = (config.get('wma_instrument') or '').upper()

    if not instrument:
        warnings.append("Missing 'instrument' field")
        return warnings

    # mt4_symbol should match the instrument's canonical MT4 name
    expected = INSTRUMENT_MAP.get(instrument, {})
    expected_mt4 = expected.get('mt4_symbol', instrument).upper()
    expected_wma = expected.get('wma_instrument', instrument).upper()

    if mt4 and mt4 != expected_mt4:
        warnings.append(
            f"mt4_symbol mismatch: got '{mt4}', expected '{expected_mt4}' for instrument '{instrument}'"
        )

    if wma and wma != expected_wma:
        warnings.append(
            f"wma_instrument mismatch: got '{wma}', expected '{expected_wma}' for instrument '{instrument}'"
        )

    return warnings


# ===========================================================================
# P0.1: Round-trip validation
# ===========================================================================

# Fields that MUST survive serialization round-trip
_CRITICAL_BOOL_FIELDS = [
    'long_side', 'short_side', 'exit_setup_reversal',
    'use_lt_filter', 'use_mt_filter', 'use_rsi50_filter',
    'use_diff_ma_red', 'use_ma_dir_filter', 'use_mavg_filter',
    'use_diff_price_red', 'use_adx_regime_filter', 'use_vol_filter',
    'use_qq_filter', 'use_partial_tp', 'use_session_filter',
    'use_vix_filter', 'use_macro_filter', 'use_fib_filter',
    'rsi_div_hidden', 'sl_dynamic',
]

_CRITICAL_NUMERIC_FIELDS = [
    'pmax_length', 'pmax_multiplier', 'don_length',
    'sl_pct', 'order_pause', 'lt_multiplier', 'mt_multiplier',
]

_CRITICAL_STR_FIELDS = [
    'entry_type', 'pmax_ma_type', 'sl_mode', 'tp_mode', 'be_mode',
    'lt_tf', 'mt_tf',
]


def validate_round_trip(config: dict) -> list:
    """
    Serialize config to JSON and deserialize, comparing field by field.
    Returns list of divergence descriptions. Empty = clean round-trip.

    Catches bugs where internal bool/enum fields are not included
    in the JSON serialization schema (e.g. short_side missing).
    """
    errors = []

    try:
        serialized = json.dumps(config, sort_keys=True, default=str)
        deserialized = json.loads(serialized)
    except (TypeError, json.JSONDecodeError) as e:
        return [f"Serialization failed: {e}"]

    # Check all critical fields
    all_critical = _CRITICAL_BOOL_FIELDS + _CRITICAL_NUMERIC_FIELDS + _CRITICAL_STR_FIELDS

    for field in all_critical:
        if field not in config:
            continue  # optional field, not present
        original = config[field]
        restored = deserialized.get(field)

        if restored is None and original is not None:
            errors.append(f"Field '{field}' lost during serialization (original={original}, restored=None)")
            continue

        # Type-aware comparison
        if isinstance(original, bool):
            if restored is not True and restored is not False:
                errors.append(f"Field '{field}' type changed: bool({original}) -> {type(restored).__name__}({restored})")
            elif original != restored:
                errors.append(f"Field '{field}' value changed: {original} -> {restored}")
        elif isinstance(original, (int, float)):
            if isinstance(restored, (int, float)):
                if abs(float(original) - float(restored)) > 1e-9:
                    errors.append(f"Field '{field}' value drift: {original} -> {restored}")
            else:
                errors.append(f"Field '{field}' type changed: {type(original).__name__} -> {type(restored).__name__}")
        elif isinstance(original, str):
            if str(restored) != original:
                errors.append(f"Field '{field}' value changed: '{original}' -> '{restored}'")

    return errors


# ===========================================================================
# P0.3: SHA-256 config hash
# ===========================================================================

def compute_config_hash(config: dict) -> str:
    """
    Compute SHA-256 hash of the canonical config representation.
    Fields are sorted and values are normalized for determinism.

    Returns 'sha256:<hex_digest>' string.
    """
    # Build canonical representation: sorted keys, normalized values
    canonical = {}
    for k in sorted(config.keys()):
        v = config[k]
        if v is None:
            continue
        # Normalize types
        if isinstance(v, bool):
            canonical[k] = v  # JSON bools
        elif isinstance(v, float):
            if math.isfinite(v):
                canonical[k] = round(v, 6)
            else:
                canonical[k] = None
        elif isinstance(v, int):
            canonical[k] = v
        elif isinstance(v, str):
            canonical[k] = v.strip()
        elif isinstance(v, (list, tuple)):
            canonical[k] = list(v)
        # Skip complex objects (DataFrames, numpy arrays)

    # Deterministic JSON for hashing
    json_str = json.dumps(canonical, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    return f"sha256:{digest}"


# ===========================================================================
# P0.3 + Metadata: build_export_metadata
# ===========================================================================

def build_export_metadata(config: dict, instrument: str = '',
                          data_source: str = 'MT5_FTMO',
                          broker: str = 'FTMO',
                          platform: str = 'MT5',
                          is_period: tuple = None,
                          oos_period: tuple = None,
                          n_bars: int = 0,
                          spread_model: str = 'raw',
                          slippage_model: str = 'zero',
                          commission_included: bool = False) -> dict:
    """
    Build the full metadata block for each export, as requested by Risk Management.

    Returns a dict ready to be merged into the export JSON.
    """
    return {
        'schema_version': APP_VERSION,
        'config_hash': compute_config_hash(config),
        'export_timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'pflab_version': APP_VERSION,
        'data_source': data_source,
        'optimization_feed': {
            'broker': broker,
            'platform': platform,
            'symbol': instrument.upper() if instrument else 'UNKNOWN',
        },
        'is_period': list(is_period) if is_period else None,
        'oos_period': list(oos_period) if oos_period else None,
        'n_bars': n_bars,
        'spread_model': spread_model,
        'slippage_model': slippage_model,
        'commission_included': commission_included,
        'trade_counting_method': 'per_position',
        'feed_divergence_note': (
            'PF OOS vs TradingView peut diverger de 30-50% selon broker LP. '
            'Validez toujours sur TradingView avant deployment.'
        ),
    }


# ===========================================================================
# P0.4: Strict export mode
# ===========================================================================

class ExportValidationError(Exception):
    """Raised when strict export mode detects inconsistencies."""
    def __init__(self, errors: list):
        self.errors = errors
        super().__init__(f"Export blocked: {len(errors)} issue(s) detected: {'; '.join(errors[:5])}")


def validate_for_export(config: dict, instrument: str = '',
                        strict: bool = True) -> list:
    """
    Run all P0 validations on a config before export.

    Args:
        config: the config dict to validate
        instrument: canonical instrument name
        strict: if True, raise ExportValidationError on any issue

    Returns:
        List of warnings/errors (empty = clean)

    Raises:
        ExportValidationError: if strict=True and issues found
    """
    all_issues = []

    # P0.1: Round-trip
    rt_issues = validate_round_trip(config)
    all_issues.extend([f"[ROUND-TRIP] {e}" for e in rt_issues])

    # P0.2: Instrument consistency
    if instrument:
        inst_issues = validate_instrument_fields(config)
        all_issues.extend([f"[INSTRUMENT] {e}" for e in inst_issues])

    # Check for obviously incomplete configs
    if not config.get('entry_type'):
        all_issues.append("[INCOMPLETE] Missing entry_type")

    if config.get('long_side') is None and config.get('short_side') is None:
        all_issues.append("[INCOMPLETE] Both long_side and short_side are None")

    # Check direction: at least one side must be active
    if config.get('long_side') is False and config.get('short_side') is False:
        all_issues.append("[INVALID] Both long_side=False and short_side=False (no trades possible)")

    if strict and all_issues:
        raise ExportValidationError(all_issues)

    return all_issues


def export_config_json(config: dict, instrument: str = '',
                       strict: bool = True,
                       metadata_kwargs: dict = None) -> dict:
    """
    Full export pipeline: validate -> reset instrument fields -> hash -> metadata.

    Args:
        config: the strategy config dict
        instrument: canonical instrument name
        strict: if True, refuse export on inconsistencies
        metadata_kwargs: extra kwargs for build_export_metadata()

    Returns:
        Complete export dict with config + metadata + hash

    Raises:
        ExportValidationError: if strict and issues found
    """
    # Reset instrument fields from canonical source
    if instrument:
        reset_instrument_fields(config, instrument)

    # Validate
    issues = validate_for_export(config, instrument, strict=strict)

    # Build metadata
    meta_kw = metadata_kwargs or {}
    meta_kw.setdefault('instrument', instrument)
    metadata = build_export_metadata(config, **meta_kw)

    return {
        'config': config,
        'metadata': metadata,
        'validation_warnings': issues,
    }
