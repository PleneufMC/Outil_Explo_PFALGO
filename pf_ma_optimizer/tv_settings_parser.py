"""
PF AI Lab 5.4.1 -- Alert-Based Config Export

The approach: modify the PineScript alert messages so that every alert
(buy, sell, close) includes the full configuration JSON as a second line.

Format of enriched alert:
    Line 1: LicenseID,buy,BTCUSDT,sl=123.45,risk=1.0    (normal PineConnector)
    Line 2: PFLAB_CONFIG:{"pmax_ma_type":"EMA","pmax_length":10, ...}

When the user pastes any alert text into PF AI Lab, the parser:
    1. Detects the PFLAB_CONFIG: prefix
    2. Extracts and parses the JSON
    3. Transforms mode values (SL/PT/BE) to internal format
    4. Returns a clean PF AI Lab config dict

The user also has 3 alternative methods:
    A. Paste raw JSON (from table export or manual)
    B. Use the mapping guide (Pine variable -> JSON key)
    C. Copy from Audit/DB pages

This module also generates the PineScript code block that builds the
config_json variable and appends it to alert messages.
"""

import re
import json


# -----------------------------------------------------------------------
# COMPLETE mapping: PineScript variable name -> PF AI Lab config key
# This is the AUTHORITATIVE mapping between the two systems.
# -----------------------------------------------------------------------
PINE_TO_PFLAB = {
    # PMax
    'Multiplier':           'pmax_multiplier',
    'mav':                  'pmax_ma_type',
    'length':               'pmax_length',
    # Entry
    'entry_type':           'entry_type',
    'long_side':            'long_side',
    'short_side':           'short_side',
    'order_pause':          'order_pause',
    # VWAP
    'vwap_stddev_mult':     'vwap_stddev_mult',
    'vwap_length':          'vwap_length',
    'vwap_confirmation':    'vwap_confirmation',
    # Fractal
    'fractal_buffer_point': 'fractal_buffer',
    # Boll+SMA
    'length_bb_ma':         'length_bb_ma',
    'mult_bb_ma':           'mult_bb_ma',
    'length_bb_sma':        'length_bb_sma',
    # ORB
    'openingRangeMinutes':  'orb_minutes',
    # Donchian
    'don_length':           'don_length',
    # RSI Divergence
    'rsi_div_length':       'rsi_div_length',
    'rsi_div_pivot_lookback': 'rsi_div_pivot',
    'rsi_div_overbought':   'rsi_div_overbought',
    'rsi_div_oversold':     'rsi_div_oversold',
    'rsi_div_detect_hidden': 'rsi_div_hidden',
    # MM Cross
    'mm_cross_ma1_type':    'mm_cross_ma1_type',
    'mm_cross_ma1_len':     'mm_cross_ma1_len',
    'mm_cross_ma2_type':    'mm_cross_ma2_type',
    'mm_cross_ma2_len':     'mm_cross_ma2_len',
    # LT Trend Filter
    'lt_trend_filter':      'use_lt_filter',
    'lt_trend_filter_tf':   'lt_tf',
    'lt_trend_filter_mult': 'lt_multiplier',
    'lt_trend_filter_mav':  'lt_ma_type',
    'lt_trend_filter_length': 'lt_length',
    # MT Trend Filter
    'mt_trend_filter':      'use_mt_filter',
    'mt_trend_filter_tf':   'mt_tf',
    'mt_trend_filter_mult': 'mt_multiplier',
    'mt_trend_filter_mav':  'mt_ma_type',
    'mt_trend_filter_length': 'mt_length',
    # Session
    'Session':              'session1_raw',
    'Session2':             'session2_raw',
    # WMA/SMA
    'wma_sma_filter':       'use_mavg_filter',
    # Macro
    'macro_filter':         'use_macro_filter',
    'macro_mode':           'macro_mode',
    # RSI filters
    'snab_rsi_50_Filter':   'use_rsi50_filter',
    'snab_rsi_ema_filter':  'use_rsi_ema_filter',
    'snab_rsi_len':         'rsi_length',
    'snab_rsi_ema_len':     'rsi_ema_length',
    # QQE
    'qq_estimation_filter': 'use_qqe_filter',
    'qq_estimation_filter_length': 'qqe_length',
    # VIX
    'vix_filter':           'use_vix_filter',
    'vix_threshold_low':    'vix_threshold_low',
    'vix_threshold_high':   'vix_threshold_high',
    # Volatility ATR
    'vol_adaptive_filter':  'use_vol_filter',
    'vol_percentile':       'vol_percentile_threshold',
    # Correlation
    'correlation_filter':   'use_correlation_filter',
    'corr_length':          'corr_length',
    'corr_threshold':       'corr_threshold',
    # Max Hold Days
    'max_hold_days_filter': 'use_max_hold_days',
    'max_hold_days':        'max_hold_days',
    'max_hold_days_profit_exception': 'max_hold_days_profit_exception',
    'max_hold_days_profit_pct': 'max_hold_days_profit_pct',
    # MTF Momentum
    'mtf_momentum_filter':  'use_mtf_momentum',
    # Gap Protection
    'gap_protection':       'use_gap_protection',
    'max_gap_pct':          'max_gap_pct',
    # ATR EMA
    'use_atr_ema_allowed':  'use_atr_ema_filter',
    'atr_ema_len':          'atr_ema_len',
    # Max Candles
    'use_max_candles_allowed': 'use_max_candles_filter',
    'max_candles_allowed':  'max_candles',
    # MA Direction
    'ma_dir_filter':        'use_ma_dir_filter',
    'ma_dir_len':           'ma_dir_len',
    # MAvg
    'mavg_filter':          'use_mavg_filter',
    # Diff MA/Red
    'diff_ma_red_filter':   'use_diff_ma_red',
    'diff_ma_red_filter_percentage': 'diff_ma_red_pct',
    # Diff Price/Red
    'diff_ma_price_filter': 'use_diff_price_red',
    'diff_ma_price_filter_percentage': 'diff_price_red_pct',
    # ADX Regime
    'use_adx_regime_filter': 'use_adx_regime_filter',
    'adx_hint_length':      'adx_period',
    'adx_trend_threshold':  'adx_trend_threshold',
    'adx_reversion_threshold': 'adx_reversion_threshold',
    # -- Risk Management --
    'sl_mode':              'sl_mode',
    'sl_percent':           'sl_pct',
    'atr_sl_mult':          'atr_sl_mult',
    'sl_dynamic_adjust':    'sl_dynamic',
    'sl_vol_min_mult':      'sl_vol_min_mult',
    'sl_vol_max_mult':      'sl_vol_max_mult',
    # PT
    'pt_mode':              'tp_mode',
    'pt_percent':           'tp_pct',
    'pt_rl_percent':        'tp_rl_pct',
    'pt_rr_ratio':          'tp_sl_ratio',
    # Partial TP
    'partial_tp':           'use_partial_tp',
    'tp1_pct':              'tp1_pct',
    'tp1_size':             'tp1_size',
    'tp2_pct':              'tp2_pct',
    'tp2_size':             'tp2_size',
    # BE / TSL
    'be_mode':              'be_mode',
    'be_percent':           'be_trigger_pct',
    'dynamic_tsl':          'use_tsl',
    'tsl_atr_mult':         'tsl_atr_mult',
    # Bars in loss
    'bars_in_loss_enable':  'use_bars_in_loss',
    'bars_in_loss':         'bars_in_loss',
    # Exit
    'exit_at_reverse_setup': 'exit_setup_reversal',
}

# Reverse mapping for quick lookup
PFLAB_TO_PINE = {v: k for k, v in PINE_TO_PFLAB.items()}

# PineScript SL Mode display -> PF AI Lab internal
SL_MODE_MAP = {
    'No':              'no',
    'Static - %':      'static_pct',
    'Red Line':        'red_line',
    'ATR-Based':       'atr_based',
}

# PineScript PT Mode display -> PF AI Lab internal
PT_MODE_MAP = {
    'No':                   'no',
    'Static - %':           'static_pct',
    'Static - Red Line %':  'red_line_pct',
    'SL Ratio':             'sl_ratio',
}

# PineScript BE Mode display -> PF AI Lab internal
BE_MODE_MAP = {
    'No':             'no',
    'BE + TP':        'be_only',
    'BE + TP + TSL':  'be_tsl',
    'BE + TPx2 + TSL': 'be_tsl',
}

# Mode fields that need value transformation
MODE_FIELDS = {
    'sl_mode': SL_MODE_MAP,
    'tp_mode': PT_MODE_MAP,
    'be_mode': BE_MODE_MAP,
}

# Alert config prefix - used to detect config in alert text
ALERT_CONFIG_PREFIX = 'PFLAB_CONFIG:'


# =====================================================================
# PARAMETERS LIST - shared between export block and alert enrichment
# Format: (pine_var, json_key, type)
# type: 's'=string, 'i'=int, 'f'=float, 'b'=bool
# =====================================================================
_EXPORT_PARAMS = [
    # PMax
    ('mav',                  'pmax_ma_type',      's'),
    ('length',               'pmax_length',       'i'),
    ('Multiplier',           'pmax_multiplier',   'f'),
    # Entry
    ('entry_type',           'entry_type',        's'),
    ('long_side',            'long_side',         'b'),
    ('short_side',           'short_side',        'b'),
    ('order_pause',          'order_pause',       'i'),
    # Donchian
    ('don_length',           'don_length',        'i'),
    # RSI Divergence
    ('rsi_div_length',       'rsi_div_length',    'i'),
    ('rsi_div_pivot_lookback', 'rsi_div_pivot',   'i'),
    ('rsi_div_detect_hidden', 'rsi_div_hidden',   'b'),
    # Fractal
    ('fractal_buffer_point', 'fractal_buffer',    'f'),
    # Boll+SMA
    ('length_bb_ma',         'length_bb_ma',      'i'),
    ('mult_bb_ma',           'mult_bb_ma',        'f'),
    ('length_bb_sma',        'length_bb_sma',     'i'),
    # MM Cross
    ('mm_cross_ma1_type',    'mm_cross_ma1_type', 's'),
    ('mm_cross_ma1_len',     'mm_cross_ma1_len',  'i'),
    ('mm_cross_ma2_type',    'mm_cross_ma2_type', 's'),
    ('mm_cross_ma2_len',     'mm_cross_ma2_len',  'i'),
    # LT Trend
    ('lt_trend_filter',      'use_lt_filter',     'b'),
    ('lt_trend_filter_tf',   'lt_tf',             's'),
    ('lt_trend_filter_mav',  'lt_ma_type',        's'),
    ('lt_trend_filter_length', 'lt_length',       'i'),
    ('lt_trend_filter_mult', 'lt_multiplier',     'f'),
    # MT Trend
    ('mt_trend_filter',      'use_mt_filter',     'b'),
    ('mt_trend_filter_tf',   'mt_tf',             's'),
    ('mt_trend_filter_mav',  'mt_ma_type',        's'),
    ('mt_trend_filter_length', 'mt_length',       'i'),
    ('mt_trend_filter_mult', 'mt_multiplier',     'f'),
    # RSI filters
    ('snab_rsi_50_Filter',   'use_rsi50_filter',  'b'),
    # Diff MA/Red
    ('diff_ma_red_filter',   'use_diff_ma_red',   'b'),
    ('diff_ma_red_filter_percentage', 'diff_ma_red_pct', 'f'),
    # MA Dir
    ('ma_dir_filter',        'use_ma_dir_filter', 'b'),
    ('ma_dir_len',           'ma_dir_len',        'i'),
    # MAvg
    ('mavg_filter',          'use_mavg_filter',   'b'),
    # Diff Price/Red
    ('diff_ma_price_filter', 'use_diff_price_red', 'b'),
    ('diff_ma_price_filter_percentage', 'diff_price_red_pct', 'f'),
    # ADX Regime
    ('use_adx_regime_filter', 'use_adx_regime_filter', 'b'),
    ('adx_hint_length',      'adx_period',        'i'),
    ('adx_trend_threshold',  'adx_trend_threshold', 'i'),
    # QQE
    ('qq_estimation_filter', 'use_qqe_filter',    'b'),
    # VIX
    ('vix_filter',           'use_vix_filter',    'b'),
    ('vix_threshold_high',   'vix_threshold_high', 'i'),
    # Vol ATR Percentile
    ('vol_adaptive_filter',  'use_vol_filter',    'b'),
    ('vol_percentile',       'vol_percentile_threshold', 'i'),
    # Max Hold Days
    ('max_hold_days_filter', 'use_max_hold_days', 'b'),
    ('max_hold_days',        'max_hold_days',     'i'),
    # Macro
    ('macro_filter',         'use_macro_filter',  'b'),
    # -- Risk Management --
    ('sl_mode',              'sl_mode',           's'),
    ('sl_percent',           'sl_pct',            'f'),
    ('atr_sl_mult',          'atr_sl_mult',       'f'),
    ('sl_dynamic_adjust',    'sl_dynamic',        'b'),
    ('sl_vol_min_mult',      'sl_vol_min_mult',   'f'),
    ('sl_vol_max_mult',      'sl_vol_max_mult',   'f'),
    # TP
    ('pt_mode',              'tp_mode',           's'),
    ('pt_percent',           'tp_pct',            'f'),
    ('pt_rr_ratio',          'tp_sl_ratio',       'f'),
    # Partial TP
    ('partial_tp',           'use_partial_tp',    'b'),
    ('tp1_pct',              'tp1_pct',           'f'),
    ('tp1_size',             'tp1_size',          'i'),
    ('tp2_pct',              'tp2_pct',           'f'),
    ('tp2_size',             'tp2_size',          'i'),
    # BE
    ('be_mode',              'be_mode',           's'),
    ('be_percent',           'be_trigger_pct',    'f'),
    ('dynamic_tsl',          'use_tsl',           'b'),
    ('tsl_atr_mult',         'tsl_atr_mult',      'f'),
    # Exit
    ('exit_at_reverse_setup', 'exit_setup_reversal', 'b'),
]


def _dedupe_params(params):
    """Remove duplicate json_keys from params list."""
    seen = set()
    unique = []
    for p in params:
        if p[1] not in seen:
            seen.add(p[1])
            unique.append(p)
    return unique


def _build_json_concat_lines(params, var_name='cfg_json', indent='    '):
    """
    Build PineScript string concatenation lines for JSON construction.

    Returns list of PineScript lines that build the JSON string.
    """
    lines = []
    lines.append(f'{indent}{var_name} = "{{"')

    for i, (pine_var, json_key, typ) in enumerate(params):
        comma = ', ' if i < len(params) - 1 else ''

        if typ == 's':
            lines.append(
                f'{indent}{var_name} := {var_name} + '
                f'\'"{json_key}": "\' + {pine_var} + \'"{comma}\''
            )
        elif typ == 'b':
            lines.append(
                f'{indent}{var_name} := {var_name} + '
                f'\'"{json_key}": \' + '
                f'({pine_var} ? "true" : "false") + \'{comma}\''
            )
        elif typ == 'i':
            lines.append(
                f'{indent}{var_name} := {var_name} + '
                f'\'"{json_key}": \' + '
                f'str.tostring({pine_var}) + \'{comma}\''
            )
        elif typ == 'f':
            lines.append(
                f'{indent}{var_name} := {var_name} + '
                f'\'"{json_key}": \' + '
                f'str.tostring({pine_var}, "#.####") + \'{comma}\''
            )

    lines.append(f'{indent}{var_name} := {var_name} + "}}"')
    return lines


def generate_pine_alert_block() -> str:
    """
    Generate PineScript V5 code block that ENRICHES alert messages
    with the full configuration JSON.

    The code:
    1. Builds a cfg_json string from all input variables
    2. Appends it to every alert message with the PFLAB_CONFIG: prefix
    3. Optionally displays it in a table on the chart

    The alert format becomes:
        LicenseID,buy,BTCUSDT,sl=123.45,risk=1.0
        PFLAB_CONFIG:{"pmax_ma_type":"EMA","pmax_length":10, ...}

    This way, every alert automatically carries the full config.
    The user simply pastes the alert text into PF AI Lab.

    Returns the PineScript code to INSERT into PF_Algo.
    """
    params = _dedupe_params(_EXPORT_PARAMS)

    lines = []
    lines.append('')
    lines.append('// ===================================================================')
    lines.append('// PF AI Lab -- ALERT CONFIG EXPORT V2')
    lines.append('// Chaque alerte contient automatiquement la config JSON complete.')
    lines.append('// Collez le texte de n\'importe quelle alerte dans PF AI Lab.')
    lines.append('// ===================================================================')
    lines.append('')
    lines.append("export_config_in_alerts = input.bool(true, "
                 "'Inclure config JSON dans les alertes', "
                 "group='Debug & Export', "
                 "tooltip='Ajoute automatiquement le JSON de la config a chaque "
                 "alerte.\\nCollez le texte d\\'une alerte dans PF AI Lab "
                 "(page Test) pour importer la config.')")
    lines.append("show_config_table = input.bool(false, "
                 "'Afficher la table config JSON', "
                 "group='Debug & Export', "
                 "tooltip='Affiche le JSON dans une table sur le graphique "
                 "(pour copie manuelle).')")
    lines.append('')

    # Build config JSON variable
    lines.append('// Build config JSON string')
    lines.append('var string cfg_json = ""')
    lines.append('if barstate.islast')
    lines.extend(_build_json_concat_lines(params, 'cfg_json', '    '))
    lines.append('')

    # Enrich alert messages
    lines.append('// Enrich alert messages with config JSON')
    lines.append(f'config_suffix = export_config_in_alerts '
                 f'? "\\n{ALERT_CONFIG_PREFIX}" + cfg_json : ""')
    lines.append('')
    lines.append('// --- REPLACE your alert message lines with these: ---')
    lines.append('// For buy alerts:')
    lines.append('//   buy_alert_msg := buy_alert_msg + config_suffix')
    lines.append('// For sell alerts:')
    lines.append('//   sell_alert_msg := sell_alert_msg + config_suffix')
    lines.append('// For close alerts:')
    lines.append('//   buy_close_alert_msg  := buy_close_alert_msg + config_suffix')
    lines.append('//   sell_close_alert_msg := sell_close_alert_msg + config_suffix')
    lines.append('')

    # Display table (optional)
    lines.append('// Optional: display config JSON in table')
    lines.append('if show_config_table and barstate.islast')
    lines.append('    n_rows = math.ceil(str.length(cfg_json) / 120)')
    lines.append('    var table json_table = table.new(position.bottom_left, '
                 '1, 25, bgcolor=color.new(color.black, 20), '
                 'border_width=1, border_color=color.gray)')
    lines.append('    table.cell(json_table, 0, 0, '
                 '"PF AI Lab Config JSON -- Copiez ou utilisez les alertes", '
                 'text_color=color.yellow, bgcolor=color.new(color.navy, 0), '
                 'text_size=size.small)')
    lines.append('    row_idx = 1')
    lines.append('    pos = 0')
    lines.append('    while pos < str.length(cfg_json) and row_idx < 25')
    lines.append('        chunk = str.substring(cfg_json, pos, '
                 'math.min(pos + 120, str.length(cfg_json)))')
    lines.append('        table.cell(json_table, 0, row_idx, chunk, '
                 'text_color=color.lime, text_size=size.tiny, '
                 'text_halign=text.align_left)')
    lines.append('        row_idx := row_idx + 1')
    lines.append('        pos := pos + 120')

    return '\n'.join(lines)


# Keep backward compat
generate_pine_export_block = generate_pine_alert_block


def generate_pine_patch_instructions() -> str:
    """
    Generate step-by-step instructions for patching PF_Algo to include
    config JSON in alerts. This is a MINIMAL change (4 lines).

    Returns instructions as plain text.
    """
    return """
INSTRUCTIONS: Ajouter le config JSON dans les alertes PF_Algo
================================================================

Methode rapide (4 lignes a ajouter) :

1. Cherchez dans votre PF_Algo la section "// ALERTS /{"
   (vers la ligne 2347)

2. JUSTE APRES la construction des alert_msg, AVANT les strategy.close,
   ajoutez ces lignes :

   // --- PF AI Lab: config dans les alertes ---
   config_suffix = "\\nPFLAB_CONFIG:" + cfg_json
   buy_alert_msg  := buy_alert_msg + config_suffix
   sell_alert_msg := sell_alert_msg + config_suffix
   buy_close_alert_msg  := buy_close_alert_msg + config_suffix
   sell_close_alert_msg := sell_close_alert_msg + config_suffix

3. Ajoutez aussi le bloc de construction de cfg_json AVANT la section
   ALERTS. Copiez le code genere par PF AI Lab (bouton "Code PineScript").

4. Sauvegardez. Desormais chaque alerte contient le JSON de la config.
   Collez le texte d'une alerte dans PF AI Lab pour l'importer.
"""


def parse_alert_with_config(alert_text: str) -> dict:
    """
    Parse alert text that may contain a PFLAB_CONFIG: line.

    Handles multiple formats:
    1. Alert with embedded config:
       "LicenseID,buy,BTCUSDT,sl=123.45,risk=1.0
        PFLAB_CONFIG:{...json...}"
    2. Raw JSON (direct paste)
    3. JSON with surrounding text (table copy)
    4. Webhook JSON payload (from TradingView webhook)

    Returns:
        dict with:
        - config: PF AI Lab config dict
        - alert_command: parsed PineConnector command (if present)
        - source: 'alert', 'json', 'webhook', or 'unknown'
        - n_parsed: number of config keys found
        - error: error message or None
    """
    text = alert_text.strip()
    if not text:
        return {'config': {}, 'alert_command': None,
                'source': 'unknown', 'n_parsed': 0,
                'error': 'Empty input'}

    # Clean up common artifacts
    text = text.replace('\u200b', '').replace('\xa0', ' ')

    # Method 1: Check for PFLAB_CONFIG: prefix in alert text
    config_match = re.search(
        rf'{re.escape(ALERT_CONFIG_PREFIX)}\s*(\{{.*?\}})',
        text, re.DOTALL
    )
    if config_match:
        json_str = config_match.group(1)
        alert_cmd = _parse_pineconnector_command(text)
        config = _parse_json_to_config(json_str)
        if config is not None:
            return {
                'config': config,
                'alert_command': alert_cmd,
                'source': 'alert',
                'n_parsed': len(config),
                'error': None,
            }

    # Method 2: Check for webhook JSON payload
    # TradingView webhooks can send JSON like {"action":"buy","config":{...}}
    webhook_config = _try_parse_webhook(text)
    if webhook_config:
        return {
            'config': webhook_config,
            'alert_command': None,
            'source': 'webhook',
            'n_parsed': len(webhook_config),
            'error': None,
        }

    # Method 3: Try to find raw JSON in text
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        config = _parse_json_to_config(json_match.group(0))
        if config is not None:
            alert_cmd = _parse_pineconnector_command(text)
            return {
                'config': config,
                'alert_command': alert_cmd,
                'source': 'json',
                'n_parsed': len(config),
                'error': None,
            }

    # Method 4: Try multi-line JSON (copied from table rows)
    joined = re.sub(r'\s*\n\s*', '', text)
    json_match = re.search(r'\{[^{}]*\}', joined, re.DOTALL)
    if json_match:
        config = _parse_json_to_config(json_match.group(0))
        if config is not None:
            return {
                'config': config,
                'alert_command': None,
                'source': 'json',
                'n_parsed': len(config),
                'error': None,
            }

    return {
        'config': {},
        'alert_command': _parse_pineconnector_command(text),
        'source': 'unknown',
        'n_parsed': 0,
        'error': 'No configuration JSON found in the pasted text. '
                 'Make sure your PF_Algo includes the config export patch.',
    }


def _parse_pineconnector_command(text: str) -> dict | None:
    """
    Parse a PineConnector command from alert text.
    Format: LicenseID,action,SYMBOL[,sl=X][,tp=X][,risk=X]

    Returns dict with action, symbol, sl, tp, risk or None.
    """
    # First line is typically the PineConnector command
    first_line = text.strip().split('\n')[0].strip()

    # Split by comma to parse fields
    parts = [p.strip() for p in first_line.split(',')]
    if len(parts) < 3:
        return None

    # First part must be a numeric license ID
    if not parts[0].isdigit():
        return None

    # Second part must be a valid action
    action = parts[1].lower()
    valid_actions = {'buy', 'sell', 'closelong', 'closeshort',
                     'closelongshort', 'closelongpct', 'closeshortpct'}
    if action not in valid_actions:
        return None

    # Third part is the symbol
    symbol = parts[2]

    cmd = {
        'license_id': parts[0],
        'action': action,
        'symbol': symbol,
    }

    # Parse optional key=value params from remaining parts
    for part in parts[3:]:
        if '=' in part:
            key, _, val = part.partition('=')
            key = key.strip().lower()
            try:
                cmd[key] = float(val.strip())
            except ValueError:
                cmd[key] = val.strip()

    return cmd


def _try_parse_webhook(text: str) -> dict | None:
    """
    Try to parse a TradingView webhook JSON payload.
    Possible formats:
    - {"action":"buy", "config": {...}}
    - {"PFLAB_CONFIG": {...}}
    """
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None

        # Check for nested config
        if 'config' in data and isinstance(data['config'], dict):
            return _transform_config(data['config'])
        if 'PFLAB_CONFIG' in data and isinstance(data['PFLAB_CONFIG'], dict):
            return _transform_config(data['PFLAB_CONFIG'])

        # Maybe the whole payload IS the config
        if 'entry_type' in data or 'pmax_ma_type' in data:
            return _transform_config(data)

        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_json_to_config(json_str: str) -> dict | None:
    """
    Parse a JSON string and transform it into a PF AI Lab config.
    Handles PineScript formatting artifacts.
    """
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError:
        # Fix common PineScript artifacts: "3." -> "3.0"
        fixed = re.sub(r':\s*(\d+)\.\s*([,}])', r': \1.0\2', json_str)
        try:
            raw = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    if not isinstance(raw, dict) or not raw:
        return None

    return _transform_config(raw)


def _transform_config(raw: dict) -> dict:
    """
    Transform raw config values:
    - SL/PT/BE mode value mappings (TV display -> internal)
    - Boolean coercion ("true"/"false" -> bool)
    """
    config = {}
    for key, val in raw.items():
        # Mode transformations
        if key in MODE_FIELDS and isinstance(val, str):
            mode_map = MODE_FIELDS[key]
            config[key] = mode_map.get(val, val.lower().replace(' ', '_'))
        # Boolean coercion
        elif isinstance(val, str) and val.lower() in ('true', 'false'):
            config[key] = val.lower() == 'true'
        else:
            config[key] = val
    return config


# Backward compat alias
def parse_pine_json(json_text: str) -> dict:
    """
    Parse JSON/alert text. Wraps parse_alert_with_config for backward compat.
    """
    result = parse_alert_with_config(json_text)
    return {
        'config': result['config'],
        'n_parsed': result['n_parsed'],
        'error': result['error'],
    }


def generate_tv_mapping_guide() -> list:
    """
    Generate a mapping guide showing TV Pine variable -> PF AI Lab JSON key.
    """
    sections = {
        'PMax': ['pmax_multiplier', 'pmax_ma_type', 'pmax_length'],
        'Entry': ['entry_type', 'long_side', 'short_side', 'order_pause'],
        'Entry: Donc+Chikou': ['don_length'],
        'Entry: RSI Divergence': ['rsi_div_length', 'rsi_div_pivot',
                                   'rsi_div_hidden'],
        'Entry: Fractal': ['fractal_buffer'],
        'Entry: Boll+SMA': ['length_bb_ma', 'mult_bb_ma', 'length_bb_sma'],
        'Entry: MM Cross': ['mm_cross_ma1_type', 'mm_cross_ma1_len',
                            'mm_cross_ma2_type', 'mm_cross_ma2_len'],
        'LT Trend Filter': ['use_lt_filter', 'lt_tf', 'lt_ma_type',
                            'lt_length', 'lt_multiplier'],
        'MT Trend Filter': ['use_mt_filter', 'mt_tf', 'mt_ma_type',
                            'mt_length', 'mt_multiplier'],
        'Filters': ['use_rsi50_filter', 'use_diff_ma_red', 'diff_ma_red_pct',
                    'use_ma_dir_filter', 'ma_dir_len', 'use_mavg_filter',
                    'use_diff_price_red', 'diff_price_red_pct'],
        'ADX Regime': ['use_adx_regime_filter', 'adx_period',
                       'adx_trend_threshold'],
        'Security Layers': ['use_vix_filter', 'vix_threshold_high',
                           'use_vol_filter', 'vol_percentile_threshold',
                           'use_max_hold_days', 'max_hold_days'],
        'Risk: SL': ['sl_mode', 'sl_pct', 'atr_sl_mult', 'sl_dynamic',
                     'sl_vol_min_mult', 'sl_vol_max_mult'],
        'Risk: TP': ['tp_mode', 'tp_pct', 'tp_sl_ratio', 'use_partial_tp',
                     'tp1_pct', 'tp1_size', 'tp2_pct', 'tp2_size'],
        'Risk: BE/TSL': ['be_mode', 'be_trigger_pct', 'use_tsl',
                         'tsl_atr_mult'],
        'Other': ['exit_setup_reversal', 'use_macro_filter',
                  'use_qqe_filter'],
    }

    guide = []
    for section, keys in sections.items():
        for json_key in keys:
            pine_var = PFLAB_TO_PINE.get(json_key, '?')
            guide.append({
                'section': section,
                'pine_var': pine_var,
                'json_key': json_key,
            })
    return guide
