"""
Configuration Parameters — Full V.69.2 parameter set with sensitivity metadata.

Sensitivity zones:
  - SENSITIVE: Small changes cause large signal divergence between Pine/Python
  - ROBUST: Parameters in this range give reproducible results cross-engine

Rule of thumb:
  If optimizer converges to EMA(6)×2.2 → works in Python, diverges in TV.
  Prefer EMA(15+)×3.0+ for cross-engine reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json


@dataclass
class ParamMeta:
    """Metadata for a single parameter."""
    name: str
    default: Any
    min_val: Any = None
    max_val: Any = None
    sensitive_zone: str = ''  # e.g., "< 10"
    robust_zone: str = ''    # e.g., ">= 15"
    description: str = ''
    pine_name: str = ''      # Corresponding Pine input name
    category: str = ''       # setup, pmax, entry, filter, sl, risk


# Sensitivity matrix from PineGuard spec
SENSITIVITY_MATRIX = {
    'pmax_length': {
        'sensitive': '< 10', 'robust': '>= 15',
        'reason': 'EMA courte = α élevé = très réactif aux écarts',
    },
    'multiplier': {
        'sensitive': '< 2.5', 'robust': '>= 3.0',
        'reason': 'PMax proche du prix = crossovers fréquents "sur le fil"',
    },
    'don_length': {
        'sensitive': '< 15', 'robust': '>= 20',
        'reason': 'Donchian basé sur moins de barres = plus volatile',
    },
    'sl_pct': {
        'sensitive': '< 2%', 'robust': '3-8%',
        'reason': 'SL serré = touché par micro-écarts de High/Low',
    },
    'lt_filter_tf': {
        'sensitive': 'Daily', 'robust': 'Weekly',
        'reason': 'Weekly = 5× moins de flip opportunities',
    },
    'mt_filter_tf': {
        'sensitive': '1H (same TF)', 'robust': '4H+',
        'reason': 'Même TF = pas de lissage',
    },
}


@dataclass
class PineGuardConfig:
    """
    Complete configuration matching V.69.2 inputs.

    Organized in 7 blocks:
      1. Setup (instrument, timeframe)
      2. PMax (MA type, length, multiplier)
      3. Entries (entry type + type-specific params, including MM Cross V69.2)
      4. Filters (LT, MT, RSI, Macro, ADX)
      5. Security Layers (0-5)
      6. Timing (session, timezone)
      7. Risk (SL, position sizing)
    """

    # === Block 1: Setup ===
    instrument: str = 'US500'
    timeframe: str = '1H'
    data_source: str = 'mt5'  # 'mt5' or 'tv'

    # === Block 2: PMax ===
    pmax_length: int = 10
    pmax_multiplier: float = 3.0
    pmax_ma_type: str = 'EMA'  # SMA, EMA, RMA, WMA, VWMA, DEMA, TEMA, HMA, ZLEMA

    # === Block 3: Entries ===
    entry_type: str = 'Donc+Chikou'  # One of 7 types (V69.2: +MM Cross)

    # Donchian+Chikou params
    don_length: int = 20
    chikou_shift: int = 26

    # Bollinger+SMA params
    boll_length: int = 20
    boll_mult: float = 2.0
    boll_sma_length: int = 50

    # RSI Divergence params
    rsi_div_length: int = 14
    rsi_div_lookback: int = 5
    rsi_div_max_distance: int = 100

    # Fractal params
    fractal_patterns: str = 'all'  # 'all', 'classic', '1,2,3'

    # VWAP params
    vwap_session_start: int = 0

    # ORB params
    orb_minutes: int = 30
    orb_session_hour: int = 9
    orb_session_minute: int = 30

    # MM Cross params (V69.2)
    mm_cross_ma1_type: str = 'EMA'   # 'EMA' or 'TRIMA'
    mm_cross_ma1_len: int = 9
    mm_cross_ma2_type: str = 'EMA'   # 'EMA' or 'TRIMA'
    mm_cross_ma2_len: int = 21

    # === Block 4: Filters ===
    lt_filter_enabled: bool = True
    lt_filter_tf: str = '1W'
    lt_pmax_length: int = 10
    lt_pmax_mult: float = 3.0
    lt_pmax_ma_type: str = 'EMA'

    mt_filter_enabled: bool = True
    mt_filter_tf: str = '4H'
    mt_pmax_length: int = 10
    mt_pmax_mult: float = 3.0
    mt_pmax_ma_type: str = 'EMA'

    rsi_filter_enabled: bool = False
    rsi_filter_length: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    adx_filter_enabled: bool = False
    adx_length: int = 14
    adx_smoothing: int = 14
    adx_threshold: float = 25.0

    macro_filter_enabled: bool = False  # Not implemented in Python

    # === Block 5: Security Layers ===
    security_layer: int = 0  # 0-5, not yet implemented in Python

    # === Block 6: Timing ===
    timezone_offset: int = 0  # UTC offset
    session_start: str = '00:00'
    session_end: str = '23:59'

    # === Block 7: Risk ===
    sl_mode: str = 'atr14'  # 'atr14', 'atr50', 'fixed_pct', 'none'
    sl_pct: float = 5.0
    sl_atr_mult: float = 1.5
    initial_capital: float = 100000.0
    position_size_pct: float = 1.0
    enable_reversal: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> 'PineGuardConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    @classmethod
    def from_json(cls, s: str) -> 'PineGuardConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(s))

    def sensitivity_warnings(self) -> List[str]:
        """Check parameters against sensitivity matrix and return warnings."""
        warnings = []

        if self.pmax_length < 10:
            warnings.append(
                f"⚠️ pmax_length={self.pmax_length} is in SENSITIVE zone (< 10). "
                f"Prefer >= 15 for cross-engine reproducibility."
            )
        if self.pmax_multiplier < 2.5:
            warnings.append(
                f"⚠️ multiplier={self.pmax_multiplier} is in SENSITIVE zone (< 2.5). "
                f"Prefer >= 3.0 for stable crossover detection."
            )
        if self.don_length < 15:
            warnings.append(
                f"⚠️ don_length={self.don_length} is in SENSITIVE zone (< 15). "
                f"Prefer >= 20 for robust Donchian channels."
            )
        if self.sl_pct < 2.0:
            warnings.append(
                f"⚠️ sl_pct={self.sl_pct}% is in SENSITIVE zone (< 2%). "
                f"Prefer 3-8% to avoid SL hits from micro-differences."
            )
        if self.entry_type not in ['Donc+Chikou', 'RSI Divergence', 'MM Cross']:
            warnings.append(
                f"⚠️ entry_type='{self.entry_type}' is NOT in SAFE_ENTRY_TYPES. "
                f"Results may diverge significantly from TradingView."
            )
        if self.data_source == 'mt5':
            warnings.append(
                f"⚠️ data_source='mt5' — Bid vs Mid-price divergence (D1). "
                f"Recommend 'tv' for better Pine coherence."
            )

        return warnings


# Default configuration
DEFAULT_CONFIG = PineGuardConfig()


def create_config(**kwargs) -> PineGuardConfig:
    """Create a config with custom parameters."""
    return PineGuardConfig(**kwargs)


def validate_config(config: PineGuardConfig) -> Dict:
    """
    Validate a configuration and return warnings/errors.

    Returns:
        Dict with 'valid', 'warnings', 'errors'
    """
    errors = []
    warnings = config.sensitivity_warnings()

    # Validation rules
    if config.pmax_length < 1:
        errors.append("pmax_length must be >= 1")
    if config.pmax_multiplier <= 0:
        errors.append("multiplier must be > 0")
    if config.don_length < 1:
        errors.append("don_length must be >= 1")
    if config.boll_length < 1:
        errors.append("boll_length must be >= 1")
    if config.initial_capital <= 0:
        errors.append("initial_capital must be > 0")
    if config.position_size_pct <= 0 or config.position_size_pct > 100:
        errors.append("position_size_pct must be between 0 and 100")
    if config.security_layer < 0 or config.security_layer > 5:
        errors.append("security_layer must be 0-5")

    # Params/trades ratio warning
    active_params = sum([
        3,  # PMax (length, mult, ma_type) — always active
        2 if config.entry_type == 'Donc+Chikou' else 0,
        3 if config.entry_type == 'Boll+SMA' else 0,
        3 if config.entry_type == 'RSI Divergence' else 0,
        3 if config.lt_filter_enabled else 0,
        3 if config.mt_filter_enabled else 0,
        2 if config.rsi_filter_enabled else 0,
        2 if config.adx_filter_enabled else 0,
        2,  # SL params
    ])
    if active_params > 15:
        warnings.append(
            f"⚠️ {active_params} active parameters — risk of overfitting. "
            f"Ensure params/trades ratio < 0.15"
        )

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'active_params': active_params,
    }
