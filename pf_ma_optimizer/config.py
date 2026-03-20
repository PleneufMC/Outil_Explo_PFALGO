"""
PF AI Lab 6.0.3 — Configuration & Constants
All parameter ranges, defaults, and gates for the exploration tool.

V5.6 CHANGES (Divergence Reduction):
  - Runtime guard: pmax_ma_mode forced to 'classic' in backtest engine
  - Runtime guard: use_ma_cross_signal forced OFF in backtest engine
  - Direction column added to Explore results table
  - Estimated AI Lab vs TV divergence reduced to <15% (within 25% tolerance)

V5.4.9 CHANGES (Performance Divergence Audit):
  - FIX #1: VIX/Macro filters REMOVED from search space (phantom filters —
    optimizer explored them but _vix_series/_macro_scores were NEVER provided,
    so Python silently ignored them while TradingView actively filters)
  - FIX #3: Fib filter explicitly disabled in defaults AND fixed_params (was
    contradictory: OFF in FOCUSED_FIXED but ON in get_default_config)
  - FIX #4: Risk Management LOCKED — SL=30% static, TP=no, BE=no.
    RM is NOT optimized by AI Lab. User validates RM exclusively on TradingView.
    Philosophy: "SL Plein" = SL is a safety net (rarely hit), exits come from
    setup reversal or filter signals, not tight stops.
  - FIX #5: annual_bars made configurable (was hardcoded 252*24, wrong for D/H4)
  - FIX #6: Perturbation robustness increased (5→20 runs, 10→15%)

V5.4 CHANGES:
  - SAFE_ENTRY_TYPES: imported from pineguard (single source of truth)
  - get_transaction_cost(): partial matching for timeframe suffixes
  - Stock CFD transaction costs added
  - WFE calculation fix (floor/cap) in backtest_db.py

V5.3 ADDITIONS (Tier 3):
  - Session/Timing Filter: day-of-week + session hours
  - VIX Filter (Security Layer 0): external volatility gate
  - Partial Profit Taking: TP1/TP2 levels with position sizing
  - QQ Estimation Filter: exact Pine V69.2 port
  - Macro Filter Simplifie: reduced RS + Climate score

RISK MANAGEMENT PHILOSOPHY ("SL Plein"):
  The PF Algo strategy philosophy is that trades should exit via:
    1. Setup Reversal (PMax direction change) — PREFERRED
    2. Filter-driven exits (trend filters flip) — PREFERRED
    3. SL Plein (stop-loss hit) — SAFETY NET ONLY
  A "SL Plein" means the full stop-loss is hit (e.g. 1% SL actually triggered).
  The strategy is sized so that very few trades hit their SL. Tight SL (like
  ATR*1) contradicts this philosophy and produces catastrophic results on TV.
  Therefore, AI Lab does NOT optimize SL/TP/BE — it uses SL=30% (filet de
  securite) and lets the user tune RM exclusively in TradingView.
"""

# ==============================================================================
# INSTRUMENT TRANSACTION COSTS (round-turn: spread + slippage)
# Full FTMO instrument list — all symbols available on FTMO MT5
# ==============================================================================
TRANSACTION_COSTS = {
    # ── Forex Majors ──
    'EURUSD':  0.00015,
    'GBPUSD':  0.00020,
    'USDJPY':  0.015,
    'USDCHF':  0.00020,
    'USDCAD':  0.00020,
    'AUDUSD':  0.00018,
    'NZDUSD':  0.00020,
    # ── Forex Minors / Crosses ──
    'EURGBP':  0.00020,
    'EURJPY':  0.020,
    'EURCHF':  0.00025,
    'EURAUD':  0.00025,
    'EURCAD':  0.00025,
    'EURNZD':  0.00035,
    'GBPJPY':  0.030,
    'GBPCHF':  0.00030,
    'GBPAUD':  0.00030,
    'GBPNZD':  0.00045,
    'GBPCAD':  0.00030,
    'AUDJPY':  0.020,
    'AUDNZD':  0.00025,
    'AUDCAD':  0.00025,
    'AUDCHF':  0.00025,
    'NZDJPY':  0.020,
    'NZDCAD':  0.00030,
    'NZDCHF':  0.00030,
    'CADJPY':  0.020,
    'CADCHF':  0.00025,
    'CHFJPY':  0.025,
    # ── Forex Exotics ──
    'EURTRY':  0.50,
    'USDTRY':  0.40,
    'EURPLN':  0.0040,
    'USDPLN':  0.0035,
    'EURNOK':  0.0060,
    'USDNOK':  0.0050,
    'EURSEK':  0.0060,
    'USDSEK':  0.0050,
    'USDMXN':  0.020,
    'USDZAR':  0.015,
    'EURCZK':  0.10,
    'EURHUF':  0.50,
    'USDSGD':  0.00030,
    'GBPSGD':  0.00050,
    'USDHKD':  0.0030,
    'USDCNH':  0.0040,
    # ── Indices (Cash CFDs) ──
    'US30':    3.0,
    'US100':   1.0,
    'US500':   0.5,
    'UK100':   1.5,
    'DEU40':   1.5,
    'GER40':   1.5,      # alias for DEU40
    'FRA40':   1.0,
    'ESP35':   5.0,
    'JPN225':  10.0,
    'AUS200':  2.0,
    'EUSTX50': 2.0,
    'HK50':    8.0,
    'CHINA50': 10.0,
    'US2000':  1.0,
    'SUI20':   2.0,
    'CHINAH':  8.0,
    'NTH25':   2.0,
    # ── Metals ──
    'XAUUSD':  0.35,
    'XAGUSD':  0.025,
    'XPTUSD':  1.50,
    'XPDUSD':  5.00,
    # ── Energy ──
    'USOIL':   0.04,
    'UKOIL':   0.04,
    'NATGAS':  0.010,
    # ── Crypto ──
    'BTCUSD':  30.0,
    'ETHUSD':  2.0,
    'LTCUSD':  0.50,
    'XRPUSD':  0.005,
    'ADAUSD':  0.003,
    'DOTUSD':  0.05,
    'DOGEUSD': 0.001,
    'SOLUSD':  0.30,
    'BNBUSD':  1.00,
    'AVAXUSD': 0.10,
    'LINKUSD': 0.05,
    'XLMUSD':  0.002,
    'AAVEUSD': 0.50,
    'XMRUSD':  0.50,
    'NEOUSD':  0.05,
    'DASHUSD': 0.30,
    'MATICUSD': 0.003,
    'ATOMUSD': 0.03,
    'ALGOUSD': 0.002,
    'ICPUSD':  0.05,
    'FILUSD':  0.03,
    'APTUSD':  0.03,
    'ARBUSD':  0.003,
    'OPUSD':   0.005,
    'NEARUSD': 0.02,
    'FTMUSD':  0.003,
    'TRXUSD':  0.001,
    'UNIUSD':  0.03,
    'TONUSD':  0.02,
    'PEPEUSD': 0.0000005,
    'SUIUSD':  0.005,
    'SHIBAUSD': 0.0000001,
    # ── Stock CFDs (per share, approximate) ──
    'NVDA':    0.10,
    'AMZN':    0.15,
    'AAPL':    0.08,
    'TSLA':    0.15,
    'MSFT':    0.08,
    'GOOG':    0.10,
    'META':    0.10,
}

DEFAULT_TRANSACTION_COST = 1.0  # fallback for unknown instruments


def get_transaction_cost(instrument: str, config: dict = None) -> float:
    """Get transaction cost with partial matching for timeframe suffixes.

    Handles cases like 'NVDA H4' → matches 'NVDA', 'EURUSD M15' → matches 'EURUSD'.
    """
    # 1. Explicit override from config
    if config:
        custom = config.get('custom_transaction_cost')
        if custom:
            return custom
    # 2. Exact match
    inst = instrument.upper().strip()
    if inst in TRANSACTION_COSTS:
        return TRANSACTION_COSTS[inst]
    # 3. Partial match: strip timeframe suffix ('NVDA H4' → 'NVDA')
    base = inst.split()[0] if ' ' in inst else inst
    if base in TRANSACTION_COSTS:
        return TRANSACTION_COSTS[base]
    return DEFAULT_TRANSACTION_COST


def register_custom_instrument(name: str, cost: float = None, group: str = 'Custom'):
    """
    Register a custom instrument at runtime (e.g. AAPL, MY_PORTFOLIO).
    Adds to TRANSACTION_COSTS and INSTRUMENT_GROUPS so it appears in dropdowns
    and gets the correct spread in backtests.
    """
    name = name.strip().upper()
    if not name:
        return
    if cost is not None and cost > 0:
        TRANSACTION_COSTS[name] = cost
    if group not in INSTRUMENT_GROUPS:
        INSTRUMENT_GROUPS[group] = []
    if name not in INSTRUMENT_GROUPS[group]:
        INSTRUMENT_GROUPS[group].append(name)

# Instrument categories for UI dropdown (grouped by asset class)
INSTRUMENT_GROUPS = {
    'Indices': ['US30', 'US100', 'US500', 'US2000', 'UK100', 'DEU40', 'GER40', 'FRA40',
                'ESP35', 'JPN225', 'AUS200', 'EUSTX50', 'HK50', 'CHINA50',
                'SUI20', 'CHINAH', 'NTH25'],
    'Forex Majors': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD'],
    'Forex Minors': ['EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD', 'EURNZD',
                     'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPNZD', 'GBPCAD',
                     'AUDJPY', 'AUDNZD', 'AUDCAD', 'AUDCHF',
                     'NZDJPY', 'NZDCAD', 'NZDCHF',
                     'CADJPY', 'CADCHF', 'CHFJPY'],
    'Forex Exotics': ['EURTRY', 'USDTRY', 'EURPLN', 'USDPLN', 'EURNOK', 'USDNOK',
                      'EURSEK', 'USDSEK', 'USDMXN', 'USDZAR', 'EURCZK', 'EURHUF',
                      'USDSGD', 'GBPSGD', 'USDHKD', 'USDCNH'],
    'Metals': ['XAUUSD', 'XAGUSD', 'XPTUSD', 'XPDUSD'],
    'Energy': ['USOIL', 'UKOIL', 'NATGAS'],
    'Crypto Majors': ['BTCUSD', 'ETHUSD', 'SOLUSD', 'BNBUSD', 'XRPUSD', 'ADAUSD'],
    'Crypto Altcoins': ['LTCUSD', 'DOTUSD', 'DOGEUSD', 'AVAXUSD', 'LINKUSD',
                        'XLMUSD', 'AAVEUSD', 'XMRUSD', 'NEOUSD', 'DASHUSD',
                        'MATICUSD', 'ATOMUSD', 'ALGOUSD', 'ICPUSD', 'FILUSD',
                        'APTUSD', 'ARBUSD', 'OPUSD', 'NEARUSD', 'FTMUSD',
                        'TRXUSD', 'UNIUSD', 'TONUSD', 'PEPEUSD', 'SUIUSD', 'SHIBAUSD'],
}

# ==============================================================================
# PMAX DEFAULTS
# ==============================================================================
PMAX_DEFAULTS = {
    'ma_type': 'EMA',
    'length': 10,
    'multiplier': 3.0,
}

MA_TYPES = ['SMA', 'EMA', 'WMA', 'TMA', 'TRIMA', 'VAR', 'WWMA', 'ZLEMA', 'TSF']

# ==============================================================================
# ENTRY TYPES — SAFE_ENTRY_TYPES imported from PineGuard (single source of truth)
# ==============================================================================
try:
    from pineguard import SAFE_ENTRY_TYPES  # V5.4: single source of truth
except ImportError:
    SAFE_ENTRY_TYPES = ['Donc+Chikou', 'RSI Divergence', 'Fractal', 'Boll+SMA', 'MM Cross']
ALL_ENTRY_TYPES = ['Fractal', 'Boll+SMA', 'VWAP', 'ORB', 'Donc+Chikou', 'RSI Divergence', 'MM Cross']

BUGGY_ENTRY_TYPES = {
    # Fractal: FIXED — 5/5 patterns, dedup with PMax signal reset, buffer support
    # Boll+SMA: FIXED — ddof=0 matching Pine ta.stdev, SMA signal line
    'VWAP': 'Session-dependent, non-reproducible in Python for CFDs',
    'ORB': 'Session-dependent, non-reproducible',
}

# ==============================================================================
# VALID TIMEFRAMES for LT / MT filters
# Must match HTF_RESAMPLE_MAP keys and PineScript V69.1 supported periods.
# ==============================================================================
LT_TIMEFRAMES = ['M', 'W', 'D', '4H', '2H', '1H']   # LT = Monthly down to 1H
MT_TIMEFRAMES = ['W', 'D', '4H', '2H', '1H', '45min', '30min', '15min']  # MT = any sub-LT TF

# ==============================================================================
# FOCUSED MODE SEARCH SPACE (recommended: ~12 free params)
#
# LT/MT Trend filter design (PineScript V69.1 alignment):
#   - FOCUSED: lt_tf='W', lt_length=20, mt_tf='4H', mt_length=10 are FIXED
#     (matches PineScript defaults; only multipliers are tunable)
#   - FULL: lt_tf, lt_length, mt_tf, mt_length, mt_ma_type all become tunable
#     (allows exploring alternative timeframes and MA lengths)
#   - lt_multiplier is ALWAYS tunable (ATR band width, low overfitting risk)
#
# V5.2 ADDITIONS (Roadmap Sprint 1-2):
#   - sl_mode: 'static_pct' | 'atr_based' (ATR-Based SL from Pine V69.2)
#   - sl_pct now VARIABLE per asset class (was fixed 30%)
#   - tp_mode: 'no' | 'static_pct' | 'sl_ratio' (Take Profit modes)
#   - use_adx_regime_filter: ADX Regime trending/ranging gate
#   - max_hold_days: Security Layer 3
#   - use_vol_filter: Security Layer 1 (ATR percentile volatility)
#   - be_mode: Break-Even + Trailing Stop
#   - long_side / short_side: optimizable direction
# ==============================================================================
FOCUSED_SEARCH_SPACE = {
    'pmax_length':       {'type': 'int',   'low': 5,   'high': 30},
    'pmax_multiplier':   {'type': 'float', 'low': 1.5, 'high': 5.0, 'step': 0.1},
    # ══ V5.4.9 FIX: pmax_ma_mode LOCKED to 'classic' ══
    # Modes 'mult_int', 'mult_dec', 'additive' do NOT exist in Pine Script V69.3.
    # Optimizing them creates configs impossible to replicate on TradingView,
    # which is a major source of AI Lab vs TV divergence.
    # ══ V5.4.9 FIX: use_ma_cross_signal REMOVED ══
    # This feature does NOT exist in Pine Script V69.3.
    # It added an independent MA crossover AND-filter that cannot be set on TV.
    'entry_type':        {'type': 'categorical', 'choices': SAFE_ENTRY_TYPES},
    # MM Cross params
    'mm_cross_ma1_type': {'type': 'categorical', 'choices': ['EMA', 'TRIMA'], 'condition': "entry_type == 'MM Cross'"},
    'mm_cross_ma1_len':  {'type': 'int', 'low': 3, 'high': 50, 'condition': "entry_type == 'MM Cross'"},
    'mm_cross_ma2_type': {'type': 'categorical', 'choices': ['EMA', 'TRIMA'], 'condition': "entry_type == 'MM Cross'"},
    'mm_cross_ma2_len':  {'type': 'int', 'low': 10, 'high': 200, 'condition': "entry_type == 'MM Cross'"},
    # Donc+Chikou params
    'don_length':        {'type': 'int',   'low': 10,  'high': 55, 'condition': "entry_type == 'Donc+Chikou'"},
    # RSI Divergence params
    'rsi_div_length':    {'type': 'int',   'low': 7,   'high': 30, 'condition': "entry_type == 'RSI Divergence'"},
    'rsi_div_pivot':     {'type': 'int',   'low': 2,   'high': 10, 'condition': "entry_type == 'RSI Divergence'"},
    'rsi_div_hidden':    {'type': 'categorical', 'choices': [True, False], 'condition': "entry_type == 'RSI Divergence'"},
    # Fractal params
    'fractal_buffer':    {'type': 'float', 'low': 0.0, 'high': 5.0, 'step': 0.5, 'condition': "entry_type == 'Fractal'"},
    # Boll+SMA params
    'length_bb_ma':      {'type': 'int',   'low': 10,  'high': 40, 'condition': "entry_type == 'Boll+SMA'"},
    'mult_bb_ma':        {'type': 'float', 'low': 1.0, 'high': 3.5, 'step': 0.1, 'condition': "entry_type == 'Boll+SMA'"},
    'length_bb_sma':     {'type': 'int',   'low': 20,  'high': 100, 'condition': "entry_type == 'Boll+SMA'"},
    # ── LT / MT Filters ──
    # In Focused mode: only on/off toggle + ATR multiplier are free;
    # lt_tf, lt_length, mt_tf, mt_length are locked to PineScript V69.1 defaults.
    'use_lt_filter':     {'type': 'categorical', 'choices': [True, False]},
    'lt_multiplier':     {'type': 'categorical', 'choices': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 'condition': 'use_lt_filter == True'},
    'use_mt_filter':     {'type': 'categorical', 'choices': [True, False]},
    'use_rsi50_filter':  {'type': 'categorical', 'choices': [True, False]},
    'use_diff_ma_red':   {'type': 'categorical', 'choices': [True, False]},
    'diff_ma_red_pct':   {'type': 'float', 'low': 0.5, 'high': 5.0, 'step': 0.5, 'condition': 'use_diff_ma_red == True'},
    # ══════════════════════════════════════════════════════════════════════
    # V5.4.9: Risk Management LOCKED — NOT optimized.
    # Philosophy: SL=30% is a safety net ("filet de securite"). Exits come
    # from setup reversal or filter changes, NOT tight stops.
    # User tunes RM exclusively in TradingView after validating the config.
    # ALL RM params below are REMOVED from the search space.
    # ══════════════════════════════════════════════════════════════════════
    # (sl_mode, sl_pct, atr_sl_mult, sl_dynamic, sl_vol_min/max — REMOVED)
    # (tp_mode, tp_pct, tp_sl_ratio — REMOVED)
    # (be_mode, be_trigger_pct, tsl_pct — REMOVED)
    # (use_partial_tp, tp1_pct/size, tp2_pct/size — REMOVED)
    # ── V5.2: Max Hold Days — kept, useful for exploration ──
    'max_hold_days':     {'type': 'int', 'low': 0, 'high': 20,
                          'comment': '0 = disabled'},
    # ── V5.2: ADX Regime Filter (Sprint 2) ──
    'use_adx_regime_filter': {'type': 'categorical', 'choices': [True, False]},
    'adx_trend_threshold':   {'type': 'int', 'low': 20, 'high': 30, 'step': 5,
                              'condition': 'use_adx_regime_filter == True'},
    # ── V5.2: Volatility ATR Percentile Filter (Sprint 2) ──
    'use_vol_filter':    {'type': 'categorical', 'choices': [True, False]},
    'vol_percentile_threshold': {'type': 'int', 'low': 50, 'high': 90, 'step': 10,
                                 'condition': 'use_vol_filter == True'},
    # ── V5.2: Long/Short direction (Sprint 1) ──
    'long_side':         {'type': 'categorical', 'choices': [True, False]},
    'short_side':        {'type': 'categorical', 'choices': [True, False]},
    # ══════════════════════════════════════════════════════════════════════
    # V5.4.9 FIX #1: VIX/Macro/Partial TP/Session filters REMOVED.
    # VIX & Macro: _vix_series/_macro_scores are NEVER provided, so the
    # optimizer explored these as phantom params (always silently no-op).
    # Session: timezone mismatch between data and TV broker.
    # Partial TP: part of RM, user-validated on TV only.
    # These are now FIXED in FOCUSED_FIXED_PARAMS (forced OFF).
    # ══════════════════════════════════════════════════════════════════════
    # (use_vix_filter, vix_threshold_high — REMOVED from search space)
    # (use_macro_filter, macro_min, macro_max — REMOVED from search space)
    # (use_session_filter, session1_start, session1_end — REMOVED)
    # (use_partial_tp, tp1/tp2 — REMOVED, part of RM)
    # ── V5.3: QQ Estimation Filter (Tier 3) — kept, computed locally ──
    'use_qq_filter':     {'type': 'categorical', 'choices': [True, False]},
    'qq_length':         {'type': 'int', 'low': 7, 'high': 21, 'step': 7,
                          'condition': 'use_qq_filter == True'},
}

# Fixed params in FOCUSED mode
# These mirror PineScript V69.1 defaults. They are NOT optimized in Focused mode
# to keep the search space small (~12 free params) and reduce overfitting risk.
# To unlock them, switch to Full mode.
FOCUSED_FIXED_PARAMS = {
    'pmax_ma_type': 'EMA',
    # ══ V5.4.9: pmax_ma_mode LOCKED to classic (only mode in Pine Script) ══
    'pmax_ma_mode': 'classic',
    'pmax_ma_factor': 1.0,       # unused (classic mode), kept for compat
    'pmax_ma_distance': 0.0,     # unused (classic mode), kept for compat
    'use_ma_cross_signal': False, # does NOT exist in Pine Script V69.3
    'sig_ma_type': 'EMA',        # unused, kept for compat
    'sig_ma_fast': 9,            # unused, kept for compat
    'sig_ma_slow': 21,           # unused, kept for compat
    # ── LT Trend filter (fixed in Focused) ──
    'lt_ma_type': 'EMA',        # PineScript default
    'lt_length': 20,             # EMA period on Weekly
    'lt_tf': 'W',                # Weekly timeframe
    # ── MT Trend filter (fixed in Focused) ──
    'mt_ma_type': 'EMA',        # PineScript default
    'mt_length': 10,             # EMA period on 4H
    'mt_tf': '4H',               # 4-Hour timeframe
    'mt_multiplier': 1.0,        # ATR multiplier (tunable in Full)
    # ── Risk / execution ──
    'exit_setup_reversal': True,
    'order_pause': 5,
    # ══ V5.4.9: Risk Management LOCKED ("SL Plein" philosophy) ══
    # SL = 30% safety net — exits via setup reversal, NOT tight SL
    'sl_mode': 'static_pct',
    'sl_pct': 30.0,
    'sl_dynamic': False,
    'tp_mode': 'no',
    'be_mode': 'no',
    'use_partial_tp': False,
    # ══ V5.4.9 FIX #1: Phantom filters forced OFF ══
    'use_vix_filter': False,      # No VIX data available in Python
    'use_macro_filter': False,    # No macro scores available in Python
    'use_session_filter': False,  # Timezone mismatch risk
    # ══ V5.4.9 FIX #3: Fib filter explicitly OFF ══
    'use_fib_filter': False,      # Fib disabled in exploration (TV uses its own)
}

# ==============================================================================
# FULL MODE SEARCH SPACE (~27 params — higher overfitting risk)
#
# Adds tunability for LT/MT timeframes, MA types, and lengths.
# ⚠️  WARNING: more free params = higher curve-fitting risk.
# Use Full mode only when Focused fails to find acceptable OOS results,
# or when testing alternative LT/MT configurations for specific instruments.
# ==============================================================================
FULL_SEARCH_SPACE = {
    **FOCUSED_SEARCH_SPACE,
    'pmax_ma_type':      {'type': 'categorical', 'choices': ['SMA', 'EMA', 'WMA', 'ZLEMA', 'TSF']},
    # ── LT Trend (unlocked in Full) ──
    'lt_tf':             {'type': 'categorical', 'choices': LT_TIMEFRAMES, 'condition': 'use_lt_filter == True'},
    'lt_length':         {'type': 'int',   'low': 10,  'high': 30, 'condition': 'use_lt_filter == True'},
    'lt_ma_type':        {'type': 'categorical', 'choices': ['SMA', 'EMA'], 'condition': 'use_lt_filter == True'},
    # ── MT Trend (unlocked in Full) ──
    'mt_tf':             {'type': 'categorical', 'choices': MT_TIMEFRAMES, 'condition': 'use_mt_filter == True'},
    'mt_length':         {'type': 'int',   'low': 5,   'high': 20, 'condition': 'use_mt_filter == True'},
    'mt_ma_type':        {'type': 'categorical', 'choices': ['SMA', 'EMA'], 'condition': 'use_mt_filter == True'},
    'mt_multiplier':     {'type': 'categorical', 'choices': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0], 'condition': 'use_mt_filter == True'},
    # ── Other filters (Full only) ──
    'use_ma_dir_filter': {'type': 'categorical', 'choices': [True, False]},
    'ma_dir_len':        {'type': 'int', 'low': 2, 'high': 5, 'condition': 'use_ma_dir_filter == True'},
    'use_mavg_filter':   {'type': 'categorical', 'choices': [True, False]},
    'use_diff_price_red':{'type': 'categorical', 'choices': [True, False]},
    'diff_price_red_pct':{'type': 'float', 'low': 0.5, 'high': 5.0, 'step': 0.5, 'condition': 'use_diff_price_red == True'},
    'order_pause':       {'type': 'int',   'low': 3,   'high': 10},
    # ── V5.4.9: Inherit remaining search params from Focused ──
    # NOTE: RM params (SL/TP/BE/Partial) are NOT inherited — they are LOCKED
    # in FOCUSED_FIXED_PARAMS. VIX/Macro/Session also LOCKED (phantom filters).
    'max_hold_days':     FOCUSED_SEARCH_SPACE['max_hold_days'],
    'use_adx_regime_filter': FOCUSED_SEARCH_SPACE['use_adx_regime_filter'],
    'adx_trend_threshold': FOCUSED_SEARCH_SPACE['adx_trend_threshold'],
    'use_vol_filter':    FOCUSED_SEARCH_SPACE['use_vol_filter'],
    'vol_percentile_threshold': FOCUSED_SEARCH_SPACE['vol_percentile_threshold'],
    'long_side':         FOCUSED_SEARCH_SPACE['long_side'],
    'short_side':        FOCUSED_SEARCH_SPACE['short_side'],
    'use_qq_filter':     FOCUSED_SEARCH_SPACE['use_qq_filter'],
    'qq_length':         FOCUSED_SEARCH_SPACE['qq_length'],
}

# ==============================================================================
# OPTIMIZER SETTINGS
# ==============================================================================
OPTIMIZER_DEFAULTS = {
    'n_trials': 500,
    'n_jobs': 1,           # single-thread for reproducibility
    'is_ratio': 0.70,      # 70% in-sample
    'seed': 42,
}

# Composite Score Weights
COMPOSITE_WEIGHTS = {
    'calmar': 0.35,
    'sortino': 0.25,
    'trades_score': 0.20,
    'dd_score': 0.20,
}

# Phase 1 Anti-Overfit: IS PF cap penalties (v5.5)
# Configs with IS PF > 3.0 collapse OOS (empirical ratio 0.46).
# Cumulative: PF 4.5 → 0.80 * 0.60 = 0.48.
IS_PF_CAP_PENALTIES = {
    3.0: 0.80,   # IS PF > 3.0 → score × 0.80
    4.0: 0.60,   # IS PF > 4.0 → score × 0.60 (cumulative with 0.80)
    5.0: 0.40,   # IS PF > 5.0 → score × 0.40 (cumulative, near-reject)
}

# Phase 1 Anti-Overfit: IS/OOS coherence gates (v5.5)
# Controls for passes_oos_gates() when IS metrics are available.
OOS_COHERENCE_GATES = {
    'pf_ratio_max': 2.5,    # OOS PF / IS PF > 2.5 → inflated (regime luck)
    'pf_ratio_min': 0.30,   # OOS PF / IS PF < 0.30 → collapsed (overfit)
    'wr_delta_max': 15.0,   # |OOS WR - IS WR| > 15pp → unstable
}

# v5.6: OOS Gate Tier Thresholds (A3.1)
# Multi-level classification: GOLD > SILVER > BRONZE > FAIL
OOS_GATE_TIERS = {
    'GOLD': {
        'min_calmar': 1.0,
        'min_pf': 1.5,
        'min_trades': 40,
        'require_perturbation_stable': True,
        'min_wfe': 0.5,
    },
    'SILVER': {
        'min_calmar': 0.50,
        'min_pf': 1.30,
        'min_trades': 25,
    },
    'BRONZE': {
        'min_calmar': 0.25,
        'min_pf': 1.20,
        'min_trades': 15,
    },
}

# Hard Constraints
MIN_TRADES = 30  # Lowered from 60 for exploration (TV validates with real data)
TRADE_PENALTY = {
    (30, 59):   0.70,
    (60, 99):   0.85,
    (100, 199): 0.95,
    (200, 9999): 1.00,
}

# Robustness Bonus (length >= 15 AND multiplier >= 3.0)
ROBUSTNESS_BONUS = 1.10

# ==============================================================================
# OOS GATES (Out-of-Sample validation thresholds)
# Based on industry research (Pardo WFE, XBTO, OPTIML, StrategyQuant):
# - Calmar OOS >= 0.25 allows ~50-60% degradation vs IS (Pardo WFE rule)
# - PF >= 1.20 is standard industry minimum
# - MaxDD threshold is ADAPTIVE per asset class (see OOS_DD_LIMITS below)
# - Min 15 trades for statistical significance
# ==============================================================================
OOS_GATES = {
    'min_calmar': 0.50,   # v5.5 Phase 1.3: raised from 0.40 → 0.50 (anti-overfit)
    'min_pf': 1.30,       # v5.5 Phase 1.3: raised from 1.25 → 1.30 (anti-overfit)
    'max_dd': -8.0,       # default fallback (percent, negative)
    'min_trades': 25,     # v5.5 Phase 1.3: raised from 20 → 25 (statistical significance)
}

# Per-asset-class Max DD thresholds for the OOS gate.
# Volatile instruments (indices, crypto, metals) tolerate deeper drawdowns
# because intrabar DD is structurally worse and spread/slippage widens tails.
# Forex (tight spread, lower vol) keeps the stricter threshold.
OOS_DD_LIMITS = {
    'Indices':        -15.0,   # US30, US100, UK100, GER40… → high intrabar DD
    'Crypto Majors':  -18.0,   # BTCUSD, ETHUSD… → extreme volatility
    'Crypto Altcoins':-20.0,   # lower liquidity altcoins
    'Metals':         -12.0,   # XAUUSD, XAGUSD → moderate-high vol
    'Energy':         -12.0,   # USOIL, UKOIL → event-driven spikes
    'Forex Majors':    -8.0,   # tight spreads, lower volatility
    'Forex Minors':   -10.0,   # wider spreads, moderate vol
    'Forex Exotics':  -15.0,   # very wide spreads, high vol
}


def get_oos_dd_limit(instrument: str) -> float:
    """
    Return the adaptive Max DD threshold for a given instrument.
    Looks up the instrument in INSTRUMENT_GROUPS to find its asset class,
    then returns the corresponding OOS_DD_LIMITS value.
    Falls back to OOS_GATES['max_dd'] (-8.0%) if not found.
    """
    for group, symbols in INSTRUMENT_GROUPS.items():
        if instrument in symbols:
            return OOS_DD_LIMITS.get(group, OOS_GATES['max_dd'])
    return OOS_GATES['max_dd']

# ==============================================================================
# OOS DEGRADATION GRADING (Calmar OOS vs IS)
# Thresholds based on Walk-Forward Efficiency (WFE) research:
# - Pardo/TradeStation: WFE >= 50% = successful
# - GSBPro: WFE > 60% recommended
# - OOS/IS ratio grades the quality of out-of-sample survival
# ==============================================================================
OOS_DEGRADATION_GRADES = [
    # (min_wfe, label, description)
    (0.85, 'EXCELLENT', 'WFE ≥85% — Minimal degradation'),
    (0.70, 'VERY_GOOD', 'WFE ≥70% — Low degradation'),
    (0.50, 'GOOD',      'WFE ≥50% — Acceptable (Pardo minimum)'),
    (0.40, 'ACCEPTABLE','WFE ≥40% — Moderate degradation'),
    (0.00, 'WEAK',      'WFE <40% — High degradation'),
]

# ==============================================================================
# MONTE CARLO SETTINGS
# ==============================================================================
MC_DEFAULTS = {
    'n_sims': 10_000,
    'seed': 42,
    'ruin_threshold': -20.0,  # percent
}

MC_CLASSIFICATION = {
    'ROBUST':  {'max_p_ruin': 5.0},
    'STABLE':  {'max_p_ruin': 15.0},
    'FRAGILE': {'max_p_ruin': 30.0},
    'REJECT':  {'max_p_ruin': 100.0},
}

# ==============================================================================
# HTF RESAMPLING RULES
# ==============================================================================
HTF_RESAMPLE_MAP = {
    'M':     'ME',        # Monthly (Month-End)
    'W':     'W-FRI',     # Weekly closes Friday
    'D':     'D',         # Daily (=B for business days)
    '4H':    '4h',
    '3H':    '3h',
    '2H':    '2h',
    '1H':    '1h',
    '45min': '45min',
    '30min': '30min',
    '15min': '15min',
    '10min': '10min',
    '5min':  '5min',
    # Numeric aliases (Pine style)
    '240':   '4h',
    '180':   '3h',
    '120':   '2h',
    '60':    '1h',
    '45':    '45min',
    '30':    '30min',
    '15':    '15min',
    '10':    '10min',
    '5':     '5min',
}

# ==============================================================================
# FIBONACCI RANGE FILTER DEFAULTS
# ==============================================================================
FIB_DEFAULTS = {
    'tf': 'D',
    'fib_length': 265,
    'zones': [True, True, True, True, True, True],  # all 6 zones ON
}

# ==============================================================================
# V5.2: ADX REGIME COMPATIBILITY MATRIX
# Pine V69.2: entry type <-> market regime compatibility
# ==============================================================================
ADX_REGIME_COMPATIBILITY = {
    'Donc+Chikou':    {'trending': True, 'ranging': True},    # Polyvalent
    'RSI Divergence': {'trending': True, 'ranging': True},    # Polyvalent
    'Boll+SMA':       {'trending': True, 'ranging': False},   # Trending
    'Fractal':        {'trending': True, 'ranging': False},   # Trending
    'MM Cross':       {'trending': True, 'ranging': False},   # Trending
    'VWAP':           {'trending': False, 'ranging': True},   # Ranging
    'ORB':            {'trending': False, 'ranging': True},   # Ranging
}

# ==============================================================================
# V5.2: ADAPTIVE SL DEFAULTS PER ASSET CLASS
# Replaces the hardcoded 30% SL that never triggers.
# These define the SEARCH RANGE for sl_pct in static mode,
# used when optimizer suggests sl_mode='static_pct'.
# ==============================================================================
SL_DEFAULTS_BY_CLASS = {
    'Indices':        {'sl_pct_low': 2.0,  'sl_pct_high': 15.0},
    'Forex Majors':   {'sl_pct_low': 0.5,  'sl_pct_high': 5.0},
    'Forex Minors':   {'sl_pct_low': 0.5,  'sl_pct_high': 5.0},
    'Forex Exotics':  {'sl_pct_low': 1.0,  'sl_pct_high': 8.0},
    'Metals':         {'sl_pct_low': 1.0,  'sl_pct_high': 10.0},
    'Energy':         {'sl_pct_low': 1.0,  'sl_pct_high': 10.0},
    'Crypto Majors':  {'sl_pct_low': 3.0,  'sl_pct_high': 20.0},
    'Crypto Altcoins':{'sl_pct_low': 3.0,  'sl_pct_high': 20.0},
}


def get_sl_range_for_instrument(instrument: str) -> dict:
    """Return adaptive SL search range for a given instrument."""
    for group, symbols in INSTRUMENT_GROUPS.items():
        if instrument in symbols:
            return SL_DEFAULTS_BY_CLASS.get(group, {'sl_pct_low': 1.0, 'sl_pct_high': 20.0})
    return {'sl_pct_low': 1.0, 'sl_pct_high': 20.0}

# ==============================================================================
# DEFAULT COMPLETE CONFIG (production-like)
# ==============================================================================
# ==============================================================================
# PERTURBATION ROBUSTNESS (±10%)
# ==============================================================================
# V5.4.9 FIX #6: Increased from 10%/5runs to 15%/20runs for better robustness assessment
PERTURBATION_PCT = 15  # percent variation for robustness check
PERTURBATION_N_RUNS = 20  # number of perturbed runs


def get_default_config():
    """Return a complete default configuration dict."""
    return {
        # PMax
        'pmax_ma_type': 'EMA',
        'pmax_length': 10,
        'pmax_multiplier': 3.0,
        # PMax MA mode: 'classic' | 'mult_int' | 'mult_dec' | 'additive'
        'pmax_ma_mode': 'classic',
        'pmax_ma_factor': 1.0,      # for mult_int / mult_dec
        'pmax_ma_distance': 0.0,    # for additive
        # Independent MA Cross Signal (AND with PMax)
        'use_ma_cross_signal': False,
        'sig_ma_type': 'EMA',       # EMA or TRIMA
        'sig_ma_fast': 9,
        'sig_ma_slow': 21,
        # Entry
        'entry_type': 'Donc+Chikou',
        'don_length': 20,
        # MM Cross (entry type)
        'mm_cross_ma1_type': 'EMA',
        'mm_cross_ma1_len': 9,
        'mm_cross_ma2_type': 'EMA',
        'mm_cross_ma2_len': 21,
        'rsi_div_length': 14,
        'rsi_div_pivot': 5,
        'rsi_div_hidden': True,
        # Fractal
        'fractal_buffer': 0.0,
        # Boll+SMA
        'length_bb_ma': 20,
        'mult_bb_ma': 2.0,
        'length_bb_sma': 50,
        # Filters
        'use_lt_filter': False,
        'lt_ma_type': 'EMA',
        'lt_length': 20,
        'lt_tf': 'W',
        'lt_multiplier': 1.5,
        'use_mt_filter': False,
        'mt_ma_type': 'EMA',
        'mt_length': 10,
        'mt_tf': '4H',
        'mt_multiplier': 1.5,
        'use_rsi50_filter': False,
        'rsi50_length': 14,
        'use_rsi_ema_filter': False,
        'rsi_ema_length': 12,
        'use_diff_ma_red': False,
        'diff_ma_red_pct': 1.0,
        'use_diff_price_red': False,
        'diff_price_red_pct': 1.0,
        'use_ma_dir_filter': False,
        'ma_dir_len': 3,
        'use_mavg_filter': False,
        # Risk — V5.4.9: LOCKED ("SL Plein" philosophy)
        # SL = 30% safety net, exits via setup reversal. NOT optimized.
        'sl_mode': 'static_pct',       # LOCKED: only static_pct
        'sl_pct': 30.0,                # LOCKED: 30% safety net
        'atr_sl_mult': 2.0,           # (kept for compatibility, not used)
        'sl_dynamic': False,           # LOCKED: no dynamic SL
        'sl_vol_min_mult': 0.8,
        'sl_vol_max_mult': 1.5,
        # V5.4.9: TP/BE LOCKED (user validates on TV)
        'tp_mode': 'no',               # LOCKED: no TP in exploration
        'tp_pct': 2.0,
        'tp_sl_ratio': 2.0,
        'be_mode': 'no',               # LOCKED: no BE in exploration
        'be_trigger_pct': 1.0,
        'tsl_pct': 1.0,
        # V5.2: Max Hold Days (Security Layer 3)
        'max_hold_days': 0,            # 0 = disabled
        # V5.2: ADX Regime Filter
        'use_adx_regime_filter': False,
        'adx_trend_threshold': 25,
        # V5.2: Volatility ATR Percentile Filter
        'use_vol_filter': False,
        'vol_percentile_threshold': 70,
        # Execution
        'exit_setup_reversal': True,
        'order_pause': 5,
        'long_side': True,
        'short_side': True,
        # V5.3: Partial Profit Taking — V5.4.9: LOCKED OFF
        'use_partial_tp': False,       # LOCKED: no partial TP in exploration
        'tp1_pct': 1.0,              # % for first partial TP
        'tp1_size': 30,              # % of position to close at TP1
        'tp2_pct': 2.0,              # % for second partial TP
        'tp2_size': 30,              # % of position to close at TP2
        # V5.3: Session / Timing Filter — V5.4.9: LOCKED OFF (timezone mismatch risk)
        'use_session_filter': False,   # LOCKED
        'session_allowed_days': None, # None = all days, or list [0..6]
        'session1_start': 0,         # HHMM start (0 = 00:00)
        'session1_end': 2400,         # HHMM end (2400 = 24:00)
        'session2_start': 0,
        'session2_end': 0,
        # V5.3: VIX Filter (Security Layer 0) — V5.4.9: LOCKED OFF (no data)
        'use_vix_filter': False,       # LOCKED: no VIX data available
        'vix_threshold_high': 30,
        'vix_threshold_low': 10,
        '_vix_series': None,          # np.ndarray, set externally
        # V5.3: Macro Filter — V5.4.9: LOCKED OFF (no data)
        'use_macro_filter': False,     # LOCKED: no macro data available
        'macro_min': -50.0,
        'macro_max': 50.0,
        '_macro_scores': None,        # np.ndarray, set externally
        # V5.3: QQ Estimation Filter
        'use_qq_filter': False,
        'qq_length': 14,
        # Fib range — V5.4.9 FIX #3: explicitly OFF (was contradictory)
        'use_fib_filter': False,       # OFF in exploration (TV uses its own Fib)
        'fib_tf': 'D',
        'fib_length': 265,
        'fib_zones': [True, True, True, True, True, True],
    }
