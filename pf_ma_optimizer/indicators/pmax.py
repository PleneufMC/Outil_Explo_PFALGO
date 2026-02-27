"""
PF AI Lab 5.0 — PMax Indicator (core of the algorithm)
PMax = Moving Average ± ATR × Multiplier (modified Supertrend)

CRITICAL IMPLEMENTATION NOTES:
- src = hl2 = (High + Low) / 2  (NOT close!)
- ATR MUST use RMA (Wilder smoothing), NOT SMA
- longStop/shortStop are STATEFUL (cumulative max/min) — requires iterative loop
- Direction depends on PREVIOUS stops — NOT vectorizable
"""

import numpy as np
import pandas as pd
from .ma_types import get_ma, rma


def compute_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """True Range — Pine Script ta.tr."""
    n = len(high)
    tr = np.full(n, np.nan, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    return tr


def compute_atr_rma(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    ATR using RMA (Wilder smoothing) — matches Pine Script ta.atr(length).
    CRITICAL: Pine uses ta.rma(ta.tr, length), NOT rolling().mean()
    """
    tr = compute_true_range(high, low, close)
    return rma(tr, length)


def compute_pmax(df: pd.DataFrame, length: int, multiplier: float,
                 ma_type: str = 'EMA',
                 ma_mode: str = 'classic',
                 ma_factor: float = 1.0,
                 ma_distance: float = 0.0):
    """
    Compute PMax indicator on OHLCV DataFrame.

    Parameters:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'hl2']
        length: MA period AND ATR period (same value — Pine design)
        multiplier: ATR multiplier for stops
        ma_type: one of 9 MA types (used for classic mode)
        ma_mode: how the MA line for PMax is built:
            'classic'     — standard MA(close, length) using ma_type
            'mult_int'    — close × int(factor)    (multiplicative integer)
            'mult_dec'    — close × factor          (multiplicative decimal)
            'additive'    — close + distance         (additive offset)
        ma_factor: factor for mult_int / mult_dec modes (default 1.0)
        ma_distance: distance for additive mode (default 0.0)

    Returns:
        dict with keys:
            'MAvg': np.ndarray — Moving Average values (or synthetic MA)
            'PMax': np.ndarray — PMax line (red line)
            'dir': np.ndarray — Direction: 1=bullish, -1=bearish
            'buy_signal': np.ndarray (bool) — crossover(MAvg, PMax) events
            'sell_signal': np.ndarray (bool) — crossunder(MAvg, PMax) events
            'buy_trigger': np.ndarray (bool) — MAvg > PMax (continuous state)
            'sell_trigger': np.ndarray (bool) — MAvg < PMax (continuous state)
            'ATR': np.ndarray — ATR(length) using RMA
    """
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)

    # Pine uses `src` which defaults to hl2 = (High + Low) / 2 in PF Algo.
    # V5.4.8 FIX #2: Was incorrectly using close. The PMax MA source in
    # Pine Script PF_Algo is hl2, not close. This was a CRITICAL divergence
    # causing all entry/exit levels to differ from TradingView.
    src = (high + low) / 2.0  # hl2 — matches Pine Script PF_Algo default

    n = len(df)

    # Step 1: Compute ATR with RMA (Wilder)
    atr = compute_atr_rma(high, low, close, length)

    # Step 2: Compute Moving Average (supports 4 modes)
    if ma_mode == 'mult_int':
        mavg = close * int(round(ma_factor))
    elif ma_mode == 'mult_dec':
        mavg = close * ma_factor
    elif ma_mode == 'additive':
        mavg = close + ma_distance
    else:  # 'classic' — standard MA
        mavg = get_ma(src, length, ma_type)

    # Step 3: Compute nLoss
    nloss = multiplier * atr

    # Step 4: Iterative PMax calculation (stateful — NOT vectorizable)
    long_stop = np.full(n, np.nan, dtype=float)
    short_stop = np.full(n, np.nan, dtype=float)
    direction = np.ones(n, dtype=int)  # 1=bullish, -1=bearish
    pmax = np.full(n, np.nan, dtype=float)

    for i in range(1, n):
        if np.isnan(mavg[i]) or np.isnan(nloss[i]):
            long_stop[i] = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else 0
            short_stop[i] = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else 0
            direction[i] = direction[i - 1]
            pmax[i] = pmax[i - 1]
            continue

        # Long stop
        ls = mavg[i] - nloss[i]
        ls_prev = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else ls
        if mavg[i] > ls_prev:
            long_stop[i] = max(ls, ls_prev)
        else:
            long_stop[i] = ls

        # Short stop
        ss = mavg[i] + nloss[i]
        ss_prev = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else ss
        if mavg[i] < ss_prev:
            short_stop[i] = min(ss, ss_prev)
        else:
            short_stop[i] = ss

        # Direction
        prev_dir = direction[i - 1]
        ss_prev_val = short_stop[i - 1] if not np.isnan(short_stop[i - 1]) else float('inf')
        ls_prev_val = long_stop[i - 1] if not np.isnan(long_stop[i - 1]) else float('-inf')

        if prev_dir == -1 and mavg[i] > ss_prev_val:
            direction[i] = 1
        elif prev_dir == 1 and mavg[i] < ls_prev_val:
            direction[i] = -1
        else:
            direction[i] = prev_dir

        # PMax
        pmax[i] = long_stop[i] if direction[i] == 1 else short_stop[i]

    # Step 5: Signals
    buy_signal = np.zeros(n, dtype=bool)
    sell_signal = np.zeros(n, dtype=bool)
    buy_trigger = np.zeros(n, dtype=bool)
    sell_trigger = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(mavg[i]) or np.isnan(pmax[i]):
            continue
        # Crossover: MAvg crosses above PMax
        m_prev = mavg[i - 1] if not np.isnan(mavg[i - 1]) else 0
        p_prev = pmax[i - 1] if not np.isnan(pmax[i - 1]) else 0

        buy_signal[i] = mavg[i] > pmax[i] and m_prev <= p_prev
        sell_signal[i] = mavg[i] < pmax[i] and m_prev >= p_prev
        buy_trigger[i] = mavg[i] > pmax[i]
        sell_trigger[i] = mavg[i] < pmax[i]

    return {
        'MAvg': mavg,
        'PMax': pmax,
        'dir': direction,
        'buy_signal': buy_signal,
        'sell_signal': sell_signal,
        'buy_trigger': buy_trigger,
        'sell_trigger': sell_trigger,
        'ATR': atr,
        'longStop': long_stop,
        'shortStop': short_stop,
    }


def compute_pmax_htf_state(df_htf: pd.DataFrame, length: int, multiplier: float,
                           ma_type: str = 'EMA',
                           ma_mode: str = 'classic',
                           ma_factor: float = 1.0,
                           ma_distance: float = 0.0) -> np.ndarray:
    """
    Compute PMax on HTF data and return CONTINUOUS STATE (V68.8 fix).

    Returns:
        np.ndarray (bool): True where MAvg[i-1] > PMax[i-1] (bullish HTF trend)

    V68.8 CRITICAL FIX:
        ❌ BEFORE: crossover signal (true for 1 bar only)
        ✅ AFTER: continuous state (true while bullish)
    """
    result = compute_pmax(df_htf, length, multiplier, ma_type,
                          ma_mode=ma_mode, ma_factor=ma_factor,
                          ma_distance=ma_distance)
    mavg = result['MAvg']
    pmax_line = result['PMax']

    n = len(df_htf)
    state = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if not np.isnan(mavg[i - 1]) and not np.isnan(pmax_line[i - 1]):
            state[i] = mavg[i - 1] > pmax_line[i - 1]

    return state
