"""
PF AI Lab 5.0 — MM Cross Entry (V69.2)

Generates long/short signals when two independent moving averages cross.
  - LONG: MA1 (fast) crosses above MA2 (slow) → golden cross
  - SHORT: MA1 (fast) crosses below MA2 (slow) → death cross

MA types supported: EMA, TRIMA (true double SMA, fixed in V69.2).
These MAs are independent of the PMax MA.
"""

import numpy as np
import pandas as pd
import math
from ..indicators.ma_types import ema, sma


def _compute_trima(src: np.ndarray, length: int) -> np.ndarray:
    """True Triangular MA = SMA(SMA(src, ceil(N/2)), floor(N/2)+1)."""
    first_len = math.ceil(length / 2)
    second_len = math.floor(length / 2) + 1
    first = sma(src, first_len)
    return sma(first, second_len)


def _get_mm_cross_ma(src: np.ndarray, length: int, ma_type: str) -> np.ndarray:
    """Compute MA for MM Cross. Only EMA and TRIMA are supported."""
    if ma_type == 'TRIMA':
        return _compute_trima(src, length)
    return ema(src, length)


def compute_mm_cross_entry(df: pd.DataFrame,
                           ma1_type: str = 'EMA',
                           ma1_len: int = 9,
                           ma2_type: str = 'EMA',
                           ma2_len: int = 21) -> dict:
    """
    Compute MM Cross entry signals.

    Parameters:
        df: DataFrame with 'Close' column
        ma1_type: type for fast MA ('EMA' or 'TRIMA')
        ma1_len: length for fast MA (default 9)
        ma2_type: type for slow MA ('EMA' or 'TRIMA')
        ma2_len: length for slow MA (default 21)

    Returns:
        dict with:
            'buy_mm_cross': np.ndarray (bool) — MA1 crosses above MA2
            'sell_mm_cross': np.ndarray (bool) — MA1 crosses below MA2
            'mm_cross_ma1': np.ndarray — fast MA values
            'mm_cross_ma2': np.ndarray — slow MA values
    """
    close = df['Close'].values.astype(float)
    n = len(close)

    ma1 = _get_mm_cross_ma(close, ma1_len, ma1_type)
    ma2 = _get_mm_cross_ma(close, ma2_len, ma2_type)

    buy = np.zeros(n, dtype=bool)
    sell = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if np.isnan(ma1[i]) or np.isnan(ma2[i]) or np.isnan(ma1[i-1]) or np.isnan(ma2[i-1]):
            continue
        # Crossover: MA1 crosses above MA2
        buy[i] = ma1[i] > ma2[i] and ma1[i-1] <= ma2[i-1]
        # Crossunder: MA1 crosses below MA2
        sell[i] = ma1[i] < ma2[i] and ma1[i-1] >= ma2[i-1]

    return {
        'buy_mm_cross': buy,
        'sell_mm_cross': sell,
        'mm_cross_ma1': ma1,
        'mm_cross_ma2': ma2,
    }
