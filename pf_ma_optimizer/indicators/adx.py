"""
PF AI Lab 5.2 — ADX (Average Directional Index) for optimizer engine.

Pine V69.2: [diPlus, diMinus, adx] = ta.dmi(length, adxSmoothing)
Used by ADX Regime Filter to classify trending vs ranging markets.
"""

import numpy as np
from .pmax import compute_true_range
from .ma_types import rma


def compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                length: int = 14, adx_smoothing: int = 14) -> dict:
    """
    Compute ADX indicator matching Pine Script ta.dmi(length, adxSmoothing).

    Returns:
        dict with 'di_plus', 'di_minus', 'adx' arrays (np.ndarray)
    """
    n = len(high)

    # Directional Movement
    up_move = np.full(n, 0.0, dtype=float)
    down_move = np.full(n, 0.0, dtype=float)
    for i in range(1, n):
        up_move[i] = high[i] - high[i - 1]
        down_move[i] = low[i - 1] - low[i]

    # +DM and -DM
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # True Range
    tr = compute_true_range(high, low, close)

    # Smooth with RMA (Wilder's)
    atr_smooth = rma(tr, length)
    plus_dm_smooth = rma(plus_dm, length)
    minus_dm_smooth = rma(minus_dm, length)

    # +DI and -DI
    di_plus = np.full(n, np.nan, dtype=float)
    di_minus = np.full(n, np.nan, dtype=float)
    dx = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if (not np.isnan(atr_smooth[i]) and atr_smooth[i] > 0
                and not np.isnan(plus_dm_smooth[i])
                and not np.isnan(minus_dm_smooth[i])):
            di_plus[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            di_minus[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
            di_sum = di_plus[i] + di_minus[i]
            if di_sum > 0:
                dx[i] = abs(di_plus[i] - di_minus[i]) / di_sum * 100

    # ADX = RMA(DX, adxSmoothing)
    adx = rma(dx, adx_smoothing)

    return {
        'di_plus': di_plus,
        'di_minus': di_minus,
        'dx': dx,
        'adx': adx,
    }
