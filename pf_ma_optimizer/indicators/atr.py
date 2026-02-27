"""
PF AI Lab 5.2 — ATR (Average True Range) for optimizer engine.

Provides compute_atr_14_50() for Dynamic SL calculation (ATR14/ATR50 ratio)
and compute_atr_percentile() for Volatility Filter (Security Layer 1).

Uses existing compute_true_range/rma from pmax.py to avoid code duplication.
"""

import numpy as np
from .pmax import compute_true_range, compute_atr_rma


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                length: int = 14) -> np.ndarray:
    """Compute ATR using RMA (Wilder's smoothing) — Pine ta.atr(length)."""
    return compute_atr_rma(high, low, close, length)


def compute_atr_14_50(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> dict:
    """
    Compute ATR(14) and ATR(50) for Dynamic SL calculation.
    Pine V69.2: vol_ratio = ATR(14) / ATR(50)

    Returns:
        dict with 'atr14', 'atr50', 'vol_ratio' arrays
    """
    atr14 = compute_atr(high, low, close, 14)
    atr50 = compute_atr(high, low, close, 50)

    n = len(high)
    vol_ratio = np.ones(n, dtype=float)
    for i in range(n):
        if (not np.isnan(atr14[i]) and not np.isnan(atr50[i])
                and atr50[i] > 0):
            vol_ratio[i] = atr14[i] / atr50[i]

    return {
        'atr14': atr14,
        'atr50': atr50,
        'vol_ratio': vol_ratio,
    }


def compute_atr_percentile(high: np.ndarray, low: np.ndarray,
                            close: np.ndarray,
                            atr_length: int = 14,
                            lookback: int = 100) -> np.ndarray:
    """
    Compute ATR percentile rank over a rolling window.
    Security Layer 1 — Volatility Filter.

    Returns:
        np.ndarray of percentile values [0..100] for each bar.
        Values < threshold means entry is allowed (low volatility).
    """
    atr = compute_atr(high, low, close, atr_length)
    n = len(atr)
    percentile = np.full(n, 50.0, dtype=float)

    for i in range(lookback, n):
        window = atr[i - lookback:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            continue
        rank = np.sum(valid < atr[i]) / len(valid) * 100
        percentile[i] = rank

    return percentile
