"""
Moving Averages — 9 types matching Pine Script v6 ta.* functions.

Pine↔Python equivalence rules:
  - ta.sma(src, n)  → rolling(n).mean()
  - ta.ema(src, n)  → ewm(span=n, adjust=False).mean()
  - ta.rma(src, n)  → ewm(alpha=1/n, adjust=False).mean()    # Wilder's smoothing
  - ta.wma(src, n)  → manual weighted sum
  - ta.vwma(src, n) → volume-weighted: sma(src*vol, n) / sma(vol, n)
  - DEMA(src, n)    → 2*EMA - EMA(EMA)
  - TEMA(src, n)    → 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))
  - HMA(src, n)     → WMA(2*WMA(n/2) - WMA(n), sqrt(n))
  - ZLEMA(src, n)   → EMA(2*src - src[lag], n) where lag = (n-1)/2

CRITICAL: RMA ≠ EMA. RMA uses alpha=1/n, EMA uses span=n (alpha=2/(n+1)).
"""

import numpy as np
import pandas as pd
from typing import Union

Series = pd.Series


def compute_sma(src: Series, length: int) -> Series:
    """Pine: ta.sma(src, length) — Simple Moving Average."""
    return src.rolling(window=length, min_periods=length).mean()


def compute_ema(src: Series, length: int) -> Series:
    """
    Pine: ta.ema(src, length) — Exponential Moving Average.

    CRITICAL: adjust=False to match Pine Script behavior.
    adjust=True gives a different result (bias-corrected).
    """
    return src.ewm(span=length, adjust=False).mean()


def compute_rma(src: Series, length: int) -> Series:
    """
    Pine: ta.rma(src, length) — Wilder's Smoothing (Running Moving Average).

    This is NOT the same as EMA:
      - RMA: alpha = 1/length
      - EMA: alpha = 2/(length+1)

    For ATR(14): RMA alpha=1/14≈0.0714, EMA alpha=2/15≈0.1333
    → RMA is smoother (lower alpha) than EMA of same length.
    """
    return src.ewm(alpha=1.0 / length, adjust=False).mean()


def compute_wma(src: Series, length: int) -> Series:
    """
    Pine: ta.wma(src, length) — Weighted Moving Average.

    Weights: [1, 2, 3, ..., length], most recent gets highest weight.
    """
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()

    def _wma(window):
        return np.dot(window, weights) / weight_sum

    return src.rolling(window=length, min_periods=length).apply(_wma, raw=True)


def compute_vwma(src: Series, length: int, volume: Series = None) -> Series:
    """
    Pine: ta.vwma(src, length) — Volume-Weighted Moving Average.

    Formula: sma(src * volume, length) / sma(volume, length)
    Requires volume series. Falls back to SMA if volume is None.
    """
    if volume is None:
        return compute_sma(src, length)
    numerator = compute_sma(src * volume, length)
    denominator = compute_sma(volume, length)
    return numerator / denominator


def compute_dema(src: Series, length: int) -> Series:
    """DEMA — Double EMA: 2 * EMA(src) - EMA(EMA(src))."""
    ema1 = compute_ema(src, length)
    ema2 = compute_ema(ema1, length)
    return 2 * ema1 - ema2


def compute_tema(src: Series, length: int) -> Series:
    """TEMA — Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))."""
    ema1 = compute_ema(src, length)
    ema2 = compute_ema(ema1, length)
    ema3 = compute_ema(ema2, length)
    return 3 * ema1 - 3 * ema2 + ema3


def compute_hma(src: Series, length: int) -> Series:
    """
    HMA — Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)).

    Faster than EMA, less lag, but more overshoot.
    """
    half_length = max(int(length / 2), 1)
    sqrt_length = max(int(np.sqrt(length)), 1)
    wma_half = compute_wma(src, half_length)
    wma_full = compute_wma(src, length)
    diff = 2 * wma_half - wma_full
    return compute_wma(diff, sqrt_length)


def compute_zlema(src: Series, length: int) -> Series:
    """
    ZLEMA — Zero-Lag EMA: EMA(2*src - src[lag], length) where lag = (length-1)//2.

    Reduces lag by using price momentum as correction.
    """
    lag = (length - 1) // 2
    adjusted_src = 2 * src - src.shift(lag)
    return compute_ema(adjusted_src, length)


# MA type mapping matching Pine Script V.69.1 input options
MA_TYPES = {
    'SMA': compute_sma,
    'EMA': compute_ema,
    'RMA': compute_rma,
    'WMA': compute_wma,
    'VWMA': compute_vwma,
    'DEMA': compute_dema,
    'TEMA': compute_tema,
    'HMA': compute_hma,
    'ZLEMA': compute_zlema,
}


def get_ma(src: Series, length: int, ma_type: str = 'EMA',
           volume: Series = None) -> Series:
    """
    Pine: getMA(src, length, mav) — Universal MA dispatcher.

    Args:
        src: Source series (typically hl2 for PMax)
        length: Lookback period
        ma_type: One of 9 types: SMA, EMA, RMA, WMA, VWMA, DEMA, TEMA, HMA, ZLEMA
        volume: Volume series (required for VWMA, ignored for others)

    Returns:
        Moving average series

    Raises:
        ValueError: If ma_type is not recognized
    """
    ma_type = ma_type.upper().strip()
    if ma_type not in MA_TYPES:
        valid = ', '.join(sorted(MA_TYPES.keys()))
        raise ValueError(f"Unknown MA type '{ma_type}'. Valid types: {valid}")

    if ma_type == 'VWMA':
        return compute_vwma(src, length, volume)
    return MA_TYPES[ma_type](src, length)
