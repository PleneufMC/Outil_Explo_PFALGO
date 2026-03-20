"""
MM Cross Entry — Moving Average Crossover (V69.2).

Pine V.69.2:
  getMM_Cross_MA(src, length, ma_type) =>
      ma_type == 'TRIMA' ? ta.sma(ta.sma(src, ceil(length/2)), floor(length/2)+1)
                         : ta.ema(src, length)

  mm_cross_ma1 = getMM_Cross_MA(close, mm_cross_ma1_len, mm_cross_ma1_type)
  mm_cross_ma2 = getMM_Cross_MA(close, mm_cross_ma2_len, mm_cross_ma2_type)

  buy_mm_cross  = ta.crossover(mm_cross_ma1, mm_cross_ma2)
  sell_mm_cross = ta.crossunder(mm_cross_ma1, mm_cross_ma2)

Signal Logic:
  LONG:  MM1 (fast) crosses ABOVE MM2 (slow) — golden cross
  SHORT: MM1 (fast) crosses BELOW MM2 (slow) — death cross

These MAs are INDEPENDENT from the PMax MA.
Entry at market price (no limit/stop).
"""

import numpy as np
import pandas as pd
from typing import Dict
import math

Series = pd.Series


def _compute_ema(src: Series, length: int) -> Series:
    """Compute EMA matching Pine's ta.ema()."""
    return src.ewm(span=length, adjust=False).mean()


def _compute_trima(src: Series, length: int) -> Series:
    """
    Compute TRIMA (Triangular Moving Average) matching Pine V69.2 fix.

    Pine: ta.sma(ta.sma(src, math.ceil(length / 2)), math.floor(length / 2) + 1)

    This is a double SMA smoothing (true TRIMA), NOT a WMA.
    Bug in V69.1 getMA: TRIMA was implemented as ta.wma() — fixed in V69.2.
    """
    first_len = math.ceil(length / 2)
    second_len = math.floor(length / 2) + 1
    first_sma = src.rolling(window=first_len, min_periods=first_len).mean()
    trima = first_sma.rolling(window=second_len, min_periods=second_len).mean()
    return trima


def _compute_ma(src: Series, length: int, ma_type: str) -> Series:
    """
    Compute MA based on type (EMA or TRIMA only).

    Matches Pine V69.2 getMM_Cross_MA().
    """
    if ma_type == 'TRIMA':
        return _compute_trima(src, length)
    else:  # Default to EMA
        return _compute_ema(src, length)


def compute_mm_cross_signals(
    close: Series,
    ma1_type: str = 'EMA',
    ma1_length: int = 9,
    ma2_type: str = 'EMA',
    ma2_length: int = 21,
) -> Dict[str, Series]:
    """
    MM Cross entry signals matching Pine V.69.2.

    Crossover of 2 independent moving averages:
      - LONG:  MM1 crosses above MM2 (ta.crossover)
      - SHORT: MM1 crosses below MM2 (ta.crossunder)

    Args:
        close: Close prices
        ma1_type: Type for fast MA ('EMA' or 'TRIMA', default 'EMA')
        ma1_length: Period for fast MA (default 9)
        ma2_type: Type for slow MA ('EMA' or 'TRIMA', default 'EMA')
        ma2_length: Period for slow MA (default 21)

    Returns:
        Dict with:
            'mm_cross_ma1': Fast MA values
            'mm_cross_ma2': Slow MA values
            'mm_cross_long': Bullish crossover signals (boolean)
            'mm_cross_short': Bearish crossunder signals (boolean)
            'mm_cross_long_state': MM1 > MM2 state (for cancel logic)
            'mm_cross_short_state': MM1 < MM2 state (for cancel logic)
    """
    # Compute both MAs
    ma1 = _compute_ma(close, ma1_length, ma1_type)
    ma2 = _compute_ma(close, ma2_length, ma2_type)

    # Crossover: MA1 crosses above MA2
    # Pine ta.crossover(a, b) = a > b AND a[1] <= b[1]
    cross_long = (ma1 > ma2) & (ma1.shift(1) <= ma2.shift(1))

    # Crossunder: MA1 crosses below MA2
    # Pine ta.crossunder(a, b) = a < b AND a[1] >= b[1]
    cross_short = (ma1 < ma2) & (ma1.shift(1) >= ma2.shift(1))

    # State for cancel logic
    long_state = ma1 > ma2
    short_state = ma1 < ma2

    return {
        'mm_cross_ma1': ma1.rename('mm_cross_ma1'),
        'mm_cross_ma2': ma2.rename('mm_cross_ma2'),
        'mm_cross_long': cross_long.fillna(False).rename('mm_cross_long'),
        'mm_cross_short': cross_short.fillna(False).rename('mm_cross_short'),
        'mm_cross_long_state': long_state.fillna(False).rename('mm_cross_long_state'),
        'mm_cross_short_state': short_state.fillna(False).rename('mm_cross_short_state'),
    }
