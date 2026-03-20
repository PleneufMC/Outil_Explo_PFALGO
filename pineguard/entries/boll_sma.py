"""
Bollinger + SMA Entry — ✅ ~90% fiable (D10 fixed: ddof=0, entry_allowed available).

Pine V.69.1:
  [basis, upper, lower] = ta.bb(close, boll_length, boll_mult)
  smaLine = ta.sma(close, boll_sma_length)

  boll_long = ta.crossover(close, lower) and close > smaLine
  boll_short = ta.crossunder(close, upper) and close < smaLine

FIXES APPLIED (D10):
  - stdev ddof=0 (population) via compute_stdev — matches Pine ta.stdev ✅
  - entry_allowed gate available in filters/entry_allowed.py ✅

RESIDUAL ~10% ÉCART:
  - entry_allowed must be applied externally by the caller/engine
  - Edge cases around SMA warmup period
"""

import numpy as np
import pandas as pd
from typing import Dict

Series = pd.Series


def compute_boll_sma_signals(
    close: Series,
    boll_length: int = 20,
    boll_mult: float = 2.0,
    sma_length: int = 50
) -> Dict[str, Series]:
    """
    Bollinger + SMA entry signals matching Pine V.69.1.

    Long: close crosses above lower Bollinger band AND close > SMA
    Short: close crosses below upper Bollinger band AND close < SMA

    Args:
        close: Close prices
        boll_length: Bollinger period (default 20)
        boll_mult: Bollinger stdev multiplier (default 2.0)
        sma_length: SMA filter period (default 50)

    Returns:
        Dict with:
            'bb_upper', 'bb_lower', 'bb_basis': Bollinger bands
            'sma_filter': SMA line
            'boll_long': Long signal (crossover lower + above SMA)
            'boll_short': Short signal (crossunder upper + below SMA)

    STATUS:
        - ✅ compute_bollinger uses ddof=0 (population stdev) via compute_stdev
        - ✅ entry_allowed gating available in filters/entry_allowed.py (apply externally)
    """
    from pineguard.indicators.bollinger import compute_bollinger
    from pineguard.indicators.moving_averages import compute_sma

    # Bollinger Bands (uses ddof=0 via compute_stdev)
    bb = compute_bollinger(close, boll_length, boll_mult)
    upper = bb['upper']
    lower = bb['lower']
    basis = bb['basis']

    # SMA filter
    sma_filter = compute_sma(close, sma_length)

    # Crossover: close crosses above lower band
    cross_above_lower = (close > lower) & (close.shift(1) <= lower.shift(1))

    # Crossunder: close crosses below upper band
    cross_below_upper = (close < upper) & (close.shift(1) >= upper.shift(1))

    # Combined signals with SMA filter
    boll_long = cross_above_lower & (close > sma_filter)
    boll_short = cross_below_upper & (close < sma_filter)

    return {
        'bb_upper': upper,
        'bb_lower': lower,
        'bb_basis': basis,
        'sma_filter': sma_filter.rename(f'SMA_{sma_length}'),
        'boll_long': boll_long.rename('boll_long'),
        'boll_short': boll_short.rename('boll_short'),
    }
