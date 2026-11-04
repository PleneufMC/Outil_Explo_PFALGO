"""
Donchian + Chikou Entry — 100% fiable.

Pine V.69.1:
  don_upperBand = ta.highest(high, don_length)
  don_lowerBand = ta.lowest(low, don_length)
  don_long = ta.crossover(close, don_upperBand[26])    // Chikou shift = 26
  don_short = ta.crossunder(close, don_lowerBand[26])

WHY 100% RELIABLE:
  - Rolling max/min are deterministic
  - Chikou shift = simple .shift(26)
  - Crossover = strict binary comparison
  - Only source of divergence = data differences (D1/D2/D3)
"""

import pandas as pd
from typing import Dict

Series = pd.Series


def compute_donchian_chikou_signals(
    high: Series, low: Series, close: Series,
    don_length: int = 20, chikou_shift: int = 26
) -> Dict[str, Series]:
    """
    Donchian + Chikou entry signals matching Pine V.69.1.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        don_length: Donchian channel period (default 20)
        chikou_shift: Chikou displacement bars (default 26, Ichimoku standard)

    Returns:
        Dict with:
            'don_upper': Upper Donchian band
            'don_lower': Lower Donchian band
            'shifted_upper': Upper band shifted by chikou_shift
            'shifted_lower': Lower band shifted by chikou_shift
            'don_long': Long entry signal (crossover)
            'don_short': Short entry signal (crossunder)

    Pine equivalence:
        don_long = ta.crossover(close, don_upperBand[26])
        → (close > shifted_upper) & (close.shift(1) <= shifted_upper.shift(1))
    """
    # Donchian channel
    don_upper = high.rolling(window=don_length, min_periods=don_length).max()
    don_lower = low.rolling(window=don_length, min_periods=don_length).min()

    # Chikou shift (shift forward = look back in time)
    shifted_upper = don_upper.shift(chikou_shift)
    shifted_lower = don_lower.shift(chikou_shift)

    # Crossover: close crosses ABOVE shifted upper band
    # Pine: ta.crossover(close, don_upperBand[26])
    # = close > shifted_upper AND close[1] <= shifted_upper[1]
    don_long = (close > shifted_upper) & (close.shift(1) <= shifted_upper.shift(1))

    # Crossunder: close crosses BELOW shifted lower band
    # Pine: ta.crossunder(close, don_lowerBand[26])
    # = close < shifted_lower AND close[1] >= shifted_lower[1]
    don_short = (close < shifted_lower) & (close.shift(1) >= shifted_lower.shift(1))

    return {
        'don_upper': don_upper.rename(f'Don_upper_{don_length}'),
        'don_lower': don_lower.rename(f'Don_lower_{don_length}'),
        'shifted_upper': shifted_upper.rename(f'Don_upper_shifted_{chikou_shift}'),
        'shifted_lower': shifted_lower.rename(f'Don_lower_shifted_{chikou_shift}'),
        'don_long': don_long.rename('don_long'),
        'don_short': don_short.rename('don_short'),
    }
