"""
Donchian Channel — Pine-equivalent implementation.

Pine:
  upper = ta.highest(high, length)
  lower = ta.lowest(low, length)
  basis = (upper + lower) / 2

Simple and deterministic — no smoothing, no edge cases.
100% reproducible between Pine and Python if data matches.
"""

import pandas as pd
from typing import Dict

Series = pd.Series


def compute_donchian(high: Series, low: Series,
                     length: int = 20) -> Dict[str, Series]:
    """
    Pine Donchian Channel.

    Args:
        high: High prices
        low: Low prices
        length: Lookback period (default 20)

    Returns:
        Dict with 'upper', 'lower', 'basis'
    """
    upper = high.rolling(window=length, min_periods=length).max()
    lower = low.rolling(window=length, min_periods=length).min()
    basis = (upper + lower) / 2

    return {
        'upper': upper.rename(f'Don_upper_{length}'),
        'lower': lower.rename(f'Don_lower_{length}'),
        'basis': basis.rename(f'Don_basis_{length}'),
    }
