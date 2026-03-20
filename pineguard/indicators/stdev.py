"""
Standard Deviation — Pine-equivalent implementation.

CRITICAL: Pine ta.stdev(src, n) uses POPULATION stdev (ddof=0).
  - Pine: ta.stdev(close, 20) → ddof=0
  - Python correct: rolling(20).std(ddof=0)
  - ❌ Python default: rolling(20).std() → ddof=1 (sample stdev)

Impact: ddof=1 gives slightly WIDER Bollinger Bands than Pine,
which means fewer crossover signals in Python.
For n=20: factor = sqrt(20/19) ≈ 1.026 → ~2.6% wider bands.
"""

import numpy as np
import pandas as pd

Series = pd.Series


def compute_stdev(src: Series, length: int) -> Series:
    """
    Pine: ta.stdev(src, length) — Population standard deviation (ddof=0).

    Args:
        src: Source series
        length: Lookback period

    Returns:
        Standard deviation series (population, ddof=0)
    """
    return src.rolling(window=length, min_periods=length).std(ddof=0)


def compute_stdev_sample(src: Series, length: int) -> Series:
    """
    Sample stdev (ddof=1) — FOR COMPARISON/DEBUGGING ONLY.

    This is the WRONG implementation for Pine compatibility.
    Pandas default: rolling().std() uses ddof=1.
    """
    return src.rolling(window=length, min_periods=length).std(ddof=1)


def stdev_divergence_factor(length: int) -> float:
    """
    Calculate the divergence factor between ddof=0 and ddof=1.

    factor = sqrt(n / (n-1))

    For n=20: 1.0263 → Bollinger bands ~2.6% wider with ddof=1
    For n=10: 1.0541 → ~5.4% wider
    For n=50: 1.0102 → ~1.0% wider

    Smaller n → bigger divergence → more impact on signals.
    """
    if length <= 1:
        return float('inf')
    return np.sqrt(length / (length - 1))
