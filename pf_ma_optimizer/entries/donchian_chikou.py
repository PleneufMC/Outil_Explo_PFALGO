"""
PF AI Lab 5.0 — Donchian + Chikou Entry (100% reliable in Python)

Logic:
    don_upper = highest(high, don_length)
    don_lower = lowest(low, don_length)
    don_long = crossover(close, don_upper.shift(26))   # Chikou = shift 26
    don_short = crossunder(close, don_lower.shift(26))
"""

import numpy as np
import pandas as pd


def compute_donchian_chikou(df: pd.DataFrame, don_length: int = 20) -> dict:
    """
    Compute Donchian Channel + Chikou Span entry signals.

    Parameters:
        df: DataFrame with ['High', 'Low', 'Close']
        don_length: Donchian channel period

    Returns:
        dict with:
            'don_upper': np.ndarray — upper band
            'don_lower': np.ndarray — lower band
            'don_long': np.ndarray (bool) — long entry signals
            'don_short': np.ndarray (bool) — short entry signals
    """
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    n = len(df)

    # Donchian Channel
    don_upper = pd.Series(high).rolling(window=don_length, min_periods=don_length).max().values
    don_lower = pd.Series(low).rolling(window=don_length, min_periods=don_length).min().values

    # Chikou Span = shift 26 (look back 26 bars)
    # In Pine: don_long = ta.crossover(close, don_upperBand[26])
    # don_upperBand[26] means the value 26 bars ago
    don_upper_shifted = np.full(n, np.nan, dtype=float)
    don_lower_shifted = np.full(n, np.nan, dtype=float)

    for i in range(26, n):
        don_upper_shifted[i] = don_upper[i - 26]
        don_lower_shifted[i] = don_lower[i - 26]

    # Crossover/Crossunder
    don_long = np.zeros(n, dtype=bool)
    don_short = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if (not np.isnan(don_upper_shifted[i]) and not np.isnan(don_upper_shifted[i - 1])):
            don_long[i] = (close[i] > don_upper_shifted[i]) and (close[i - 1] <= don_upper_shifted[i - 1])

        if (not np.isnan(don_lower_shifted[i]) and not np.isnan(don_lower_shifted[i - 1])):
            don_short[i] = (close[i] < don_lower_shifted[i]) and (close[i - 1] >= don_lower_shifted[i - 1])

    return {
        'don_upper': don_upper,
        'don_lower': don_lower,
        'don_upper_shifted': don_upper_shifted,
        'don_lower_shifted': don_lower_shifted,
        'don_long': don_long,
        'don_short': don_short,
    }
