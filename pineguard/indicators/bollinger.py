"""
Bollinger Bands — Pine-equivalent implementation.

Pine: ta.bb(close, length, mult)
  basis = ta.sma(close, length)
  dev = mult * ta.stdev(close, length)    // ddof=0 (population)
  upper = basis + dev
  lower = basis - dev

CRITICAL: Pine uses ddof=0 for stdev. Pandas default is ddof=1.
Fix D10: Always use ddof=0 in compute_stdev.
"""

import pandas as pd
from typing import Dict

Series = pd.Series


def compute_bollinger(close: Series, length: int = 20,
                      mult: float = 2.0) -> Dict[str, Series]:
    """
    Pine: ta.bb(close, length, mult) — Bollinger Bands.

    Args:
        close: Close prices
        length: SMA period (default 20)
        mult: Standard deviation multiplier (default 2.0)

    Returns:
        Dict with 'basis', 'upper', 'lower', 'bandwidth', 'pct_b'
    """
    from pineguard.indicators.moving_averages import compute_sma
    from pineguard.indicators.stdev import compute_stdev

    basis = compute_sma(close, length)
    dev = mult * compute_stdev(close, length)  # ddof=0!
    upper = basis + dev
    lower = basis - dev

    # Bandwidth = (upper - lower) / basis * 100
    bandwidth = ((upper - lower) / basis) * 100

    # %B = (close - lower) / (upper - lower)
    pct_b = (close - lower) / (upper - lower)

    return {
        'basis': basis.rename('BB_basis'),
        'upper': upper.rename('BB_upper'),
        'lower': lower.rename('BB_lower'),
        'bandwidth': bandwidth.rename('BB_bandwidth'),
        'pct_b': pct_b.rename('BB_pctB'),
    }
