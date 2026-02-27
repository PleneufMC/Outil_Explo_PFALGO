"""
ADX Regime Filter — Trending vs Ranging market classification.

Pine V.69.2:
  ADX > threshold → Trending regime (favor Fractal, Boll, MM Cross)
  ADX < threshold → Ranging regime (favor ORB, VWAP)

Some entry types perform better in specific regimes.
This filter gates entries based on market regime.
"""

import pandas as pd
from typing import Dict, Optional

Series = pd.Series
DataFrame = pd.DataFrame


# Entry type / regime compatibility matrix
REGIME_COMPATIBILITY = {
    'Donc+Chikou': {'trending': True, 'ranging': True},    # Works in both
    'RSI Divergence': {'trending': True, 'ranging': True},  # Works in both (polyvalent)
    'Boll+SMA': {'trending': True, 'ranging': False},       # Better trending
    'Fractal': {'trending': True, 'ranging': False},         # Better trending
    'VWAP': {'trending': False, 'ranging': True},            # Better ranging
    'ORB': {'trending': False, 'ranging': True},             # Better ranging
    'MM Cross': {'trending': True, 'ranging': False},        # V69.2: Trend signal
}


def compute_adx_regime_filter(
    high: Series, low: Series, close: Series,
    adx_length: int = 14,
    adx_smoothing: int = 14,
    trending_threshold: float = 25.0,
    entry_type: str = 'Donc+Chikou'
) -> Dict[str, Series]:
    """
    ADX Regime Filter — gates entries based on market regime.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        adx_length: ADX DI period (default 14)
        adx_smoothing: ADX smoothing period (default 14)
        trending_threshold: ADX level above which market is 'trending' (default 25)
        entry_type: Entry type to check regime compatibility

    Returns:
        Dict with:
            'adx': ADX values
            'is_trending': Boolean — ADX > threshold
            'is_ranging': Boolean — ADX <= threshold
            'regime_allowed': Boolean — entry type compatible with current regime
    """
    from pineguard.indicators.adx import compute_adx

    adx_result = compute_adx(high, low, close, adx_length, adx_smoothing)
    adx = adx_result['adx']

    is_trending = adx > trending_threshold
    is_ranging = ~is_trending

    # Check regime compatibility for the entry type
    compat = REGIME_COMPATIBILITY.get(entry_type, {'trending': True, 'ranging': True})

    regime_allowed = pd.Series(False, index=close.index)
    if compat['trending']:
        regime_allowed = regime_allowed | is_trending
    if compat['ranging']:
        regime_allowed = regime_allowed | is_ranging

    return {
        'adx': adx,
        'di_plus': adx_result['di_plus'],
        'di_minus': adx_result['di_minus'],
        'is_trending': is_trending.rename('is_trending'),
        'is_ranging': is_ranging.rename('is_ranging'),
        'regime_allowed': regime_allowed.rename('regime_allowed'),
        'regime': is_trending.map({True: 'trending', False: 'ranging'}).rename('regime'),
    }
