"""
ADX (Average Directional Index) — Pine-equivalent implementation.

Pine:
  [diPlus, diMinus, adx] = ta.dmi(length, adxSmoothing)

Used in V.69.1 for ADX Regime Filter:
  - Trending regime: ADX > threshold (typically 25)
  - Ranging regime: ADX < threshold
  - Certain entry types work better in trending vs ranging markets
"""

import numpy as np
import pandas as pd
from typing import Dict

Series = pd.Series


def compute_adx(high: Series, low: Series, close: Series,
                length: int = 14, adx_smoothing: int = 14) -> Dict[str, Series]:
    """
    Pine: ta.dmi(length, adxSmoothing) — Directional Movement Index + ADX.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: DI period (default 14)
        adx_smoothing: ADX smoothing period (default 14)

    Returns:
        Dict with 'di_plus', 'di_minus', 'adx', 'dx'
    """
    from pineguard.indicators.moving_averages import compute_rma

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    # +DM and -DM
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index
    )

    # True Range
    from pineguard.indicators.atr import compute_tr
    tr = compute_tr(high, low, close)

    # Smoothed with RMA (Wilder's)
    atr_smooth = compute_rma(tr, length)
    plus_dm_smooth = compute_rma(plus_dm, length)
    minus_dm_smooth = compute_rma(minus_dm, length)

    # +DI and -DI
    di_plus = 100 * plus_dm_smooth / atr_smooth
    di_minus = 100 * minus_dm_smooth / atr_smooth

    # DX = |+DI - -DI| / (+DI + -DI) * 100
    di_sum = di_plus + di_minus
    dx = (di_plus - di_minus).abs() / di_sum.replace(0, np.nan) * 100

    # ADX = RMA(DX, adxSmoothing)
    adx = compute_rma(dx, adx_smoothing)

    return {
        'di_plus': di_plus.rename('DI_plus'),
        'di_minus': di_minus.rename('DI_minus'),
        'dx': dx.rename('DX'),
        'adx': adx.rename(f'ADX_{length}'),
    }
