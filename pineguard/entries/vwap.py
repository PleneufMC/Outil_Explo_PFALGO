"""
VWAP Entry — ❓ Non audité (session-dependent, non reproductible).

Pine V.69.1:
  vwap_val = ta.vwap(hlc3)  // session-anchored VWAP
  vwap_long = ta.crossover(close, vwap_val)
  vwap_short = ta.crossunder(close, vwap_val)

ISSUES:
  - VWAP resets at session start (exchange-specific)
  - Session boundaries differ between TV and Python
  - Not in SAFE_ENTRY_TYPES — use with caution
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

Series = pd.Series


def compute_vwap(
    high: Series, low: Series, close: Series, volume: Series,
    session_start_hour: int = 0
) -> Series:
    """
    Session-anchored VWAP matching Pine ta.vwap(hlc3).

    VWAP resets at each session start (defined by session_start_hour).
    Formula: cumsum(typical_price * volume) / cumsum(volume)
    where typical_price = (high + low + close) / 3

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        session_start_hour: Hour when session resets (default 0 = midnight)

    Returns:
        VWAP series
    """
    hlc3 = (high + low + close) / 3

    # Detect session boundaries
    hours = pd.Series(high.index.hour, index=high.index)
    session_mask = hours == session_start_hour
    # Also detect when hour changes TO session_start_hour
    prev_mask = session_mask.shift(1)
    prev_mask = prev_mask.fillna(False).infer_objects(copy=False).astype(bool)
    session_start = session_mask & (~prev_mask)

    # Assign session groups
    session_id = session_start.cumsum()

    # Cumulative sums within each session
    tp_vol = hlc3 * volume
    cum_tp_vol = tp_vol.groupby(session_id).cumsum()
    cum_vol = volume.groupby(session_id).cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    vwap.name = 'VWAP'
    return vwap


def compute_vwap_signals(
    high: Series, low: Series, close: Series, volume: Series,
    session_start_hour: int = 0
) -> Dict[str, Series]:
    """
    VWAP entry signals.

    Long: close crosses above VWAP
    Short: close crosses below VWAP

    ⚠️ NOT in SAFE_ENTRY_TYPES — session dependency makes
    Pine↔Python comparison unreliable.
    """
    vwap = compute_vwap(high, low, close, volume, session_start_hour)

    vwap_long = (close > vwap) & (close.shift(1) <= vwap.shift(1))
    vwap_short = (close < vwap) & (close.shift(1) >= vwap.shift(1))

    return {
        'vwap': vwap,
        'vwap_long': vwap_long.rename('vwap_long'),
        'vwap_short': vwap_short.rename('vwap_short'),
    }
