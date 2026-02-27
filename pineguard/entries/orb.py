"""
ORB (Opening Range Breakout) Entry — 🔴 BLOQUANT (signal inversé D13).

Pine V.69.1:
  orb_long = close < orbHigh   // ← BUG: should be close > orbHigh
  orb_short = close > orbLow   // ← BUG: should be close < orbLow

The signal is INVERTED in Pine V.69.1 (D13).
This module implements the CORRECT logic, with a flag to reproduce the bug.

Excluded from SAFE_ENTRY_TYPES. Do NOT use for production backtesting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

Series = pd.Series


def compute_orb_signals(
    high: Series, low: Series, close: Series, open_: Series,
    orb_minutes: int = 30,
    session_start_hour: int = 9, session_start_minute: int = 30,
    reproduce_pine_bug: bool = False
) -> Dict[str, Series]:
    """
    Opening Range Breakout entry signals.

    The Opening Range is defined as the high/low of the first `orb_minutes`
    of each trading session.

    CORRECT logic:
      orb_long = close > orb_high (breakout above range)
      orb_short = close < orb_low (breakdown below range)

    PINE BUG (D13) - reproduce_pine_bug=True:
      orb_long = close < orb_high (INVERTED)
      orb_short = close > orb_low (INVERTED)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        open_: Open prices
        orb_minutes: Duration of opening range in minutes (default 30)
        session_start_hour: Session start hour (default 9)
        session_start_minute: Session start minute (default 30)
        reproduce_pine_bug: If True, reproduce the inverted signal bug (default False)

    Returns:
        Dict with:
            'orb_high': Opening range high
            'orb_low': Opening range low
            'orb_long': Long signal
            'orb_short': Short signal
            'is_orb_period': Boolean mask for ORB calculation period
    """
    # Identify session start bars
    idx = close.index
    hours = pd.Series(idx.hour, index=idx)
    minutes = pd.Series(idx.minute, index=idx)

    is_session_start = (hours == session_start_hour) & (minutes == session_start_minute)
    session_id = is_session_start.cumsum()

    # Calculate time since session start (in minutes)
    # For simplicity, use bar index within session
    time_in_session = pd.Series(0, index=idx, dtype=int)
    for sid in session_id.unique():
        mask = session_id == sid
        bars = mask.sum()
        time_in_session[mask] = range(bars)

    # Infer bar frequency in minutes
    if len(idx) > 1:
        freq_minutes = (idx[1] - idx[0]).total_seconds() / 60
    else:
        freq_minutes = 60  # default

    orb_bars = max(1, int(orb_minutes / freq_minutes))

    # Identify ORB period
    is_orb_period = time_in_session < orb_bars
    is_orb_period = pd.Series(is_orb_period, index=idx, name='is_orb_period')

    # Calculate ORB high and low for each session
    orb_high = pd.Series(np.nan, index=idx, name='orb_high')
    orb_low = pd.Series(np.nan, index=idx, name='orb_low')

    for sid in session_id.unique():
        session_mask = session_id == sid
        orb_mask = session_mask & is_orb_period

        if orb_mask.any():
            oh = high[orb_mask].max()
            ol = low[orb_mask].min()

            # Apply ORB levels after ORB period ends
            post_orb = session_mask & ~is_orb_period
            orb_high[post_orb] = oh
            orb_low[post_orb] = ol

    # Generate signals
    if reproduce_pine_bug:
        # D13: INVERTED signals (as in Pine V.69.1)
        orb_long = (close < orb_high) & (close.shift(1) >= orb_high.shift(1))
        orb_short = (close > orb_low) & (close.shift(1) <= orb_low.shift(1))
    else:
        # CORRECT logic
        orb_long = (close > orb_high) & (close.shift(1) <= orb_high.shift(1))
        orb_short = (close < orb_low) & (close.shift(1) >= orb_low.shift(1))

    return {
        'orb_high': orb_high,
        'orb_low': orb_low,
        'orb_long': orb_long.rename('orb_long'),
        'orb_short': orb_short.rename('orb_short'),
        'is_orb_period': is_orb_period,
    }
