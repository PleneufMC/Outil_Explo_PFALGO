"""
entry_allowed — THE 4-component gate that controls trade entries.

This is bug B9 (HAUTE sévérité): 15-25 phantom trades when absent.

Pine V.69.1 entry_allowed has 4 components:
  1. PMax direction filter (dir must match trade direction)
  2. Not already in position (no double entries)
  3. LT trend filter alignment (if enabled)
  4. MT trend filter alignment (if enabled)

All 4 must be True for an entry to be allowed.

entry_allowed_long = (pmax_dir == 1)         // PMax bullish
                   AND (not in_long_position) // No existing long
                   AND (lt_filter_long OR not lt_enabled)
                   AND (mt_filter_long OR not mt_enabled)

entry_allowed_short = mirror logic for shorts
"""

import pandas as pd
from typing import Dict, Optional

Series = pd.Series


def compute_entry_allowed(
    pmax_dir: Series,
    in_position_long: Optional[Series] = None,
    in_position_short: Optional[Series] = None,
    lt_filter_long: Optional[Series] = None,
    lt_filter_short: Optional[Series] = None,
    mt_filter_long: Optional[Series] = None,
    mt_filter_short: Optional[Series] = None,
    lt_enabled: bool = True,
    mt_enabled: bool = True,
    rsi_filter_long: Optional[Series] = None,
    rsi_filter_short: Optional[Series] = None,
    rsi_enabled: bool = False,
) -> Dict[str, Series]:
    """
    Compute entry_allowed gate — all 4+ components combined.

    This is the FIX for bug B9. Without this gate, the backtest engine
    generates 15-25 phantom trades per 23K-bar period.

    Args:
        pmax_dir: PMax direction series (1=bullish, -1=bearish)
        in_position_long: Boolean — True if already in long position
        in_position_short: Boolean — True if already in short position
        lt_filter_long: LT trend filter — True when bullish trend
        lt_filter_short: LT trend filter — True when bearish trend
        mt_filter_long: MT trend filter — True when bullish trend
        mt_filter_short: MT trend filter — True when bearish trend
        lt_enabled: Whether LT filter is active (default True)
        mt_enabled: Whether MT filter is active (default True)
        rsi_filter_long: RSI-based filter (optional additional gate)
        rsi_filter_short: RSI-based filter (optional additional gate)
        rsi_enabled: Whether RSI filter is active (default False)

    Returns:
        Dict with:
            'entry_allowed_long': Boolean series — True when long entry is permitted
            'entry_allowed_short': Boolean series — True when short entry is permitted
            'blocked_by': Dict of block reasons per component
    """
    idx = pmax_dir.index

    # Component 1: PMax direction
    pmax_long = (pmax_dir == 1)
    pmax_short = (pmax_dir == -1)

    # Component 2: Not already in position (default: always allowed)
    if in_position_long is None:
        not_in_long = pd.Series(True, index=idx)
    else:
        not_in_long = ~in_position_long

    if in_position_short is None:
        not_in_short = pd.Series(True, index=idx)
    else:
        not_in_short = ~in_position_short

    # Component 3: LT filter (pass-through if disabled)
    if lt_enabled and lt_filter_long is not None:
        lt_pass_long = lt_filter_long
    else:
        lt_pass_long = pd.Series(True, index=idx)

    if lt_enabled and lt_filter_short is not None:
        lt_pass_short = lt_filter_short
    else:
        lt_pass_short = pd.Series(True, index=idx)

    # Component 4: MT filter (pass-through if disabled)
    if mt_enabled and mt_filter_long is not None:
        mt_pass_long = mt_filter_long
    else:
        mt_pass_long = pd.Series(True, index=idx)

    if mt_enabled and mt_filter_short is not None:
        mt_pass_short = mt_filter_short
    else:
        mt_pass_short = pd.Series(True, index=idx)

    # Optional: RSI filter
    if rsi_enabled and rsi_filter_long is not None:
        rsi_pass_long = rsi_filter_long
    else:
        rsi_pass_long = pd.Series(True, index=idx)

    if rsi_enabled and rsi_filter_short is not None:
        rsi_pass_short = rsi_filter_short
    else:
        rsi_pass_short = pd.Series(True, index=idx)

    # Combine all components (AND logic)
    entry_allowed_long = (
        pmax_long & not_in_long & lt_pass_long & mt_pass_long & rsi_pass_long
    )
    entry_allowed_short = (
        pmax_short & not_in_short & lt_pass_short & mt_pass_short & rsi_pass_short
    )

    # Debug: count blocks by component
    total_bars = len(idx)
    blocked_by = {
        'pmax': {
            'long_blocked': int((~pmax_long).sum()),
            'short_blocked': int((~pmax_short).sum()),
        },
        'position': {
            'long_blocked': int((~not_in_long).sum()),
            'short_blocked': int((~not_in_short).sum()),
        },
        'lt_filter': {
            'long_blocked': int((~lt_pass_long).sum()) if lt_enabled else 0,
            'short_blocked': int((~lt_pass_short).sum()) if lt_enabled else 0,
        },
        'mt_filter': {
            'long_blocked': int((~mt_pass_long).sum()) if mt_enabled else 0,
            'short_blocked': int((~mt_pass_short).sum()) if mt_enabled else 0,
        },
        'rsi_filter': {
            'long_blocked': int((~rsi_pass_long).sum()) if rsi_enabled else 0,
            'short_blocked': int((~rsi_pass_short).sum()) if rsi_enabled else 0,
        },
        'total_bars': total_bars,
    }

    return {
        'entry_allowed_long': entry_allowed_long.rename('entry_allowed_long'),
        'entry_allowed_short': entry_allowed_short.rename('entry_allowed_short'),
        'blocked_by': blocked_by,
    }
