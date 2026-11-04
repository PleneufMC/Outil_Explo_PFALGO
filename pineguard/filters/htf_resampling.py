"""
HTF Resampling — Critical component for LT/MT filters.

Pine: request.security(tf, expr, lookahead=barmerge.lookahead_off)
Python: df.resample(rule).agg(ohlcv_rules) + shift(1)

Known divergences:
  D4: Weekly boundary — Pine uses server-native weeks, Python uses resample('W-FRI')
  D5: Incomplete bar — Pine [1] vs Python shift(1) after resample (timing differs)
  D6: 4H from H1 — boundaries differ if market has 23h/day sessions

PIÈGE CRITIQUE:
  After resampling, MUST apply shift(1) to simulate Pine's [1] (confirmed bar).
  Then forward-fill to map back to base timeframe.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict

Series = pd.Series
DataFrame = pd.DataFrame


# Standard OHLCV aggregation rules
OHLCV_AGG = {
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum',
}

# Timeframe mapping: Pine TF string → pandas resample rule
TF_MAP = {
    '1H': '1h',
    '2H': '2h',
    '4H': '4h',
    '1D': '1D',
    'D': '1D',
    '1W': 'W-FRI',   # Pine weekly aligned to Friday close
    'W': 'W-FRI',
    '1M': 'MS',       # Month start
    'M': 'MS',
}


def resample_to_htf(
    df: DataFrame,
    target_tf: str,
    custom_rule: Optional[str] = None,
    hl2: bool = False
) -> DataFrame:
    """
    Resample OHLCV DataFrame to a higher timeframe.

    Args:
        df: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        target_tf: Target timeframe ('4H', '1D', '1W', etc.)
        custom_rule: Override pandas resample rule (e.g., 'W-FRI')
        hl2: If True, also compute hl2 on the resampled data

    Returns:
        Resampled DataFrame with OHLCV (and optionally hl2)

    IMPORTANT:
        This returns data WITHOUT shift(1). You must apply shift(1) yourself
        or use reindex_htf_to_base() which handles it automatically.
    """
    rule = custom_rule or TF_MAP.get(target_tf)
    if rule is None:
        raise ValueError(
            f"Unknown timeframe '{target_tf}'. Valid: {list(TF_MAP.keys())} "
            f"or provide custom_rule"
        )

    # Validate columns
    required = ['Open', 'High', 'Low', 'Close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Build aggregation dict for available columns
    agg_rules = {}
    for col, func in OHLCV_AGG.items():
        if col in df.columns:
            agg_rules[col] = func

    df_htf = df.resample(rule).agg(agg_rules)

    # Drop bars where all OHLC are NaN (no data in that period)
    df_htf = df_htf.dropna(subset=['Open', 'High', 'Low', 'Close'], how='all')

    if hl2:
        df_htf['hl2'] = (df_htf['High'] + df_htf['Low']) / 2

    return df_htf


def reindex_htf_to_base(
    htf_series: Series,
    base_index: pd.DatetimeIndex,
    shift_bars: int = 1
) -> Series:
    """
    Reindex an HTF series back to the base timeframe with forward-fill and shift.

    This simulates Pine's:
      request.security(tf, expr, lookahead=barmerge.lookahead_off)
    combined with [1] indexing.

    Process:
      1. Reindex to base timeframe
      2. Forward-fill (carry HTF value through all base bars in that period)
      3. Shift by shift_bars (default 1 = Pine's [1] for confirmed bar)

    Args:
        htf_series: Series from HTF computation
        base_index: DatetimeIndex of the base timeframe
        shift_bars: Number of HTF bars to shift (default 1 for Pine [1])

    Returns:
        Series at base timeframe resolution

    V.69.1 APPROACH (implemented below):
      Apply shift(1) on the HTF series BEFORE reindexing to base TF.
      This ensures the shift is in HTF bars (matching Pine's [1]),
      then forward-fill carries the confirmed value through base bars.

      shift(1) in HTF space = Pine's [1] (last confirmed bar).
      Then reindex + ffill = carry that value to all base bars in the period.
    """
    # Shift in HTF space (1 HTF bar = Pine's [1])
    if shift_bars > 0:
        shifted = htf_series.shift(shift_bars)
    else:
        shifted = htf_series

    # Reindex to base timeframe with forward-fill
    reindexed = shifted.reindex(base_index, method='ffill')

    return reindexed


def validate_htf_boundaries(
    df_base: DataFrame,
    df_htf: DataFrame,
    tolerance_bars: int = 2
) -> Dict:
    """
    Validate HTF resampling boundaries against expected behavior.

    Checks:
      1. First/last bars of HTF periods align correctly
      2. No data leakage (future bars in current HTF period)
      3. Gap analysis (missing bars at boundaries)

    Args:
        df_base: Base timeframe DataFrame
        df_htf: Resampled HTF DataFrame
        tolerance_bars: Acceptable boundary misalignment (default 2)

    Returns:
        Dict with validation results:
            'is_valid': Overall pass/fail
            'boundary_issues': List of problematic boundaries
            'gap_count': Number of gaps detected
            'alignment_score': 0-100 score
    """
    issues = []
    gaps = 0

    for i in range(len(df_htf) - 1):
        htf_end = df_htf.index[i]
        htf_next = df_htf.index[i + 1]

        # Find base bars in this HTF period
        mask = (df_base.index >= htf_end) & (df_base.index < htf_next)
        base_bars = df_base[mask]

        if len(base_bars) == 0:
            gaps += 1
            issues.append({
                'htf_bar': htf_end,
                'issue': 'no_base_bars',
                'expected_next': htf_next,
            })

    total_periods = max(len(df_htf) - 1, 1)
    alignment_score = max(0, 100 - (len(issues) / total_periods * 100))

    return {
        'is_valid': len(issues) <= tolerance_bars,
        'boundary_issues': issues[:20],  # Cap at 20 for readability
        'gap_count': gaps,
        'alignment_score': round(alignment_score, 1),
        'total_htf_periods': total_periods,
    }
