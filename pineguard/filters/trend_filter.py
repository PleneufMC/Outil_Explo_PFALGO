"""
LT/MT Trend Filters — CONTINUOUS STATE (V.68.8+).

CRITICAL DISTINCTION:
  ❌ BEFORE V.68.8: crossover (true 1 single bar) → filter quasi-inoperant
  ✅ V.68.8+: continuous state → filter active as long as condition is true

Pine V.69.1:
  [MAvg_HTF, PMax_HTF] = request.security(tf, pmax_filter(hl2, length, mult))
  lt_trend_filter_long = MAvg_HTF[1] > PMax_HTF[1]    // confirmed bar
  lt_trend_filter_short = MAvg_HTF[1] < PMax_HTF[1]

Python:
  1. Resample to HTF
  2. Compute PMax on HTF data
  3. Shift(1) in HTF space (confirmed bar)
  4. Forward-fill to base timeframe
  5. Compare: MAvg > PMax = long filter, MAvg < PMax = short filter
"""

import pandas as pd
from typing import Dict, Optional

Series = pd.Series
DataFrame = pd.DataFrame


def compute_lt_filter(
    df: DataFrame,
    lt_tf: str = '1W',
    pmax_length: int = 10,
    pmax_mult: float = 3.0,
    pmax_ma_type: str = 'EMA'
) -> Dict[str, Series]:
    """
    Long-Term (LT) trend filter using PMax on Weekly (default).

    Pine: lt_trend_filter_long = MAvg_W[1] > PMax_W[1]

    Args:
        df: Base timeframe OHLCV DataFrame
        lt_tf: LT timeframe (default '1W')
        pmax_length: PMax period (default 10)
        pmax_mult: PMax multiplier (default 3.0)
        pmax_ma_type: MA type for PMax (default 'EMA')

    Returns:
        Dict with:
            'lt_long': Boolean — True when LT trend is bullish
            'lt_short': Boolean — True when LT trend is bearish
            'lt_mavg': LT PMax MAvg (at base TF resolution)
            'lt_pmax': LT PMax (at base TF resolution)
    """
    return _compute_trend_filter(
        df, lt_tf, pmax_length, pmax_mult, pmax_ma_type, prefix='lt'
    )


def compute_mt_filter(
    df: DataFrame,
    mt_tf: str = '4H',
    pmax_length: int = 10,
    pmax_mult: float = 3.0,
    pmax_ma_type: str = 'EMA'
) -> Dict[str, Series]:
    """
    Medium-Term (MT) trend filter using PMax on 4H (default).

    Pine: mt_trend_filter_long = MAvg_4H[1] > PMax_4H[1]

    Same logic as LT but on shorter timeframe.
    """
    return _compute_trend_filter(
        df, mt_tf, pmax_length, pmax_mult, pmax_ma_type, prefix='mt'
    )


def _compute_trend_filter(
    df: DataFrame,
    target_tf: str,
    pmax_length: int,
    pmax_mult: float,
    pmax_ma_type: str,
    prefix: str
) -> Dict[str, Series]:
    """Internal: compute trend filter for any timeframe."""
    from pineguard.filters.htf_resampling import resample_to_htf, reindex_htf_to_base
    from pineguard.indicators.pmax import compute_pmax

    # Step 1: Resample to HTF
    df_htf = resample_to_htf(df, target_tf, hl2=True)

    # Step 2: Compute PMax on HTF data
    pmax_result = compute_pmax(
        high=df_htf['High'],
        low=df_htf['Low'],
        close=df_htf['Close'],
        length=pmax_length,
        multiplier=pmax_mult,
        ma_type=pmax_ma_type,
    )

    mavg_htf = pmax_result['MAvg']
    pmax_htf = pmax_result['PMax']

    # Step 3: Shift(1) in HTF space (= Pine's [1] for confirmed bar)
    # Then reindex to base TF with forward-fill
    mavg_base = reindex_htf_to_base(mavg_htf, df.index, shift_bars=1)
    pmax_base = reindex_htf_to_base(pmax_htf, df.index, shift_bars=1)

    # Step 4: CONTINUOUS STATE comparison (V.68.8+)
    # NOT crossover — state is true for entire trend duration
    filter_long = mavg_base > pmax_base
    filter_short = mavg_base < pmax_base

    return {
        f'{prefix}_long': filter_long.rename(f'{prefix}_trend_long'),
        f'{prefix}_short': filter_short.rename(f'{prefix}_trend_short'),
        f'{prefix}_mavg': mavg_base.rename(f'{prefix}_mavg'),
        f'{prefix}_pmax': pmax_base.rename(f'{prefix}_pmax'),
    }
