"""
Filters module — LT/MT Trend, HTF Resampling, entry_allowed, ADX Regime.

CRITICAL: V.68.8+ uses CONTINUOUS STATE for LT/MT filters, NOT crossover.
  ✅ lt_trend_filter_long = MAvg_HTF[1] > PMax_HTF[1]  (true for entire trend)
  ❌ lt_trend_filter_long = ta.crossover(MAvg_HTF, PMax_HTF)  (true only 1 bar)
"""

from pineguard.filters.htf_resampling import resample_to_htf, reindex_htf_to_base
from pineguard.filters.trend_filter import compute_lt_filter, compute_mt_filter
from pineguard.filters.entry_allowed import compute_entry_allowed
from pineguard.filters.adx_regime import compute_adx_regime_filter

__all__ = [
    'resample_to_htf', 'reindex_htf_to_base',
    'compute_lt_filter', 'compute_mt_filter',
    'compute_entry_allowed', 'compute_adx_regime_filter',
]
