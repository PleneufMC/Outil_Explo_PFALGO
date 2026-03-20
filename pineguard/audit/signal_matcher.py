"""
Signal Matcher — Compare raw signals between Pine and Python.

Operates at the signal level (before trade execution), which isolates
signal generation issues from execution engine issues.

For each bar, compares:
  - buy_trigger (PMax crossover)
  - sell_trigger (PMax crossunder)
  - entry signals (per entry type)
  - filter states (LT, MT, RSI, etc.)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

Series = pd.Series
DataFrame = pd.DataFrame


class SignalMatcher:
    """
    Compare signal series between Pine overlay export and Python computation.

    Usage:
        matcher = SignalMatcher()
        result = matcher.compare_signals(pine_signals, python_signals)
    """

    def __init__(self, time_tolerance_bars: int = 1):
        """
        Args:
            time_tolerance_bars: Allow ±N bars shift for signal matching (default 1)
        """
        self.time_tolerance = time_tolerance_bars

    def compare_signals(
        self,
        pine_signal: Series,
        python_signal: Series,
        signal_name: str = 'signal'
    ) -> Dict:
        """
        Compare two boolean signal series bar-by-bar.

        Args:
            pine_signal: Boolean series from Pine (exported)
            python_signal: Boolean series from Python computation
            signal_name: Name of the signal being compared

        Returns:
            Dict with:
                'both_true': Bars where both fire (matched)
                'python_only': Bars where only Python fires (phantom)
                'pine_only': Bars where only Pine fires (missed)
                'both_false': Bars where neither fires (agreement)
                'match_rate': Percentage of signal bars that match
                'summary': Statistics
        """
        # Align indices
        common_idx = pine_signal.index.intersection(python_signal.index)
        pine = pine_signal.reindex(common_idx).fillna(False)
        python = python_signal.reindex(common_idx).fillna(False)

        both_true = pine & python
        python_only = python & ~pine
        pine_only = pine & ~python
        both_false = ~pine & ~python

        total_signals = int(pine.sum() + python_only.sum())
        matched = int(both_true.sum())
        match_rate = (matched / max(int(pine.sum()), 1)) * 100

        # With tolerance: check if Python signal fires within ±N bars of Pine
        matched_with_tolerance = 0
        if self.time_tolerance > 0:
            for i in range(len(common_idx)):
                if pine.iloc[i]:
                    window_start = max(0, i - self.time_tolerance)
                    window_end = min(len(common_idx), i + self.time_tolerance + 1)
                    if python.iloc[window_start:window_end].any():
                        matched_with_tolerance += 1
            match_rate_tolerant = (matched_with_tolerance / max(int(pine.sum()), 1)) * 100
        else:
            match_rate_tolerant = match_rate

        return {
            'signal_name': signal_name,
            'both_true': both_true,
            'python_only': python_only,
            'pine_only': pine_only,
            'both_false': both_false,
            'summary': {
                'pine_signals': int(pine.sum()),
                'python_signals': int(python.sum()),
                'matched_exact': matched,
                'matched_tolerant': matched_with_tolerance if self.time_tolerance > 0 else matched,
                'python_only_count': int(python_only.sum()),
                'pine_only_count': int(pine_only.sum()),
                'match_rate_exact': round(match_rate, 1),
                'match_rate_tolerant': round(match_rate_tolerant, 1),
                'total_bars': len(common_idx),
            },
        }

    def compare_multi_signals(
        self,
        pine_signals: Dict[str, Series],
        python_signals: Dict[str, Series],
    ) -> Dict:
        """
        Compare multiple signal series at once.

        Args:
            pine_signals: Dict of {signal_name: Series} from Pine
            python_signals: Dict of {signal_name: Series} from Python

        Returns:
            Dict of comparison results per signal + overall summary
        """
        results = {}
        for name in pine_signals:
            if name in python_signals:
                results[name] = self.compare_signals(
                    pine_signals[name], python_signals[name], name
                )

        # Overall summary
        total_pine = sum(r['summary']['pine_signals'] for r in results.values())
        total_matched = sum(r['summary']['matched_exact'] for r in results.values())
        overall_match = (total_matched / max(total_pine, 1)) * 100

        return {
            'per_signal': results,
            'overall': {
                'total_pine_signals': total_pine,
                'total_python_signals': sum(
                    r['summary']['python_signals'] for r in results.values()
                ),
                'total_matched': total_matched,
                'overall_match_rate': round(overall_match, 1),
                'signals_compared': list(results.keys()),
            },
        }

    def compare_indicator_values(
        self,
        pine_values: Series,
        python_values: Series,
        indicator_name: str = 'indicator',
        tolerance_pct: float = 0.01,
    ) -> Dict:
        """
        Compare continuous indicator values (not boolean signals).

        Useful for comparing PMax, MAvg, ATR, RSI, etc.

        Args:
            pine_values: Numeric series from Pine
            python_values: Numeric series from Python
            indicator_name: Name for reporting
            tolerance_pct: Acceptable difference as % of Pine value

        Returns:
            Dict with comparison statistics
        """
        common_idx = pine_values.index.intersection(python_values.index)
        pine = pine_values.reindex(common_idx).dropna()
        python = python_values.reindex(common_idx).dropna()

        # Align after dropna
        common = pine.index.intersection(python.index)
        pine = pine.reindex(common)
        python = python.reindex(common)

        if len(common) == 0:
            return {'indicator_name': indicator_name, 'error': 'No overlapping data'}

        diff = python - pine
        diff_pct = (diff / pine.abs().replace(0, np.nan)) * 100

        within_tolerance = (diff_pct.abs() <= tolerance_pct).sum()
        match_rate = (within_tolerance / len(common)) * 100

        return {
            'indicator_name': indicator_name,
            'bars_compared': len(common),
            'mean_diff': round(diff.mean(), 6),
            'std_diff': round(diff.std(), 6),
            'max_diff': round(diff.abs().max(), 6),
            'mean_diff_pct': round(diff_pct.mean(), 4),
            'max_diff_pct': round(diff_pct.abs().max(), 4),
            'within_tolerance_pct': round(match_rate, 1),
            'tolerance_used': tolerance_pct,
            'correlation': round(pine.corr(python), 6),
        }
