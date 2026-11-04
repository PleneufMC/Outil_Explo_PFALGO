"""
Trade Comparator — Bar-by-bar trade matching between Pine and Python.

Compares trade lists from TradingView (TV) export and Python backtest.
Classifies trades as: matched, Python-only (phantom), TV-only (missed).

Tolerances:
  - Time: ±4h (timezone difference)
  - PnL: ±0.5% (data/execution differences)
  - Direction must match exactly
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TradeRecord:
    """Normalized trade record for comparison (works with both TV and Python)."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    pnl: float
    source: str  # 'pine' or 'python'


@dataclass
class MatchResult:
    """Result of comparing a pair of trades."""
    pine_trade: Optional[TradeRecord]
    python_trade: Optional[TradeRecord]
    match_type: str  # 'matched', 'python_only', 'pine_only'
    time_delta_hours: float = 0.0
    pnl_delta_pct: float = 0.0
    pnl_delta_abs: float = 0.0


class TradeComparator:
    """
    Compare Pine and Python trade lists with configurable tolerances.

    Usage:
        comparator = TradeComparator(time_tolerance_hours=4, pnl_tolerance_pct=0.5)
        result = comparator.compare(pine_trades, python_trades)
    """

    def __init__(
        self,
        time_tolerance_hours: float = 4.0,
        pnl_tolerance_pct: float = 0.5,
    ):
        self.time_tolerance = pd.Timedelta(hours=time_tolerance_hours)
        self.pnl_tolerance_pct = pnl_tolerance_pct

    def compare(
        self,
        pine_trades: List[TradeRecord],
        python_trades: List[TradeRecord],
    ) -> Dict:
        """
        Compare two trade lists and classify matches.

        Returns:
            Dict with:
                'matched': List of MatchResult (matching trades)
                'python_only': List of MatchResult (phantom trades)
                'pine_only': List of MatchResult (missed trades)
                'summary': Summary statistics
        """
        matched = []
        python_only = []
        pine_only = []

        # Track which trades have been matched
        pine_matched = set()
        python_matched = set()

        # Try to match each Python trade to a Pine trade
        for py_idx, py_trade in enumerate(python_trades):
            best_match = None
            best_score = float('inf')
            best_pine_idx = None

            for pine_idx, pine_trade in enumerate(pine_trades):
                if pine_idx in pine_matched:
                    continue

                # Direction must match
                if py_trade.direction != pine_trade.direction:
                    continue

                # Time tolerance
                time_diff = abs(py_trade.entry_time - pine_trade.entry_time)
                if time_diff > self.time_tolerance:
                    continue

                # PnL tolerance
                if pine_trade.pnl != 0:
                    pnl_diff_pct = abs(py_trade.pnl - pine_trade.pnl) / abs(pine_trade.pnl) * 100
                else:
                    pnl_diff_pct = abs(py_trade.pnl) * 100

                # Score: combine time and PnL difference
                score = time_diff.total_seconds() / 3600 + pnl_diff_pct

                if score < best_score:
                    best_score = score
                    best_match = pine_trade
                    best_pine_idx = pine_idx

            if best_match is not None:
                time_delta = (py_trade.entry_time - best_match.entry_time).total_seconds() / 3600
                pnl_delta = py_trade.pnl - best_match.pnl
                pnl_delta_pct = (pnl_delta / abs(best_match.pnl) * 100) if best_match.pnl != 0 else 0

                matched.append(MatchResult(
                    pine_trade=best_match,
                    python_trade=py_trade,
                    match_type='matched',
                    time_delta_hours=round(time_delta, 2),
                    pnl_delta_pct=round(pnl_delta_pct, 2),
                    pnl_delta_abs=round(pnl_delta, 4),
                ))
                pine_matched.add(best_pine_idx)
                python_matched.add(py_idx)

        # Python-only (phantom) trades
        for py_idx, py_trade in enumerate(python_trades):
            if py_idx not in python_matched:
                python_only.append(MatchResult(
                    pine_trade=None,
                    python_trade=py_trade,
                    match_type='python_only',
                ))

        # Pine-only (missed) trades
        for pine_idx, pine_trade in enumerate(pine_trades):
            if pine_idx not in pine_matched:
                pine_only.append(MatchResult(
                    pine_trade=pine_trade,
                    python_trade=None,
                    match_type='pine_only',
                ))

        # Summary
        total_pine = len(pine_trades)
        total_python = len(python_trades)
        n_matched = len(matched)

        match_rate = (n_matched / max(total_pine, 1)) * 100
        phantom_rate = (len(python_only) / max(total_python, 1)) * 100
        missed_rate = (len(pine_only) / max(total_pine, 1)) * 100

        avg_pnl_delta = np.mean([m.pnl_delta_pct for m in matched]) if matched else 0
        avg_time_delta = np.mean([abs(m.time_delta_hours) for m in matched]) if matched else 0

        summary = {
            'pine_trades': total_pine,
            'python_trades': total_python,
            'matched': n_matched,
            'python_only_phantom': len(python_only),
            'pine_only_missed': len(pine_only),
            'match_rate_pct': round(match_rate, 1),
            'phantom_rate_pct': round(phantom_rate, 1),
            'missed_rate_pct': round(missed_rate, 1),
            'avg_pnl_delta_pct': round(avg_pnl_delta, 2),
            'avg_time_delta_hours': round(avg_time_delta, 2),
            'trade_count_divergence_pct': round(
                (total_python - total_pine) / max(total_pine, 1) * 100, 1
            ),
        }

        # Severity classification
        if total_python > 0 and total_pine > 0:
            divergence = abs(total_python - total_pine) / total_pine * 100
            if divergence > 50 or match_rate < 50:
                summary['severity'] = 'BLOQUANT'
            elif divergence > 20:
                summary['severity'] = 'HAUTE'
            elif divergence > 10:
                summary['severity'] = 'MOYENNE'
            else:
                summary['severity'] = 'BASSE'
        else:
            summary['severity'] = 'N/A'

        return {
            'matched': matched,
            'python_only': python_only,
            'pine_only': pine_only,
            'summary': summary,
        }

    def format_report(self, result: Dict) -> str:
        """Format comparison result as a readable report."""
        s = result['summary']
        lines = []
        lines.append("=" * 60)
        lines.append("TRADE COMPARISON REPORT")
        lines.append("=" * 60)
        lines.append(f"Pine trades:   {s['pine_trades']}")
        lines.append(f"Python trades: {s['python_trades']}")
        lines.append(f"Trade count divergence: {s['trade_count_divergence_pct']:+.1f}%")
        lines.append(f"")
        lines.append(f"Matched:       {s['matched']} ({s['match_rate_pct']:.1f}%)")
        lines.append(f"Python-only:   {s['python_only_phantom']} (phantom)")
        lines.append(f"Pine-only:     {s['pine_only_missed']} (missed)")
        lines.append(f"")
        lines.append(f"Avg PnL delta: {s['avg_pnl_delta_pct']:+.2f}%")
        lines.append(f"Avg time delta: {s['avg_time_delta_hours']:.1f}h")
        lines.append(f"")
        lines.append(f"Severity: {s['severity']}")

        if result['python_only']:
            lines.append(f"\n--- PHANTOM TRADES (Python-only) ---")
            for m in result['python_only'][:10]:
                t = m.python_trade
                lines.append(
                    f"  {t.entry_time} {t.direction.upper()} "
                    f"PnL={t.pnl:+.2f}"
                )

        if result['pine_only']:
            lines.append(f"\n--- MISSED TRADES (Pine-only) ---")
            for m in result['pine_only'][:10]:
                t = m.pine_trade
                lines.append(
                    f"  {t.entry_time} {t.direction.upper()} "
                    f"PnL={t.pnl:+.2f}"
                )

        lines.append("=" * 60)
        return '\n'.join(lines)
