"""
PF AI Lab 5.0 — Walk-Forward Analysis
Validates strategy stability over time using sliding windows.
"""

import numpy as np
import pandas as pd
from ..backtest_engine import run_backtest
from ..metrics import compute_metrics


def run_walk_forward(df: pd.DataFrame, config: dict,
                     instrument: str = 'UK100',
                     n_windows: int = 5,
                     is_pct: float = 0.60,
                     overlap: float = 0.5) -> dict:
    """
    Walk-Forward Analysis: slide windows through data and check stability.

    Parameters:
        df: Full OHLCV DataFrame
        config: strategy configuration
        instrument: for transaction costs
        n_windows: number of windows
        is_pct: in-sample fraction of each window
        overlap: overlap between consecutive windows

    Returns:
        dict with:
            'windows': list of window results
            'stability_score': float (0-100)
            'degradation': bool (performance degrading over time)
            'avg_calmar': float
            'avg_pf': float
            'consistency': float (% of windows with PF > 1)
    """
    n = len(df)

    # Calculate window parameters
    window_size = int(n / (1 + (n_windows - 1) * (1 - overlap)))
    step = int(window_size * (1 - overlap))

    if window_size < 200:
        return {
            'windows': [],
            'stability_score': 0,
            'degradation': False,
            'avg_calmar': 0,
            'avg_pf': 0,
            'consistency': 0,
            'error': 'Not enough data for walk-forward analysis'
        }

    windows = []

    for i in range(n_windows):
        start = i * step
        end = min(start + window_size, n)

        if end - start < 100:
            break

        window_df = df.iloc[start:end].copy()

        # Split IS/OOS within window
        split = int(len(window_df) * is_pct)
        df_is = window_df.iloc[:split]
        df_oos = window_df.iloc[split:]

        # Run backtest on IS and OOS
        try:
            trades_is = run_backtest(df_is, config, instrument)
            metrics_is = compute_metrics(trades_is)

            trades_oos = run_backtest(df_oos, config, instrument)
            metrics_oos = compute_metrics(trades_oos)
        except Exception as e:
            metrics_is = {'n_trades': 0, 'calmar': 0, 'profit_factor': 0, 'max_dd': 0}
            metrics_oos = {'n_trades': 0, 'calmar': 0, 'profit_factor': 0, 'max_dd': 0}

        windows.append({
            'window': i + 1,
            'start': str(window_df.index[0]),
            'end': str(window_df.index[-1]),
            'n_bars': len(window_df),
            'metrics_is': metrics_is,
            'metrics_oos': metrics_oos,
        })

    if not windows:
        return {
            'windows': [],
            'stability_score': 0,
            'degradation': False,
            'avg_calmar': 0,
            'avg_pf': 0,
            'consistency': 0,
        }

    # Analyze stability
    oos_calmars = [w['metrics_oos']['calmar'] for w in windows]
    oos_pfs = [w['metrics_oos']['profit_factor'] for w in windows]
    oos_trades = [w['metrics_oos']['n_trades'] for w in windows]

    avg_calmar = np.mean(oos_calmars) if oos_calmars else 0
    avg_pf = np.mean(oos_pfs) if oos_pfs else 0

    # Consistency: % of windows with PF > 1
    profitable_windows = sum(1 for pf in oos_pfs if pf > 1)
    consistency = (profitable_windows / len(windows) * 100) if windows else 0

    # Degradation: is the performance declining over time?
    if len(oos_calmars) >= 3:
        # Simple trend: compare first half vs second half
        half = len(oos_calmars) // 2
        first_half = np.mean(oos_calmars[:half])
        second_half = np.mean(oos_calmars[half:])
        degradation = second_half < first_half * 0.5  # 50% decline
    else:
        degradation = False

    # Stability score (0-100)
    stability = min(100, max(0,
        consistency * 0.4 +
        min(avg_calmar * 20, 30) +
        min(avg_pf * 10, 20) +
        (10 if not degradation else 0)
    ))

    return {
        'windows': windows,
        'stability_score': round(stability, 1),
        'degradation': degradation,
        'avg_calmar': round(avg_calmar, 3),
        'avg_pf': round(avg_pf, 3),
        'consistency': round(consistency, 1),
    }


def format_walk_forward_report(wf_result: dict) -> str:
    """Format walk-forward analysis as readable text."""
    lines = []
    lines.append("=" * 60)
    lines.append("WALK-FORWARD ANALYSIS")
    lines.append("=" * 60)

    lines.append(f"\n  Stability Score: {wf_result['stability_score']:.1f}/100")
    lines.append(f"  Avg OOS Calmar:  {wf_result['avg_calmar']:.3f}")
    lines.append(f"  Avg OOS PF:      {wf_result['avg_pf']:.3f}")
    lines.append(f"  Consistency:     {wf_result['consistency']:.1f}% windows profitable")
    lines.append(f"  Degradation:     {'⚠️ YES' if wf_result['degradation'] else '✅ NO'}")

    lines.append(f"\n  {'Window':<8} {'Period':<35} {'OOS Trades':<12} {'OOS Calmar':<12} {'OOS PF':<10}")
    lines.append("  " + "-" * 77)

    for w in wf_result.get('windows', []):
        m = w['metrics_oos']
        lines.append(f"  #{w['window']:<7} {w['start'][:10]}→{w['end'][:10]:<23} "
                     f"{m['n_trades']:<12} {m['calmar']:<12.3f} {m['profit_factor']:<10.3f}")

    lines.append("")
    return '\n'.join(lines)
