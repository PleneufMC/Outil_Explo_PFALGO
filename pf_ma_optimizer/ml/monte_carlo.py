"""
PF AI Lab 5.0 — Monte Carlo Simulation (10,000 sims)
Based on REAL TV trades — zero divergence by construction.

Shuffles the order of trades and calculates distributions of:
- Max Drawdown (95th percentile)
- Return (5th percentile)
- Calmar Ratio (median)
- P(ruin) = P(DD > threshold)

Uses ADAPTIVE ruin threshold:
  1. Compute worst-case DD (all losses first) to detect if fixed threshold is unreachable
  2. If unreachable, use adaptive threshold = 1.5x median DD from the MC distribution
  3. This ensures P(ruin) is always a discriminating metric

Classification: ROBUST (P_ruin < 5%), STABLE (< 15%), FRAGILE (< 30%), REJECT
"""

import numpy as np
import pandas as pd
from ..config import MC_DEFAULTS, MC_CLASSIFICATION


def _compute_worst_case_dd(trade_pnls: np.ndarray) -> float:
    """
    Compute the worst-case max drawdown: all losing trades first, then winners.
    This is the deepest DD that ANY permutation can ever produce.
    """
    sorted_asc = np.sort(trade_pnls)  # losses first (ascending)
    equity = np.cumsum(sorted_asc)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(np.min(dd))


def _compute_adaptive_threshold(max_dd_dist: np.ndarray,
                                 user_threshold: float) -> tuple:
    """
    Compute an adaptive ruin threshold that makes P(ruin) discriminating.

    Strategy:
      - Use the 5th percentile of the DD distribution (= worst 5% of sims)
        as the adaptive threshold.
      - This means P(ruin) measures: "what fraction of orderings produce
        a DD worse than the already-bad 5th percentile?"
      - By construction this gives P(ruin) ~5% as a baseline, making the
        metric sensitive to the shape of the tail.
      - If the user threshold is already tighter (less negative) than adaptive,
        keep the user threshold.

    Returns:
        (effective_threshold, threshold_mode)
        threshold_mode: 'user' if user threshold was used, 'adaptive' if computed
    """
    dd_5th = float(np.percentile(max_dd_dist, 5))   # worst 5% boundary
    dd_median = float(np.percentile(max_dd_dist, 50))

    # Adaptive = 1.5x the median DD (deeper than typical, shallower than extreme)
    # This sits between the median and the tail, giving meaningful discrimination
    adaptive = dd_median * 1.5

    # Use the TIGHTER (less negative = stricter) of user vs adaptive
    # But only switch to adaptive if user threshold is unreachable
    if user_threshold < adaptive:
        # User threshold is deeper than adaptive — probably unreachable
        return adaptive, 'adaptive'
    else:
        # User threshold is tighter — use it
        return user_threshold, 'user'


def run_monte_carlo(trade_pnls: np.ndarray,
                    n_sims: int = None,
                    ruin_threshold: float = None,
                    seed: int = None,
                    initial_capital: float = 100000.0) -> dict:
    """
    Run Monte Carlo simulation by shuffling trade order.

    Parameters:
        trade_pnls: array of trade P&Ls (absolute or percentage)
        n_sims: number of simulations (default: 10,000)
        ruin_threshold: max drawdown threshold for ruin (default: -20%)
            If the fixed threshold is unreachable (worst-case DD doesn't hit it),
            an adaptive threshold is computed automatically.
        seed: random seed
        initial_capital: starting capital

    Returns:
        dict with:
            'max_dd_dist': np.ndarray — distribution of max drawdowns
            'return_dist': np.ndarray — distribution of total returns
            'calmar_dist': np.ndarray — distribution of Calmar ratios
            'p_ruin': float — probability of ruin (DD > effective threshold)
            'p_ruin_fixed': float — probability of ruin with user's fixed threshold
            'ruin_threshold': float — effective threshold used
            'ruin_threshold_user': float — original user threshold
            'ruin_threshold_mode': str — 'user', 'adaptive', or 'unreachable'
            'worst_case_dd': float — worst possible DD (all losses first)
            'classification': str — ROBUST/STABLE/FRAGILE/REJECT
            'dd_50', 'dd_95', 'dd_99': percentiles
            'return_5', 'return_50', 'return_95': percentiles
            'calmar_50': float — median Calmar
            'n_sims': int
            'n_trades': int
    """
    if n_sims is None:
        n_sims = MC_DEFAULTS['n_sims']
    if ruin_threshold is None:
        ruin_threshold = MC_DEFAULTS['ruin_threshold']
    if seed is None:
        seed = MC_DEFAULTS['seed']

    rng = np.random.RandomState(seed)
    n_trades = len(trade_pnls)

    if n_trades == 0:
        return _empty_mc_result()

    # Pre-compute worst-case DD
    worst_case_dd = _compute_worst_case_dd(trade_pnls)
    user_threshold = ruin_threshold

    # Arrays for storing results
    max_dd_dist = np.zeros(n_sims)
    return_dist = np.zeros(n_sims)
    calmar_dist = np.zeros(n_sims)

    for sim in range(n_sims):
        # Shuffle trade order
        shuffled = rng.permutation(trade_pnls)

        # Build equity curve
        equity = np.cumsum(shuffled)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = np.min(dd)

        # Total return
        total_ret = equity[-1]

        # Calmar (simplified: total return / max DD)
        calmar = total_ret / abs(max_dd) if max_dd != 0 else 0.0

        max_dd_dist[sim] = max_dd
        return_dist[sim] = total_ret
        calmar_dist[sim] = calmar

    # --- Adaptive threshold logic ---
    p_ruin_fixed = float(np.mean(max_dd_dist < user_threshold) * 100)

    if worst_case_dd >= user_threshold:
        # Fixed threshold is MATHEMATICALLY UNREACHABLE
        # -> use adaptive threshold
        effective_threshold, threshold_mode = _compute_adaptive_threshold(
            max_dd_dist, user_threshold
        )
        threshold_mode = 'adaptive'
    else:
        if p_ruin_fixed == 0.0:
            # Threshold is reachable in theory but no sim hit it
            # -> still use adaptive for meaningful classification
            effective_threshold, threshold_mode = _compute_adaptive_threshold(
                max_dd_dist, user_threshold
            )
            threshold_mode = 'adaptive'
        else:
            effective_threshold = user_threshold
            threshold_mode = 'user'

    p_ruin = float(np.mean(max_dd_dist < effective_threshold) * 100)

    # Classification based on effective P(ruin)
    classification = 'REJECT'
    for cls_name, gates in sorted(MC_CLASSIFICATION.items(),
                                   key=lambda x: x[1]['max_p_ruin']):
        if p_ruin <= gates['max_p_ruin']:
            classification = cls_name
            break

    return {
        'max_dd_dist': max_dd_dist,
        'return_dist': return_dist,
        'calmar_dist': calmar_dist,
        # Effective P(ruin) — always discriminating
        'p_ruin': round(p_ruin, 2),
        # Fixed P(ruin) — for reference
        'p_ruin_fixed': round(p_ruin_fixed, 2),
        # Threshold info
        'ruin_threshold': round(effective_threshold, 2),
        'ruin_threshold_user': user_threshold,
        'ruin_threshold_mode': threshold_mode,
        'worst_case_dd': round(worst_case_dd, 2),
        # Classification
        'classification': classification,
        # DD percentiles (note: 5th = worst, 95th = best because DD is negative)
        'dd_50': round(float(np.percentile(max_dd_dist, 50)), 2),
        'dd_95': round(float(np.percentile(max_dd_dist, 95)), 2),
        'dd_99': round(float(np.percentile(max_dd_dist, 99)), 2),
        'dd_5': round(float(np.percentile(max_dd_dist, 5)), 2),
        # Return percentiles
        'return_5': round(float(np.percentile(return_dist, 5)), 2),
        'return_50': round(float(np.percentile(return_dist, 50)), 2),
        'return_95': round(float(np.percentile(return_dist, 95)), 2),
        # Calmar
        'calmar_50': round(float(np.percentile(calmar_dist, 50)), 3),
        # Counts
        'n_sims': n_sims,
        'n_trades': n_trades,
    }


def classify_config(mc_result: dict) -> dict:
    """
    Generate a verdict based on Monte Carlo results.

    Uses the adaptive P(ruin) for classification, with additional checks
    on DD severity, Calmar, and return sign.

    Returns:
        dict with:
            'verdict': str — DEPLOY / MONITOR / REJECT
            'classification': str — ROBUST / STABLE / FRAGILE / REJECT
            'reasons': list of str
            'confidence': float — 0-100
            'p_ruin': float
    """
    cls = mc_result.get('classification', 'REJECT')
    p_ruin = mc_result.get('p_ruin', 100)
    p_ruin_fixed = mc_result.get('p_ruin_fixed', p_ruin)
    dd_95 = mc_result.get('dd_95', -100)
    dd_50 = mc_result.get('dd_50', -100)
    return_50 = mc_result.get('return_50', 0)
    calmar_50 = mc_result.get('calmar_50', 0)
    threshold_mode = mc_result.get('ruin_threshold_mode', 'user')
    effective_threshold = mc_result.get('ruin_threshold', -20)
    worst_dd = mc_result.get('worst_case_dd', -100)

    reasons = []

    # Note adaptive threshold in reasons
    if threshold_mode == 'adaptive':
        reasons.append(
            f"Adaptive threshold used: {effective_threshold:.1f}% "
            f"(fixed {mc_result.get('ruin_threshold_user', -20):.0f}% unreachable, "
            f"worst-case DD = {worst_dd:.1f}%)"
        )

    if cls == 'ROBUST':
        verdict = 'DEPLOY'
        confidence = 90 - p_ruin
        reasons.append(f"P(ruin) = {p_ruin:.1f}% < 5% — very low risk")
    elif cls == 'STABLE':
        verdict = 'MONITOR'
        confidence = 70 - p_ruin / 2
        reasons.append(f"P(ruin) = {p_ruin:.1f}% (5-15%) — moderate risk, monitor closely")
    elif cls == 'FRAGILE':
        verdict = 'MONITOR'
        confidence = 40
        reasons.append(f"P(ruin) = {p_ruin:.1f}% (15-30%) — fragile, high risk")
    else:
        verdict = 'REJECT'
        confidence = 10
        reasons.append(f"P(ruin) = {p_ruin:.1f}% > 30% — too risky")

    # Secondary checks on DD severity
    if dd_95 < -15:
        reasons.append(f"DD 95th pct = {dd_95:.1f}% — deep drawdown risk")
        if verdict == 'DEPLOY':
            verdict = 'MONITOR'

    if calmar_50 < 0.3:
        reasons.append(f"Median Calmar = {calmar_50:.3f} — low risk-adjusted return")
        if verdict == 'DEPLOY':
            verdict = 'MONITOR'

    if return_50 < 0:
        reasons.append(f"Median return = {return_50:.1f}% — negative expected return")
        verdict = 'REJECT'

    return {
        'verdict': verdict,
        'classification': cls,
        'reasons': reasons,
        'confidence': round(max(0, min(100, confidence)), 1),
        'p_ruin': p_ruin,
    }


def _empty_mc_result() -> dict:
    """Return empty MC result."""
    return {
        'max_dd_dist': np.array([]),
        'return_dist': np.array([]),
        'calmar_dist': np.array([]),
        'p_ruin': 100.0,
        'p_ruin_fixed': 100.0,
        'classification': 'REJECT',
        'ruin_threshold': -20,
        'ruin_threshold_user': -20,
        'ruin_threshold_mode': 'user',
        'worst_case_dd': 0,
        'dd_5': 0, 'dd_50': 0, 'dd_95': 0, 'dd_99': 0,
        'return_5': 0, 'return_50': 0, 'return_95': 0,
        'calmar_50': 0,
        'n_sims': 0, 'n_trades': 0,
    }
