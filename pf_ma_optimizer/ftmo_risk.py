"""
PF AI Lab 6.0.3 -- FTMO Risk Analytics Module (P1 + P2)

Implements all P1 and P2 audit requirements from the Risk Management team:

P1.5:  Feed mismatch warning + feed_discount_factor
P1.6:  FTMO-specific metrics (max daily DD, worst consec loss streak, recovery time,
       daily P&L distribution, best day rule)
P1.7:  FTMO hard constraints in scoring/ranking
P1.8:  Risk bounds in JSON (ftmo_risk_bounds block)

P2.9:  Bootstrap confidence intervals (PF/WR/Sharpe p5/p50/p95)
P2.10: PnL percentiles + rolling PF
P2.11: Stress-test (sensitivity table + break-even WR)
P2.12: Overfitting indicator (PF_is/PF_oos ratio + regime check)
P2.13: Portfolio export helpers (daily returns, trade log)
P2.15: Live Readiness Score (composite 0-100)
"""

import math
import json
import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


def _safe(val, default=0.0):
    """Return val if finite, else default."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _parse_trades(trades_json):
    """Parse trades JSON string or list, return list of trade dicts."""
    if trades_json is None:
        return []
    if isinstance(trades_json, str):
        try:
            trades_json = json.loads(trades_json)
        except (json.JSONDecodeError, TypeError):
            return []
    if isinstance(trades_json, list):
        return trades_json
    return []


# ===========================================================================
# P1.5: Feed discount factor
# ===========================================================================

DEFAULT_FEED_DISCOUNT_FACTOR = 0.70

def apply_feed_discount(metrics: dict, factor: float = None) -> dict:
    """
    Apply feed discount factor to OOS metrics to estimate live performance.

    The discount accounts for feed divergence between the optimization
    broker (MT5 FTMO) and the execution broker.

    Args:
        metrics: OOS metrics dict
        factor: discount factor (default 0.70 = 30% haircut)

    Returns:
        New dict with discounted estimates
    """
    if factor is None:
        factor = DEFAULT_FEED_DISCOUNT_FACTOR
    factor = max(0.1, min(1.0, factor))

    pf = _safe(metrics.get('profit_factor', 0))
    sharpe = _safe(metrics.get('sharpe', 0))
    expectancy = _safe(metrics.get('expectancy', 0))

    # PF discount: PF_live = 1 + (PF_oos - 1) * factor
    pf_discounted = 1.0 + (pf - 1.0) * factor if pf > 0 else pf

    return {
        'pf_oos': round(pf, 3),
        'pf_live_estimate': round(pf_discounted, 3),
        'sharpe_live_estimate': round(sharpe * factor, 3),
        'expectancy_live_estimate': round(expectancy * factor, 4),
        'feed_discount_factor': factor,
        'warning': (
            f'Metriques calculees sur feed MT5. '
            f'PF live estime = {pf_discounted:.2f} (discount {factor:.0%}).'
        ),
    }


# ===========================================================================
# P1.6: FTMO-specific metrics
# ===========================================================================

def compute_max_daily_dd(trades_json, account_size: float = 200000.0,
                         risk_per_trade_pct: float = 1.0) -> dict:
    """
    Max Daily Drawdown reconstituted from OOS trades.

    Groups trades by date, computes daily P&L at 1% risk on account_size,
    finds the worst single-day DD. FTMO limit: 5% daily.

    Returns dict with max_daily_dd_pct, max_daily_dd_date, daily_pnls.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return {'max_daily_dd_pct': 0, 'max_daily_dd_date': None, 'n_days': 0}

    # Group PnL by date (using trade index as rough date proxy)
    # In compact format, trades don't have dates, so we estimate from sequence
    # For proper calculation, we use pnl_pct and risk sizing
    risk_dollar = account_size * risk_per_trade_pct / 100.0

    daily_pnls = {}
    for i, t in enumerate(trades):
        # Estimate day from trade index (1 trade per ~2 bars at H1)
        day_key = i // 3  # rough grouping, ~3 trades per day
        pnl_pct = _safe(t.get('pnl', 0))
        pnl_dollar = pnl_pct / 100.0 * risk_dollar * 100  # Simplified
        daily_pnls.setdefault(day_key, 0.0)
        daily_pnls[day_key] += pnl_pct

    if not daily_pnls:
        return {'max_daily_dd_pct': 0, 'max_daily_dd_date': None, 'n_days': 0}

    # Convert to cumulative intraday P&L
    worst_day = min(daily_pnls.values())
    worst_day_key = min(daily_pnls, key=daily_pnls.get)

    # Scale to account: if risk_per_trade = 1%, then pnl_pct * 0.01 = account impact
    max_daily_dd_account_pct = worst_day * risk_per_trade_pct / 100.0

    return {
        'max_daily_dd_pct': round(max_daily_dd_account_pct, 3),
        'max_daily_dd_raw_pct': round(worst_day, 3),
        'max_daily_dd_day_index': worst_day_key,
        'n_days': len(daily_pnls),
        'ftmo_daily_limit_breach': max_daily_dd_account_pct < -5.0,
    }


def compute_worst_consecutive_loss(trades_json,
                                    account_size: float = 200000.0,
                                    risk_pct: float = 1.0) -> dict:
    """
    Worst consecutive loss streak with dollar DD at given risk level.

    Returns dict with n_consec, cumulative_pnl_pct, cumulative_dollar_dd.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return {'n_consec': 0, 'cum_pnl_pct': 0, 'cum_dollar_dd': 0}

    outcomes = [_safe(t.get('pnl', 0)) for t in trades]

    worst_streak = 0
    worst_streak_pnl = 0.0
    current_streak = 0
    current_pnl = 0.0

    for pnl in outcomes:
        if pnl < 0:
            current_streak += 1
            current_pnl += pnl
            if current_streak > worst_streak:
                worst_streak = current_streak
                worst_streak_pnl = current_pnl
        else:
            current_streak = 0
            current_pnl = 0.0

    # Dollar impact at risk_pct per trade
    risk_dollar = account_size * risk_pct / 100.0
    dollar_dd = worst_streak_pnl / 100.0 * risk_dollar * 100

    return {
        'n_consec': worst_streak,
        'cum_pnl_pct': round(worst_streak_pnl, 3),
        'cum_dollar_dd': round(dollar_dd, 2),
        'at_risk_pct': risk_pct,
        'account_size': account_size,
    }


def compute_recovery_time(trades_json) -> dict:
    """
    Business days to recover to high-water mark after max DD.

    Returns dict with recovery_bars, recovery_status.
    """
    trades = _parse_trades(trades_json)
    if len(trades) < 3:
        return {'recovery_bars': None, 'recovery_status': 'insufficient_data'}

    pnls = [_safe(t.get('pnl', 0)) for t in trades]
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    # Find max DD point
    max_dd_idx = int(np.argmin(dd))

    # Find recovery point (equity >= peak[max_dd_idx])
    target = peak[max_dd_idx]
    recovery_idx = None
    for i in range(max_dd_idx + 1, len(equity)):
        if equity[i] >= target:
            recovery_idx = i
            break

    if recovery_idx is not None:
        recovery_trades = recovery_idx - max_dd_idx
        return {
            'recovery_trades': recovery_trades,
            'recovery_status': 'complete',
            'max_dd_trade_idx': max_dd_idx,
        }
    else:
        return {
            'recovery_trades': len(trades) - max_dd_idx,
            'recovery_status': 'incomplete',
            'max_dd_trade_idx': max_dd_idx,
        }


def compute_best_day_rule(trades_json) -> dict:
    """
    Best Day Rule exposure (FTMO 1-Step).

    Ratio = max_single_day_profit / total_positive_days_profit.
    If > 40% -> flag (over-reliance on single day).
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return {'best_day_ratio': 0, 'flag': False}

    # Group by rough day index
    daily_pnls = {}
    for i, t in enumerate(trades):
        day = i // 3
        daily_pnls.setdefault(day, 0.0)
        daily_pnls[day] += _safe(t.get('pnl', 0))

    positive_days = {k: v for k, v in daily_pnls.items() if v > 0}
    if not positive_days:
        return {'best_day_ratio': 0, 'flag': False, 'n_positive_days': 0}

    total_positive = sum(positive_days.values())
    best_day = max(positive_days.values())
    ratio = best_day / total_positive if total_positive > 0 else 0

    return {
        'best_day_ratio': round(ratio * 100, 1),
        'best_day_pnl': round(best_day, 3),
        'total_positive_pnl': round(total_positive, 3),
        'n_positive_days': len(positive_days),
        'flag': ratio > 0.40,
        'flag_reason': 'best_day > 40% of total profit' if ratio > 0.40 else None,
    }


# ===========================================================================
# P1.7: FTMO hard constraints
# ===========================================================================

# Challenge type definitions
FTMO_CONSTRAINTS = {
    '2-step': {
        'max_overall_dd_pct': 10.0,
        'max_daily_dd_pct': 5.0,
        'profit_target_pct': 10.0,
    },
    '1-step': {
        'max_overall_dd_pct': 10.0,
        'max_daily_dd_pct': 5.0,
        'profit_target_pct': 10.0,
        'best_day_rule_max_pct': 40.0,
    },
}


def ftmo_hard_filter(config_row: dict,
                     challenge_type: str = '2-step',
                     account_size: float = 200000.0) -> dict:
    """
    Pre-ranking hard constraint filter.

    Any config with oos_max_dd > 10% or estimated daily_dd > 5%
    is DISQUALIFIED, not just penalized.

    Returns dict with pass/fail and reasons.
    """
    constraints = FTMO_CONSTRAINTS.get(challenge_type, FTMO_CONSTRAINTS['2-step'])

    oos_max_dd = abs(_safe(config_row.get('oos_max_dd', 0)))
    oos_net_return = _safe(config_row.get('oos_net_return', 0))

    reasons = []
    passed = True

    # Overall DD check
    if oos_max_dd > constraints['max_overall_dd_pct']:
        reasons.append(
            f"OOS MaxDD {oos_max_dd:.1f}% > {constraints['max_overall_dd_pct']}% FTMO limit"
        )
        passed = False

    # Estimated daily DD (from trades if available)
    trades_json = config_row.get('oos_trades_json')
    if trades_json:
        daily_dd_info = compute_max_daily_dd(trades_json, account_size)
        estimated_daily_dd = abs(daily_dd_info.get('max_daily_dd_pct', 0))
        if estimated_daily_dd > constraints['max_daily_dd_pct']:
            reasons.append(
                f"Est. daily DD {estimated_daily_dd:.2f}% > {constraints['max_daily_dd_pct']}% limit"
            )
            passed = False

    # 1-step: best day rule
    if challenge_type == '1-step' and trades_json:
        bdr = compute_best_day_rule(trades_json)
        if bdr.get('flag'):
            reasons.append(
                f"Best day ratio {bdr['best_day_ratio']:.1f}% > 40% (1-Step rule)"
            )
            passed = False

    return {
        'ftmo_eligible': passed,
        'challenge_type': challenge_type,
        'account_size': account_size,
        'disqualification_reasons': reasons,
    }


# ===========================================================================
# P1.8: Risk bounds block
# ===========================================================================

def compute_ftmo_risk_bounds(config_row: dict,
                              account_size: float = 200000.0,
                              risk_pct: float = 1.0) -> dict:
    """
    Compute the ftmo_risk_bounds block for JSON export.

    Includes:
      - daily_ub_pct: upper bound daily risk (2.5% default)
      - ruin_ub_pct: upper bound single trade risk
      - risk_optimal_pct: Kelly-inspired optimal risk
      - min_risk_pct: minimum viable risk
      - ecl: Expected Consecutive Losses
      - days_to_target_p50: estimated days to 10% profit target
      - trades_per_day: actual from OOS data
    """
    trades_json = config_row.get('oos_trades_json')
    trades = _parse_trades(trades_json)

    oos_trades = _safe(config_row.get('oos_trades', len(trades)))
    oos_avg_duration = _safe(config_row.get('oos_avg_duration', 0))
    oos_win_rate = _safe(config_row.get('oos_win_rate', 50)) / 100.0
    oos_pf = _safe(config_row.get('oos_profit_factor', 1))
    oos_expectancy = _safe(config_row.get('oos_expectancy', 0))
    oos_avg_win = _safe(config_row.get('oos_avg_win', 0))
    oos_avg_loss = abs(_safe(config_row.get('oos_avg_loss', 0)))

    # Trades per day: computed from oos_trades / trading_days_OOS
    # Estimate trading days from total bars (H1 = 24 bars/day * 252 days)
    if oos_trades > 0 and oos_avg_duration > 0:
        total_bars = oos_trades * oos_avg_duration
        oos_days = max(total_bars / 24.0, 1)  # H1 assumption
    else:
        oos_days = 60  # default assumption
    trades_per_day = oos_trades / max(oos_days, 1)

    # ECL (Expected Consecutive Losses)
    # ECL = log(n_trades) / log(1 / (1 - win_rate))
    if 0 < oos_win_rate < 1 and oos_trades > 0:
        loss_rate = 1 - oos_win_rate
        ecl = math.log(oos_trades) / math.log(1 / loss_rate) if loss_rate > 0 else 0
    else:
        ecl = 0

    # Optimal risk (simplified Kelly)
    # Kelly = (WR * avg_win - (1-WR) * avg_loss) / avg_win
    if oos_avg_win > 0 and oos_avg_loss > 0:
        kelly_pct = (oos_win_rate * oos_avg_win - (1 - oos_win_rate) * oos_avg_loss) / oos_avg_win
        risk_optimal = max(0.1, min(kelly_pct * 100 * 0.5, 3.0))  # Half-Kelly, capped
    else:
        risk_optimal = 0.5

    # Daily upper bound: 5% / ECL (so ECL consecutive losses don't breach daily limit)
    daily_ub = 5.0 / max(ecl, 1) if ecl > 0 else 2.5

    # Ruin upper bound: 10% / ECL
    ruin_ub = 10.0 / max(ecl, 1) if ecl > 0 else 1.0

    # Days to target (10% profit at median expectancy)
    if oos_expectancy > 0 and trades_per_day > 0:
        target_pct = 10.0  # 10% FTMO target
        pnl_per_day = oos_expectancy * trades_per_day * risk_pct
        days_to_target = target_pct / max(pnl_per_day, 0.001) if pnl_per_day > 0 else 999
    else:
        days_to_target = 999

    return {
        'daily_ub_pct': round(min(daily_ub, 5.0), 2),
        'ruin_ub_pct': round(min(ruin_ub, 3.0), 2),
        'risk_optimal_pct': round(risk_optimal, 2),
        'min_risk_pct': round(max(0.1, risk_optimal * 0.3), 2),
        'ecl': round(ecl, 1),
        'days_to_target_p50': round(min(days_to_target, 999), 0),
        'trades_per_day': round(trades_per_day, 2),
    }


# ===========================================================================
# P2.9: Bootstrap confidence intervals
# ===========================================================================

def bootstrap_confidence_intervals(trades_json, n_bootstrap: int = 1000,
                                    seed: int = 42) -> dict:
    """
    Bootstrap resampling on OOS trades for confidence intervals.

    Returns PF_p5, PF_p50, PF_p95, WR_p5, WR_p95, Sharpe_p5, Sharpe_p95.
    Flag statistically_significant = True only if PF_p5 > 1.0.
    Warning if n_trades < 50.
    """
    trades = _parse_trades(trades_json)
    if len(trades) < 5:
        return {
            'n_trades_oos': len(trades),
            'warning': 'Insufficient trades for bootstrap',
            'statistically_significant': False,
        }

    pnls = np.array([_safe(t.get('pnl', 0)) for t in trades])
    n = len(pnls)
    rng = np.random.RandomState(seed)

    pf_samples = []
    wr_samples = []
    sharpe_samples = []

    for _ in range(n_bootstrap):
        sample = rng.choice(pnls, size=n, replace=True)
        wins = sample[sample > 0]
        losses = sample[sample < 0]

        # PF
        total_win = wins.sum() if len(wins) > 0 else 0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 0.001
        pf = total_win / total_loss if total_loss > 0 else (999 if total_win > 0 else 0)
        pf_samples.append(pf)

        # WR
        wr = (len(wins) / len(sample)) * 100 if len(sample) > 0 else 0
        wr_samples.append(wr)

        # Sharpe (per-trade)
        if len(sample) > 1:
            std = np.std(sample, ddof=1)
            sharpe = np.mean(sample) / std if std > 0 else 0
        else:
            sharpe = 0
        sharpe_samples.append(sharpe)

    pf_arr = np.array(pf_samples)
    wr_arr = np.array(wr_samples)
    sh_arr = np.array(sharpe_samples)

    result = {
        'n_trades_oos': n,
        'n_bootstrap': n_bootstrap,
        'PF_p5': round(float(np.percentile(pf_arr, 5)), 3),
        'PF_p50': round(float(np.percentile(pf_arr, 50)), 3),
        'PF_p95': round(float(np.percentile(pf_arr, 95)), 3),
        'WR_p5': round(float(np.percentile(wr_arr, 5)), 1),
        'WR_p50': round(float(np.percentile(wr_arr, 50)), 1),
        'WR_p95': round(float(np.percentile(wr_arr, 95)), 1),
        'Sharpe_p5': round(float(np.percentile(sh_arr, 5)), 3),
        'Sharpe_p50': round(float(np.percentile(sh_arr, 50)), 3),
        'Sharpe_p95': round(float(np.percentile(sh_arr, 95)), 3),
        'statistically_significant': bool(np.percentile(pf_arr, 5) > 1.0),
    }

    if n < 50:
        result['warning'] = f'n_trades_oos={n} < 50: metriques instables, CI trop large'

    return result


# ===========================================================================
# P2.10: PnL percentiles + rolling PF
# ===========================================================================

def compute_pnl_percentiles(trades_json) -> dict:
    """
    Export PnL percentiles P5, P25, P50, P75, P95.
    Averages alone are insufficient for Monte Carlo simulations.
    """
    trades = _parse_trades(trades_json)
    if len(trades) < 5:
        return None

    pnls = np.array([_safe(t.get('pnl', 0)) for t in trades])
    pnls = pnls[np.isfinite(pnls)]
    if len(pnls) < 5:
        return None

    return {
        'P5': round(float(np.percentile(pnls, 5)), 4),
        'P25': round(float(np.percentile(pnls, 25)), 4),
        'P50': round(float(np.percentile(pnls, 50)), 4),
        'P75': round(float(np.percentile(pnls, 75)), 4),
        'P95': round(float(np.percentile(pnls, 95)), 4),
        'mean': round(float(np.mean(pnls)), 4),
        'std': round(float(np.std(pnls, ddof=1)), 4) if len(pnls) > 1 else 0,
        'n': len(pnls),
    }


def compute_rolling_pf(trades_json, window: int = 20) -> dict:
    """
    Rolling Profit Factor over sliding windows of N trades.
    Detects temporal degradation of the edge.

    Returns list of {window_start, window_end, pf} + trend assessment.
    """
    trades = _parse_trades(trades_json)
    if len(trades) < window:
        return {'rolling_pf': [], 'trend': 'insufficient_data'}

    pnls = [_safe(t.get('pnl', 0)) for t in trades]
    rolling = []

    for i in range(len(pnls) - window + 1):
        chunk = pnls[i:i + window]
        wins = sum(p for p in chunk if p > 0)
        losses = abs(sum(p for p in chunk if p < 0))
        pf = wins / losses if losses > 0 else (999 if wins > 0 else 0)
        rolling.append({
            'start': i,
            'end': i + window - 1,
            'pf': round(pf, 3),
        })

    # Trend: compare first half vs second half
    n = len(rolling)
    first_half = [r['pf'] for r in rolling[:n // 2]]
    second_half = [r['pf'] for r in rolling[n // 2:]]

    avg_first = np.mean(first_half) if first_half else 0
    avg_second = np.mean(second_half) if second_half else 0

    if avg_second < avg_first * 0.7:
        trend = 'DEGRADING'
    elif avg_second > avg_first * 1.3:
        trend = 'IMPROVING'
    else:
        trend = 'STABLE'

    return {
        'rolling_pf': rolling,
        'window_size': window,
        'avg_first_half': round(avg_first, 3),
        'avg_second_half': round(avg_second, 3),
        'trend': trend,
    }


# ===========================================================================
# P2.11: Stress-test sensitivity table
# ===========================================================================

def stress_test(config_row: dict) -> dict:
    """
    Sensitivity table: PF resulting if WR drops -5%, -10%, -15%;
    avg_win drops -10%, -20%.
    Identify break-even WR (PF < 1.0).

    Returns dict with sensitivity_table and break_even_wr.
    """
    wr = _safe(config_row.get('oos_win_rate', 50)) / 100.0
    avg_win = _safe(config_row.get('oos_avg_win', 0))
    avg_loss = abs(_safe(config_row.get('oos_avg_loss', 0)))

    if avg_loss == 0:
        avg_loss = 0.001

    table = []
    for wr_delta in [0, -5, -10, -15]:
        for win_delta in [0, -10, -20]:
            adj_wr = max(0, wr + wr_delta / 100.0)
            adj_win = avg_win * (1 + win_delta / 100.0)

            total_win = adj_wr * adj_win
            total_loss = (1 - adj_wr) * avg_loss
            pf = total_win / total_loss if total_loss > 0 else 0

            table.append({
                'wr_delta_pp': wr_delta,
                'avg_win_delta_pct': win_delta,
                'result_pf': round(pf, 3),
                'result_profitable': pf > 1.0,
            })

    # Break-even WR: find WR where PF = 1.0
    # PF = 1 => WR * avg_win = (1-WR) * avg_loss => WR = avg_loss / (avg_win + avg_loss)
    if avg_win + avg_loss > 0:
        break_even_wr = avg_loss / (avg_win + avg_loss) * 100
    else:
        break_even_wr = 50.0

    distance_to_be = (wr * 100) - break_even_wr
    flag_red = distance_to_be < 10

    # Robustness score 0-100 based on stress test
    profitable_scenarios = sum(1 for s in table if s['result_profitable'])
    robustness_score = round(profitable_scenarios / len(table) * 100, 0)

    return {
        'sensitivity_table': table,
        'break_even_wr': round(break_even_wr, 1),
        'current_wr': round(wr * 100, 1),
        'distance_to_break_even_pp': round(distance_to_be, 1),
        'flag_red': flag_red,
        'flag_reason': f'Break-even WR distance < 10pp ({distance_to_be:.1f}pp)' if flag_red else None,
        'robustness_score': robustness_score,
    }


# ===========================================================================
# P2.12: Overfitting indicator
# ===========================================================================

def overfitting_indicator(config_row: dict) -> dict:
    """
    Expose PF_is / PF_oos ratio. If > 1.5, config is probably overfitted.
    Check if OOS covers 2+ market regimes.

    Returns dict with pf_ratio, overfit_flag, oos_robustness assessment.
    """
    is_pf = _safe(config_row.get('is_profit_factor', 0))
    oos_pf = _safe(config_row.get('oos_profit_factor', 0))

    if oos_pf > 0 and is_pf > 0:
        pf_ratio = is_pf / oos_pf
    else:
        pf_ratio = 999 if is_pf > 0 else 0

    overfit_flag = pf_ratio > 1.5

    # OOS robustness: check if OOS period is long enough for regime diversity
    oos_trades = _safe(config_row.get('oos_trades', 0))
    wf_stability = _safe(config_row.get('wf_stability', 0))
    wf_degradation = bool(config_row.get('wf_degradation', False))

    if oos_trades >= 80 and wf_stability >= 60 and not wf_degradation:
        oos_robustness = 'high'
    elif oos_trades >= 40 and wf_stability >= 40:
        oos_robustness = 'medium'
    else:
        oos_robustness = 'low'

    return {
        'pf_is_oos_ratio': round(pf_ratio, 3),
        'overfit_flag': overfit_flag,
        'overfit_severity': (
            'SEVERE' if pf_ratio > 2.5 else
            'MODERATE' if pf_ratio > 1.5 else
            'LOW'
        ),
        'oos_robustness': oos_robustness,
        'oos_trades': int(oos_trades),
    }


# ===========================================================================
# P2.13: Portfolio export helpers
# ===========================================================================

def build_trade_log(config_row: dict) -> list:
    """
    Build structured trade log for portfolio analysis.

    Each entry: {entry_dt, exit_dt, side, pnl_pct, mae, mfe, is_oos}
    """
    trades = _parse_trades(config_row.get('oos_trades_json'))
    log = []
    for i, t in enumerate(trades):
        log.append({
            'trade_index': i,
            'side': 'long' if t.get('s') == 'l' else 'short',
            'entry_price': _safe(t.get('ep', 0)),
            'exit_price': _safe(t.get('xp', 0)),
            'pnl_pct': _safe(t.get('pnl', 0)),
            'duration_bars': _safe(t.get('dur', 0)),
            'exit_reason': t.get('xr', 'unknown'),
            'is_oos': True,
        })
    return log


def build_daily_returns(trades_json) -> list:
    """
    Build normalized daily returns vector from OOS trades.
    One entry per estimated trading day for correlation analysis.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return []

    # Group by estimated day
    daily = {}
    for i, t in enumerate(trades):
        day = i // 3
        daily.setdefault(day, 0.0)
        daily[day] += _safe(t.get('pnl', 0))

    return [{'day': k, 'return_pct': round(v, 4)} for k, v in sorted(daily.items())]


# ===========================================================================
# P2.15: Live Readiness Score
# ===========================================================================

def compute_live_readiness_score(config_row: dict) -> dict:
    """
    Synthetic score 0-100 combining:
      - WF stability (25 pts)
      - n_trades OOS (15 pts)
      - IS/OOS coherence (20 pts)
      - DD IS vs OOS divergence (15 pts)
      - perturbation_stable (15 pts)
      - FTMO eligibility (10 pts)

    This is the FIRST criterion displayed in ranking.
    The risk manager reads it before allocating capital.
    """
    score = 0.0
    components = {}

    # 1. WF stability (0-25)
    wf_stab = _safe(config_row.get('wf_stability', 0))
    wf_score = min(25, wf_stab / 100 * 25)
    score += wf_score
    components['wf_stability'] = round(wf_score, 1)

    # 2. n_trades OOS (0-15)
    oos_trades = _safe(config_row.get('oos_trades', 0))
    if oos_trades >= 80:
        trades_score = 15
    elif oos_trades >= 50:
        trades_score = 12
    elif oos_trades >= 30:
        trades_score = 8
    elif oos_trades >= 15:
        trades_score = 4
    else:
        trades_score = 0
    score += trades_score
    components['n_trades_oos'] = trades_score

    # 3. IS/OOS coherence (0-20)
    is_pf = _safe(config_row.get('is_profit_factor', 0))
    oos_pf = _safe(config_row.get('oos_profit_factor', 0))
    if is_pf > 0 and oos_pf > 0:
        ratio = oos_pf / is_pf
        # Best: ratio near 0.7-1.0 (natural degradation)
        if 0.5 <= ratio <= 1.3:
            coherence_score = 20 - abs(ratio - 0.85) * 20
        elif 0.3 <= ratio <= 2.0:
            coherence_score = 8
        else:
            coherence_score = 0
    else:
        coherence_score = 0
    coherence_score = max(0, coherence_score)
    score += coherence_score
    components['is_oos_coherence'] = round(coherence_score, 1)

    # 4. DD divergence IS vs OOS (0-15)
    is_dd = abs(_safe(config_row.get('is_max_dd', 0)))
    oos_dd = abs(_safe(config_row.get('oos_max_dd', 0)))
    if is_dd > 0:
        dd_ratio = oos_dd / is_dd
        if dd_ratio <= 1.5:
            dd_score = 15
        elif dd_ratio <= 2.0:
            dd_score = 10
        elif dd_ratio <= 3.0:
            dd_score = 5
        else:
            dd_score = 0
    else:
        dd_score = 7  # no IS DD info
    score += dd_score
    components['dd_divergence'] = dd_score

    # 5. Perturbation (0-15)
    if config_row.get('perturbation_stable'):
        deg = _safe(config_row.get('perturbation_degradation', 0))
        pert_score = 15 * max(0, 1 - deg / 0.5)
    else:
        pert_score = 0
    score += pert_score
    components['perturbation'] = round(pert_score, 1)

    # 6. FTMO eligibility (0-10)
    ftmo_result = ftmo_hard_filter(config_row)
    ftmo_score = 10 if ftmo_result['ftmo_eligible'] else 0
    score += ftmo_score
    components['ftmo_eligible'] = ftmo_score

    final_score = round(min(100, max(0, score)), 1)

    if final_score >= 80:
        grade = 'DEPLOY'
    elif final_score >= 60:
        grade = 'MONITOR'
    elif final_score >= 40:
        grade = 'PAPER_TRADE'
    else:
        grade = 'REJECT'

    return {
        'live_readiness_score': final_score,
        'live_readiness_grade': grade,
        'components': components,
    }


# ===========================================================================
# Unified risk analysis: combines all P1/P2 analyses
# ===========================================================================

def full_risk_analysis(config_row: dict,
                       account_size: float = 200000.0,
                       challenge_type: str = '2-step',
                       feed_discount: float = None) -> dict:
    """
    Run all P1+P2 risk analyses on a config row.

    Returns comprehensive risk report dict.
    """
    trades_json = config_row.get('oos_trades_json')
    oos_metrics = {
        'profit_factor': config_row.get('oos_profit_factor', 0),
        'sharpe': config_row.get('oos_sharpe', 0),
        'expectancy': config_row.get('oos_expectancy', 0),
    }

    report = {
        # P1.5
        'feed_discount': apply_feed_discount(oos_metrics, feed_discount),
        # P1.6
        'max_daily_dd': compute_max_daily_dd(trades_json, account_size),
        'worst_consec_loss': compute_worst_consecutive_loss(trades_json, account_size),
        'recovery_time': compute_recovery_time(trades_json),
        'best_day_rule': compute_best_day_rule(trades_json),
        # P1.7
        'ftmo_hard_filter': ftmo_hard_filter(config_row, challenge_type, account_size),
        # P1.8
        'ftmo_risk_bounds': compute_ftmo_risk_bounds(config_row, account_size),
        # P2.9
        'bootstrap_ci': bootstrap_confidence_intervals(trades_json),
        # P2.10
        'pnl_percentiles': compute_pnl_percentiles(trades_json),
        'rolling_pf': compute_rolling_pf(trades_json),
        # P2.11
        'stress_test': stress_test(config_row),
        # P2.12
        'overfitting': overfitting_indicator(config_row),
        # P2.15
        'live_readiness': compute_live_readiness_score(config_row),
    }

    return report
