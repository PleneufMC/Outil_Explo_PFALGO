"""
PF AI Lab 6.0.3 -- Trade Analytics Module
Advanced analysis functions operating on oos_trades_json data.

Features (v5.6):
  - Omega Ratio & Tail Ratio (A1.1, A1.2)
  - PnL Distribution Analysis (B2.1)
  - Trade Autocorrelation (B2.2)
  - Metrics by Exit Reason (B2.4)
  - Run-Length Analysis (B2.5)
  - FTMO-Readiness Score (B5.2)
  - IS/OOS Degradation Vector (B5.1)
  - Long/Short Ratio Warning (B1.5)
  - Stability Decay Index (B5.3)
  - OOS Gate Tiers: GOLD / SILVER / BRONZE / FAIL (A3.1)

Features (v6.0.3):
  - Full risk analysis integration (P1+P2 audit)
  - Live Readiness Score as primary ranking criterion
"""

import json
import math
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


# ==========================================================================
# A1.1  Omega Ratio
# ==========================================================================

def omega_ratio(daily_returns, threshold=0.0):
    """
    Omega Ratio = sum(max(r - threshold, 0)) / sum(max(threshold - r, 0)).

    Captures the full return distribution (skew + kurtosis sensitive).
    A higher Omega is better.  Omega > 1.0 at threshold=0 means net positive.

    Parameters:
        daily_returns: array-like of per-trade PnL percentages
        threshold: return threshold (default 0.0)

    Returns:
        float -- Omega ratio (inf if no losses below threshold)
    """
    if daily_returns is None or len(daily_returns) == 0:
        return 0.0
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    excess = r - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess < 0].sum())
    if losses == 0:
        return 999.0 if gains > 0 else 0.0
    return float(gains / losses)


# ==========================================================================
# A1.2  Tail Ratio
# ==========================================================================

def tail_ratio(daily_returns):
    """
    Tail Ratio = percentile_95(returns) / abs(percentile_5(returns)).

    Measures tail symmetry. > 1.0 means large gains exceed large losses.
    Critical for detecting strategies with dangerous left-tail risk.

    Parameters:
        daily_returns: array-like of per-trade PnL percentages

    Returns:
        float -- Tail ratio (inf if p5 == 0)
    """
    if daily_returns is None or len(daily_returns) < 5:
        return 0.0
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 5:
        return 0.0
    p95 = float(np.percentile(r, 95))
    p5 = float(np.percentile(r, 5))
    if p5 == 0:
        return 999.0 if p95 > 0 else 0.0
    return float(p95 / abs(p5))


# ==========================================================================
# B2.1  PnL Distribution Analysis
# ==========================================================================

def analyze_pnl_distribution(trades_json):
    """
    Comprehensive PnL distribution analysis.

    Returns dict with: mean, std, skew, kurtosis, outliers_2std,
    top5_contribution, worst5_contribution, omega, tail_ratio.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return None

    pnls = np.array([_safe(t.get('pnl', 0)) for t in trades])
    pnls = pnls[np.isfinite(pnls)]
    if len(pnls) < 3:
        return None

    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0

    # Skewness and kurtosis (scipy if available, else manual)
    if scipy_stats is not None and len(pnls) >= 3:
        skew_val = float(scipy_stats.skew(pnls))
        kurt_val = float(scipy_stats.kurtosis(pnls))  # excess kurtosis
    else:
        skew_val = 0.0
        kurt_val = 0.0

    # Outliers: trades > 2 std from mean
    outliers_2std = int(np.sum(np.abs(pnls - mean_pnl) > 2 * std_pnl)) if std_pnl > 0 else 0

    # Top-5 / Worst-5 contribution
    sorted_pnls = np.sort(pnls)
    total_gains = float(np.sum(pnls[pnls > 0])) if np.any(pnls > 0) else 0.001
    total_losses = float(np.sum(pnls[pnls < 0])) if np.any(pnls < 0) else -0.001

    top5_sum = float(sorted_pnls[-5:].sum()) if len(sorted_pnls) >= 5 else float(sorted_pnls.sum())
    worst5_sum = float(sorted_pnls[:5].sum()) if len(sorted_pnls) >= 5 else float(sorted_pnls.sum())

    top5_pct = (top5_sum / total_gains * 100) if total_gains > 0 else 0.0
    worst5_pct = (worst5_sum / total_losses * 100) if total_losses < 0 else 0.0

    return {
        'mean': round(mean_pnl, 4),
        'std': round(std_pnl, 4),
        'skew': round(skew_val, 3),
        'kurtosis': round(kurt_val, 3),
        'outliers_2std': outliers_2std,
        'outliers_2std_pct': round(outliers_2std / len(pnls) * 100, 1),
        'top5_contribution_pct': round(top5_pct, 1),
        'worst5_contribution_pct': round(worst5_pct, 1),
        'omega': round(omega_ratio(pnls), 3),
        'tail_ratio': round(tail_ratio(pnls), 3),
        'n_trades': len(pnls),
    }


# ==========================================================================
# B2.2  Trade Autocorrelation
# ==========================================================================

def trade_autocorrelation(trades_json, max_lag=5):
    """
    Compute autocorrelation of win/loss outcomes at lags 1..max_lag.

    High positive autocorrelation at lag-1 means losses come in clusters
    (dangerous for FTMO challenges).

    Returns dict: {'lag_1': float, 'lag_2': float, ..., 'cluster_risk': str}
    """
    trades = _parse_trades(trades_json)
    if len(trades) < max_lag + 2:
        return None

    outcomes = np.array([1.0 if _safe(t.get('pnl', 0)) > 0 else 0.0 for t in trades])

    result = {}
    import pandas as pd
    series = pd.Series(outcomes)
    for lag in range(1, max_lag + 1):
        ac = series.autocorr(lag=lag)
        result[f'lag_{lag}'] = round(_safe(ac), 3)

    # Cluster risk assessment
    lag1 = result.get('lag_1', 0)
    if lag1 > 0.3:
        result['cluster_risk'] = 'HIGH'
    elif lag1 > 0.15:
        result['cluster_risk'] = 'MODERATE'
    elif lag1 > 0.0:
        result['cluster_risk'] = 'LOW'
    else:
        result['cluster_risk'] = 'NONE'

    return result


# ==========================================================================
# B2.4  Metrics by Exit Reason
# ==========================================================================

def metrics_by_exit_reason(trades_json):
    """
    Break down PF, win rate, avg PnL per exit reason.

    Returns dict: {exit_reason: {count, pct, win_rate, avg_pnl, total_pnl, pf}}
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return None

    total = len(trades)
    by_reason = {}

    for t in trades:
        xr = t.get('xr', 'unknown')
        if xr not in by_reason:
            by_reason[xr] = []
        by_reason[xr].append(_safe(t.get('pnl', 0)))

    result = {}
    for xr, pnls in by_reason.items():
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gains = sum(wins)
        loss_sum = abs(sum(losses))
        pf = (gains / loss_sum) if loss_sum > 0 else (999.0 if gains > 0 else 0.0)

        result[xr] = {
            'count': len(pnls),
            'pct': round(len(pnls) / total * 100, 1),
            'win_rate': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
            'avg_pnl': round(float(np.mean(pnls)), 4) if pnls else 0,
            'total_pnl': round(sum(pnls), 4),
            'profit_factor': round(pf, 3),
        }

    return result


# ==========================================================================
# B2.5  Run-Length Analysis
# ==========================================================================

def run_length_analysis(trades_json):
    """
    Compute max / avg consecutive winning and losing streaks.

    Critical for FTMO: a 10-loss streak on M5 can blow a 5% DD limit.

    Returns dict with max/avg win/loss streaks.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return None

    outcomes = [1 if _safe(t.get('pnl', 0)) > 0 else -1 for t in trades]

    win_streaks = []
    loss_streaks = []
    current = 0
    current_type = 0  # 1 = win, -1 = loss

    for o in outcomes:
        if o == current_type:
            current += 1
        else:
            if current > 0:
                if current_type == 1:
                    win_streaks.append(current)
                elif current_type == -1:
                    loss_streaks.append(current)
            current = 1
            current_type = o

    # Don't forget the last streak
    if current > 0:
        if current_type == 1:
            win_streaks.append(current)
        elif current_type == -1:
            loss_streaks.append(current)

    return {
        'max_win_streak': max(win_streaks) if win_streaks else 0,
        'avg_win_streak': round(float(np.mean(win_streaks)), 1) if win_streaks else 0,
        'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
        'avg_loss_streak': round(float(np.mean(loss_streaks)), 1) if loss_streaks else 0,
        'n_win_streaks': len(win_streaks),
        'n_loss_streaks': len(loss_streaks),
    }


# ==========================================================================
# B5.2  FTMO-Readiness Score
# ==========================================================================

def ftmo_readiness(config_row, oos_days=None):
    """
    Compute FTMO-readiness score from a DB config row or dict.

    Criteria (mapped to FTMO challenge rules):
      1. Max DD OOS >= -5%  (FTMO: max 5% trailing DD)
      2. Max consecutive losses <= 5
      3. OOS net return >= 10% (FTMO: 10% profit target in 30 days)
      4. Trades per day >= 0.5 (at least 1 trade every 2 days)
      5. Expectancy > 0.02% per trade

    Returns dict with individual checks and overall ftmo_score [0-100].
    """
    trades_json = config_row.get('oos_trades_json')
    trades = _parse_trades(trades_json)

    oos_max_dd = _safe(config_row.get('oos_max_dd', 0))
    oos_net_return = _safe(config_row.get('oos_net_return', 0))
    oos_expectancy = _safe(config_row.get('oos_expectancy', 0))
    oos_trades = _safe(config_row.get('oos_trades', 0))
    oos_avg_duration = _safe(config_row.get('oos_avg_duration', 0))

    # Estimate OOS days from trade count and duration if not provided
    if oos_days is None and oos_trades > 0 and oos_avg_duration > 0:
        # Rough estimate: total bars / bars_per_day (M15 = 96 bars/day)
        total_bars = oos_trades * oos_avg_duration
        oos_days = max(total_bars / 96.0, 1)  # assume M15 if unknown
    oos_days = max(oos_days or 30, 1)

    # 1. Max DD check
    dd_ok = oos_max_dd >= -5.0
    dd_score = min(1.0, max(0, (oos_max_dd + 10.0) / 5.0))  # scale [-10, -5] -> [0, 1]

    # 2. Max consecutive losses
    run_info = run_length_analysis(trades_json)
    max_consec_loss = run_info['max_loss_streak'] if run_info else 0
    consec_ok = max_consec_loss <= 5
    consec_score = min(1.0, max(0, 1.0 - (max_consec_loss - 3) / 5.0))

    # 3. Profit target
    profit_ok = oos_net_return >= 10.0
    profit_score = min(1.0, max(0, oos_net_return / 10.0))

    # 4. Trade frequency
    trades_per_day = oos_trades / oos_days if oos_days > 0 else 0
    frequency_ok = trades_per_day >= 0.5
    frequency_score = min(1.0, trades_per_day / 0.5) if trades_per_day < 1.0 else 1.0

    # 5. Expectancy
    expectancy_ok = oos_expectancy > 0.02
    expectancy_score = min(1.0, max(0, oos_expectancy / 0.1))

    # Overall score (weighted)
    weights = [0.30, 0.25, 0.20, 0.10, 0.15]
    scores = [dd_score, consec_score, profit_score, frequency_score, expectancy_score]
    ftmo_score = round(sum(w * s for w, s in zip(weights, scores)) * 100, 1)

    return {
        'dd_ok': dd_ok,
        'dd_value': round(oos_max_dd, 2),
        'consec_loss_ok': consec_ok,
        'max_consec_loss': max_consec_loss,
        'profit_ok': profit_ok,
        'profit_value': round(oos_net_return, 2),
        'frequency_ok': frequency_ok,
        'trades_per_day': round(trades_per_day, 2),
        'expectancy_ok': expectancy_ok,
        'expectancy_value': round(oos_expectancy, 4),
        'ftmo_score': ftmo_score,
        'ftmo_grade': (
            'READY' if ftmo_score >= 80 else
            'PROMISING' if ftmo_score >= 60 else
            'NEEDS_WORK' if ftmo_score >= 40 else
            'NOT_READY'
        ),
    }


# ==========================================================================
# B5.1  IS/OOS Degradation Vector
# ==========================================================================

def degradation_vector(config_row):
    """
    Compute IS->OOS degradation for key metrics.

    A uniform ~30% degradation is healthy. Asymmetric degradation
    (e.g., Calmar drops 80% but WR is stable) signals a specific problem.

    Returns dict with degradation percentages per metric.
    """
    is_calmar = _safe(config_row.get('is_calmar', 0))
    oos_calmar = _safe(config_row.get('oos_calmar', 0))
    is_pf = _safe(config_row.get('is_profit_factor', 0))
    oos_pf = _safe(config_row.get('oos_profit_factor', 0))
    is_wr = _safe(config_row.get('is_win_rate', 0))
    oos_wr = _safe(config_row.get('oos_win_rate', 0))
    is_trades = _safe(config_row.get('is_trades', 0))
    oos_trades = _safe(config_row.get('oos_trades', 0))
    is_sharpe = _safe(config_row.get('is_sharpe', 0))
    oos_sharpe = _safe(config_row.get('oos_sharpe', 0))

    def _deg(is_val, oos_val):
        if is_val == 0:
            return 0.0
        return round((is_val - oos_val) / abs(is_val) * 100, 1)

    calmar_deg = _deg(is_calmar, oos_calmar)
    pf_deg = _deg(is_pf, oos_pf)
    wr_deg = _deg(is_wr, oos_wr)
    sharpe_deg = _deg(is_sharpe, oos_sharpe)
    trades_ratio = round(oos_trades / max(is_trades, 1) * 100, 1)

    degs = [abs(calmar_deg), abs(pf_deg), abs(wr_deg), abs(sharpe_deg)]
    degs = [d for d in degs if d > 0]
    avg_deg = round(float(np.mean(degs)), 1) if degs else 0.0
    std_deg = round(float(np.std(degs)), 1) if len(degs) > 1 else 0.0

    # Asymmetry detection
    if std_deg > 20:
        symmetry = 'ASYMMETRIC'
    elif std_deg > 10:
        symmetry = 'MODERATE'
    else:
        symmetry = 'UNIFORM'

    return {
        'calmar_deg': calmar_deg,
        'pf_deg': pf_deg,
        'wr_deg': wr_deg,
        'sharpe_deg': sharpe_deg,
        'trades_ratio_pct': trades_ratio,
        'avg_degradation': avg_deg,
        'degradation_std': std_deg,
        'symmetry': symmetry,
    }


# ==========================================================================
# B1.5  Long/Short Ratio Warning
# ==========================================================================

def long_short_analysis(trades_json):
    """
    Analyze long/short distribution.

    Warns if > 90% of trades are on one side (potential directional bias).

    Returns dict with counts, ratio, and warning.
    """
    trades = _parse_trades(trades_json)
    if not trades:
        return None

    n_long = sum(1 for t in trades if t.get('s') == 'l')
    n_short = sum(1 for t in trades if t.get('s') == 's')
    total = n_long + n_short
    if total == 0:
        return None

    long_pct = round(n_long / total * 100, 1)
    short_pct = round(n_short / total * 100, 1)

    # PnL by side
    long_pnl = sum(_safe(t.get('pnl', 0)) for t in trades if t.get('s') == 'l')
    short_pnl = sum(_safe(t.get('pnl', 0)) for t in trades if t.get('s') == 's')

    # Warning threshold
    if long_pct > 90 or short_pct > 90:
        warning = 'EXTREME_BIAS'
    elif long_pct > 75 or short_pct > 75:
        warning = 'MODERATE_BIAS'
    else:
        warning = 'BALANCED'

    return {
        'n_long': n_long,
        'n_short': n_short,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'long_total_pnl': round(long_pnl, 4),
        'short_total_pnl': round(short_pnl, 4),
        'bias_warning': warning,
    }


# ==========================================================================
# B5.3  Stability Decay Index
# ==========================================================================

def stability_decay(trades_json):
    """
    Divide OOS trades into 4 chronological quartiles and compute
    metrics per quartile.  Declining performance Q1->Q4 = strategy decay.

    Returns dict with per-quartile stats and overall decay_index.
    """
    trades = _parse_trades(trades_json)
    if len(trades) < 8:  # need at least 2 per quartile
        return None

    n = len(trades)
    quartiles = [trades[i * n // 4:(i + 1) * n // 4] for i in range(4)]

    result = {'quartiles': {}}
    win_rates = []
    avg_pnls = []

    for i, q in enumerate(quartiles):
        if not q:
            continue
        pnls = [_safe(t.get('pnl', 0)) for t in q]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / len(pnls) * 100 if pnls else 0
        avg = float(np.mean(pnls)) if pnls else 0
        win_rates.append(wr)
        avg_pnls.append(avg)

        result['quartiles'][f'Q{i + 1}'] = {
            'n_trades': len(q),
            'win_rate': round(wr, 1),
            'avg_pnl': round(avg, 4),
            'total_pnl': round(sum(pnls), 4),
        }

    # Decay index: compare Q1 vs Q4
    if len(win_rates) == 4 and len(avg_pnls) == 4:
        wr_decay = win_rates[0] - win_rates[3]  # positive = decaying
        pnl_decay = avg_pnls[0] - avg_pnls[3]   # positive = decaying

        # Monotonic check: Q1 > Q2 > Q3 > Q4?
        wr_monotonic = all(win_rates[i] >= win_rates[i + 1] for i in range(3))
        pnl_monotonic = all(avg_pnls[i] >= avg_pnls[i + 1] for i in range(3))

        result['wr_decay'] = round(wr_decay, 1)
        result['pnl_decay'] = round(pnl_decay, 4)
        result['wr_monotonic_decline'] = wr_monotonic
        result['pnl_monotonic_decline'] = pnl_monotonic

        # Overall decay assessment
        if wr_monotonic and pnl_monotonic and wr_decay > 10:
            result['decay_status'] = 'DECAYING'
        elif wr_decay > 15 or pnl_decay > 0.1:
            result['decay_status'] = 'DECLINING'
        elif abs(wr_decay) < 5:
            result['decay_status'] = 'STABLE'
        else:
            result['decay_status'] = 'MIXED'
    else:
        result['decay_status'] = 'INSUFFICIENT_DATA'

    return result


# ==========================================================================
# A3.1  OOS Gate Tiers: GOLD / SILVER / BRONZE / FAIL
# ==========================================================================

def oos_gate_tier(config_row):
    """
    Multi-level OOS gate classification.

    GOLD:   Calmar >= 1.0 AND PF >= 1.5 AND trades >= 40 AND perturb stable AND WFE >= 0.5
    SILVER: Calmar >= 0.50 AND PF >= 1.30 AND trades >= 25
    BRONZE: Calmar >= 0.25 AND PF >= 1.20 AND trades >= 15
    FAIL:   below BRONZE thresholds

    Parameters:
        config_row: dict with OOS metrics (from DB row or similar)

    Returns:
        dict with tier, tier_reasons, tier_score
    """
    oos_calmar = _safe(config_row.get('oos_calmar', 0))
    oos_pf = _safe(config_row.get('oos_profit_factor', 0))
    oos_trades = int(_safe(config_row.get('oos_trades', 0)))
    perturb_stable = bool(config_row.get('perturbation_stable', False))
    wfe = config_row.get('wfe')
    wfe_val = _safe(wfe) if wfe is not None else None

    reasons = []

    # ── GOLD ──
    gold_checks = {
        'calmar >= 1.0': oos_calmar >= 1.0,
        'pf >= 1.5': oos_pf >= 1.5,
        'trades >= 40': oos_trades >= 40,
        'perturbation_stable': perturb_stable,
        'wfe >= 0.5': wfe_val is not None and wfe_val >= 0.5,
    }
    if all(gold_checks.values()):
        return {
            'tier': 'GOLD',
            'tier_checks': gold_checks,
            'tier_score': 3,
        }

    # ── SILVER ──
    silver_checks = {
        'calmar >= 0.50': oos_calmar >= 0.50,
        'pf >= 1.30': oos_pf >= 1.30,
        'trades >= 25': oos_trades >= 25,
    }
    if all(silver_checks.values()):
        return {
            'tier': 'SILVER',
            'tier_checks': silver_checks,
            'tier_score': 2,
        }

    # ── BRONZE ──
    bronze_checks = {
        'calmar >= 0.25': oos_calmar >= 0.25,
        'pf >= 1.20': oos_pf >= 1.20,
        'trades >= 15': oos_trades >= 15,
    }
    if all(bronze_checks.values()):
        return {
            'tier': 'BRONZE',
            'tier_checks': bronze_checks,
            'tier_score': 1,
        }

    # ── FAIL ──
    fail_reasons = {}
    if oos_calmar < 0.25:
        fail_reasons['calmar'] = f'{oos_calmar:.3f} < 0.25'
    if oos_pf < 1.20:
        fail_reasons['pf'] = f'{oos_pf:.3f} < 1.20'
    if oos_trades < 15:
        fail_reasons['trades'] = f'{oos_trades} < 15'

    return {
        'tier': 'FAIL',
        'tier_checks': fail_reasons,
        'tier_score': 0,
    }


# ==========================================================================
# B1.2 + B1.3  Enhanced Gate: avg_win/loss ratio + expectancy checks
# ==========================================================================

def enhanced_gate_checks(config_row):
    """
    Additional gate checks using under-exploited fields (B1.2, B1.3).

    Returns dict with:
      - rr_ok: Reward/Risk ratio check (WR < 50% AND RR < 1.0 => FAIL)
      - expectancy_ok: OOS expectancy > 0.02
      - avg_duration_ok: avg duration < 200 bars (B1.1)
      - reasons: list of failure reasons
    """
    oos_avg_win = _safe(config_row.get('oos_avg_win', 0))
    oos_avg_loss = _safe(config_row.get('oos_avg_loss', 0))
    oos_wr = _safe(config_row.get('oos_win_rate', 0))
    oos_expectancy = _safe(config_row.get('oos_expectancy', 0))
    oos_avg_duration = _safe(config_row.get('oos_avg_duration', 0))

    reasons = []

    # B1.2: Reward/Risk ratio
    rr = (oos_avg_win / abs(oos_avg_loss)) if oos_avg_loss != 0 else 999.0
    rr_ok = not (oos_wr < 50 and rr < 1.0)
    if not rr_ok:
        reasons.append(f'RR={rr:.2f} with WR={oos_wr:.1f}% (mathematically losing)')

    # B1.3: Expectancy
    expectancy_ok = oos_expectancy > 0.02
    if not expectancy_ok:
        reasons.append(f'Expectancy={oos_expectancy:.4f} < 0.02 (insufficient edge)')

    # B1.1: Duration
    avg_duration_ok = oos_avg_duration < 200 or oos_avg_duration == 0
    if not avg_duration_ok:
        reasons.append(f'Avg duration={oos_avg_duration:.0f} bars (overnight risk)')

    return {
        'rr_ratio': round(rr, 3),
        'rr_ok': rr_ok,
        'expectancy_ok': expectancy_ok,
        'avg_duration_ok': avg_duration_ok,
        'enhanced_pass': rr_ok and expectancy_ok,
        'reasons': reasons,
    }


# ==========================================================================
# Convenience: Full v5.6 analysis on a single config row
# ==========================================================================

def full_v56_analysis(config_row):
    """
    Run all v5.6 + v6.0.3 analytics on a config row (from DB or API).

    Returns a comprehensive dict with all analysis results.
    """
    trades_json = config_row.get('oos_trades_json')

    result = {
        'tier': oos_gate_tier(config_row),
        'enhanced_gate': enhanced_gate_checks(config_row),
        'degradation': degradation_vector(config_row),
    }

    # Trade-level analytics (require oos_trades_json)
    if trades_json:
        result['pnl_distribution'] = analyze_pnl_distribution(trades_json)
        result['autocorrelation'] = trade_autocorrelation(trades_json)
        result['exit_reasons'] = metrics_by_exit_reason(trades_json)
        result['run_length'] = run_length_analysis(trades_json)
        result['long_short'] = long_short_analysis(trades_json)
        result['stability'] = stability_decay(trades_json)
        result['ftmo'] = ftmo_readiness(config_row)

    # v6.0.3: Integrate full risk analysis (P1+P2 audit)
    try:
        from .ftmo_risk import (
            full_risk_analysis, compute_live_readiness_score,
        )
        result['risk_analysis'] = full_risk_analysis(config_row)
        result['live_readiness'] = compute_live_readiness_score(config_row)
    except ImportError:
        pass

    return result
