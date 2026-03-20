"""
PF AI Lab 5.0 — Performance Metrics
Calmar, Sortino, Sharpe, Profit Factor, Drawdown, etc.

CRITICAL FIXES (v5.0.2):
- Max Drawdown: BAR-BY-BAR equity curve using percentage-based P&L
  (unrealized mark-to-market on every bar, matching TradingView intrabar DD)
- Sharpe/Sortino: daily-resampled returns, annualized with sqrt(252)
  (matching TradingView's methodology, NOT per-trade)
- Equity computed in PERCENTAGE terms relative to entry price
  (not absolute price differences, which are scale-dependent)
- Clean interval-based position tracking (no buggy state machine)
- DataFrame auto-detected from trades.__df_ref__
"""

import math
import numpy as np
import pandas as pd

from .analytics import omega_ratio, tail_ratio


def _finite(val, default=0.0):
    """Return val if finite, else default. Handles NaN, Inf, None."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _detect_annual_bars(df) -> int:
    """
    V5.4.9 FIX #5: Auto-detect annual bar count from DataFrame index.

    Detects the median bar interval and maps to annual count:
      M1  = 252 * 24 * 60 = 362880 (but rarely used)
      M5  = 252 * 24 * 12 = 72576
      M15 = 252 * 24 * 4  = 24192
      M30 = 252 * 24 * 2  = 12096
      H1  = 252 * 24       = 6048
      H2  = 252 * 12       = 3024
      H4  = 252 * 6        = 1512
      D   = 252
      W   = 52
      M   = 12

    Falls back to 252 (daily) if detection fails.
    """
    if df is None or len(df) < 10:
        return 252  # default to daily

    try:
        idx = pd.to_datetime(df.index)
        diffs = idx[1:] - idx[:-1]
        # Use median to avoid outliers (weekends, gaps)
        median_diff = np.median(diffs.total_seconds())

        if median_diff <= 90:        # ~1min
            return 252 * 24 * 60
        elif median_diff <= 400:     # ~5min
            return 252 * 24 * 12
        elif median_diff <= 1200:    # ~15min
            return 252 * 24 * 4
        elif median_diff <= 2400:    # ~30min
            return 252 * 24 * 2
        elif median_diff <= 5400:    # ~1H
            return 252 * 24
        elif median_diff <= 10800:   # ~2H
            return 252 * 12
        elif median_diff <= 21600:   # ~4H
            return 252 * 6
        elif median_diff <= 100000:  # ~Daily
            return 252
        elif median_diff <= 700000:  # ~Weekly
            return 52
        else:                        # Monthly
            return 12
    except Exception:
        return 252  # fallback daily


def compute_metrics(trades: list, initial_capital: float = 100000.0,
                    annual_bars: int = None, df: pd.DataFrame = None) -> dict:
    """
    Compute comprehensive performance metrics from a list of trades.

    Parameters:
        trades: list of dicts from run_backtest()
        initial_capital: starting capital (used for absolute equity display)
        annual_bars: bars per year for annualization.
                     If None, auto-detected from df index frequency.
                     V5.4.9 FIX #5: was hardcoded 252*24 which is wrong for D/H4.
        df: OHLCV DataFrame (optional, auto-detected from trades.__df_ref__)

    Returns:
        dict of metrics
    """
    if not trades:
        return _empty_metrics()

    # Auto-detect DataFrame from backtest engine
    if df is None:
        df = getattr(trades, '__df_ref__', None)

    # V5.4.9 FIX #5: Auto-detect annual_bars from data frequency
    if annual_bars is None:
        annual_bars = _detect_annual_bars(df)

    pnls = np.array([t['pnl'] for t in trades])
    pnl_pcts = np.array([t['pnl_pct'] for t in trades])

    n_trades = len(trades)
    n_wins = int(np.sum(pnl_pcts > 0))
    n_losses = int(np.sum(pnl_pcts < 0))

    # Profit factor uses pnl_pct for consistency
    wins_pct = pnl_pcts[pnl_pcts > 0]
    losses_pct = pnl_pcts[pnl_pcts < 0]
    total_profit_pct = float(np.sum(wins_pct)) if len(wins_pct) > 0 else 0.0
    total_loss_pct = float(np.abs(np.sum(losses_pct))) if len(losses_pct) > 0 else 0.0

    # Keep absolute profit/loss for display
    total_profit = float(np.sum(pnls[pnls > 0])) if np.sum(pnls > 0) > 0 else 0.0
    total_loss = float(np.abs(np.sum(pnls[pnls < 0]))) if np.sum(pnls < 0) > 0 else 0.0

    # ------------------------------------------------------------------
    # BAR-BY-BAR EQUITY + MAX DRAWDOWN (TradingView-aligned)
    # ------------------------------------------------------------------
    if df is not None and len(df) > 0:
        equity_pct_curve, max_dd_pct = _compute_bar_by_bar_equity(trades, df)
    else:
        # Fallback: trade-level equity
        equity_pct_curve = None
        cum_pnl_pct = np.cumsum(pnl_pcts)
        peak_t = np.maximum.accumulate(cum_pnl_pct)
        dd_t = cum_pnl_pct - peak_t
        # Apply 1.5x correction for missing intrabar data
        max_dd_pct = float(np.min(dd_t)) * 1.5 if len(dd_t) > 0 else 0.0

    # Trade-level equity (for display charts)
    equity = np.cumsum(pnl_pcts)
    equity_abs = initial_capital * (1 + np.cumsum(pnl_pcts) / 100)

    # Drawdown curve (trade-level, for display)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak

    # Net return (sum of pnl_pct)
    net_return = float(equity[-1]) if len(equity) > 0 else 0.0

    # Duration in bars and years
    sorted_t = sorted(trades, key=lambda t: t['entry_bar'])
    total_bars = sorted_t[-1]['exit_bar'] - sorted_t[0]['entry_bar']
    years = max(total_bars / annual_bars, 0.01)
    annual_return = net_return / years

    # Calmar Ratio = Annual Return / |Max Drawdown (bar-by-bar)|
    calmar = annual_return / abs(max_dd_pct) if max_dd_pct != 0 else 0.0
    calmar = _finite(calmar)

    # Profit Factor (percentage-based)
    profit_factor = total_profit_pct / total_loss_pct if total_loss_pct > 0 else (
        999.0 if total_profit_pct > 0 else 0.0)
    profit_factor = _finite(profit_factor)

    # ------------------------------------------------------------------
    # SHARPE / SORTINO — Daily returns (TradingView-aligned)
    # ------------------------------------------------------------------
    sharpe, sortino = _compute_daily_sharpe_sortino(trades, df, initial_capital)

    # Win rate
    win_rate = (n_wins / n_trades * 100) if n_trades > 0 else 0.0

    # Average win/loss (percentage)
    avg_win = float(np.mean(wins_pct)) if len(wins_pct) > 0 else 0.0
    avg_loss = float(np.mean(losses_pct)) if len(losses_pct) > 0 else 0.0

    # Expectancy
    expectancy = (win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss)

    # Max consecutive wins/losses
    max_consec_wins = _max_consecutive(pnl_pcts > 0)
    max_consec_losses = _max_consecutive(pnl_pcts < 0)

    # Average duration
    durations = [t.get('duration_bars', 0) for t in trades]
    avg_duration = np.mean(durations) if durations else 0

    # ── v5.6: Omega Ratio & Tail Ratio ──
    omega = omega_ratio(pnl_pcts, threshold=0.0)
    tail = tail_ratio(pnl_pcts)

    return {
        'n_trades': n_trades,
        'n_wins': n_wins,
        'n_losses': n_losses,
        'win_rate': round(_finite(win_rate), 1),
        'net_return': round(_finite(net_return), 2),
        'annual_return': round(_finite(annual_return), 2),
        'max_dd': round(_finite(max_dd_pct), 2),
        'calmar': round(_finite(calmar), 3),
        'profit_factor': round(_finite(profit_factor), 3),
        'sharpe': round(_finite(sharpe), 3),
        'sortino': round(_finite(sortino), 3),
        'expectancy': round(_finite(expectancy), 4),
        'avg_win': round(_finite(avg_win), 3),
        'avg_loss': round(_finite(avg_loss), 3),
        'total_profit': round(_finite(total_profit), 2),
        'total_loss': round(_finite(total_loss), 2),
        'max_consec_wins': max_consec_wins,
        'max_consec_losses': max_consec_losses,
        'avg_duration_bars': round(_finite(avg_duration), 1),
        'omega_ratio': round(_finite(omega), 3),
        'tail_ratio': round(_finite(tail), 3),
        'equity_curve': [_finite(v) for v in equity.tolist()],
        'equity_abs': [_finite(v, 100000.0) for v in equity_abs.tolist()],
        'drawdown_curve': [_finite(v) for v in dd.tolist()],
        'years': round(_finite(years, 0.01), 2),
    }


# ==============================================================================
# BAR-BY-BAR EQUITY + INTRABAR DRAWDOWN (TradingView-aligned)
# ==============================================================================

def _compute_bar_by_bar_equity(trades, df):
    """
    Build a bar-by-bar equity curve with unrealized P&L during open positions.
    All values in PERCENTAGE terms (relative to entry price for each trade).

    Methodology (matching TradingView):
    - equity_pct[i] = sum of realized trade pnl_pcts + unrealized pnl_pct
    - unrealized_pct = (price - entry) / entry * 100 for long
    - For DD: use High/Low to get best/worst equity per bar (intrabar)
    - Max DD = min( equity_low - running_peak(equity_high) ) in percentage

    Uses interval-based tracking: each trade defines [entry_bar, exit_bar].
    No complex state machine — simple linear scan with sorted intervals.

    Returns:
        equity_pct: array of equity in cumulative % (same scale as pnl_pct)
        max_dd_pct: float, maximum intrabar drawdown in % (negative number)
    """
    n = len(df)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)

    sorted_trades = sorted(trades, key=lambda t: t['entry_bar'])
    n_trades = len(sorted_trades)

    # Build trade interval list: (entry_bar, exit_bar, side_mult, entry_price, pnl_pct)
    # side_mult: +1 for long, -1 for short
    intervals = []
    for t in sorted_trades:
        side_mult = 1.0 if t['side'] == 'long' else -1.0
        intervals.append((
            t['entry_bar'], t['exit_bar'], side_mult,
            t['entry_price'], t['pnl_pct']
        ))

    # Arrays for equity at close, high (best), low (worst) per bar
    eq_close = np.zeros(n)
    eq_high = np.zeros(n)
    eq_low = np.zeros(n)

    cum_realized_pct = 0.0
    t_idx = 0

    for i in range(n):
        # Check if current trade exits on this bar
        if t_idx < n_trades:
            eb, xb, sm, ep, ppct = intervals[t_idx]

            if i == xb:
                # Trade exits on this bar — add realized P&L
                cum_realized_pct += ppct
                t_idx += 1

                # Check if next trade starts on same bar
                if t_idx < n_trades and intervals[t_idx][0] == i:
                    eb2, xb2, sm2, ep2, ppct2 = intervals[t_idx]
                    if ep2 > 0:
                        unr_close = sm2 * (close[i] - ep2) / ep2 * 100
                        unr_high = sm2 * ((high[i] if sm2 > 0 else low[i]) - ep2) / ep2 * 100
                        unr_low = sm2 * ((low[i] if sm2 > 0 else high[i]) - ep2) / ep2 * 100
                    else:
                        unr_close = unr_high = unr_low = 0.0
                    eq_close[i] = cum_realized_pct + unr_close
                    eq_high[i] = cum_realized_pct + max(unr_high, unr_low)
                    eq_low[i] = cum_realized_pct + min(unr_high, unr_low)
                else:
                    eq_close[i] = cum_realized_pct
                    eq_high[i] = cum_realized_pct
                    eq_low[i] = cum_realized_pct
                continue

            elif eb <= i < xb:
                # In position — compute unrealized P&L in %
                if ep > 0:
                    unr_close = sm * (close[i] - ep) / ep * 100
                    # For long: high is best, low is worst
                    # For short: low is best, high is worst
                    if sm > 0:  # long
                        unr_best = (high[i] - ep) / ep * 100
                        unr_worst = (low[i] - ep) / ep * 100
                    else:  # short
                        unr_best = (ep - low[i]) / ep * 100
                        unr_worst = (ep - high[i]) / ep * 100
                else:
                    unr_close = unr_best = unr_worst = 0.0

                eq_close[i] = cum_realized_pct + unr_close
                eq_high[i] = cum_realized_pct + unr_best
                eq_low[i] = cum_realized_pct + unr_worst
                continue

        # Flat — no position
        eq_close[i] = cum_realized_pct
        eq_high[i] = cum_realized_pct
        eq_low[i] = cum_realized_pct

    # Compute max drawdown: worst equity vs running peak
    # running_peak uses the BEST equity (high) seen so far
    running_peak = np.maximum.accumulate(eq_high)
    # DD at each bar is worst equity (low) minus running peak
    dd = eq_low - running_peak
    max_dd_pct = float(np.min(dd))

    return eq_close, max_dd_pct


# ==============================================================================
# DAILY SHARPE / SORTINO (TradingView-aligned)
# ==============================================================================

def _compute_daily_sharpe_sortino(trades, df, initial_capital):
    """
    Compute Sharpe and Sortino ratios using DAILY returns on the
    percentage-based equity curve.

    Methodology:
    1. Build bar-by-bar equity in cumulative % (same as DD calculation)
    2. Convert to equity index: base + cumulative_pct (base=100)
    3. Resample to DAILY close (last value per calendar day)
    4. Compute daily percent returns via pct_change()
    5. Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252)
    6. Sortino = mean(daily_returns) / downside_std * sqrt(252)

    NOTE: TradingView's Sharpe depends on position sizing (contracts, capital).
    Our Sharpe uses percentage-of-entry returns (100% face value), which is
    the correct metric for parameter comparison. Expect higher absolute values
    than TV's Sharpe due to different leverage assumptions.

    CRITICAL: For intraday data (M15, M30, H1, H4), resample to daily first.
    """
    if df is not None and len(df) > 0 and len(trades) > 0:
        # Use percentage-based equity (same basis as DD calculation)
        eq_pct, _ = _compute_bar_by_bar_equity(trades, df)

        # Convert to equity index: base=100, so 100 + cumulative_pct
        eq_val = 100.0 + eq_pct

        # Resample to daily (critical for intraday data)
        equity_series = pd.Series(eq_val, index=df.index)
        daily_equity = equity_series.resample('D').last().dropna()

        if len(daily_equity) > 2:
            daily_returns = daily_equity.pct_change().dropna().values * 100

            mean_ret = np.mean(daily_returns)
            std_ret = np.std(daily_returns, ddof=1)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

            downside = daily_returns[daily_returns < 0]
            if len(downside) > 1:
                downside_std = np.std(downside, ddof=1)
                sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0
            else:
                sortino = 0.0
        else:
            sharpe = 0.0
            sortino = 0.0
    else:
        # Fallback: per-trade (less accurate)
        pnl_pcts = np.array([t['pnl_pct'] for t in trades])
        if len(pnl_pcts) > 1:
            avg_dur = np.mean([t.get('duration_bars', 10) for t in trades])
            trades_per_year = 252 * 24 / max(avg_dur, 1)
            mean_ret = np.mean(pnl_pcts)
            std_ret = np.std(pnl_pcts, ddof=1)
            sharpe = (mean_ret / std_ret * np.sqrt(trades_per_year)) if std_ret > 0 else 0.0

            downside = pnl_pcts[pnl_pcts < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 0.001
            sortino = (mean_ret / downside_std * np.sqrt(trades_per_year)) if downside_std > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

    return sharpe, sortino


def compute_composite_score(metrics: dict, weights: dict = None,
                            metrics_is: dict = None) -> float:
    """
    Compute the composite optimization score.

    Score = w1*Calmar + w2*Sortino + w3*Trades_score + w4*DD_score

    Applies:
        - Hard minimum: MIN_TRADES
        - Progressive penalty for low trade count
        - Robustness bonus for large parameters
        - v5.5 Phase 1.2: IS PF cap penalty (PF > 3.0 → penalize)
        - v5.5 Phase 1.4: IS/OOS coherence bonus (if metrics_is provided)

    Parameters:
        metrics: IS metrics dict (used for scoring)
        weights: optional weight overrides
        metrics_is: IS metrics (when scoring OOS, pass IS here for coherence)
    """
    from .config import (COMPOSITE_WEIGHTS, MIN_TRADES, TRADE_PENALTY,
                          IS_PF_CAP_PENALTIES)

    if weights is None:
        weights = COMPOSITE_WEIGHTS

    n_trades = metrics.get('n_trades', 0)

    # Hard minimum
    if n_trades < MIN_TRADES:
        return -999.0

    calmar = max(metrics.get('calmar', 0), 0)
    sortino = max(metrics.get('sortino', 0), 0)
    max_dd = metrics.get('max_dd', 0)
    profit_factor = metrics.get('profit_factor', 0)

    # Trades score: normalized (logarithmic, diminishing returns)
    trades_score = min(np.log(n_trades / MIN_TRADES + 1) * 2, 3.0)

    # DD score: penalize deep drawdowns
    dd_score = max(0, 2.0 + max_dd / 5.0)  # max_dd is negative

    # Composite
    score = (
        weights.get('calmar', 0.35) * calmar +
        weights.get('sortino', 0.25) * sortino +
        weights.get('trades_score', 0.20) * trades_score +
        weights.get('dd_score', 0.20) * dd_score
    )

    # Progressive trade count penalty
    for (lo, hi), mult in TRADE_PENALTY.items():
        if lo <= n_trades <= hi:
            score *= mult
            break

    # ── v5.5 Phase 1.2: IS PF cap penalty ──
    # Configs with IS PF > 3.0 collapse OOS (empirical ratio 0.46).
    # Apply cumulative penalty for "too-good-to-be-true" IS performance.
    for pf_threshold, penalty in sorted(IS_PF_CAP_PENALTIES.items()):
        if profit_factor > pf_threshold:
            score *= penalty

    # ── v5.5 Phase 1.4: IS/OOS coherence bonus ──
    # If IS metrics available, reward configs where OOS ≈ IS (ratio ~1.0).
    if metrics_is is not None:
        is_pf = metrics_is.get('profit_factor', 0) or 0
        oos_pf = profit_factor
        if is_pf > 0 and oos_pf > 0:
            ratio = oos_pf / is_pf
            # ratio=1.0 → +0.5, ratio=2.0 or 0.5 → +0.0
            coherence_bonus = max(0.0, 0.5 - abs(ratio - 1.0) * 0.5)
            score += coherence_bonus

    return round(score, 4)


def passes_oos_gates(metrics: dict, gates: dict = None,
                     instrument: str = None,
                     metrics_is: dict = None) -> tuple:
    """
    Check if OOS metrics pass validation gates.

    The Max DD threshold is ADAPTIVE per instrument asset class.
    If `instrument` is provided, the DD threshold is looked up from
    OOS_DD_LIMITS (e.g. -15% for Indices, -8% for Forex Majors).
    Otherwise, the default from OOS_GATES['max_dd'] is used.

    v5.5 Phase 1.1: IS/OOS coherence gates when metrics_is is provided.
    Detects both classic overfit (OOS << IS) and regime inflation (OOS >> IS).

    Args:
        metrics: OOS metrics dict with 'calmar', 'profit_factor', 'max_dd', etc.
        gates: optional override dict (defaults to OOS_GATES)
        instrument: optional instrument name for adaptive DD threshold
        metrics_is: optional IS metrics for IS/OOS ratio coherence checks

    Returns:
        (passed: bool, reasons: list of str)
    """
    from .config import OOS_GATES, get_oos_dd_limit, OOS_COHERENCE_GATES

    if gates is None:
        gates = OOS_GATES

    # Adaptive DD threshold: per-instrument if available, else gates default
    if instrument:
        dd_limit = get_oos_dd_limit(instrument)
    else:
        dd_limit = gates.get('max_dd', -8.0)

    reasons = []
    passed = True

    if metrics.get('calmar', 0) < gates.get('min_calmar', 0.40):
        reasons.append(f"Calmar {metrics.get('calmar', 0):.3f} < {gates.get('min_calmar', 0.40)}")
        passed = False

    if metrics.get('profit_factor', 0) < gates.get('min_pf', 1.25):
        reasons.append(f"PF {metrics.get('profit_factor', 0):.3f} < {gates.get('min_pf', 1.25)}")
        passed = False

    if metrics.get('max_dd', 0) < dd_limit:
        reasons.append(f"MaxDD {metrics.get('max_dd', 0):.2f}% < {dd_limit}%")
        passed = False

    if metrics.get('n_trades', 0) < gates.get('min_trades', 20):
        reasons.append(f"Trades {metrics.get('n_trades', 0)} < {gates.get('min_trades', 20)}")
        passed = False

    # ── v5.5 Phase 1.1: IS/OOS coherence gates ──
    if metrics_is is not None:
        is_pf = _finite(metrics_is.get('profit_factor', 0))
        oos_pf = _finite(metrics.get('profit_factor', 0))

        if is_pf > 0 and oos_pf > 0:
            pf_ratio = oos_pf / is_pf
            # Reject if OOS >> IS (probable regime-favorable, not real edge)
            if pf_ratio > OOS_COHERENCE_GATES['pf_ratio_max']:
                reasons.append(f"OOS/IS PF ratio {pf_ratio:.2f} > {OOS_COHERENCE_GATES['pf_ratio_max']} (inflated)")
                passed = False
            # Reject if OOS << IS (classic overfit collapse)
            if pf_ratio < OOS_COHERENCE_GATES['pf_ratio_min']:
                reasons.append(f"OOS/IS PF ratio {pf_ratio:.2f} < {OOS_COHERENCE_GATES['pf_ratio_min']} (collapsed)")
                passed = False

        # Win rate divergence check
        is_wr = _finite(metrics_is.get('win_rate', 0))
        oos_wr = _finite(metrics.get('win_rate', 0))
        if is_wr > 0 and oos_wr > 0:
            wr_delta = abs(oos_wr - is_wr)
            if wr_delta > OOS_COHERENCE_GATES['wr_delta_max']:
                reasons.append(f"WR delta {wr_delta:.1f}pp > {OOS_COHERENCE_GATES['wr_delta_max']}pp")
                passed = False

    return passed, reasons


def _max_consecutive(mask: np.ndarray) -> int:
    """Count max consecutive True values in a bool array."""
    max_count = 0
    current = 0
    for val in mask:
        if val:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count


def _empty_metrics() -> dict:
    """Return empty metrics dict."""
    return {
        'n_trades': 0, 'n_wins': 0, 'n_losses': 0,
        'win_rate': 0, 'net_return': 0, 'annual_return': 0,
        'max_dd': 0, 'calmar': 0, 'profit_factor': 0,
        'sharpe': 0, 'sortino': 0, 'expectancy': 0,
        'avg_win': 0, 'avg_loss': 0,
        'total_profit': 0, 'total_loss': 0,
        'max_consec_wins': 0, 'max_consec_losses': 0,
        'avg_duration_bars': 0,
        'omega_ratio': 0, 'tail_ratio': 0,
        'equity_curve': [], 'equity_abs': [], 'drawdown_curve': [],
        'years': 0,
    }


# ==============================================================================
# V5.4: QUALITY SCORE V2 — Robustness Score + Quality Grade
# ==============================================================================

def compute_robustness_score(wfe, perturbation_stable, perturbation_degradation,
                              wf_stability, wf_consistency, wf_degradation) -> float:
    """
    Score de robustesse [0-100] basé sur WFE, perturbation et Walk-Forward.

    Composantes :
      - WFE (0-35 pts) : optimal at ~0.70, 0 pts if >2.0 or <0.4 or None
      - Perturbation (0-30 pts) : 30 pts if stable with 0% degradation
      - Walk-Forward (0-35 pts) : WF stability + WF consistency + no degradation
    """
    score = 0.0

    # WFE (0-35 points) — zone optimale [0.4, 1.3]
    if wfe is not None and 0.4 <= wfe <= 1.3:
        # Bell curve : max at 0.70 (normal IS→OOS degradation)
        wfe_score = 35.0 * (1.0 - abs(wfe - 0.70) / 0.60)
        score += max(0, wfe_score)
    elif wfe is not None and 1.3 < wfe <= 2.0:
        score += 10.0  # Acceptable mais suspect
    # wfe > 2.0, < 0.4, or None → 0 points

    # Perturbation (0-30 points)
    if perturbation_stable:
        deg = perturbation_degradation if perturbation_degradation else 0.0
        score += 30.0 * max(0, 1.0 - deg / 0.50)

    # Walk-Forward (0-35 points)
    wf_stab = wf_stability or 0
    wf_cons = wf_consistency or 0
    wf_deg = wf_degradation or False

    if wf_stab >= 70 and not wf_deg:
        score += 25.0 * (wf_stab / 100.0)
        score += 10.0 * (wf_cons / 100.0)
    elif wf_stab >= 40:
        score += 15.0 * (wf_stab / 100.0)
    # wf_stab < 40 → 0 points

    return round(min(score, 100.0), 1)


def compute_quality_grade(perf_score: float, robustness_score: float,
                           entry_type: str, instrument: str,
                           safe_entry_types: list = None,
                           transaction_costs: dict = None,
                           default_cost: float = 1.0,
                           metrics_is: dict = None,
                           metrics_oos: dict = None) -> dict:
    """
    Quality grade final [0-100] avec grade A/B/C/D et flags.

    perf_score: OOS performance score (from existing quality calc, max ~12)
    robustness_score: from compute_robustness_score() [0-100]
    metrics_is: IS metrics dict (optional, for v5.5 overfit detection)
    metrics_oos: OOS metrics dict (optional, for v5.5 overfit detection)

    v5.5 Phase 1: adds overfit detection flags and penalties:
      - OVERFIT_PF_RATIO: OOS/IS PF ratio outside [0.30, 2.50]
      - IS_PF_TOO_HIGH: IS PF > 3.0 (empirically collapses OOS)
      - OVERFIT_WR_DELTA: |OOS WR - IS WR| > 15pp
      - IS_OOS_COHERENT: bonus flag when ratio is near 1.0

    Returns dict with: quality_score, grade, perf_normalized, robustness_score,
                       flags, multiplier, pf_ratio, wr_delta, coherence_bonus
    """
    from .config import OOS_COHERENCE_GATES, IS_PF_CAP_PENALTIES

    # Normalize perf_score to [0-100] (current max is ~12-13 pts)
    perf_normalized = min(perf_score / 12.0 * 100.0, 100.0)

    # Weighted average
    raw_score = 0.55 * perf_normalized + 0.45 * robustness_score

    # Flags and penalties
    flags = []
    multiplier = 1.0
    pf_ratio = None
    wr_delta = None
    coherence_bonus = 0.0

    # SAFE check
    if safe_entry_types is None:
        try:
            from pineguard import SAFE_ENTRY_TYPES
            safe_entry_types = SAFE_ENTRY_TYPES
        except ImportError:
            safe_entry_types = ['Donc+Chikou', 'RSI Divergence', 'Fractal', 'Boll+SMA', 'MM Cross']
    if entry_type and entry_type not in safe_entry_types:
        flags.append('NON_SAFE_ENTRY')
        multiplier *= 0.60

    # Cost check
    if transaction_costs is not None:
        # Check partial match (strip timeframe suffix)
        base = instrument.split()[0] if instrument and ' ' in instrument else (instrument or '')
        inst_upper = instrument.upper().strip() if instrument else ''
        base_upper = base.upper().strip()
        if inst_upper not in transaction_costs and base_upper not in transaction_costs:
            flags.append('DEFAULT_COST')
            multiplier *= 0.80

    # ── v5.5 Phase 1: Overfit detection flags and penalties ──
    if metrics_is is not None and metrics_oos is not None:
        is_pf = _finite(metrics_is.get('profit_factor', 0))
        oos_pf = _finite(metrics_oos.get('profit_factor', 0))

        # IS PF too-good-to-be-true flag
        if is_pf > 3.0:
            flags.append('IS_PF_TOO_HIGH')
            # Progressive penalty matching IS_PF_CAP_PENALTIES
            for pf_threshold, penalty in sorted(IS_PF_CAP_PENALTIES.items()):
                if is_pf > pf_threshold:
                    multiplier *= penalty

        # PF ratio check
        if is_pf > 0 and oos_pf > 0:
            pf_ratio = round(oos_pf / is_pf, 3)
            if pf_ratio > OOS_COHERENCE_GATES['pf_ratio_max']:
                flags.append('OVERFIT_PF_RATIO')
                multiplier *= 0.70  # severe penalty for inflated OOS
            elif pf_ratio < OOS_COHERENCE_GATES['pf_ratio_min']:
                flags.append('OVERFIT_PF_RATIO')
                multiplier *= 0.70  # severe penalty for collapsed OOS
            elif 0.60 <= pf_ratio <= 1.40:
                # Good coherence zone — reward
                flags.append('IS_OOS_COHERENT')
                coherence_bonus = max(0.0, 5.0 - abs(pf_ratio - 1.0) * 10.0)
                raw_score += coherence_bonus

        # Win rate divergence check
        is_wr = _finite(metrics_is.get('win_rate', 0))
        oos_wr = _finite(metrics_oos.get('win_rate', 0))
        if is_wr > 0 and oos_wr > 0:
            wr_delta = round(abs(oos_wr - is_wr), 1)
            if wr_delta > OOS_COHERENCE_GATES['wr_delta_max']:
                flags.append('OVERFIT_WR_DELTA')
                multiplier *= 0.85

    final_score = round(raw_score * multiplier, 1)

    # Grade
    if final_score >= 70 and 'NON_SAFE_ENTRY' not in flags:
        grade = 'A'   # Phase 2 TV candidate immédiat
    elif final_score >= 50:
        grade = 'B'   # Prometteur, validation complémentaire requise
    elif final_score >= 30:
        grade = 'C'   # Exploratoire, R&D uniquement
    else:
        grade = 'D'   # Rejet

    return {
        'quality_score': final_score,
        'grade': grade,
        'perf_normalized': round(perf_normalized, 1),
        'robustness_score': round(robustness_score, 1),
        'flags': flags,
        'multiplier': multiplier,
        'pf_ratio': pf_ratio,
        'wr_delta': wr_delta,
        'coherence_bonus': round(coherence_bonus, 2),
    }
