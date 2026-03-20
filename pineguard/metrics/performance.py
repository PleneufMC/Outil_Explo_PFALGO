"""
Performance Metrics — Standard trading metrics.

All metrics follow standard financial conventions:
  - Calmar = annualized return / max drawdown
  - Sharpe = annualized: mean(returns) / std(returns) × √(periods_per_year)
  - Max Drawdown = peak-to-trough equity drop
  - Profit Factor = gross profit / gross loss
  - Win Rate = winning trades / total trades
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

Series = pd.Series


def max_drawdown(equity: Series) -> Dict[str, float]:
    """
    Maximum Drawdown — peak-to-trough equity decline.

    Returns:
        Dict with:
            'max_dd_pct': Maximum drawdown as percentage
            'max_dd_abs': Maximum drawdown in absolute terms
            'dd_start': Timestamp of peak
            'dd_end': Timestamp of trough
    """
    if len(equity) == 0:
        return {'max_dd_pct': 0.0, 'max_dd_abs': 0.0, 'dd_start': None, 'dd_end': None}

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    drawdown_abs = equity - running_max

    max_dd_pct = drawdown.min()
    max_dd_abs = drawdown_abs.min()

    # Find peak and trough
    trough_idx = drawdown.idxmin()
    peak_idx = equity[:trough_idx].idxmax() if trough_idx is not None else None

    return {
        'max_dd_pct': abs(max_dd_pct) * 100,  # As positive percentage
        'max_dd_abs': abs(max_dd_abs),
        'dd_start': peak_idx,
        'dd_end': trough_idx,
        'dd_series': drawdown,  # Full drawdown series for plotting
    }


def annualized_return(equity: Series, periods_per_year: float = 252) -> float:
    """
    Annualized return from equity curve.

    Args:
        equity: Equity curve
        periods_per_year: Trading periods per year (252 daily, 52 weekly, ~6240 H1)
    """
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0

    total_return = equity.iloc[-1] / equity.iloc[0]
    n_periods = len(equity)
    years = n_periods / periods_per_year

    if years <= 0 or total_return <= 0:
        return 0.0

    return (total_return ** (1 / years) - 1) * 100


def calmar_ratio(equity: Series, periods_per_year: float = 252) -> float:
    """
    Calmar Ratio = Annualized Return / Max Drawdown.

    Higher is better. > 1.0 is acceptable, > 2.0 is good, > 3.0 is excellent.
    """
    ann_ret = annualized_return(equity, periods_per_year)
    dd = max_drawdown(equity)
    max_dd = dd['max_dd_pct']

    if max_dd == 0:
        return 0.0
    return ann_ret / max_dd


def sharpe_ratio(equity: Series, risk_free_rate: float = 0.0,
                 periods_per_year: float = 252) -> float:
    """
    Annualized Sharpe Ratio.

    Sharpe = (mean_return - risk_free) / std_return × √periods_per_year

    Args:
        equity: Equity curve
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: For annualization (252=daily, 52=weekly)
    """
    if len(equity) < 2:
        return 0.0

    returns = equity.pct_change().dropna()
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Period risk-free rate
    rf_period = risk_free_rate / periods_per_year

    excess_returns = returns - rf_period
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(equity: Series, risk_free_rate: float = 0.0,
                  periods_per_year: float = 252) -> float:
    """
    Sortino Ratio — like Sharpe but only penalizes downside volatility.

    Sortino = (mean_return - risk_free) / downside_std × √periods_per_year
    """
    if len(equity) < 2:
        return 0.0

    returns = equity.pct_change().dropna()
    rf_period = risk_free_rate / periods_per_year
    excess_returns = returns - rf_period

    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0

    return (excess_returns.mean() / downside.std()) * np.sqrt(periods_per_year)


def profit_factor(trades: list) -> float:
    """
    Profit Factor = Gross Profit / Gross Loss.

    > 1.0 is profitable. > 1.5 is good. > 2.0 is excellent.
    """
    gross_profit = sum(t.pnl_points for t in trades if t.pnl_points > 0)
    gross_loss = abs(sum(t.pnl_points for t in trades if t.pnl_points < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def win_rate(trades: list) -> float:
    """Win Rate = winning trades / total trades × 100."""
    if len(trades) == 0:
        return 0.0
    winners = sum(1 for t in trades if t.pnl_points > 0)
    return (winners / len(trades)) * 100


def expectancy(trades: list) -> float:
    """
    Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss).

    Positive expectancy = profitable system over time.
    """
    if len(trades) == 0:
        return 0.0

    winners = [t.pnl_points for t in trades if t.pnl_points > 0]
    losers = [t.pnl_points for t in trades if t.pnl_points <= 0]

    win_pct = len(winners) / len(trades) if trades else 0
    loss_pct = len(losers) / len(trades) if trades else 0

    avg_win = np.mean(winners) if winners else 0
    avg_loss = abs(np.mean(losers)) if losers else 0

    return (win_pct * avg_win) - (loss_pct * avg_loss)


def compute_all_metrics(
    equity: Series,
    trades: list,
    periods_per_year: float = 252,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Compute all performance metrics.

    Args:
        equity: Equity curve
        trades: List of Trade objects
        periods_per_year: For annualization
        risk_free_rate: Annual risk-free rate

    Returns:
        Comprehensive metrics dictionary
    """
    dd = max_drawdown(equity)

    return {
        'total_trades': len(trades),
        'win_rate': round(win_rate(trades), 2),
        'profit_factor': round(profit_factor(trades), 3),
        'expectancy': round(expectancy(trades), 4),
        'max_drawdown_pct': round(dd['max_dd_pct'], 2),
        'max_drawdown_abs': round(dd['max_dd_abs'], 2),
        'annualized_return': round(annualized_return(equity, periods_per_year), 2),
        'calmar_ratio': round(calmar_ratio(equity, periods_per_year), 3),
        'sharpe_ratio': round(sharpe_ratio(equity, risk_free_rate, periods_per_year), 3),
        'sortino_ratio': round(sortino_ratio(equity, risk_free_rate, periods_per_year), 3),
        'total_pnl': round(sum(t.pnl_points for t in trades), 2),
        'avg_trade_pnl': round(np.mean([t.pnl_points for t in trades]), 4) if trades else 0,
        'avg_bars_held': round(np.mean([t.bars_held for t in trades]), 1) if trades else 0,
        'longest_trade': max((t.bars_held for t in trades), default=0),
        'shortest_trade': min((t.bars_held for t in trades), default=0),
        'consecutive_wins': _max_consecutive(trades, winning=True),
        'consecutive_losses': _max_consecutive(trades, winning=False),
        'dd_start': dd['dd_start'],
        'dd_end': dd['dd_end'],
    }


def _max_consecutive(trades: list, winning: bool) -> int:
    """Count max consecutive wins or losses."""
    if not trades:
        return 0
    max_streak = 0
    current = 0
    for t in trades:
        is_win = t.pnl_points > 0
        if is_win == winning:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak
