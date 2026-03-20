"""
Tests for performance metrics.
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.metrics.performance import (
    max_drawdown, annualized_return, calmar_ratio,
    sharpe_ratio, profit_factor, win_rate, expectancy,
    compute_all_metrics,
)
from pineguard.engine.backtest_engine import Trade, Direction


@pytest.fixture
def equity_curve():
    idx = pd.date_range('2024-01-01', periods=252, freq='1D')
    # 10% annual return with some volatility
    returns = np.random.RandomState(42).normal(0.0004, 0.01, 252)
    equity = 100000 * np.cumprod(1 + returns)
    return pd.Series(equity, index=idx)


@pytest.fixture
def sample_trades():
    trades = []
    for i, pnl in enumerate([10, -5, 15, -3, 8, -7, 20, -2, 12, -6]):
        t = Trade(
            entry_bar=i * 10, entry_time=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i*10),
            entry_price=5000, direction=Direction.LONG,
            exit_bar=i * 10 + 5,
            exit_time=pd.Timestamp('2024-01-01') + pd.Timedelta(hours=i*10+5),
            exit_price=5000 + pnl, pnl_points=pnl, bars_held=5,
        )
        trades.append(t)
    return trades


class TestMaxDrawdown:
    def test_no_drawdown(self):
        """Monotonically increasing equity → 0 drawdown."""
        idx = pd.date_range('2024-01-01', periods=100, freq='1D')
        equity = pd.Series(range(100, 200), index=idx, dtype=float)
        dd = max_drawdown(equity)
        assert dd['max_dd_pct'] == pytest.approx(0.0)

    def test_known_drawdown(self):
        """Known drawdown: 100 → 90 → 110 = 10% dd."""
        idx = pd.date_range('2024-01-01', periods=3, freq='1D')
        equity = pd.Series([100.0, 90.0, 110.0], index=idx)
        dd = max_drawdown(equity)
        assert dd['max_dd_pct'] == pytest.approx(10.0)


class TestWinRate:
    def test_all_winners(self, sample_trades):
        """Modify all to winners → 100%."""
        for t in sample_trades:
            t.pnl_points = abs(t.pnl_points)
        assert win_rate(sample_trades) == pytest.approx(100.0)

    def test_known_win_rate(self, sample_trades):
        """5 winners, 5 losers out of 10 → 50%."""
        assert win_rate(sample_trades) == pytest.approx(50.0)


class TestProfitFactor:
    def test_profitable(self, sample_trades):
        """Sum of wins > sum of losses → PF > 1."""
        pf = profit_factor(sample_trades)
        assert pf > 1.0

    def test_no_losses(self):
        """All winners → PF = inf."""
        trades = [Trade(0, pd.Timestamp('2024-01-01'), 100, Direction.LONG,
                       pnl_points=10, bars_held=5)]
        assert profit_factor(trades) == float('inf')


class TestComputeAll:
    def test_all_metrics_present(self, equity_curve, sample_trades):
        metrics = compute_all_metrics(equity_curve, sample_trades)
        required = [
            'total_trades', 'win_rate', 'profit_factor', 'max_drawdown_pct',
            'sharpe_ratio', 'calmar_ratio', 'total_pnl',
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"
