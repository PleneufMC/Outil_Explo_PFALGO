"""
Tests for PF AI Lab 5.0.4 Monte Carlo simulation.
"""

import pytest
import numpy as np

from pineguard.engine.backtest_engine import Trade, Direction
import pandas as pd


@pytest.fixture
def sample_trades():
    """Create sample trades for MC testing."""
    trades = []
    base_time = pd.Timestamp('2024-01-01')
    pnls = [10, -5, 15, -8, 20, -3, 12, -7, 25, -10, 8, -4, 18, -6, 30]

    for i, pnl in enumerate(pnls):
        t = Trade(
            entry_bar=i * 10,
            entry_time=base_time + pd.Timedelta(hours=i * 10),
            entry_price=5000.0,
            direction=Direction.LONG if pnl > 0 else Direction.SHORT,
            exit_bar=i * 10 + 5,
            exit_time=base_time + pd.Timedelta(hours=i * 10 + 5),
            exit_price=5000.0 + pnl,
            exit_reason='signal',
            pnl_points=pnl,
            bars_held=5,
        )
        trades.append(t)

    return trades


class TestMonteCarloTradeShuffle:
    """Test Monte Carlo trade shuffling."""

    def test_basic_mc(self, sample_trades):
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle
        result = monte_carlo_trade_shuffle(
            sample_trades, n_simulations=100, seed=42,
        )

        assert result.n_simulations == 100
        assert result.method == 'trade_shuffle'
        assert result.pnl_mean != 0  # Should have non-zero PnL
        assert result.max_dd_mean > 0  # Drawdown should be positive
        assert result.sharpe_ci_95[0] <= result.sharpe_ci_95[1]

    def test_mc_percentiles_ordered(self, sample_trades):
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle
        result = monte_carlo_trade_shuffle(
            sample_trades, n_simulations=500, seed=42,
        )

        pct = result.pnl_percentiles
        assert pct['5'] <= pct['25'] <= pct['50'] <= pct['75'] <= pct['95']

    def test_mc_empty_trades(self):
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle
        result = monte_carlo_trade_shuffle([], n_simulations=100)
        assert result.pnl_mean == 0
        assert result.max_dd_mean == 0

    def test_mc_store_curves(self, sample_trades):
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle
        result = monte_carlo_trade_shuffle(
            sample_trades, n_simulations=50,
            store_curves=True, seed=42,
        )
        assert result.equity_curves is not None
        assert result.equity_curves.shape == (50, len(sample_trades) + 1)

    def test_mc_report_format(self, sample_trades):
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle, format_mc_report
        result = monte_carlo_trade_shuffle(
            sample_trades, n_simulations=100, seed=42,
        )
        report = format_mc_report(result)
        assert 'Monte Carlo' in report
        assert 'PnL Distribution' in report
        assert 'Drawdown' in report
        assert 'Sharpe' in report
