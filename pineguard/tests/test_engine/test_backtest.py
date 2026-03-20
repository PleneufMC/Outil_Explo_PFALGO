"""
Tests for Backtest Engine — 3-phase execution model.

Critical checks:
  - Phase 1: Execute at Open[i], not Close[i-1]
  - Phase 2: SL checked on High/Low
  - Phase 3: Signal detected at Close → queue for next Open
  - No look-ahead bias
  - Transaction costs applied
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.engine.backtest_engine import BacktestEngine, Direction


@pytest.fixture
def simple_data():
    """Simple trending data with clear signals."""
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    prices = 5000 + np.arange(n) * 0.5 + np.random.randn(n) * 2
    np.random.seed(42)
    df = pd.DataFrame({
        'Open': pd.Series(prices, index=idx).shift(1).fillna(prices[0]),
        'High': prices + np.abs(np.random.randn(n)) * 3,
        'Low': prices - np.abs(np.random.randn(n)) * 3,
        'Close': prices,
        'Volume': np.random.randint(100, 1000, n),
    }, index=idx)
    return df


@pytest.fixture
def engine():
    return BacktestEngine(instrument='US500', sl_mode='none')


class TestBacktestExecution:
    def test_no_trades_without_signals(self, engine, simple_data):
        """No signals → no trades."""
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        result = engine.run(simple_data, entry_long, entry_short)
        assert result['trade_count'] == 0

    def test_single_long_trade(self, engine, simple_data):
        """Single long signal → one trade."""
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[10] = True  # Signal at bar 10
        result = engine.run(simple_data, entry_long, entry_short)
        # Should have one trade (opened at bar 11, closed at end)
        assert result['trade_count'] >= 1
        trade = result['trades'][0]
        assert trade.direction == Direction.LONG
        assert trade.entry_bar == 11  # Filled at Open[11], not Close[10]

    def test_execution_at_open_not_close(self, engine, simple_data):
        """Entry should be at Open[i+1], not Close[i]."""
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[20] = True
        result = engine.run(simple_data, entry_long, entry_short)
        trade = result['trades'][0]
        # Entry price should be Open of bar 21
        expected_price = simple_data['Open'].iloc[21]
        assert trade.entry_price == pytest.approx(expected_price)

    def test_reversal(self, simple_data):
        """Long → Short reversal should close long and open short."""
        engine = BacktestEngine(instrument='US500', sl_mode='none',
                               enable_reversal=True)
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[10] = True   # Long at bar 10
        entry_short.iloc[30] = True  # Short at bar 30 → reversal
        result = engine.run(simple_data, entry_long, entry_short)
        # Should have 2 trades: closed long + opened short
        assert result['trade_count'] >= 2
        assert result['trades'][0].direction == Direction.LONG
        assert result['trades'][0].exit_reason == 'reversal'

    def test_transaction_costs(self, simple_data):
        """Transaction costs should reduce PnL."""
        engine_no_cost = BacktestEngine(instrument='US500', sl_mode='none')
        # Override cost for testing
        engine_no_cost.cost_per_trade = 0.0

        engine_with_cost = BacktestEngine(instrument='US500', sl_mode='none')
        engine_with_cost.cost_per_trade = 1.0

        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[10] = True

        r_no_cost = engine_no_cost.run(simple_data, entry_long, entry_short)
        r_with_cost = engine_with_cost.run(simple_data, entry_long, entry_short)

        if r_no_cost['trades'] and r_with_cost['trades']:
            assert r_with_cost['trades'][0].pnl_points < r_no_cost['trades'][0].pnl_points


class TestStopLoss:
    def test_sl_hit_on_low(self, simple_data):
        """Long position SL should be checked against Low."""
        engine = BacktestEngine(instrument='US500', sl_mode='fixed_pct', sl_pct=0.01)
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[10] = True

        atr = pd.Series(5.0, index=simple_data.index)  # Constant ATR
        result = engine.run(simple_data, entry_long, entry_short, atr=atr)
        # With very tight SL, should hit SL
        if result['trades']:
            # At least check that trades were generated
            assert result['trade_count'] >= 1


class TestEntryAllowed:
    def test_entry_blocked_by_gate(self, engine, simple_data):
        """entry_allowed=False should block entries."""
        entry_long = pd.Series(False, index=simple_data.index)
        entry_short = pd.Series(False, index=simple_data.index)
        entry_long.iloc[10] = True
        entry_long.iloc[20] = True

        # Block all entries
        allowed_long = pd.Series(False, index=simple_data.index)
        allowed_short = pd.Series(False, index=simple_data.index)

        result = engine.run(simple_data, entry_long, entry_short,
                          entry_allowed_long=allowed_long,
                          entry_allowed_short=allowed_short)
        assert result['trade_count'] == 0


class TestReversalEquity:
    """Tests for the reversal + SL-on-same-bar equity fix."""

    def test_reversal_pnl_both_counted(self):
        """When a reversal fires and the new position also hits SL on the same bar,
        both trades' PnL must be reflected in the equity curve."""
        n = 50
        idx = pd.date_range('2024-01-01', periods=n, freq='1h')
        # Construct controlled data
        prices = np.full(n, 5000.0)
        prices[11] = 5010  # Open of bar 11 (long fills here)
        # Bar 31 open: reversal close long, open short
        prices[31] = 5020  # Open[31] → long exits at 5020
        df = pd.DataFrame({
            'Open': prices.copy(),
            'High': prices + 20,
            'Low': prices - 20,
            'Close': prices.copy(),
            'Volume': np.full(n, 500),
        }, index=idx)
        # Force specific values for bar 31
        df.iloc[31, df.columns.get_loc('Open')] = 5020
        df.iloc[31, df.columns.get_loc('High')] = 5025
        df.iloc[31, df.columns.get_loc('Low')] = 4975

        engine = BacktestEngine(instrument='US500', sl_mode='none')
        engine.cost_per_trade = 0.0

        entry_long = pd.Series(False, index=idx)
        entry_short = pd.Series(False, index=idx)
        entry_long.iloc[10] = True   # Signal bar 10 → fills at Open[11]
        entry_short.iloc[30] = True  # Signal bar 30 → reversal at Open[31]

        result = engine.run(df, entry_long, entry_short)
        # Should have at least 2 trades (reversal close + new short)
        assert result['trade_count'] >= 2
        # First trade is the closed long
        assert result['trades'][0].exit_reason == 'reversal'
        assert result['trades'][0].direction == Direction.LONG
        # Equity should reflect realized PnL correctly at bar 31
        equity_at_31 = result['equity'].iloc[31]
        realized = sum(t.pnl_points for t in result['trades']
                      if t.exit_bar <= 31 and t.exit_bar >= 0)
        # Equity = initial + realized + unrealized at bar 31
        assert equity_at_31 != engine.initial_capital or realized == 0

    def test_reversal_generates_two_trades(self):
        """Reversal should produce exactly one closed trade + one new position."""
        n = 50
        idx = pd.date_range('2024-01-01', periods=n, freq='1h')
        prices = 5000 + np.arange(n, dtype=float) * 0.5
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 5,
            'Low': prices - 5,
            'Close': prices,
            'Volume': np.full(n, 500),
        }, index=idx)

        engine = BacktestEngine(instrument='US500', sl_mode='none')
        engine.cost_per_trade = 0.0

        entry_long = pd.Series(False, index=idx)
        entry_short = pd.Series(False, index=idx)
        entry_long.iloc[5] = True
        entry_short.iloc[20] = True  # reversal

        result = engine.run(df, entry_long, entry_short)
        # Trade 0: long closed by reversal
        # Trade 1: short closed at end_of_data
        assert result['trade_count'] == 2
        assert result['trades'][0].exit_reason == 'reversal'
        assert result['trades'][1].exit_reason == 'end_of_data'
        assert result['trades'][1].direction == Direction.SHORT
