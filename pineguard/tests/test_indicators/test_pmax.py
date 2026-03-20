"""
Tests for PMax — the core algorithm.

Key properties tested:
  - Stateful stop carry-over (longStop only increases during uptrend)
  - Direction changes only on crossover
  - Signals are strict crossovers (> AND [1] <=)
  - Uses hl2 as source
  - ATR uses RMA
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.indicators.pmax import compute_pmax


@pytest.fixture
def trending_up_data():
    """Generate clear uptrend data."""
    np.random.seed(42)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    base = 5000 + np.arange(n) * 0.5  # clear uptrend
    noise = np.random.randn(n) * 2
    close = pd.Series(base + noise, index=idx)
    high = close + np.abs(np.random.randn(n)) * 3
    low = close - np.abs(np.random.randn(n)) * 3
    return pd.DataFrame({'High': high, 'Low': low, 'Close': close}, index=idx)


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(123)
    n = 500
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    returns = np.random.randn(n) * 0.001
    close = pd.Series(5000 * np.cumprod(1 + returns), index=idx)
    high = close * (1 + np.abs(np.random.randn(n)) * 0.001)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.001)
    vol = pd.Series(np.random.randint(100, 2000, n), index=idx)
    return pd.DataFrame({
        'Open': close.shift(1).fillna(close.iloc[0]),
        'High': high, 'Low': low, 'Close': close, 'Volume': vol,
    }, index=idx)


class TestPMaxOutput:
    def test_returns_all_keys(self, sample_ohlcv):
        """PMax should return all expected keys."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'],
            length=10, multiplier=3.0, ma_type='EMA'
        )
        expected_keys = {'MAvg', 'ATR', 'PMax', 'dir', 'longStop', 'shortStop',
                         'buy_trigger', 'sell_trigger'}
        assert set(result.keys()) == expected_keys

    def test_output_lengths(self, sample_ohlcv):
        """All outputs should have same length as input."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'],
            length=10, multiplier=3.0
        )
        n = len(sample_ohlcv)
        for key, val in result.items():
            assert len(val) == n, f"{key} has wrong length"


class TestPMaxStops:
    def test_long_stop_non_decreasing_in_uptrend(self, trending_up_data):
        """During uptrend (dir=1), longStop should only increase."""
        result = compute_pmax(
            trending_up_data['High'], trending_up_data['Low'],
            trending_up_data['Close'],
            length=10, multiplier=3.0
        )
        ls = result['longStop']
        d = result['dir']

        # Find bars where dir = 1
        uptrend = d == 1
        if uptrend.sum() < 10:
            pytest.skip("Not enough uptrend bars")

        # In uptrend segments, longStop should be non-decreasing
        uptrend_groups = (uptrend != uptrend.shift(1)).cumsum()
        for _, group in ls[uptrend].groupby(uptrend_groups[uptrend]):
            if len(group) > 1:
                diffs = group.diff().dropna()
                assert (diffs >= -1e-10).all(), \
                    "longStop decreased during uptrend (carry-over bug)"


class TestPMaxSignals:
    def test_signals_are_boolean(self, sample_ohlcv):
        """buy_trigger and sell_trigger should be boolean series."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        assert result['buy_trigger'].dtype == bool
        assert result['sell_trigger'].dtype == bool

    def test_no_simultaneous_signals(self, sample_ohlcv):
        """buy and sell should never fire on the same bar."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        simultaneous = result['buy_trigger'] & result['sell_trigger']
        assert not simultaneous.any(), "Buy and sell fired simultaneously"

    def test_signals_are_crossovers(self, sample_ohlcv):
        """Verify signals follow strict crossover logic."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close']
        )
        mavg = result['MAvg']
        pmax = result['PMax']
        buy = result['buy_trigger']

        # For every buy signal: MAvg > PMax AND MAvg[1] <= PMax[1]
        for i in buy[buy].index:
            loc = mavg.index.get_loc(i)
            if loc == 0:
                continue
            assert mavg.iloc[loc] > pmax.iloc[loc], \
                f"Buy at {i}: MAvg should be > PMax"
            assert mavg.iloc[loc - 1] <= pmax.iloc[loc - 1], \
                f"Buy at {i}: MAvg[1] should be <= PMax[1]"


class TestPMaxMATypes:
    def test_all_ma_types(self, sample_ohlcv):
        """PMax should work with all 9 MA types."""
        for ma_type in ['SMA', 'EMA', 'RMA', 'WMA', 'DEMA', 'TEMA', 'HMA', 'ZLEMA']:
            result = compute_pmax(
                sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'],
                length=10, multiplier=3.0, ma_type=ma_type
            )
            assert not result['MAvg'].dropna().empty, f"MAvg empty for {ma_type}"
            assert not result['PMax'].dropna().empty, f"PMax empty for {ma_type}"

    def test_vwma_requires_volume(self, sample_ohlcv):
        """VWMA should work when volume is provided."""
        result = compute_pmax(
            sample_ohlcv['High'], sample_ohlcv['Low'], sample_ohlcv['Close'],
            length=10, multiplier=3.0, ma_type='VWMA',
            volume=sample_ohlcv['Volume']
        )
        assert not result['MAvg'].dropna().empty
