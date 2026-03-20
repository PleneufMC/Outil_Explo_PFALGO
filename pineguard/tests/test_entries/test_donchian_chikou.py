"""
Tests for Donchian+Chikou entry — the 100% reliable entry type.

Verifies:
  - Donchian channel = rolling max/min
  - Chikou shift = simple .shift(26)
  - Crossover logic is strict (> AND [1] <=)
  - Signals are deterministic
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.donchian_chikou import compute_donchian_chikou_signals


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 5), index=idx)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    return high, low, close


class TestDonchianChikou:
    def test_output_keys(self, sample_data):
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(high, low, close)
        assert 'don_upper' in result
        assert 'don_lower' in result
        assert 'don_long' in result
        assert 'don_short' in result

    def test_donchian_upper_is_rolling_max(self, sample_data):
        """Upper band should equal rolling(20).max()."""
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(high, low, close, don_length=20)
        expected = high.rolling(20).max()
        pd.testing.assert_series_equal(
            result['don_upper'].dropna(), expected.dropna(),
            check_names=False
        )

    def test_shifted_by_chikou(self, sample_data):
        """Shifted band should be .shift(26) of the band."""
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(high, low, close, chikou_shift=26)
        expected = result['don_upper'].shift(26)
        # Compare non-NaN values
        common = result['shifted_upper'].dropna().index.intersection(expected.dropna().index)
        pd.testing.assert_series_equal(
            result['shifted_upper'][common], expected[common],
            check_names=False
        )

    def test_signals_are_boolean(self, sample_data):
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(high, low, close)
        assert result['don_long'].dtype == bool
        assert result['don_short'].dtype == bool

    def test_crossover_strictness(self, sample_data):
        """Verify crossover uses > AND <= (not < or <)."""
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(high, low, close)
        shifted_upper = result['shifted_upper']

        for i in result['don_long'][result['don_long']].index:
            loc = close.index.get_loc(i)
            if loc == 0:
                continue
            # Current: close > shifted_upper
            assert close.iloc[loc] > shifted_upper.iloc[loc]
            # Previous: close[1] <= shifted_upper[1]
            assert close.iloc[loc - 1] <= shifted_upper.iloc[loc - 1]

    def test_deterministic(self, sample_data):
        """Same data should always give same signals."""
        high, low, close = sample_data
        r1 = compute_donchian_chikou_signals(high, low, close)
        r2 = compute_donchian_chikou_signals(high, low, close)
        assert (r1['don_long'] == r2['don_long']).all()
        assert (r1['don_short'] == r2['don_short']).all()

    def test_no_signals_before_warmup(self, sample_data):
        """No signals should fire before don_length + chikou_shift bars."""
        high, low, close = sample_data
        result = compute_donchian_chikou_signals(
            high, low, close, don_length=20, chikou_shift=26
        )
        warmup = 20 + 26
        assert not result['don_long'].iloc[:warmup].any()
        assert not result['don_short'].iloc[:warmup].any()
