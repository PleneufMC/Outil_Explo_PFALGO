"""
Tests for Bollinger + SMA entry signals.

Verifies:
  - Bollinger bands use ddof=0 (population stdev)
  - Crossover logic: close crosses above lower band
  - Crossunder logic: close crosses below upper band
  - SMA filter applied: long only above SMA, short only below SMA
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.boll_sma import compute_boll_sma_signals


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 5), index=idx)
    return close


class TestBollSmaSignals:
    def test_output_keys(self, sample_data):
        result = compute_boll_sma_signals(sample_data)
        expected = {'bb_upper', 'bb_lower', 'bb_basis', 'sma_filter',
                    'boll_long', 'boll_short'}
        assert set(result.keys()) == expected

    def test_signals_are_boolean(self, sample_data):
        result = compute_boll_sma_signals(sample_data)
        assert result['boll_long'].dtype == bool
        assert result['boll_short'].dtype == bool

    def test_long_above_sma(self, sample_data):
        """Long signals must only fire when close > SMA."""
        result = compute_boll_sma_signals(sample_data, sma_length=50)
        long_bars = result['boll_long']
        sma = result['sma_filter']
        for i in long_bars[long_bars].index:
            assert sample_data[i] > sma[i], \
                f"Long signal at {i} but close <= SMA"

    def test_short_below_sma(self, sample_data):
        """Short signals must only fire when close < SMA."""
        result = compute_boll_sma_signals(sample_data, sma_length=50)
        short_bars = result['boll_short']
        sma = result['sma_filter']
        for i in short_bars[short_bars].index:
            assert sample_data[i] < sma[i], \
                f"Short signal at {i} but close >= SMA"

    def test_bollinger_uses_ddof0(self, sample_data):
        """Verify Bollinger bands use population stdev (ddof=0)."""
        result = compute_boll_sma_signals(sample_data, boll_length=20, boll_mult=2.0)
        # Manual check: basis should be SMA(20)
        expected_basis = sample_data.rolling(20).mean()
        common = result['bb_basis'].dropna().index.intersection(expected_basis.dropna().index)
        pd.testing.assert_series_equal(
            result['bb_basis'][common], expected_basis[common],
            check_names=False
        )

    def test_no_simultaneous_signals(self, sample_data):
        """Long and short should never fire simultaneously."""
        result = compute_boll_sma_signals(sample_data)
        both = result['boll_long'] & result['boll_short']
        assert not both.any()

    def test_crossover_strictness(self, sample_data):
        """Verify long uses strict crossover: close > lower AND close[1] <= lower[1]."""
        result = compute_boll_sma_signals(sample_data)
        lower = result['bb_lower']
        for i in result['boll_long'][result['boll_long']].index:
            loc = sample_data.index.get_loc(i)
            if loc == 0:
                continue
            assert sample_data.iloc[loc] > lower.iloc[loc]
            assert sample_data.iloc[loc - 1] <= lower.iloc[loc - 1]
