"""
Tests for ATR, RSI, and Stdev — verify Pine equivalence.

Critical checks:
  - ATR uses RMA, not SMA
  - RSI uses RMA for gain/loss smoothing
  - Stdev uses ddof=0 (population), not ddof=1 (sample)
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.indicators.atr import compute_tr, compute_atr, compute_atr_sma
from pineguard.indicators.rsi import compute_rsi
from pineguard.indicators.stdev import compute_stdev, compute_stdev_sample, stdev_divergence_factor


@pytest.fixture
def ohlc_data():
    np.random.seed(42)
    n = 200
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 5), index=idx)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    return high, low, close


class TestTR:
    def test_first_bar(self, ohlc_data):
        """First bar TR should be high-low (no previous close)."""
        high, low, close = ohlc_data
        tr = compute_tr(high, low, close)
        # First bar should be NaN (because close.shift(1) is NaN)
        # But max of (high-low, NaN, NaN) = high-low
        expected_first = high.iloc[0] - low.iloc[0]
        assert tr.iloc[0] == pytest.approx(expected_first)

    def test_tr_always_positive(self, ohlc_data):
        """TR should always be >= 0."""
        high, low, close = ohlc_data
        tr = compute_tr(high, low, close)
        assert (tr.dropna() >= 0).all()

    def test_tr_includes_gaps(self):
        """TR should capture gap moves."""
        idx = pd.date_range('2024-01-01', periods=3, freq='1h')
        high = pd.Series([100, 105, 110], index=idx, dtype=float)
        low = pd.Series([98, 103, 108], index=idx, dtype=float)
        close = pd.Series([99, 104, 109], index=idx, dtype=float)
        tr = compute_tr(high, low, close)
        # Bar 1: max(105-103, |105-99|, |103-99|) = max(2, 6, 4) = 6
        assert tr.iloc[1] == pytest.approx(6.0)


class TestATR:
    def test_atr_rma_differs_from_sma(self, ohlc_data):
        """ATR (RMA) should differ from SMA-based ATR."""
        high, low, close = ohlc_data
        atr_rma = compute_atr(high, low, close, 14)
        atr_sma = compute_atr_sma(high, low, close, 14)
        # They should NOT be equal
        common = atr_rma.dropna().index.intersection(atr_sma.dropna().index)
        assert not np.allclose(atr_rma[common].values, atr_sma[common].values)


class TestRSI:
    def test_rsi_range(self, ohlc_data):
        """RSI should be between 0 and 100."""
        _, _, close = ohlc_data
        rsi = compute_rsi(close, 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_uptrend_high(self):
        """RSI should be high (>50) in a strong uptrend."""
        idx = pd.date_range('2024-01-01', periods=100, freq='1h')
        close = pd.Series(range(100, 200), index=idx, dtype=float)
        rsi = compute_rsi(close, 14)
        # After warmup, RSI should be close to 100
        assert rsi.iloc[-1] > 90

    def test_rsi_downtrend_low(self):
        """RSI should be low (<50) in a strong downtrend."""
        idx = pd.date_range('2024-01-01', periods=100, freq='1h')
        close = pd.Series(range(200, 100, -1), index=idx, dtype=float)
        rsi = compute_rsi(close, 14)
        assert rsi.iloc[-1] < 10


class TestStdev:
    def test_ddof0_vs_ddof1(self, ohlc_data):
        """ddof=0 (Pine) should give smaller values than ddof=1 (pandas default)."""
        _, _, close = ohlc_data
        std_pop = compute_stdev(close, 20)
        std_sample = compute_stdev_sample(close, 20)
        # Population stdev < sample stdev (always, for n>1)
        common = std_pop.dropna().index.intersection(std_sample.dropna().index)
        assert (std_pop[common] <= std_sample[common]).all()

    def test_divergence_factor(self):
        """Verify divergence factor formula."""
        # For n=20: sqrt(20/19) ≈ 1.0263
        factor = stdev_divergence_factor(20)
        assert factor == pytest.approx(np.sqrt(20/19), rel=1e-6)

        # For n=10: sqrt(10/9) ≈ 1.0541
        factor = stdev_divergence_factor(10)
        assert factor == pytest.approx(np.sqrt(10/9), rel=1e-6)

    def test_ddof0_matches_numpy(self, ohlc_data):
        """Our ddof=0 should match numpy's population stdev."""
        _, _, close = ohlc_data
        std_ours = compute_stdev(close, 20)
        # Manual check on a window
        window = close.iloc[50:70].values
        expected = np.std(window, ddof=0)
        assert std_ours.iloc[69] == pytest.approx(expected, rel=1e-6)
