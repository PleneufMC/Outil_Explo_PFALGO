"""
Tests for moving averages — verify Pine↔Python equivalence.

Key equivalence rules tested:
  - EMA: adjust=False
  - RMA: alpha=1/n (NOT span=n)
  - RMA ≠ EMA for same length
  - All 9 MA types produce valid output
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.indicators.moving_averages import (
    compute_sma, compute_ema, compute_rma, compute_wma,
    compute_dema, compute_tema, compute_hma, compute_zlema,
    compute_vwma, get_ma, MA_TYPES,
)


@pytest.fixture
def sample_series():
    """Generate a sample price series."""
    np.random.seed(42)
    idx = pd.date_range('2024-01-01', periods=100, freq='1h')
    return pd.Series(np.cumsum(np.random.randn(100)) + 100, index=idx)


@pytest.fixture
def volume_series():
    """Generate a sample volume series."""
    np.random.seed(42)
    idx = pd.date_range('2024-01-01', periods=100, freq='1h')
    return pd.Series(np.abs(np.random.randn(100) * 1000).astype(int), index=idx)


class TestSMA:
    def test_basic(self, sample_series):
        result = compute_sma(sample_series, 10)
        assert len(result) == len(sample_series)
        assert result.iloc[:9].isna().all()  # First 9 bars should be NaN
        assert not result.iloc[9:].isna().any()

    def test_known_value(self):
        """SMA of [1,2,3,4,5] with length 3 should give [NaN, NaN, 2, 3, 4]."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_sma(s, 3)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)


class TestEMA:
    def test_adjust_false(self, sample_series):
        """EMA must use adjust=False to match Pine."""
        ema = compute_ema(sample_series, 10)
        expected = sample_series.ewm(span=10, adjust=False).mean()
        pd.testing.assert_series_equal(ema, expected)

    def test_adjust_true_differs(self, sample_series):
        """Verify that adjust=True gives DIFFERENT results."""
        ema_correct = compute_ema(sample_series, 10)
        ema_wrong = sample_series.ewm(span=10, adjust=True).mean()
        # They should NOT be equal
        assert not np.allclose(ema_correct.dropna(), ema_wrong.dropna())


class TestRMA:
    def test_rma_not_equal_to_ema(self, sample_series):
        """RMA and EMA with same length should give DIFFERENT results."""
        rma = compute_rma(sample_series, 14)
        ema = compute_ema(sample_series, 14)
        # RMA uses alpha=1/14, EMA uses alpha=2/15 → different
        assert not np.allclose(rma.dropna(), ema.dropna())

    def test_rma_alpha(self, sample_series):
        """RMA should use alpha=1/length."""
        rma = compute_rma(sample_series, 14)
        expected = sample_series.ewm(alpha=1/14, adjust=False).mean()
        pd.testing.assert_series_equal(rma, expected)

    def test_rma_smoother_than_ema(self, sample_series):
        """RMA should be smoother (less reactive) than EMA of same length."""
        rma = compute_rma(sample_series, 14)
        ema = compute_ema(sample_series, 14)
        # RMA has lower alpha → less reactive → smaller variance of changes
        rma_var = rma.diff().dropna().var()
        ema_var = ema.diff().dropna().var()
        assert rma_var < ema_var


class TestWMA:
    def test_known_value(self):
        """WMA([1,2,3,4,5], 3) last value: (3*5 + 2*4 + 1*3) / 6 = 26/6."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_wma(s, 3)
        expected = (3*5 + 2*4 + 1*3) / (1+2+3)
        assert result.iloc[-1] == pytest.approx(expected, rel=1e-6)


class TestAllMATypes:
    def test_all_9_types_exist(self):
        """All 9 MA types should be registered."""
        assert len(MA_TYPES) == 9
        expected = {'SMA', 'EMA', 'RMA', 'WMA', 'VWMA', 'DEMA', 'TEMA', 'HMA', 'ZLEMA'}
        assert set(MA_TYPES.keys()) == expected

    def test_get_ma_all_types(self, sample_series, volume_series):
        """get_ma() should work for all 9 types."""
        for ma_type in MA_TYPES:
            if ma_type == 'VWMA':
                result = get_ma(sample_series, 10, ma_type, volume=volume_series)
            else:
                result = get_ma(sample_series, 10, ma_type)
            assert len(result) == len(sample_series)
            assert not result.dropna().empty

    def test_get_ma_invalid_type(self, sample_series):
        """get_ma() should raise ValueError for unknown type."""
        with pytest.raises(ValueError, match="Unknown MA type"):
            get_ma(sample_series, 10, 'INVALID')


class TestDEMA:
    def test_less_lag_than_ema(self, sample_series):
        """DEMA should have less lag than EMA."""
        dema = compute_dema(sample_series, 20)
        ema = compute_ema(sample_series, 20)
        # DEMA reacts faster — check that recent values are closer to price
        recent = sample_series.iloc[-20:]
        dema_recent = dema.iloc[-20:]
        ema_recent = ema.iloc[-20:]
        dema_dist = (recent - dema_recent).abs().mean()
        ema_dist = (recent - ema_recent).abs().mean()
        assert dema_dist < ema_dist


class TestHMA:
    def test_output_length(self, sample_series):
        """HMA output should have same length as input."""
        result = compute_hma(sample_series, 20)
        assert len(result) == len(sample_series)
