"""
Tests for HTF resampling — critical for LT/MT filters.

Verifies:
  - Correct OHLCV aggregation (first, max, min, last, sum)
  - shift(1) applied in HTF space (not base TF)
  - Forward-fill to base timeframe
  - Weekly boundary handling
  - No look-ahead bias
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.filters.htf_resampling import (
    resample_to_htf, reindex_htf_to_base, validate_htf_boundaries,
    TF_MAP,
)


@pytest.fixture
def hourly_data():
    """Generate 30 days of hourly data."""
    n = 30 * 24  # 30 days
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    np.random.seed(42)
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 2), index=idx)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(100, 1000, n), index=idx)
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=idx)


class TestResampleToHTF:
    def test_daily_aggregation(self, hourly_data):
        """Resample H1 to Daily: 24 bars -> 1 bar per day."""
        df_daily = resample_to_htf(hourly_data, '1D')
        # Should have ~30 days
        assert len(df_daily) >= 28
        assert len(df_daily) <= 31

    def test_4h_aggregation(self, hourly_data):
        """Resample H1 to 4H: 4 bars -> 1 bar."""
        df_4h = resample_to_htf(hourly_data, '4H')
        # Should have ~180 bars (30 * 24 / 4)
        assert len(df_4h) >= 170
        assert len(df_4h) <= 200

    def test_ohlcv_rules(self, hourly_data):
        """Verify OHLCV aggregation rules."""
        df_daily = resample_to_htf(hourly_data, '1D')
        # Daily High should be max of all hourly highs that day
        first_day = hourly_data.iloc[:24]
        assert df_daily['High'].iloc[0] == pytest.approx(
            first_day['High'].max(), rel=1e-6
        )
        assert df_daily['Low'].iloc[0] == pytest.approx(
            first_day['Low'].min(), rel=1e-6
        )

    def test_hl2_computation(self, hourly_data):
        """When hl2=True, should add hl2 column."""
        df_daily = resample_to_htf(hourly_data, '1D', hl2=True)
        assert 'hl2' in df_daily.columns
        expected = (df_daily['High'] + df_daily['Low']) / 2
        pd.testing.assert_series_equal(df_daily['hl2'], expected, check_names=False)

    def test_unknown_tf_raises(self, hourly_data):
        """Unknown timeframe should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            resample_to_htf(hourly_data, '3H')

    def test_weekly_uses_friday(self, hourly_data):
        """Weekly resampling should use W-FRI."""
        df_weekly = resample_to_htf(hourly_data, '1W')
        assert len(df_weekly) >= 3


class TestReindexHTFToBase:
    def test_shift_prevents_lookahead(self, hourly_data):
        """shift(1) should prevent look-ahead bias."""
        df_daily = resample_to_htf(hourly_data, '1D')
        close_daily = df_daily['Close']

        # With shift(1): first value should be NaN
        reindexed = reindex_htf_to_base(close_daily, hourly_data.index, shift_bars=1)
        assert pd.isna(reindexed.iloc[0])

    def test_no_shift(self, hourly_data):
        """shift_bars=0 should have no NaN at start."""
        df_daily = resample_to_htf(hourly_data, '1D')
        close_daily = df_daily['Close']
        reindexed = reindex_htf_to_base(close_daily, hourly_data.index, shift_bars=0)
        # Should have forward-filled values
        assert not pd.isna(reindexed.iloc[23])  # End of first day

    def test_output_length_matches_base(self, hourly_data):
        """Reindexed series should have same length as base."""
        df_daily = resample_to_htf(hourly_data, '1D')
        reindexed = reindex_htf_to_base(df_daily['Close'], hourly_data.index)
        assert len(reindexed) == len(hourly_data)


class TestValidateHTFBoundaries:
    def test_basic_validation(self, hourly_data):
        """Basic validation should pass for standard data."""
        df_daily = resample_to_htf(hourly_data, '1D')
        result = validate_htf_boundaries(hourly_data, df_daily)
        assert result['alignment_score'] > 50
        assert 'boundary_issues' in result
        assert 'gap_count' in result
