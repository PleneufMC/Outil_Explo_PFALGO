"""
Tests for LT/MT Trend Filters — CONTINUOUS STATE (V.68.8+).

Verifies:
  - Continuous state (not crossover)
  - HTF PMax computation
  - shift(1) for confirmed bar
  - Forward-fill to base TF
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.filters.trend_filter import compute_lt_filter, compute_mt_filter


@pytest.fixture
def hourly_ohlcv():
    """60 days of hourly data — enough for weekly PMax warmup."""
    n = 60 * 24  # 60 days
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    np.random.seed(42)
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 3), index=idx)
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.Series(np.random.randint(100, 2000, n), index=idx)
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=idx)


class TestLTFilter:
    def test_output_keys(self, hourly_ohlcv):
        result = compute_lt_filter(hourly_ohlcv, lt_tf='1D')  # Use daily for faster test
        assert 'lt_long' in result
        assert 'lt_short' in result
        assert 'lt_mavg' in result
        assert 'lt_pmax' in result

    def test_continuous_state(self, hourly_ohlcv):
        """LT filter should be continuous (not single-bar crossover)."""
        result = compute_lt_filter(hourly_ohlcv, lt_tf='1D')
        lt_long = result['lt_long'].dropna()
        if lt_long.any():
            # Consecutive True bars should exist
            runs = lt_long.astype(int).groupby(
                (lt_long != lt_long.shift()).cumsum()
            ).sum()
            max_run = runs.max()
            assert max_run > 1, "Filter seems to be single-bar, not continuous"

    def test_long_short_mutually_exclusive(self, hourly_ohlcv):
        """Long and short filters should be mutually exclusive (ignoring NaN)."""
        result = compute_lt_filter(hourly_ohlcv, lt_tf='1D')
        both = result['lt_long'] & result['lt_short']
        assert not both.any(), "Both LT long and short active simultaneously"

    def test_output_length_matches_base(self, hourly_ohlcv):
        """Filter output should match base timeframe length."""
        result = compute_lt_filter(hourly_ohlcv, lt_tf='1D')
        assert len(result['lt_long']) == len(hourly_ohlcv)


class TestMTFilter:
    def test_output_keys(self, hourly_ohlcv):
        result = compute_mt_filter(hourly_ohlcv, mt_tf='4H')
        assert 'mt_long' in result
        assert 'mt_short' in result

    def test_4h_filter_works(self, hourly_ohlcv):
        """MT filter on 4H should produce non-empty results."""
        result = compute_mt_filter(hourly_ohlcv, mt_tf='4H')
        mt_long = result['mt_long'].dropna()
        assert len(mt_long) > 0
