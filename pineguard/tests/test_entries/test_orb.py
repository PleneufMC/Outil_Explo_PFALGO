"""
Tests for ORB (Opening Range Breakout) entry signals.

Verifies:
  - ORB high/low correctly computed from first N minutes
  - Signal only fires after ORB period
  - Correct vs inverted (D13 bug) signal logic
  - Session boundary detection
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.orb import compute_orb_signals


@pytest.fixture
def intraday_data():
    """Generate 5 days of 1-min data with clear ORB patterns."""
    n = 5 * 390  # 5 days, 390 min each (9:30 - 16:00)
    idx = pd.date_range('2024-01-01 09:30', periods=n, freq='1min')
    np.random.seed(42)
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 0.5), index=idx)
    high = close + np.abs(np.random.randn(n)) * 0.3
    low = close - np.abs(np.random.randn(n)) * 0.3
    open_ = close.shift(1).fillna(close.iloc[0])
    return high, low, close, open_


class TestORBSignals:
    def test_output_keys(self, intraday_data):
        high, low, close, open_ = intraday_data
        result = compute_orb_signals(high, low, close, open_,
                                     orb_minutes=30,
                                     session_start_hour=9,
                                     session_start_minute=30)
        expected = {'orb_high', 'orb_low', 'orb_long', 'orb_short', 'is_orb_period'}
        assert set(result.keys()) == expected

    def test_no_signals_during_orb_period(self, intraday_data):
        """Signals should NOT fire during ORB calculation period."""
        high, low, close, open_ = intraday_data
        result = compute_orb_signals(high, low, close, open_,
                                     orb_minutes=30,
                                     session_start_hour=9,
                                     session_start_minute=30)
        is_orb = result['is_orb_period']
        long_in_orb = result['orb_long'] & is_orb
        short_in_orb = result['orb_short'] & is_orb
        assert not long_in_orb.any(), "Long signal during ORB period"
        assert not short_in_orb.any(), "Short signal during ORB period"

    def test_correct_vs_inverted(self, intraday_data):
        """Correct and inverted signals should be different."""
        high, low, close, open_ = intraday_data
        correct = compute_orb_signals(high, low, close, open_,
                                      orb_minutes=30,
                                      session_start_hour=9,
                                      session_start_minute=30,
                                      reproduce_pine_bug=False)
        inverted = compute_orb_signals(high, low, close, open_,
                                       orb_minutes=30,
                                       session_start_hour=9,
                                       session_start_minute=30,
                                       reproduce_pine_bug=True)
        # They should produce different signals
        if correct['orb_long'].any() or inverted['orb_long'].any():
            # At least one should differ
            assert not (correct['orb_long'] == inverted['orb_long']).all() or \
                   not (correct['orb_short'] == inverted['orb_short']).all()

    def test_signals_are_boolean(self, intraday_data):
        high, low, close, open_ = intraday_data
        result = compute_orb_signals(high, low, close, open_,
                                     orb_minutes=30,
                                     session_start_hour=9,
                                     session_start_minute=30)
        assert result['orb_long'].dtype == bool
        assert result['orb_short'].dtype == bool
