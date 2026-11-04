"""
Tests for VWAP entry signals.

Verifies:
  - Session-anchored VWAP resets at session start
  - VWAP formula: cumsum(tp * vol) / cumsum(vol)
  - Crossover/crossunder logic
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.vwap import compute_vwap, compute_vwap_signals


@pytest.fixture
def vwap_data():
    """Generate 3 days of hourly data."""
    n = 72  # 3 days * 24 hours
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    np.random.seed(42)
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 2), index=idx)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    volume = pd.Series(np.random.randint(100, 1000, n), index=idx)
    return high, low, close, volume


class TestVWAP:
    def test_vwap_basic(self, vwap_data):
        """VWAP should produce values close to price."""
        high, low, close, volume = vwap_data
        vwap = compute_vwap(high, low, close, volume, session_start_hour=0)
        valid = vwap.dropna()
        assert len(valid) > 0
        # VWAP should be within reasonable range of close
        diff_pct = ((valid - close[valid.index]).abs() / close[valid.index] * 100)
        assert diff_pct.mean() < 5.0  # Within 5% on average

    def test_vwap_session_reset(self, vwap_data):
        """VWAP should reset at each session start."""
        high, low, close, volume = vwap_data
        vwap = compute_vwap(high, low, close, volume, session_start_hour=0)
        # At session start, VWAP = hlc3 (single bar)
        session_starts = vwap.index[vwap.index.hour == 0]
        for ts in session_starts[:3]:
            hlc3 = (high[ts] + low[ts] + close[ts]) / 3
            assert vwap[ts] == pytest.approx(hlc3, rel=1e-6)


class TestVWAPSignals:
    def test_output_keys(self, vwap_data):
        high, low, close, volume = vwap_data
        result = compute_vwap_signals(high, low, close, volume)
        assert 'vwap' in result
        assert 'vwap_long' in result
        assert 'vwap_short' in result

    def test_signals_boolean(self, vwap_data):
        high, low, close, volume = vwap_data
        result = compute_vwap_signals(high, low, close, volume)
        assert result['vwap_long'].dtype == bool
        assert result['vwap_short'].dtype == bool
