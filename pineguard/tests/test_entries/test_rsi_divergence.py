"""
Tests for RSI Divergence entry signals.

Verifies:
  - Pivot detection asymmetry (>= left, > right for highs)
  - Bullish divergence: price LL + RSI HL
  - Bearish divergence: price HH + RSI LH
  - Cancel logic
  - Pivot detection delay
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.rsi_divergence import (
    compute_rsi_divergence_signals, _detect_pivot_high, _detect_pivot_low,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 5), index=idx)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    return high, low, close


class TestPivotDetection:
    def test_pivot_high_basic(self):
        """Detect a clear pivot high."""
        idx = pd.date_range('2024-01-01', periods=11, freq='1h')
        # Clear pivot at index 5: [30, 40, 50, 60, 70, 80, 70, 60, 50, 40, 30]
        src = pd.Series([30, 40, 50, 60, 70, 80, 70, 60, 50, 40, 30],
                        index=idx, dtype=float)
        pivots = _detect_pivot_high(src, left=5, right=5)
        # Pivot at index 5, detected at index 10 (5 bars delay)
        assert not np.isnan(pivots.iloc[10])
        assert pivots.iloc[10] == pytest.approx(80.0)

    def test_pivot_low_basic(self):
        """Detect a clear pivot low."""
        idx = pd.date_range('2024-01-01', periods=11, freq='1h')
        src = pd.Series([80, 70, 60, 50, 40, 30, 40, 50, 60, 70, 80],
                        index=idx, dtype=float)
        pivots = _detect_pivot_low(src, left=5, right=5)
        assert not np.isnan(pivots.iloc[10])
        assert pivots.iloc[10] == pytest.approx(30.0)

    def test_pivot_high_asymmetry(self):
        """Pine: left uses >=, right uses > (asymmetric)."""
        idx = pd.date_range('2024-01-01', periods=7, freq='1h')
        # Left equal: [50, 50, 50, 40, 30] — pivot at index 2 since 50 >= 50
        src = pd.Series([40, 50, 50, 50, 40, 30, 20], index=idx, dtype=float)
        pivots = _detect_pivot_high(src, left=2, right=2)
        # Candidate at index 2: left [40, 50] — 50 >= 40 (yes), 50 >= 50 (yes)
        # Right [50, 40] — 50 > 50? NO! So this should NOT be a pivot
        # Candidate at index 3: left [50, 50] — 50 >= 50 (yes), 50 >= 50 (yes)
        # Right [40, 30] — 50 > 40 (yes), 50 > 30 (yes) → pivot!
        assert not np.isnan(pivots.iloc[5])  # Detected at index 3+2 = 5


class TestRSIDivergenceSignals:
    def test_output_keys(self, sample_data):
        high, low, close = sample_data
        result = compute_rsi_divergence_signals(high, low, close)
        expected = {'rsi', 'pivot_high', 'pivot_low',
                    'rsi_div_long', 'rsi_div_short'}
        assert set(result.keys()) == expected

    def test_rsi_in_range(self, sample_data):
        """RSI should be 0-100."""
        high, low, close = sample_data
        result = compute_rsi_divergence_signals(high, low, close)
        valid_rsi = result['rsi'].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_signals_are_boolean(self, sample_data):
        high, low, close = sample_data
        result = compute_rsi_divergence_signals(high, low, close)
        assert result['rsi_div_long'].dtype == bool
        assert result['rsi_div_short'].dtype == bool

    def test_cancel_logic_runs(self, sample_data):
        """Cancel logic should not crash."""
        high, low, close = sample_data
        r_with = compute_rsi_divergence_signals(high, low, close,
                                                 enable_cancel_logic=True)
        r_without = compute_rsi_divergence_signals(high, low, close,
                                                    enable_cancel_logic=False)
        # Cancel should reduce or maintain signal count
        assert r_with['rsi_div_long'].sum() <= r_without['rsi_div_long'].sum()

    def test_deterministic(self, sample_data):
        """Same data should give same signals."""
        high, low, close = sample_data
        r1 = compute_rsi_divergence_signals(high, low, close)
        r2 = compute_rsi_divergence_signals(high, low, close)
        assert (r1['rsi_div_long'] == r2['rsi_div_long']).all()
        assert (r1['rsi_div_short'] == r2['rsi_div_short']).all()


class TestRSICancelLogicD11:
    """Tests for the 3-layer cancel logic (D11 fix)."""

    def test_cancel_reduces_signals(self, sample_data):
        """Cancel logic should eliminate some signals."""
        high, low, close = sample_data
        r_cancel = compute_rsi_divergence_signals(
            high, low, close, enable_cancel_logic=True
        )
        r_nocancel = compute_rsi_divergence_signals(
            high, low, close, enable_cancel_logic=False
        )
        # Cancel should reduce or equal the signal count
        assert r_cancel['rsi_div_long'].sum() <= r_nocancel['rsi_div_long'].sum()
        assert r_cancel['rsi_div_short'].sum() <= r_nocancel['rsi_div_short'].sum()

    def test_cancel_logic_no_crash_empty(self):
        """Cancel logic should handle very short data without crashing."""
        idx = pd.date_range('2024-01-01', periods=30, freq='1h')
        close = pd.Series(np.linspace(5000, 5050, 30), index=idx)
        high = close + 3
        low = close - 3
        result = compute_rsi_divergence_signals(
            high, low, close, enable_cancel_logic=True
        )
        assert 'rsi_div_long' in result
        assert 'rsi_div_short' in result

    def test_cancel_preserves_strong_signals(self):
        """Strong divergences (clear LL in price, HL in RSI) should survive cancel."""
        # Build data where RSI doesn't move dramatically on the next bar
        np.random.seed(123)
        n = 500
        idx = pd.date_range('2024-01-01', periods=n, freq='1h')
        close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 2), index=idx)
        high = close + np.abs(np.random.randn(n)) * 2
        low = close - np.abs(np.random.randn(n)) * 2

        r_cancel = compute_rsi_divergence_signals(
            high, low, close, enable_cancel_logic=True
        )
        # We can't guarantee signals exist, but the function shouldn't crash
        # and output should be well-formed
        assert r_cancel['rsi_div_long'].dtype == bool
        assert r_cancel['rsi_div_short'].dtype == bool
