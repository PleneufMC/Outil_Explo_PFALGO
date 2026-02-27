"""
Tests for Fractal entry — all 5 Williams patterns.
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.entries.fractal import compute_fractal_signals


@pytest.fixture
def classic_fractal_data():
    """Data with clear classic fractal pattern."""
    # High fractal: [10, 12, 15, 12, 10] → peak at index 2
    # Signal at index 4 (2 bars delay)
    n = 20
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    high = pd.Series([10, 12, 15, 12, 10] + [10]*15, index=idx, dtype=float)
    low = pd.Series([8, 10, 13, 10, 8] + [8]*15, index=idx, dtype=float)
    close = pd.Series([9, 11, 14, 11, 9] + [9]*15, index=idx, dtype=float)
    return high, low, close


class TestFractalPatterns:
    def test_classic_pattern_detected(self, classic_fractal_data):
        """Pattern 1 (classic) should detect the clear fractal."""
        high, low, close = classic_fractal_data
        result = compute_fractal_signals(high, low, close, patterns='classic',
                                         enable_dedup=False)
        # Should detect at least one fractal
        assert result['fractal_short'].any() or result['fractal_long'].any()

    def test_all_5_patterns(self, classic_fractal_data):
        """All 5 patterns should produce output."""
        high, low, close = classic_fractal_data
        result = compute_fractal_signals(high, low, close, patterns='all',
                                         enable_dedup=False)
        assert 'fractal_up_1' in result
        assert 'fractal_up_2' in result
        assert 'fractal_up_3' in result
        assert 'fractal_up_4' in result
        assert 'fractal_up_5' in result

    def test_patterns_selectable(self, classic_fractal_data):
        high, low, close = classic_fractal_data
        result = compute_fractal_signals(high, low, close, patterns='1,3')
        assert 'fractal_up_1' in result
        assert 'fractal_up_3' in result
        assert 'fractal_up_2' not in result

    def test_dedup_prevents_double(self):
        """Dedup should prevent consecutive signals in same direction."""
        n = 100
        idx = pd.date_range('2024-01-01', periods=n, freq='1h')
        np.random.seed(42)
        high = pd.Series(np.random.randn(n).cumsum() + 100, index=idx)
        low = high - np.abs(np.random.randn(n)) * 2
        close = (high + low) / 2

        result_no_dedup = compute_fractal_signals(
            high, low, close, patterns='classic', enable_dedup=False
        )
        result_dedup = compute_fractal_signals(
            high, low, close, patterns='classic', enable_dedup=True
        )

        # Deduped should have <= signals than non-deduped
        assert result_dedup['fractal_long'].sum() <= result_no_dedup['fractal_long'].sum()
