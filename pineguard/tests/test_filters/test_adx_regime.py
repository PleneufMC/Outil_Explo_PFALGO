"""
Tests for ADX Regime Filter.

Verifies:
  - ADX trending/ranging classification
  - Entry type / regime compatibility matrix
  - Filter gating
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.filters.adx_regime import (
    compute_adx_regime_filter, REGIME_COMPATIBILITY,
)


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 300
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(5000 + np.cumsum(np.random.randn(n) * 5), index=idx)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    return high, low, close


class TestADXRegime:
    def test_output_keys(self, sample_data):
        high, low, close = sample_data
        result = compute_adx_regime_filter(high, low, close)
        expected = {'adx', 'di_plus', 'di_minus', 'is_trending',
                    'is_ranging', 'regime_allowed', 'regime'}
        assert set(result.keys()) == expected

    def test_trending_ranging_exclusive(self, sample_data):
        """Trending and ranging should be mutually exclusive."""
        high, low, close = sample_data
        result = compute_adx_regime_filter(high, low, close)
        # is_trending XOR is_ranging should always be True
        valid = result['is_trending'].dropna() | result['is_ranging'].dropna()
        assert valid.all()
        both = result['is_trending'] & result['is_ranging']
        assert not both.any()

    def test_adx_range(self, sample_data):
        """ADX should be between 0 and 100."""
        high, low, close = sample_data
        result = compute_adx_regime_filter(high, low, close)
        valid_adx = result['adx'].dropna()
        assert (valid_adx >= 0).all()
        assert (valid_adx <= 100).all()

    def test_regime_compatibility(self):
        """All 7 entry types should be in the compatibility matrix."""
        assert len(REGIME_COMPATIBILITY) == 7
        for et in ['Donc+Chikou', 'RSI Divergence', 'Boll+SMA',
                    'Fractal', 'VWAP', 'ORB', 'MM Cross']:
            assert et in REGIME_COMPATIBILITY

    def test_donc_chikou_both_regimes(self, sample_data):
        """Donc+Chikou works in both regimes."""
        high, low, close = sample_data
        result = compute_adx_regime_filter(high, low, close,
                                            entry_type='Donc+Chikou')
        # Should be allowed in ALL bars (works in both regimes)
        valid = result['regime_allowed'].dropna()
        assert valid.all()
