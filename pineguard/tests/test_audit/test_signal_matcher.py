"""
Tests for Signal Matcher — Pine vs Python signal comparison.

Verifies:
  - Exact match detection
  - Tolerance-based matching
  - Python-only (phantom) and Pine-only (missed) classification
  - Multi-signal comparison
  - Indicator value comparison
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.audit.signal_matcher import SignalMatcher


@pytest.fixture
def matcher():
    return SignalMatcher(time_tolerance_bars=1)


@pytest.fixture
def signal_pair():
    idx = pd.date_range('2024-01-01', periods=100, freq='1h')
    pine = pd.Series(False, index=idx)
    python = pd.Series(False, index=idx)
    # Both fire at bar 10, 30, 50
    pine.iloc[10] = True
    pine.iloc[30] = True
    pine.iloc[50] = True
    python.iloc[10] = True
    python.iloc[30] = True
    python.iloc[50] = True
    # Python extra at bar 70 (phantom)
    python.iloc[70] = True
    # Pine extra at bar 90 (missed)
    pine.iloc[90] = True
    return pine, python


class TestSignalMatcher:
    def test_perfect_match(self, matcher):
        """Identical signals should have 100% match rate."""
        idx = pd.date_range('2024-01-01', periods=50, freq='1h')
        signal = pd.Series(False, index=idx)
        signal.iloc[10] = True
        signal.iloc[20] = True
        result = matcher.compare_signals(signal, signal, 'test')
        assert result['summary']['match_rate_exact'] == pytest.approx(100.0)
        assert result['summary']['python_only_count'] == 0
        assert result['summary']['pine_only_count'] == 0

    def test_phantom_and_missed(self, matcher, signal_pair):
        """Detect phantom (Python-only) and missed (Pine-only) signals."""
        pine, python = signal_pair
        result = matcher.compare_signals(pine, python, 'test')
        assert result['summary']['matched_exact'] == 3
        assert result['summary']['python_only_count'] == 1  # Bar 70
        assert result['summary']['pine_only_count'] == 1     # Bar 90

    def test_tolerance_matching(self, signal_pair):
        """Tolerance should match signals within +-N bars."""
        idx = pd.date_range('2024-01-01', periods=50, freq='1h')
        pine = pd.Series(False, index=idx)
        python = pd.Series(False, index=idx)
        pine.iloc[10] = True
        python.iloc[11] = True  # 1 bar offset

        matcher = SignalMatcher(time_tolerance_bars=1)
        result = matcher.compare_signals(pine, python, 'test')
        assert result['summary']['matched_tolerant'] >= 1

    def test_no_signals(self, matcher):
        """No signals → 0 counts."""
        idx = pd.date_range('2024-01-01', periods=50, freq='1h')
        signal = pd.Series(False, index=idx)
        result = matcher.compare_signals(signal, signal, 'test')
        assert result['summary']['pine_signals'] == 0
        assert result['summary']['python_signals'] == 0


class TestMultiSignalComparison:
    def test_multi_signal(self, matcher, signal_pair):
        pine_sig, python_sig = signal_pair
        pine_dict = {'buy': pine_sig}
        python_dict = {'buy': python_sig}
        result = matcher.compare_multi_signals(pine_dict, python_dict)
        assert 'per_signal' in result
        assert 'overall' in result
        assert result['overall']['total_pine_signals'] > 0


class TestIndicatorComparison:
    def test_identical_indicators(self, matcher):
        """Identical indicator values should have 100% match."""
        idx = pd.date_range('2024-01-01', periods=100, freq='1h')
        values = pd.Series(np.random.randn(100) * 10 + 50, index=idx)
        result = matcher.compare_indicator_values(values, values, 'RSI')
        assert result['within_tolerance_pct'] == pytest.approx(100.0)
        assert result['mean_diff'] == pytest.approx(0.0, abs=1e-10)
        assert result['correlation'] == pytest.approx(1.0)

    def test_shifted_indicators(self, matcher):
        """Shifted indicators should show difference."""
        idx = pd.date_range('2024-01-01', periods=100, freq='1h')
        pine = pd.Series(np.random.randn(100) * 10 + 50, index=idx)
        python = pine + 1.0  # Constant offset
        result = matcher.compare_indicator_values(pine, python, 'ATR')
        assert result['mean_diff'] == pytest.approx(1.0, rel=1e-4)
        assert result['correlation'] == pytest.approx(1.0, abs=1e-6)
