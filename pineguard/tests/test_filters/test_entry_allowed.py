"""
Tests for entry_allowed gate — the 4-component filter (Bug B9 fix).

Verifies:
  - PMax direction gating
  - Position dedup (no double entries)
  - LT/MT filter integration
  - All components AND logic
  - Blocked-by statistics
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.filters.entry_allowed import compute_entry_allowed


@pytest.fixture
def base_data():
    n = 100
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    pmax_dir = pd.Series([1] * 50 + [-1] * 50, index=idx, dtype=float)
    return idx, pmax_dir


class TestEntryAllowed:
    def test_pmax_direction_filter(self, base_data):
        """Long only when PMax dir == 1, short only when dir == -1."""
        idx, pmax_dir = base_data
        result = compute_entry_allowed(pmax_dir)
        assert result['entry_allowed_long'].iloc[0] == True   # dir=1
        assert result['entry_allowed_long'].iloc[75] == False  # dir=-1
        assert result['entry_allowed_short'].iloc[0] == False  # dir=1
        assert result['entry_allowed_short'].iloc[75] == True  # dir=-1

    def test_position_dedup(self, base_data):
        """entry_allowed=False when already in position."""
        idx, pmax_dir = base_data
        in_long = pd.Series([True] * 100, index=idx)
        result = compute_entry_allowed(pmax_dir, in_position_long=in_long)
        # All long entries blocked since already in long
        assert not result['entry_allowed_long'].any()

    def test_lt_filter_blocks(self, base_data):
        """LT filter = False should block entries."""
        idx, pmax_dir = base_data
        lt_long = pd.Series(False, index=idx)
        lt_short = pd.Series(False, index=idx)
        result = compute_entry_allowed(
            pmax_dir,
            lt_filter_long=lt_long, lt_filter_short=lt_short,
            lt_enabled=True
        )
        assert not result['entry_allowed_long'].any()
        assert not result['entry_allowed_short'].any()

    def test_lt_disabled_passes_through(self, base_data):
        """LT filter disabled should not block."""
        idx, pmax_dir = base_data
        lt_long = pd.Series(False, index=idx)
        result = compute_entry_allowed(
            pmax_dir,
            lt_filter_long=lt_long,
            lt_enabled=False
        )
        # LT is disabled, so it shouldn't block
        assert result['entry_allowed_long'].iloc[0] == True

    def test_all_components_and(self, base_data):
        """All components must be True (AND logic)."""
        idx, pmax_dir = base_data
        # PMax allows long for first 50 bars
        # But MT filter blocks everything
        mt_long = pd.Series(False, index=idx)
        result = compute_entry_allowed(
            pmax_dir,
            mt_filter_long=mt_long,
            mt_enabled=True
        )
        assert not result['entry_allowed_long'].any()

    def test_blocked_by_statistics(self, base_data):
        """blocked_by dict should have correct structure."""
        idx, pmax_dir = base_data
        result = compute_entry_allowed(pmax_dir)
        blocked = result['blocked_by']
        assert 'pmax' in blocked
        assert 'position' in blocked
        assert 'lt_filter' in blocked
        assert 'mt_filter' in blocked
        assert 'total_bars' in blocked
        assert blocked['total_bars'] == 100

    def test_rsi_filter(self, base_data):
        """RSI filter when enabled should block."""
        idx, pmax_dir = base_data
        rsi_long = pd.Series(False, index=idx)
        result = compute_entry_allowed(
            pmax_dir,
            rsi_filter_long=rsi_long,
            rsi_enabled=True
        )
        assert not result['entry_allowed_long'].any()

    def test_all_pass(self, base_data):
        """Default: all filters pass → entries allowed based on PMax dir."""
        idx, pmax_dir = base_data
        result = compute_entry_allowed(pmax_dir)
        # First 50 bars: dir=1 → long allowed
        assert result['entry_allowed_long'].iloc[:50].all()
        # Last 50 bars: dir=-1 → short allowed
        assert result['entry_allowed_short'].iloc[50:].all()
