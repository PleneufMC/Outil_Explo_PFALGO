"""
Tests for crossover/crossunder helpers — Pine-exact equivalence.

Verifies:
  - crossover: a > b AND a[1] <= b[1] (note: <=, not <)
  - crossunder: a < b AND a[1] >= b[1] (note: >=, not >)
"""

import pytest
import numpy as np
import pandas as pd
from pineguard.utils.crossovers import crossover, crossunder


@pytest.fixture
def basic_pair():
    idx = pd.date_range('2024-01-01', periods=10, freq='1h')
    a = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1, 2], index=idx, dtype=float)
    b = pd.Series([3, 3, 3, 3, 3, 3, 3, 3, 3, 3], index=idx, dtype=float)
    return a, b


class TestCrossover:
    def test_basic_crossover(self, basic_pair):
        """a crosses above b at bar 3 (a goes from 3 to 4, b=3)."""
        a, b = basic_pair
        result = crossover(a, b)
        # Bar 3: a=4 > b=3 AND a[2]=3 <= b[2]=3 → True
        assert result.iloc[3] == True
        # Bar 4: a=5 > b=3 but a[3]=4 > b[3]=3 (not <=) → False
        assert result.iloc[4] == False

    def test_equal_on_previous_bar(self):
        """When a[1] == b[1], crossover should still detect (uses <=)."""
        idx = pd.date_range('2024-01-01', periods=3, freq='1h')
        a = pd.Series([3, 3, 4], index=idx, dtype=float)
        b = pd.Series([3, 3, 3], index=idx, dtype=float)
        result = crossover(a, b)
        # Bar 2: a=4 > b=3 AND a[1]=3 <= b[1]=3 → True (Pine uses <=)
        assert result.iloc[2] == True

    def test_no_crossover_when_already_above(self):
        """If a was already above b, no crossover."""
        idx = pd.date_range('2024-01-01', periods=3, freq='1h')
        a = pd.Series([4, 5, 6], index=idx, dtype=float)
        b = pd.Series([3, 3, 3], index=idx, dtype=float)
        result = crossover(a, b)
        assert not result.iloc[1]
        assert not result.iloc[2]


class TestCrossunder:
    def test_basic_crossunder(self, basic_pair):
        """a crosses below b at bar 6 (a goes from 4 to 3, b=3)."""
        a, b = basic_pair
        result = crossunder(a, b)
        # Bar 6: a=3 < b=3? NO (equal) → False
        # Bar 7: a=2 < b=3 AND a[6]=3 >= b[6]=3 → True
        assert result.iloc[7] == True

    def test_equal_on_previous_bar_crossunder(self):
        """When a[1] == b[1], crossunder uses >= so it should detect."""
        idx = pd.date_range('2024-01-01', periods=3, freq='1h')
        a = pd.Series([3, 3, 2], index=idx, dtype=float)
        b = pd.Series([3, 3, 3], index=idx, dtype=float)
        result = crossunder(a, b)
        # Bar 2: a=2 < b=3 AND a[1]=3 >= b[1]=3 → True
        assert result.iloc[2] == True
