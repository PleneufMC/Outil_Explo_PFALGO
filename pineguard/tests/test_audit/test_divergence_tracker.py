"""
Tests for Divergence Tracker — verify all known divergences are loaded.
"""

import pytest
from pineguard.audit.divergence_tracker import (
    DivergenceTracker, Divergence, Severity, Category, Status
)


@pytest.fixture
def tracker():
    return DivergenceTracker()


class TestDivergenceTracker:
    def test_all_13_divergences_loaded(self, tracker):
        """All D1-D13 should be pre-loaded."""
        for i in range(1, 14):
            assert tracker.get(f'D{i}') is not None, f"D{i} not loaded"

    def test_severity_counts(self, tracker):
        summary = tracker.summary()
        assert summary['by_severity']['BLOQUANT'] >= 1  # D13
        assert summary['by_severity']['HAUTE'] >= 3     # D7, D8, D9
        assert summary['total'] == 13

    def test_bloquant_is_orb(self, tracker):
        """D13 (ORB signal inversé) should be BLOQUANT."""
        d13 = tracker.get('D13')
        assert d13.severity == Severity.BLOQUANT

    def test_structural_divergences(self, tracker):
        """D1-D3 should be data category."""
        for did in ['D1', 'D2', 'D3']:
            d = tracker.get(did)
            assert d.category == Category.DATA

    def test_implementation_divergences(self, tracker):
        """D7-D13 should be implementation category."""
        impl_ids = ['D7', 'D8', 'D9', 'D10', 'D11', 'D13']
        for did in impl_ids:
            d = tracker.get(did)
            assert d.category == Category.IMPLEMENTATION

    def test_list_open(self, tracker):
        """Should have open divergences (D4, D5, D6, D12, D13 remain open)."""
        open_divs = tracker.list_open()
        assert len(open_divs) >= 5  # D4, D5, D6, D12, D13 still open
        open_ids = {d.id for d in open_divs}
        assert 'D13' in open_ids  # ORB still blocking
        assert 'D4' in open_ids   # Weekly boundaries still open

    def test_format_report(self, tracker):
        """Report should be a non-empty string."""
        report = tracker.format_report()
        assert len(report) > 100
        assert 'PINEGUARD' in report

    def test_add_custom_divergence(self, tracker):
        """Should be able to add new divergences."""
        custom = Divergence(
            id='D14', title='Custom divergence',
            category=Category.IMPLEMENTATION,
            severity=Severity.BASSE,
            status=Status.OPEN,
            description='Test divergence',
            pine_behavior='Test',
            python_behavior='Test',
        )
        tracker.add(custom)
        assert tracker.get('D14') is not None
        assert tracker.summary()['total'] == 14

    def test_phantom_trade_estimate(self, tracker):
        """Should estimate phantom trades from open divergences."""
        summary = tracker.summary()
        assert summary['estimated_total_phantom_trades'] > 0
