"""
Tests for Trade Comparator — Pine vs Python trade matching.

Verifies:
  - Exact trade matching
  - Time tolerance
  - PnL tolerance
  - Direction matching
  - Phantom and missed trade classification
  - Severity classification
"""

import pytest
import pandas as pd
from pineguard.audit.trade_comparator import TradeComparator, TradeRecord


@pytest.fixture
def comparator():
    return TradeComparator(time_tolerance_hours=4.0, pnl_tolerance_pct=0.5)


@pytest.fixture
def matching_trades():
    """Create matching Pine and Python trade lists."""
    pine = [
        TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='pine',
        ),
        TradeRecord(
            entry_time=pd.Timestamp('2024-01-02 10:00'),
            exit_time=pd.Timestamp('2024-01-02 14:00'),
            direction='short', entry_price=5050, exit_price=5000, pnl=50,
            source='pine',
        ),
    ]
    python = [
        TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='python',
        ),
        TradeRecord(
            entry_time=pd.Timestamp('2024-01-02 10:00'),
            exit_time=pd.Timestamp('2024-01-02 14:00'),
            direction='short', entry_price=5050, exit_price=5000, pnl=50,
            source='python',
        ),
    ]
    return pine, python


class TestTradeComparator:
    def test_perfect_match(self, comparator, matching_trades):
        pine, python = matching_trades
        result = comparator.compare(pine, python)
        assert result['summary']['matched'] == 2
        assert result['summary']['python_only_phantom'] == 0
        assert result['summary']['pine_only_missed'] == 0
        assert result['summary']['match_rate_pct'] == pytest.approx(100.0)

    def test_phantom_trade(self, comparator, matching_trades):
        """Extra Python trade should be classified as phantom."""
        pine, python = matching_trades
        python.append(TradeRecord(
            entry_time=pd.Timestamp('2024-01-03 10:00'),
            exit_time=pd.Timestamp('2024-01-03 14:00'),
            direction='long', entry_price=5100, exit_price=5080, pnl=-20,
            source='python',
        ))
        result = comparator.compare(pine, python)
        assert result['summary']['python_only_phantom'] == 1

    def test_missed_trade(self, comparator, matching_trades):
        """Missing Python trade should be classified as Pine-only."""
        pine, python = matching_trades
        python.pop()  # Remove last trade
        result = comparator.compare(pine, python)
        assert result['summary']['pine_only_missed'] == 1

    def test_direction_mismatch(self, comparator):
        """Different direction should not match."""
        pine = [TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='pine',
        )]
        python = [TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='short', entry_price=5000, exit_price=4950, pnl=50,
            source='python',
        )]
        result = comparator.compare(pine, python)
        assert result['summary']['matched'] == 0

    def test_time_tolerance(self):
        """Trades within tolerance should match."""
        pine = [TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='pine',
        )]
        python = [TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 12:00'),  # 2h offset
            exit_time=pd.Timestamp('2024-01-01 17:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='python',
        )]
        comp = TradeComparator(time_tolerance_hours=4.0)
        result = comp.compare(pine, python)
        assert result['summary']['matched'] == 1

    def test_severity_classification(self, comparator):
        """Many extra Python trades → HIGH or BLOQUANT severity."""
        pine = [TradeRecord(
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            exit_time=pd.Timestamp('2024-01-01 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='pine',
        )]
        python = [TradeRecord(
            entry_time=pd.Timestamp(f'2024-01-0{i} 10:00'),
            exit_time=pd.Timestamp(f'2024-01-0{i} 15:00'),
            direction='long', entry_price=5000, exit_price=5050, pnl=50,
            source='python',
        ) for i in range(1, 6)]  # 5 Python trades vs 1 Pine
        result = comparator.compare(pine, python)
        assert result['summary']['severity'] in ('HAUTE', 'BLOQUANT')

    def test_format_report(self, comparator, matching_trades):
        pine, python = matching_trades
        result = comparator.compare(pine, python)
        report = comparator.format_report(result)
        assert 'TRADE COMPARISON REPORT' in report
        assert 'Pine trades:' in report

    def test_empty_lists(self, comparator):
        """Empty lists should not crash."""
        result = comparator.compare([], [])
        assert result['summary']['matched'] == 0
        assert result['summary']['pine_trades'] == 0
