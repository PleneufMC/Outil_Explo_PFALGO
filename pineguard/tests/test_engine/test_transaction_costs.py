"""
Tests for transaction costs module.

Verifies:
  - All 13 instruments have costs defined
  - Cost components (spread, commission, slippage, total)
  - Default cost for unknown instruments
  - Cost impact estimation
"""

import pytest
from pineguard.engine.transaction_costs import (
    get_cost, estimate_cost_impact, TRANSACTION_COSTS, DEFAULT_COST,
)


class TestTransactionCosts:
    def test_13_instruments(self):
        """All 13 instruments should be defined."""
        assert len(TRANSACTION_COSTS) == 13

    def test_known_instruments(self):
        """Check key instruments have non-zero costs."""
        for inst in ['US500', 'US100', 'EURUSD', 'XAUUSD', 'BTCUSD']:
            cost = get_cost(inst)
            assert cost > 0, f"{inst} has zero cost"

    def test_us500_cost(self):
        """US500 total cost should be 0.75."""
        assert get_cost('US500') == pytest.approx(0.75)

    def test_eurusd_cost(self):
        """EURUSD total cost should be 0.00015."""
        assert get_cost('EURUSD') == pytest.approx(0.00015)

    def test_unknown_instrument_default(self):
        """Unknown instrument should return default cost."""
        cost = get_cost('UNKNOWN_SYMBOL')
        assert cost == DEFAULT_COST['total']

    def test_component_access(self):
        """Should access individual cost components."""
        spread = get_cost('US500', 'spread')
        assert spread == pytest.approx(0.5)
        slippage = get_cost('US500', 'slippage')
        assert slippage == pytest.approx(0.25)

    def test_total_equals_components(self):
        """Total should equal spread + commission + slippage."""
        for inst in TRANSACTION_COSTS:
            costs = TRANSACTION_COSTS[inst]
            expected = costs['spread'] + costs['commission'] + costs['slippage']
            assert costs['total'] == pytest.approx(expected), \
                f"{inst} total mismatch"


class TestCostImpact:
    def test_impact_estimation(self):
        result = estimate_cost_impact('US500', avg_trade_pnl=10.0)
        assert result['instrument'] == 'US500'
        assert result['total_cost_per_trade'] == pytest.approx(0.75)
        assert result['cost_as_pct_of_pnl'] == pytest.approx(7.5)
        assert result['severity'] == 'LOW'

    def test_high_impact(self):
        """Small PnL relative to cost → HIGH severity."""
        result = estimate_cost_impact('US500', avg_trade_pnl=1.0)
        assert result['severity'] == 'HIGH'

    def test_zero_pnl(self):
        """Zero PnL → infinite cost impact."""
        result = estimate_cost_impact('US500', avg_trade_pnl=0.0)
        assert result['cost_as_pct_of_pnl'] == float('inf')
