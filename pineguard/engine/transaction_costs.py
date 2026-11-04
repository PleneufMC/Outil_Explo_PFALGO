"""
Transaction Costs — 13 instruments with realistic cost models.

Costs are expressed in points (or pips) per round trip.
Applied at entry and exit to simulate real trading conditions.
"""

from typing import Dict, Optional


# Transaction costs per instrument (round-trip, in points)
TRANSACTION_COSTS: Dict[str, Dict] = {
    # Indices
    'US500': {'spread': 0.5, 'commission': 0.0, 'slippage': 0.25, 'total': 0.75},
    'US100': {'spread': 1.0, 'commission': 0.0, 'slippage': 0.5, 'total': 1.5},
    'US30': {'spread': 2.0, 'commission': 0.0, 'slippage': 1.0, 'total': 3.0},
    'UK100': {'spread': 1.0, 'commission': 0.0, 'slippage': 0.5, 'total': 1.5},
    'DE40': {'spread': 1.0, 'commission': 0.0, 'slippage': 0.5, 'total': 1.5},
    'FR40': {'spread': 1.0, 'commission': 0.0, 'slippage': 0.5, 'total': 1.5},
    'JP225': {'spread': 8.0, 'commission': 0.0, 'slippage': 4.0, 'total': 12.0},

    # Forex majors
    'EURUSD': {'spread': 0.00010, 'commission': 0.0, 'slippage': 0.00005, 'total': 0.00015},
    'GBPUSD': {'spread': 0.00012, 'commission': 0.0, 'slippage': 0.00006, 'total': 0.00018},
    'USDJPY': {'spread': 0.010, 'commission': 0.0, 'slippage': 0.005, 'total': 0.015},

    # Commodities
    'XAUUSD': {'spread': 0.30, 'commission': 0.0, 'slippage': 0.15, 'total': 0.45},
    'XAGUSD': {'spread': 0.020, 'commission': 0.0, 'slippage': 0.010, 'total': 0.030},

    # Crypto
    'BTCUSD': {'spread': 15.0, 'commission': 0.0, 'slippage': 10.0, 'total': 25.0},
}

# Default cost if instrument not found
DEFAULT_COST = {'spread': 1.0, 'commission': 0.0, 'slippage': 0.5, 'total': 1.5}


def get_cost(instrument: str, component: str = 'total') -> float:
    """
    Get transaction cost for an instrument.

    Args:
        instrument: Instrument symbol (e.g., 'US500', 'XAUUSD')
        component: Cost component ('spread', 'commission', 'slippage', 'total')

    Returns:
        Cost value in instrument points
    """
    costs = TRANSACTION_COSTS.get(instrument.upper(), DEFAULT_COST)
    return costs.get(component, costs['total'])


def estimate_cost_impact(instrument: str, avg_trade_pnl: float) -> Dict:
    """
    Estimate the impact of transaction costs on trading results.

    Args:
        instrument: Instrument symbol
        avg_trade_pnl: Average trade PnL in points

    Returns:
        Dict with cost analysis
    """
    total_cost = get_cost(instrument)
    cost_pct = (total_cost / abs(avg_trade_pnl) * 100) if avg_trade_pnl != 0 else float('inf')

    return {
        'instrument': instrument,
        'total_cost_per_trade': total_cost,
        'avg_trade_pnl': avg_trade_pnl,
        'cost_as_pct_of_pnl': round(cost_pct, 2),
        'severity': 'HIGH' if cost_pct > 30 else ('MEDIUM' if cost_pct > 15 else 'LOW'),
    }
