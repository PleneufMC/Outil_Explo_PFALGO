"""
Backtest Engine — 3-phase execution matching Pine V.69.1.

Execution model (process_orders_on_close = false):
  Phase 1: Execute queued orders at Open[i]
  Phase 2: Check SL on High/Low[i] during bar
  Phase 3: Evaluate signals at Close[i] → queue for Open[i+1]

CRITICAL:
  - Entries execute at Open[i+1], NOT Close[i] (no look-ahead)
  - SL checks use High/Low of current bar (intra-bar)
  - Reversal = exit current + enter opposite
  - Transaction costs applied per instrument
"""

from pineguard.engine.backtest_engine import BacktestEngine
from pineguard.engine.transaction_costs import TRANSACTION_COSTS, get_cost

__all__ = ['BacktestEngine', 'TRANSACTION_COSTS', 'get_cost']
