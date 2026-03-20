"""
Backtest Engine — 3-phase bar-by-bar execution matching Pine V.69.1.

Pine execution model (process_orders_on_close = false):
  Phase 1 (Open[i]):   Execute queued orders from previous bar
  Phase 2 (H/L[i]):    Check stop-loss against High/Low
  Phase 3 (Close[i]):  Evaluate signals → queue for Open[i+1]

CRITICAL RULES:
  - NO execution on Close[i] — signal detected at Close, filled at Open[i+1]
  - SL is checked against High (for shorts) and Low (for longs) of current bar
  - Reversal = close current position + open opposite direction
  - Dynamic SL: ATR14 or ATR50 based (configurable)
  - entry_allowed gate must pass (4 components) — Fix for bug B9
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

Series = pd.Series
DataFrame = pd.DataFrame


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Represents a single completed trade."""
    entry_bar: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: Direction
    exit_bar: int = -1
    exit_time: Optional[pd.Timestamp] = None
    exit_price: float = 0.0
    exit_reason: str = ''  # 'sl', 'reversal', 'signal'
    pnl_points: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    sl_price: float = 0.0
    transaction_cost: float = 0.0


@dataclass
class Position:
    """Current open position state."""
    direction: Direction = Direction.FLAT
    entry_price: float = 0.0
    entry_bar: int = -1
    entry_time: Optional[pd.Timestamp] = None
    sl_price: float = 0.0
    size: float = 1.0


@dataclass
class PendingOrder:
    """Order queued for execution at next Open."""
    direction: Direction
    signal_bar: int
    signal_time: pd.Timestamp
    sl_distance: float = 0.0  # ATR-based SL distance in points


class BacktestEngine:
    """
    3-phase backtest engine matching Pine V.69.1 execution model.

    Usage:
        engine = BacktestEngine(instrument='US500', sl_mode='atr14')
        result = engine.run(df, signals)
    """

    def __init__(
        self,
        instrument: str = 'US500',
        sl_mode: str = 'atr14',
        sl_pct: Optional[float] = None,
        initial_capital: float = 100000.0,
        position_size_pct: float = 1.0,  # % of capital per trade
        enable_reversal: bool = True,
    ):
        """
        Args:
            instrument: Trading instrument for cost lookup
            sl_mode: Stop-loss mode: 'atr14', 'atr50', 'fixed_pct', 'none'
            sl_pct: Fixed SL percentage (used when sl_mode='fixed_pct')
            initial_capital: Starting capital
            position_size_pct: Position size as % of capital
            enable_reversal: Allow reversal exits (close + open opposite)
        """
        from pineguard.engine.transaction_costs import get_cost

        self.instrument = instrument
        self.sl_mode = sl_mode
        self.sl_pct = sl_pct
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.enable_reversal = enable_reversal
        self.cost_per_trade = get_cost(instrument)

    def run(
        self,
        df: DataFrame,
        entry_long: Series,
        entry_short: Series,
        entry_allowed_long: Optional[Series] = None,
        entry_allowed_short: Optional[Series] = None,
        atr: Optional[Series] = None,
    ) -> Dict:
        """
        Run backtest with 3-phase execution.

        Args:
            df: OHLCV DataFrame
            entry_long: Boolean series — long entry signals
            entry_short: Boolean series — short entry signals
            entry_allowed_long: Boolean gate for long entries (bug B9 fix)
            entry_allowed_short: Boolean gate for short entries (bug B9 fix)
            atr: ATR series for dynamic SL calculation

        Returns:
            Dict with:
                'trades': List of Trade objects
                'equity': Equity curve Series
                'positions': Position history
                'stats': Summary statistics
        """
        n = len(df)
        if n == 0:
            return self._empty_result(df)

        # Default: allow all entries if gate not provided
        if entry_allowed_long is None:
            entry_allowed_long = pd.Series(True, index=df.index)
        if entry_allowed_short is None:
            entry_allowed_short = pd.Series(True, index=df.index)

        # Initialize state
        position = Position()
        pending: Optional[PendingOrder] = None
        trades: List[Trade] = []
        equity = np.full(n, self.initial_capital)
        unrealized_pnl = 0.0

        open_ = df['Open'].values
        high = df['High'].values
        low = df['Low'].values
        close = df['Close'].values
        idx = df.index

        atr_vals = atr.values if atr is not None else np.full(n, np.nan)
        realized_pnl_total = 0.0  # Running total — O(1) incremental

        for i in range(n):
            # ═══ PHASE 1: Execute pending order at Open[i] ═══
            if pending is not None and i > 0:
                fill_price = open_[i]

                # Close existing position if reversal
                # FIX: accumulate ALL realized PnL from trades closed on this bar
                if position.direction != Direction.FLAT:
                    trade = self._close_position(
                        position, i, idx[i], fill_price, 'reversal'
                    )
                    trades.append(trade)
                    realized_pnl_total += trade.pnl_points

                # Open new position
                sl_price = self._calculate_sl(
                    fill_price, pending.direction, pending.sl_distance
                )
                position = Position(
                    direction=pending.direction,
                    entry_price=fill_price,
                    entry_bar=i,
                    entry_time=idx[i],
                    sl_price=sl_price,
                )
                pending = None

            # ═══ PHASE 2: Check stop-loss on High/Low[i] ═══
            # NOTE: SL can hit a position that was just opened in Phase 1
            # on the same bar. This correctly handles the case where a
            # reversal closes trade A and opens trade B, then B gets
            # stopped out — both PnLs are counted for this bar's equity.
            if position.direction != Direction.FLAT and position.sl_price != 0:
                sl_hit = False
                exit_price = 0.0

                if position.direction == Direction.LONG:
                    # Long SL: low touches or crosses below SL
                    if low[i] <= position.sl_price:
                        sl_hit = True
                        exit_price = position.sl_price
                elif position.direction == Direction.SHORT:
                    # Short SL: high touches or crosses above SL
                    if high[i] >= position.sl_price:
                        sl_hit = True
                        exit_price = position.sl_price

                if sl_hit:
                    trade = self._close_position(
                        position, i, idx[i], exit_price, 'sl'
                    )
                    trades.append(trade)
                    realized_pnl_total += trade.pnl_points
                    position = Position()  # Reset to flat

            # ═══ PHASE 3: Evaluate signals → queue for Open[i+1] ═══
            if i < n - 1:  # Can't queue on last bar
                # Long signal
                if (entry_long.iloc[i] and entry_allowed_long.iloc[i] and
                        position.direction != Direction.LONG):

                    sl_dist = self._get_sl_distance(atr_vals[i], close[i])
                    pending = PendingOrder(
                        direction=Direction.LONG,
                        signal_bar=i,
                        signal_time=idx[i],
                        sl_distance=sl_dist,
                    )

                # Short signal (can override long if both fire)
                elif (entry_short.iloc[i] and entry_allowed_short.iloc[i] and
                      position.direction != Direction.SHORT):

                    sl_dist = self._get_sl_distance(atr_vals[i], close[i])
                    pending = PendingOrder(
                        direction=Direction.SHORT,
                        signal_bar=i,
                        signal_time=idx[i],
                        sl_distance=sl_dist,
                    )

            # ═══ UPDATE EQUITY ═══
            if position.direction == Direction.LONG:
                unrealized_pnl = close[i] - position.entry_price
            elif position.direction == Direction.SHORT:
                unrealized_pnl = position.entry_price - close[i]
            else:
                unrealized_pnl = 0.0

            equity[i] = self.initial_capital + realized_pnl_total + unrealized_pnl

        # Close any remaining position at last close
        if position.direction != Direction.FLAT:
            trade = self._close_position(
                position, n - 1, idx[-1], close[-1], 'end_of_data'
            )
            trades.append(trade)

        equity_series = pd.Series(equity, index=idx, name='equity')

        return {
            'trades': trades,
            'equity': equity_series,
            'trade_count': len(trades),
            'instrument': self.instrument,
        }

    def _close_position(self, position: Position, bar: int,
                        time: pd.Timestamp, price: float,
                        reason: str) -> Trade:
        """Close a position and create a Trade record."""
        if position.direction == Direction.LONG:
            pnl = price - position.entry_price - self.cost_per_trade
        else:
            pnl = position.entry_price - price - self.cost_per_trade

        pnl_pct = (pnl / position.entry_price * 100) if position.entry_price != 0 else 0

        return Trade(
            entry_bar=position.entry_bar,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            direction=position.direction,
            exit_bar=bar,
            exit_time=time,
            exit_price=price,
            exit_reason=reason,
            pnl_points=pnl,
            pnl_pct=pnl_pct,
            bars_held=bar - position.entry_bar,
            sl_price=position.sl_price,
            transaction_cost=self.cost_per_trade,
        )

    def _get_sl_distance(self, atr_val: float, price: float) -> float:
        """Calculate SL distance based on mode."""
        if self.sl_mode == 'atr14' and not np.isnan(atr_val):
            return atr_val * 1.5  # 1.5× ATR14
        elif self.sl_mode == 'atr50' and not np.isnan(atr_val):
            return atr_val * 2.0  # 2× ATR50
        elif self.sl_mode == 'fixed_pct' and self.sl_pct:
            return price * self.sl_pct / 100.0
        return 0.0  # No SL

    def _calculate_sl(self, entry_price: float, direction: Direction,
                      sl_distance: float) -> float:
        """Calculate SL price from entry and distance."""
        if sl_distance == 0:
            return 0.0
        if direction == Direction.LONG:
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def _empty_result(self, df: DataFrame) -> Dict:
        """Return empty result for empty DataFrames."""
        return {
            'trades': [],
            'equity': pd.Series(dtype=float, name='equity'),
            'trade_count': 0,
            'instrument': self.instrument,
        }
