"""
PMax (Profit Maximizer) — Core algorithm, Pine-equivalent implementation.

PMax is the HEART of P.F.Algo V.69.1. It combines a Moving Average with
ATR-based trailing stops to determine trend direction.

Pine logic:
  MAvg = getMA(hl2, length, mav_type)
  ATR = ta.atr(length)                    // RMA, not SMA
  nLoss = Multiplier × ATR
  longStop = MAvg - nLoss
  shortStop = MAvg + nLoss
  // Stops are STATEFUL: carry over from previous bar
  dir = 1 if MAvg > shortStopPrev, -1 if MAvg < longStopPrev
  PMax = dir == 1 ? longStop : shortStop

CRITICAL: The stops use carry-over logic:
  longStop[i] = max(MAvg - nLoss, longStop[i-1])  if dir[i-1] == 1
  shortStop[i] = min(MAvg + nLoss, shortStop[i-1]) if dir[i-1] == -1
  → This is STATEFUL and requires a loop. Not vectorizable naively.

Signals:
  buy_trigger = ta.crossover(MAvg, PMax)   → MAvg crosses above PMax
  sell_trigger = ta.crossunder(MAvg, PMax)  → MAvg crosses below PMax
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

Series = pd.Series


def compute_pmax(high: Series, low: Series, close: Series,
                 length: int = 10, multiplier: float = 3.0,
                 ma_type: str = 'EMA', volume: Series = None
                 ) -> Dict[str, Series]:
    """
    Compute PMax indicator matching Pine Script V.69.1 behavior.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: Period for both MA and ATR (default 10)
        multiplier: ATR multiplier for stop distance (default 3.0)
        ma_type: Moving average type (default 'EMA'), one of 9 types
        volume: Volume series (required if ma_type='VWMA')

    Returns:
        Dict with keys:
            'MAvg': Moving average series
            'ATR': ATR series
            'PMax': PMax trailing stop series
            'dir': Direction series (1=long, -1=short)
            'longStop': Long stop series
            'shortStop': Short stop series
            'buy_trigger': Boolean series (crossover MAvg > PMax)
            'sell_trigger': Boolean series (crossunder MAvg < PMax)

    Implementation notes:
        - Source = hl2 (calculated internally from high/low)
        - ATR uses RMA (Wilder's), not SMA
        - Stops are stateful with carry-over logic (requires loop)
        - Signals use strict crossover: a > b AND a[1] <= b[1]
    """
    from pineguard.indicators.moving_averages import get_ma
    from pineguard.indicators.atr import compute_atr

    # Source = hl2 (standard for PMax)
    hl2 = (high + low) / 2

    # Moving Average on hl2
    mavg = get_ma(hl2, length, ma_type, volume)

    # ATR using RMA (Wilder's)
    atr = compute_atr(high, low, close, length)

    # nLoss
    nloss = multiplier * atr

    # Raw stops (before stateful carry-over)
    raw_long_stop = mavg - nloss
    raw_short_stop = mavg + nloss

    # Stateful stop computation (MUST use loop)
    n = len(mavg)
    long_stop = np.full(n, np.nan)
    short_stop = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    # Initialize
    first_valid = mavg.first_valid_index()
    if first_valid is None:
        return _empty_result(mavg, atr)

    first_idx = mavg.index.get_loc(first_valid)

    long_stop[first_idx] = raw_long_stop.iloc[first_idx]
    short_stop[first_idx] = raw_short_stop.iloc[first_idx]
    direction[first_idx] = 1  # Start bullish

    for i in range(first_idx + 1, n):
        raw_ls = raw_long_stop.iloc[i]
        raw_ss = raw_short_stop.iloc[i]
        prev_dir = direction[i - 1]
        prev_ls = long_stop[i - 1]
        prev_ss = short_stop[i - 1]
        curr_mavg = mavg.iloc[i]

        if np.isnan(raw_ls) or np.isnan(raw_ss):
            long_stop[i] = np.nan
            short_stop[i] = np.nan
            direction[i] = prev_dir
            continue

        # Carry-over logic for long stop
        if prev_dir == 1:
            long_stop[i] = max(raw_ls, prev_ls) if not np.isnan(prev_ls) else raw_ls
        else:
            long_stop[i] = raw_ls

        # Carry-over logic for short stop
        if prev_dir == -1:
            short_stop[i] = min(raw_ss, prev_ss) if not np.isnan(prev_ss) else raw_ss
        else:
            short_stop[i] = raw_ss

        # Direction logic — CRITICAL: Compare MAvg against PREVIOUS bar's stops,
        # matching Pine's: dir := MAvg > shortStopPrev ? 1 : MAvg < longStopPrev ? -1 : dir[1]
        # We use prev_ss and prev_ls (the previous bar's stop values), NOT the
        # just-computed short_stop[i] and long_stop[i].
        if curr_mavg > prev_ss:
            direction[i] = 1
        elif curr_mavg < prev_ls:
            direction[i] = -1
        else:
            direction[i] = prev_dir

    # Convert to Series
    idx = mavg.index
    long_stop_s = pd.Series(long_stop, index=idx, name='longStop')
    short_stop_s = pd.Series(short_stop, index=idx, name='shortStop')
    dir_s = pd.Series(direction, index=idx, name='dir')

    # PMax = longStop when dir=1, shortStop when dir=-1
    pmax = np.where(dir_s == 1, long_stop_s, short_stop_s)
    pmax_s = pd.Series(pmax, index=idx, name='PMax')

    # Signals: strict crossover/crossunder
    # buy_trigger = MAvg > PMax AND MAvg[1] <= PMax[1]
    buy_trigger = (mavg > pmax_s) & (mavg.shift(1) <= pmax_s.shift(1))
    buy_trigger.name = 'buy_trigger'

    # sell_trigger = MAvg < PMax AND MAvg[1] >= PMax[1]
    sell_trigger = (mavg < pmax_s) & (mavg.shift(1) >= pmax_s.shift(1))
    sell_trigger.name = 'sell_trigger'

    return {
        'MAvg': mavg,
        'ATR': atr,
        'PMax': pmax_s,
        'dir': dir_s,
        'longStop': long_stop_s,
        'shortStop': short_stop_s,
        'buy_trigger': buy_trigger,
        'sell_trigger': sell_trigger,
    }


def _empty_result(mavg: Series, atr: Series) -> Dict[str, Series]:
    """Return empty result dict when no valid data."""
    idx = mavg.index
    nan_s = pd.Series(np.nan, index=idx)
    false_s = pd.Series(False, index=idx)
    return {
        'MAvg': mavg, 'ATR': atr,
        'PMax': nan_s.rename('PMax'),
        'dir': nan_s.rename('dir'),
        'longStop': nan_s.rename('longStop'),
        'shortStop': nan_s.rename('shortStop'),
        'buy_trigger': false_s.rename('buy_trigger'),
        'sell_trigger': false_s.rename('sell_trigger'),
    }
