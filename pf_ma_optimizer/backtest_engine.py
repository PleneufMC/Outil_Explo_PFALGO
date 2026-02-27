"""
PF AI Lab 5.3 — Backtest Engine
3-phase execution model matching Pine Script (process_orders_on_close=false):

For each bar i:
  Phase 1: Execute pending orders at Open[i]
  Phase 2: Check SL/TP/BE/TSL on High/Low of bar i
  Phase 3: Evaluate signals -> queue for Open[i+1]

EXECUTION PRICES:
  Entry      = Open[i+1]   (NOT Close[i])
  Exit Setup = Open[i+1]   (Setup Reversal)
  Exit SL    = SL price    (or Open[i] if gap exceeds SL)
  Exit TP    = TP price    (or Open[i] if gap exceeds TP)
  Exit end   = Close[-1]   (last bar)

V5.2 ADDITIONS:
  - SL ATR-Based + Dynamic SL (ATR14/ATR50 vol ratio)
  - TP modes: No, Static %, SL Ratio
  - BE (Break-Even): move SL to entry after X% profit
  - TSL (Trailing Stop Loss): follow price at distance
  - Max Hold Days: forced exit after N calendar days
  - ADX Regime Filter: gate entries by trending/ranging
  - Volatility ATR Percentile Filter: block high-vol entries

V5.3 ADDITIONS:
  - Partial Profit Taking: close TP1 size at TP1%, TP2 size at TP2%
  - Session/Timing Filter: day-of-week + session hours
  - VIX Filter (Security Layer 0): external volatility gate
  - Macro Filter Simplifie: reduced RS + Climate score
  - QQ Estimation Filter: exact Pine V69.2 port (fixed)
"""

import numpy as np
import pandas as pd
import warnings
from .indicators.pmax import compute_pmax
from .indicators.atr import compute_atr, compute_atr_14_50, compute_atr_percentile
from .indicators.adx import compute_adx
from .entries.donchian_chikou import compute_donchian_chikou
from .entries.rsi_divergence import compute_rsi_divergence
from .entries.fractal import compute_fractal_entry
from .entries.boll_sma import compute_boll_sma_entry
from .entries.mm_cross import compute_mm_cross_entry
from .filters.all_filters import compute_all_filters
from .config import (TRANSACTION_COSTS, DEFAULT_TRANSACTION_COST,
                     ADX_REGIME_COMPATIBILITY, get_transaction_cost)


def _safe_pnl_pct(price_diff, entry_price, tc):
    """Compute pnl_pct safely, avoiding division by zero."""
    if entry_price == 0 or np.isnan(entry_price):
        return 0.0
    return price_diff / entry_price * 100 - tc / entry_price * 100


def run_backtest(df: pd.DataFrame, config: dict,
                 instrument: str = 'UK100',
                 initial_capital: float = 100000.0) -> list:
    """
    Run a full backtest on OHLCV data with the given configuration.

    Parameters:
        df: DataFrame with ['Open', 'High', 'Low', 'Close', 'Volume', 'hl2']
        config: dict with all strategy parameters
        instrument: for transaction costs lookup
        initial_capital: starting capital

    Returns:
        list of trade dicts:
            {
                'entry_bar': int,
                'exit_bar': int,
                'side': 'long' | 'short',
                'entry_price': float,
                'exit_price': float,
                'pnl': float,
                'pnl_pct': float,
                'exit_reason': str,
                'duration_bars': int,
                'entry_date': str,
                'exit_date': str,
            }

    NOTE (v5.0.1): The returned list also stores a reference to the original
    DataFrame via trades.__df_ref__ so that metrics.py can compute bar-by-bar
    equity and daily Sharpe/Sortino without requiring callers to pass df.
    """
    n = len(df)
    if n < 50:
        return _TradeList([])

    # Validate data — prevent NaN propagation causing OSError downstream
    if df[['Open', 'High', 'Low', 'Close']].isnull().any().any():
        warnings.warn("OHLC data contains NaN values, dropping affected rows")
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        n = len(df)
        if n < 50:
            return _TradeList([])

    open_ = df['Open'].values.astype(float)
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)

    # Transaction cost — V5.4: partial matching for timeframe suffixes
    tc = get_transaction_cost(instrument, config)

    # --- Step 1: Compute indicators ---
    pmax_result = compute_pmax(
        df,
        length=config.get('pmax_length', 10),
        multiplier=config.get('pmax_multiplier', 3.0),
        ma_type=config.get('pmax_ma_type', 'EMA'),
        ma_mode=config.get('pmax_ma_mode', 'classic'),
        ma_factor=config.get('pmax_ma_factor', 1.0),
        ma_distance=config.get('pmax_ma_distance', 0.0),
    )

    buy_signal = pmax_result['buy_signal']
    sell_signal = pmax_result['sell_signal']
    buy_trigger = pmax_result['buy_trigger']
    sell_trigger = pmax_result['sell_trigger']
    mavg = pmax_result['MAvg']
    pmax_line = pmax_result['PMax']

    # --- Step 1b: Independent MA Crossover Signal ---
    # Combines with PMax via AND logic: enter only when BOTH are aligned.
    # The MA crossover MAs are INDEPENDENT of the PMax MA.
    use_ma_cross_signal = config.get('use_ma_cross_signal', False)
    if use_ma_cross_signal:
        ma_cross = compute_mm_cross_entry(
            df,
            ma1_type=config.get('sig_ma_type', 'EMA'),
            ma1_len=config.get('sig_ma_fast', 9),
            ma2_type=config.get('sig_ma_type', 'EMA'),
            ma2_len=config.get('sig_ma_slow', 21),
        )
        # Continuous state: fast MA > slow MA = bullish
        sig_ma1 = ma_cross['mm_cross_ma1']
        sig_ma2 = ma_cross['mm_cross_ma2']
        ma_cross_bullish = np.zeros(n, dtype=bool)
        ma_cross_bearish = np.zeros(n, dtype=bool)
        for idx in range(n):
            if not np.isnan(sig_ma1[idx]) and not np.isnan(sig_ma2[idx]):
                ma_cross_bullish[idx] = sig_ma1[idx] > sig_ma2[idx]
                ma_cross_bearish[idx] = sig_ma1[idx] < sig_ma2[idx]
        # AND with PMax triggers
        buy_trigger = buy_trigger & ma_cross_bullish
        sell_trigger = sell_trigger & ma_cross_bearish

    # --- Step 2: Compute entry signals ---
    entry_type = config.get('entry_type', 'Donc+Chikou')

    entry_long_raw = np.zeros(n, dtype=bool)
    entry_short_raw = np.zeros(n, dtype=bool)

    if entry_type == 'Donc+Chikou':
        dc = compute_donchian_chikou(df, config.get('don_length', 20))
        entry_long_raw = dc['don_long']
        entry_short_raw = dc['don_short']

    elif entry_type == 'RSI Divergence':
        rd = compute_rsi_divergence(
            df,
            rsi_length=config.get('rsi_div_length', 14),
            pivot_lookback=config.get('rsi_div_pivot', 5),
            detect_hidden=config.get('rsi_div_hidden', True)
        )
        entry_long_raw = rd['buy_rsi_div']
        entry_short_raw = rd['sell_rsi_div']

    elif entry_type == 'Fractal':
        fr = compute_fractal_entry(
            df,
            fractal_buffer=config.get('fractal_buffer', 0.0),
            buy_signal=buy_signal,    # PMax buySignalk for dedup reset
            sell_signal=sell_signal,   # PMax sellSignallk for dedup reset
        )
        entry_long_raw = fr['entry_long']    # already deduped
        entry_short_raw = fr['entry_short']

    elif entry_type == 'Boll+SMA':
        bs = compute_boll_sma_entry(
            df,
            length_bb_ma=config.get('length_bb_ma', 20),
            mult_bb_ma=config.get('mult_bb_ma', 2.0),
            length_bb_sma=config.get('length_bb_sma', 50),
        )
        entry_long_raw = bs['buy_boll_ma']
        entry_short_raw = bs['sell_boll_ma']

    elif entry_type == 'MM Cross':
        mc = compute_mm_cross_entry(
            df,
            ma1_type=config.get('mm_cross_ma1_type', 'EMA'),
            ma1_len=config.get('mm_cross_ma1_len', 9),
            ma2_type=config.get('mm_cross_ma2_type', 'EMA'),
            ma2_len=config.get('mm_cross_ma2_len', 21),
        )
        entry_long_raw = mc['buy_mm_cross']
        entry_short_raw = mc['sell_mm_cross']

    # --- Step 3: Compute filters ---
    filter_long, filter_short = compute_all_filters(df, pmax_result, config)

    # --- Step 3b: V5.2 ADX Regime Filter ---
    use_adx_regime = config.get('use_adx_regime_filter', False)
    adx_filter_long = np.ones(n, dtype=bool)
    adx_filter_short = np.ones(n, dtype=bool)

    if use_adx_regime:
        adx_result = compute_adx(high, low, close, 14, 14)
        adx_vals = adx_result['adx']
        adx_threshold = config.get('adx_trend_threshold', 25)

        compat = ADX_REGIME_COMPATIBILITY.get(
            entry_type, {'trending': True, 'ranging': True})

        for i in range(n):
            if np.isnan(adx_vals[i]):
                continue
            is_trending = adx_vals[i] > adx_threshold
            # Entry allowed if entry_type is compatible with current regime
            regime_ok = (is_trending and compat['trending']) or \
                        (not is_trending and compat['ranging'])
            adx_filter_long[i] = regime_ok
            adx_filter_short[i] = regime_ok

        filter_long = filter_long & adx_filter_long
        filter_short = filter_short & adx_filter_short

    # --- Step 3c: V5.2 Volatility ATR Percentile Filter ---
    use_vol_filter = config.get('use_vol_filter', False)
    if use_vol_filter:
        vol_threshold = config.get('vol_percentile_threshold', 70)
        atr_pctile = compute_atr_percentile(high, low, close, 14, 100)
        vol_ok = atr_pctile < vol_threshold
        filter_long = filter_long & vol_ok
        filter_short = filter_short & vol_ok

    # --- Step 4: SL/TP/BE/TSL parameters ---
    sl_mode = config.get('sl_mode', 'static_pct')
    sl_pct = config.get('sl_pct', 30.0) / 100.0
    exit_setup_reversal = config.get('exit_setup_reversal', True)
    order_pause = config.get('order_pause', 5)
    long_side = config.get('long_side', True)
    short_side = config.get('short_side', True)

    # Ensure at least one side is active (guard against both False)
    if not long_side and not short_side:
        long_side = True

    # ATR-Based SL params
    atr_sl_mult = config.get('atr_sl_mult', 2.0)
    sl_dynamic = config.get('sl_dynamic', False)
    sl_vol_min_mult = config.get('sl_vol_min_mult', 0.8)
    sl_vol_max_mult = config.get('sl_vol_max_mult', 1.5)

    # Precompute ATR data for ATR-Based SL
    atr14_arr = None
    vol_ratio_arr = None
    if sl_mode == 'atr_based':
        atr_data = compute_atr_14_50(high, low, close)
        atr14_arr = atr_data['atr14']
        vol_ratio_arr = atr_data['vol_ratio']

    # TP params
    tp_mode = config.get('tp_mode', 'no')
    tp_pct_val = config.get('tp_pct', 2.0) / 100.0
    tp_sl_ratio = config.get('tp_sl_ratio', 2.0)

    # BE/TSL params
    be_mode = config.get('be_mode', 'no')
    be_trigger_pct = config.get('be_trigger_pct', 1.0) / 100.0
    tsl_pct_val = config.get('tsl_pct', 1.0) / 100.0

    # Max Hold Days
    max_hold_days = config.get('max_hold_days', 0)

    # V5.3: Partial Profit Taking
    use_partial_tp = config.get('use_partial_tp', False)
    tp1_pct_val = config.get('tp1_pct', 1.0) / 100.0
    tp1_size = config.get('tp1_size', 30) / 100.0      # fraction of position
    tp2_pct_val = config.get('tp2_pct', 2.0) / 100.0
    tp2_size = config.get('tp2_size', 30) / 100.0

    # Precompute date index for max_hold_days
    has_datetime_index = hasattr(df.index, 'date') or hasattr(df.index, 'to_pydatetime')
    dates = None
    if max_hold_days > 0 and has_datetime_index:
        try:
            dates = pd.to_datetime(df.index)
        except Exception:
            max_hold_days = 0

    # --- Step 5: 3-Phase Execution Engine ---
    trades = []
    position = 0       # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    sl_price = 0.0
    tp_price = 0.0
    be_triggered = False
    tsl_price = 0.0
    # V5.3: Partial TP state
    position_size = 1.0   # fraction of position remaining (1.0 = full)
    tp1_hit = False
    tp2_hit = False
    tp1_price = 0.0
    tp2_price = 0.0

    bars_since_last_long = order_pause + 1
    bars_since_last_short = order_pause + 1

    pending_order = None  # ('long', signal_bar) or ('short', signal_bar) or None

    def _compute_sl_for_entry(exec_price, side_long, bar_idx):
        """Compute initial SL price for a new trade."""
        if sl_mode == 'static_pct':
            if side_long:
                return exec_price * (1 - sl_pct)
            else:
                return exec_price * (1 + sl_pct)
        elif sl_mode == 'red_line':
            rl = pmax_line[bar_idx]
            if np.isnan(rl):
                return exec_price * (0.7 if side_long else 1.3)
            return rl
        elif sl_mode == 'atr_based':
            atr_val = atr14_arr[bar_idx] if atr14_arr is not None else 0
            if np.isnan(atr_val) or atr_val <= 0:
                return exec_price * (0.95 if side_long else 1.05)
            # Dynamic SL: adjust mult by vol_ratio
            mult = atr_sl_mult
            if sl_dynamic and vol_ratio_arr is not None:
                vr = vol_ratio_arr[bar_idx]
                if not np.isnan(vr):
                    vol_adj = np.clip(vr, sl_vol_min_mult, sl_vol_max_mult)
                    mult = atr_sl_mult * vol_adj
            if side_long:
                return exec_price - (atr_val * mult)
            else:
                return exec_price + (atr_val * mult)
        else:
            return 0  # No SL

    def _compute_tp_for_entry(exec_price, sl_p, side_long):
        """Compute TP price based on tp_mode."""
        if tp_mode == 'no':
            return 0.0
        elif tp_mode == 'static_pct':
            if side_long:
                return exec_price * (1 + tp_pct_val)
            else:
                return exec_price * (1 - tp_pct_val)
        elif tp_mode == 'sl_ratio':
            sl_dist = abs(exec_price - sl_p) if sl_p > 0 else exec_price * 0.02
            if side_long:
                return exec_price + sl_dist * tp_sl_ratio
            else:
                return exec_price - sl_dist * tp_sl_ratio
        return 0.0

    def _close_trade(exit_bar_idx, exit_p, side, reason, size_fraction=1.0):
        nonlocal position, entry_price, entry_bar, sl_price, tp_price
        nonlocal be_triggered, tsl_price, position_size
        nonlocal tp1_hit, tp2_hit, tp1_price, tp2_price
        if side == 'long':
            pnl = ((exit_p - entry_price) - tc) * size_fraction
            pnl_pct = _safe_pnl_pct(exit_p - entry_price, entry_price, tc) * size_fraction
        else:
            pnl = ((entry_price - exit_p) - tc) * size_fraction
            pnl_pct = _safe_pnl_pct(entry_price - exit_p, entry_price, tc) * size_fraction
        trades.append(_make_trade(
            entry_bar, exit_bar_idx, side, entry_price, exit_p,
            pnl, pnl_pct, reason, df
        ))
        if size_fraction >= position_size - 0.001:
            # Full close
            position = 0
            entry_price = 0.0
            sl_price = 0.0
            tp_price = 0.0
            be_triggered = False
            tsl_price = 0.0
            position_size = 1.0
            tp1_hit = False
            tp2_hit = False
            tp1_price = 0.0
            tp2_price = 0.0
        else:
            # Partial close — position remains open with reduced size
            position_size -= size_fraction

    for i in range(1, n):
        # --- PHASE 1: Execute pending orders at Open[i] ---
        if pending_order is not None:
            order_side = pending_order[0]
            exec_price = open_[i]

            if order_side == 'long' and long_side and position <= 0:
                # Close short if open
                if position == -1:
                    _close_trade(i, exec_price, 'short', 'reversal_to_long')

                # Open long
                position = 1
                entry_price = exec_price
                entry_bar = i
                sl_price = _compute_sl_for_entry(exec_price, True, i)
                tp_price = _compute_tp_for_entry(exec_price, sl_price, True)
                be_triggered = False
                tsl_price = 0.0
                position_size = 1.0
                tp1_hit = False
                tp2_hit = False
                # V5.3: Partial TP levels
                if use_partial_tp:
                    tp1_price = exec_price * (1 + tp1_pct_val)
                    tp2_price = exec_price * (1 + tp2_pct_val)
                else:
                    tp1_price = 0.0
                    tp2_price = 0.0
                bars_since_last_long = 0

            elif order_side == 'short' and short_side and position >= 0:
                # Close long if open
                if position == 1:
                    _close_trade(i, exec_price, 'long', 'reversal_to_short')

                # Open short
                position = -1
                entry_price = exec_price
                entry_bar = i
                sl_price = _compute_sl_for_entry(exec_price, False, i)
                tp_price = _compute_tp_for_entry(exec_price, sl_price, False)
                be_triggered = False
                tsl_price = 0.0
                position_size = 1.0
                tp1_hit = False
                tp2_hit = False
                # V5.3: Partial TP levels
                if use_partial_tp:
                    tp1_price = exec_price * (1 - tp1_pct_val)
                    tp2_price = exec_price * (1 - tp2_pct_val)
                else:
                    tp1_price = 0.0
                    tp2_price = 0.0
                bars_since_last_short = 0

            pending_order = None

        # --- PHASE 2: Check SL/TP/BE/TSL on High/Low of bar i ---
        if position != 0:
            is_long = position == 1
            cur_side = 'long' if is_long else 'short'

            # V5.2: Max Hold Days check
            if max_hold_days > 0 and dates is not None:
                try:
                    days_held = (dates[i] - dates[entry_bar]).days
                    if days_held >= max_hold_days:
                        _close_trade(i, open_[i], cur_side, 'max_hold_days', position_size)
                        continue
                except Exception:
                    pass

            # V5.2: Break-Even logic
            if be_mode != 'no' and not be_triggered and entry_price > 0:
                if is_long:
                    unrealized_pct = (high[i] - entry_price) / entry_price
                else:
                    unrealized_pct = (entry_price - low[i]) / entry_price
                if unrealized_pct >= be_trigger_pct:
                    be_triggered = True
                    # Move SL to entry price (break-even)
                    sl_price = entry_price
                    if be_mode == 'be_tsl':
                        # Initialize TSL at entry price
                        tsl_price = entry_price

            # V5.2: Trailing Stop Update
            if be_mode == 'be_tsl' and be_triggered and tsl_pct_val > 0:
                if is_long:
                    new_tsl = high[i] * (1 - tsl_pct_val)
                    if new_tsl > tsl_price:
                        tsl_price = new_tsl
                    # TSL becomes the effective SL (only if higher)
                    if tsl_price > sl_price:
                        sl_price = tsl_price
                else:
                    new_tsl = low[i] * (1 + tsl_pct_val)
                    if tsl_price == 0 or new_tsl < tsl_price:
                        tsl_price = new_tsl
                    if tsl_price < sl_price or sl_price <= 0:
                        sl_price = tsl_price

            # Check TP hit
            if tp_price > 0:
                if is_long and high[i] >= tp_price:
                    exit_p = tp_price if open_[i] < tp_price else open_[i]
                    _close_trade(i, exit_p, 'long', 'take_profit', position_size)
                    continue
                elif not is_long and low[i] <= tp_price:
                    exit_p = tp_price if open_[i] > tp_price else open_[i]
                    _close_trade(i, exit_p, 'short', 'take_profit', position_size)
                    continue

            # V5.3: Partial Profit Taking (before full SL exit)
            if use_partial_tp and position != 0:
                if not tp1_hit and tp1_price > 0:
                    if is_long and high[i] >= tp1_price:
                        exit_p = tp1_price if open_[i] < tp1_price else open_[i]
                        partial_size = min(tp1_size, position_size)
                        _close_trade(i, exit_p, cur_side, 'partial_tp1', partial_size)
                        tp1_hit = True
                        # Move SL to break-even after TP1
                        if position != 0:
                            sl_price = entry_price
                            be_triggered = True
                    elif not is_long and low[i] <= tp1_price:
                        exit_p = tp1_price if open_[i] > tp1_price else open_[i]
                        partial_size = min(tp1_size, position_size)
                        _close_trade(i, exit_p, cur_side, 'partial_tp1', partial_size)
                        tp1_hit = True
                        if position != 0:
                            sl_price = entry_price
                            be_triggered = True

                if tp1_hit and not tp2_hit and tp2_price > 0 and position != 0:
                    if is_long and high[i] >= tp2_price:
                        exit_p = tp2_price if open_[i] < tp2_price else open_[i]
                        partial_size = min(tp2_size, position_size)
                        _close_trade(i, exit_p, cur_side, 'partial_tp2', partial_size)
                        tp2_hit = True
                    elif not is_long and low[i] <= tp2_price:
                        exit_p = tp2_price if open_[i] > tp2_price else open_[i]
                        partial_size = min(tp2_size, position_size)
                        _close_trade(i, exit_p, cur_side, 'partial_tp2', partial_size)
                        tp2_hit = True

                # If all partial TPs hit and position still open, check if fully closed
                if position == 0:
                    continue

            # Check SL hit
            if sl_price > 0:
                if is_long and low[i] <= sl_price:
                    exit_p = max(sl_price, open_[i]) if open_[i] >= sl_price else open_[i]
                    reason = 'trailing_stop' if be_triggered else 'stop_loss'
                    _close_trade(i, exit_p, 'long', reason, position_size)
                    continue
                elif not is_long and high[i] >= sl_price:
                    exit_p = min(sl_price, open_[i]) if open_[i] <= sl_price else open_[i]
                    reason = 'trailing_stop' if be_triggered else 'stop_loss'
                    _close_trade(i, exit_p, 'short', reason, position_size)
                    continue

        # --- PHASE 3: Evaluate signals -> queue for Open[i+1] ---
        bars_since_last_long += 1
        bars_since_last_short += 1

        # Check Setup Reversal exit (close at next bar open)
        if exit_setup_reversal:
            if position == 1 and sell_signal[i]:
                exit_price = open_[i + 1] if i + 1 < n else close[i]
                idx_exit = i + 1 if i + 1 < n else i
                _close_trade(idx_exit, exit_price, 'long', 'setup_reversal', position_size)
                pending_order = None
                continue

            if position == -1 and buy_signal[i]:
                exit_price = open_[i + 1] if i + 1 < n else close[i]
                idx_exit = i + 1 if i + 1 < n else i
                _close_trade(idx_exit, exit_price, 'short', 'setup_reversal', position_size)
                pending_order = None
                continue

        # Check new entry conditions
        if i + 1 >= n:
            continue  # No next bar to execute on

        # Long entry condition
        if (long_side and
                entry_long_raw[i] and buy_trigger[i] and
                filter_long[i] and
                bars_since_last_long >= order_pause and
                position != 1):
            pending_order = ('long', i)

        # Short entry condition
        elif (short_side and
              entry_short_raw[i] and sell_trigger[i] and
              filter_short[i] and
              bars_since_last_short >= order_pause and
              position != -1):
            pending_order = ('short', i)

    # --- Close any open position at last bar ---
    if position != 0:
        exit_price = close[-1]
        if position == 1:
            _close_trade(n - 1, exit_price, 'long', 'end_of_data', position_size)
        else:
            _close_trade(n - 1, exit_price, 'short', 'end_of_data', position_size)

    # Attach DataFrame reference so metrics.py can compute bar-by-bar equity
    # without requiring all callers to pass df explicitly.
    trades = _TradeList(trades)
    trades.__df_ref__ = df
    return trades


class _TradeList(list):
    """
    A list subclass that can carry a __df_ref__ attribute.
    This allows run_backtest() to attach the OHLCV DataFrame to the trade list
    so that compute_metrics() can access it for bar-by-bar equity & daily Sharpe
    without changing the function signatures of all callers.
    """
    __df_ref__ = None


def _make_trade(entry_bar, exit_bar, side, entry_price, exit_price,
                pnl, pnl_pct, exit_reason, df):
    """Create a standardized trade dict."""
    entry_date = str(df.index[entry_bar]) if entry_bar < len(df) else ''
    exit_date = str(df.index[exit_bar]) if exit_bar < len(df) else ''

    return {
        'entry_bar': entry_bar,
        'exit_bar': exit_bar,
        'side': side,
        'entry_price': round(entry_price, 5),
        'exit_price': round(exit_price, 5),
        'pnl': round(pnl, 5),
        'pnl_pct': round(pnl_pct, 4),
        'exit_reason': exit_reason,
        'duration_bars': exit_bar - entry_bar,
        'entry_date': entry_date,
        'exit_date': exit_date,
    }
