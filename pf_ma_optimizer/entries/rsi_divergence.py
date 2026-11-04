"""
PF AI Lab 5.0 — RSI Divergence Entry (reliable in Python)

Detects regular and hidden divergences between price and RSI.
- Regular Bullish: price makes lower low, RSI makes higher low
- Regular Bearish: price makes higher high, RSI makes lower high
- Hidden Bullish: price makes higher low, RSI makes lower low (continuation)
- Hidden Bearish: price makes lower high, RSI makes higher high (continuation)
"""

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def compute_rsi(close: np.ndarray, length: int = 14) -> np.ndarray:
    """Compute RSI using Wilder's smoothing (RMA), matching Pine Script ta.rsi."""
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # RMA smoothing (Wilder) with SMA seed
    alpha = 1.0 / length

    avg_gain = np.full(n, np.nan, dtype=float)
    avg_loss = np.full(n, np.nan, dtype=float)

    # Seed with SMA
    if n > length:
        avg_gain[length] = np.mean(gain[1:length + 1])
        avg_loss[length] = np.mean(loss[1:length + 1])

        for i in range(length + 1, n):
            avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
            avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]

        for i in range(length, n):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def find_pivots(data: np.ndarray, lookback: int):
    """
    Find pivot highs and lows using argrelextrema (matching Pine ta.pivothigh/ta.pivotlow).
    Pine: pivot detected at index i if data[i] is max/min of [i-lookback, i+lookback].
    """
    n = len(data)
    pivot_highs = np.full(n, np.nan, dtype=float)
    pivot_lows = np.full(n, np.nan, dtype=float)

    for i in range(lookback, n - lookback):
        window = data[i - lookback:i + lookback + 1]
        if np.any(np.isnan(window)):
            continue
        # Pivot high: data[i] is the highest in the window
        if data[i] == np.max(window):
            # Check strict: must be strictly higher on both sides
            left = data[i - lookback:i]
            right = data[i + 1:i + lookback + 1]
            if np.all(data[i] > left) and np.all(data[i] > right):
                pivot_highs[i] = data[i]

        # Pivot low: data[i] is the lowest in the window
        if data[i] == np.min(window):
            left = data[i - lookback:i]
            right = data[i + 1:i + lookback + 1]
            if np.all(data[i] < left) and np.all(data[i] < right):
                pivot_lows[i] = data[i]

    return pivot_highs, pivot_lows


def compute_rsi_divergence(df: pd.DataFrame,
                           rsi_length: int = 14,
                           pivot_lookback: int = 5,
                           detect_hidden: bool = True) -> dict:
    """
    Compute RSI Divergence entry signals.

    Parameters:
        df: DataFrame with ['High', 'Low', 'Close']
        rsi_length: RSI period
        pivot_lookback: bars on each side for pivot detection
        detect_hidden: also detect hidden divergences (continuation)

    Returns:
        dict with:
            'rsi': np.ndarray — RSI values
            'buy_rsi_div': np.ndarray (bool) — long signals
            'sell_rsi_div': np.ndarray (bool) — short signals
            'bullish_regular': np.ndarray (bool)
            'bearish_regular': np.ndarray (bool)
            'bullish_hidden': np.ndarray (bool)
            'bearish_hidden': np.ndarray (bool)
    """
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)
    n = len(df)

    # Compute RSI
    rsi_values = compute_rsi(close, rsi_length)

    # Find pivots on price and RSI
    price_pivot_highs, price_pivot_lows = find_pivots(high, pivot_lookback)
    rsi_pivot_highs, rsi_pivot_lows = find_pivots(rsi_values, pivot_lookback)

    # Signals
    bullish_reg = np.zeros(n, dtype=bool)
    bearish_reg = np.zeros(n, dtype=bool)
    bullish_hid = np.zeros(n, dtype=bool)
    bearish_hid = np.zeros(n, dtype=bool)

    # Track last pivots
    last_price_low = np.nan
    last_rsi_low = np.nan
    last_price_high = np.nan
    last_rsi_high = np.nan

    for i in range(pivot_lookback, n):
        # Check for new pivot low at position (i - pivot_lookback) — pivots are delayed
        check_idx = i
        if not np.isnan(price_pivot_lows[check_idx]):
            current_price_low = price_pivot_lows[check_idx]
            current_rsi_low = rsi_pivot_lows[check_idx] if not np.isnan(rsi_pivot_lows[check_idx]) else rsi_values[check_idx]

            if not np.isnan(last_price_low) and not np.isnan(last_rsi_low):
                # Regular Bullish: Price LL, RSI HL
                if current_price_low < last_price_low and current_rsi_low > last_rsi_low:
                    # Signal appears at pivot_lookback bars after the actual pivot
                    sig_idx = min(check_idx + pivot_lookback, n - 1)
                    if sig_idx < n:
                        bullish_reg[sig_idx] = True

                # Hidden Bullish: Price HL, RSI LL (continuation)
                if detect_hidden and current_price_low > last_price_low and current_rsi_low < last_rsi_low:
                    sig_idx = min(check_idx + pivot_lookback, n - 1)
                    if sig_idx < n:
                        bullish_hid[sig_idx] = True

            last_price_low = current_price_low
            last_rsi_low = current_rsi_low

        if not np.isnan(price_pivot_highs[check_idx]):
            current_price_high = price_pivot_highs[check_idx]
            current_rsi_high = rsi_pivot_highs[check_idx] if not np.isnan(rsi_pivot_highs[check_idx]) else rsi_values[check_idx]

            if not np.isnan(last_price_high) and not np.isnan(last_rsi_high):
                # Regular Bearish: Price HH, RSI LH
                if current_price_high > last_price_high and current_rsi_high < last_rsi_high:
                    sig_idx = min(check_idx + pivot_lookback, n - 1)
                    if sig_idx < n:
                        bearish_reg[sig_idx] = True

                # Hidden Bearish: Price LH, RSI HH (continuation)
                if detect_hidden and current_price_high < last_price_high and current_rsi_high > last_rsi_high:
                    sig_idx = min(check_idx + pivot_lookback, n - 1)
                    if sig_idx < n:
                        bearish_hid[sig_idx] = True

            last_price_high = current_price_high
            last_rsi_high = current_rsi_high

    buy_rsi_div = bullish_reg | bullish_hid
    sell_rsi_div = bearish_reg | bearish_hid

    return {
        'rsi': rsi_values,
        'buy_rsi_div': buy_rsi_div,
        'sell_rsi_div': sell_rsi_div,
        'bullish_regular': bullish_reg,
        'bearish_regular': bearish_reg,
        'bullish_hidden': bullish_hid,
        'bearish_hidden': bearish_hid,
    }
