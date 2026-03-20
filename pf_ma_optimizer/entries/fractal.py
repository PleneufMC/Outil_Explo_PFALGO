"""
PF AI Lab 5.0 — Williams Fractal Entry (calibrated from PineScript V69.1)

PineScript logic (n=2):
    upFractal = 5 patterns matching a local HIGH peak at bar [n]:
        Pattern 1: h[n+2]<h[n] and h[n+1]<h[n] and h[n-1]<h[n] and h[n-2]<h[n]
        Pattern 2: h[n+3]<h[n] and h[n+2]<h[n] and h[n+1]==h[n] and h[n-1]<h[n] and h[n-2]<h[n]
        Pattern 3: h[n+4]<h[n] and h[n+3]<h[n] and h[n+2]==h[n] and h[n+1]<=h[n] and h[n-1]<h[n] and h[n-2]<h[n]
        Pattern 4: h[n+5]<h[n] and h[n+4]<h[n] and h[n+3]==h[n] and h[n+2]==h[n] and h[n+1]<=h[n] and h[n-1]<h[n] and h[n-2]<h[n]
        Pattern 5: h[n+6]<h[n] and h[n+5]<h[n] and h[n+4]==h[n] and h[n+3]<=h[n] and h[n+2]==h[n] and h[n+1]<=h[n] and h[n-1]<h[n] and h[n-2]<h[n]
    (dnFractal is the mirror for LOW)

    Fractal levels persist until a new fractal is detected:
        frac_high_level = upFractal ? high[n] + buffer : frac_high_level[1]
        frac_low_level  = dnFractal ? low[n] - buffer  : frac_low_level[1]

    Entry signals:
        buy_fractal  = close < frac_high_level   (price is BELOW the fractal breakout level)
        sell_fractal = close > frac_low_level     (price is ABOVE the fractal breakout level)

    Dedup (1 entry per unique fractal level):
        fractal_high_order_created = buy_fractal and not created[1]
          reset when: frac_high_level changes OR buySignalk (PMax buy signal)

    Entry condition:
        entry_long = buy_fractal AND NOT fractal_high_order_created[1] AND bars_pause
        → Entry via STOP ORDER at frac_high_level

    Cancel condition:
        cancel_entry_long = NOT buy_fractal
"""

import numpy as np
import pandas as pd


def _is_up_fractal(high: np.ndarray, i: int, n_param: int = 2) -> bool:
    """
    Check if bar at index i has an upward fractal (local high peak).
    PineScript checks at bar [n] looking at bars [n-2..n+6].
    In our array, 'i' corresponds to the fractal center bar.
    We need i-2 >= 0 and i+6 < len for all 5 patterns.
    Minimum: i-2 >= 0 and i+2 < len (pattern 1).
    """
    N = len(high)
    h = high

    # Pattern 1: basic 5-bar (n-2 to n+2)
    if i - 2 < 0 or i + 2 >= N:
        return False

    p1 = (h[i + 2] < h[i] and h[i + 1] < h[i] and
          h[i - 1] < h[i] and h[i - 2] < h[i])
    if p1:
        return True

    # Pattern 2: 6-bar with 1 equal (needs i+3)
    if i + 3 >= N:
        return False
    p2 = (h[i + 3] < h[i] and h[i + 2] < h[i] and h[i + 1] == h[i] and
          h[i - 1] < h[i] and h[i - 2] < h[i])
    if p2:
        return True

    # Pattern 3: 7-bar with equal (needs i+4)
    if i + 4 >= N:
        return False
    p3 = (h[i + 4] < h[i] and h[i + 3] < h[i] and h[i + 2] == h[i] and
          h[i + 1] <= h[i] and h[i - 1] < h[i] and h[i - 2] < h[i])
    if p3:
        return True

    # Pattern 4: 8-bar (needs i+5)
    if i + 5 >= N:
        return False
    p4 = (h[i + 5] < h[i] and h[i + 4] < h[i] and h[i + 3] == h[i] and
          h[i + 2] == h[i] and h[i + 1] <= h[i] and
          h[i - 1] < h[i] and h[i - 2] < h[i])
    if p4:
        return True

    # Pattern 5: 9-bar (needs i+6)
    if i + 6 >= N:
        return False
    p5 = (h[i + 6] < h[i] and h[i + 5] < h[i] and h[i + 4] == h[i] and
          h[i + 3] <= h[i] and h[i + 2] == h[i] and h[i + 1] <= h[i] and
          h[i - 1] < h[i] and h[i - 2] < h[i])
    return p5


def _is_dn_fractal(low: np.ndarray, i: int, n_param: int = 2) -> bool:
    """
    Check if bar at index i has a downward fractal (local low valley).
    Mirror of _is_up_fractal for lows.
    """
    N = len(low)
    lo = low

    # Pattern 1
    if i - 2 < 0 or i + 2 >= N:
        return False
    p1 = (lo[i + 2] > lo[i] and lo[i + 1] > lo[i] and
          lo[i - 1] > lo[i] and lo[i - 2] > lo[i])
    if p1:
        return True

    # Pattern 2
    if i + 3 >= N:
        return False
    p2 = (lo[i + 3] > lo[i] and lo[i + 2] > lo[i] and lo[i + 1] == lo[i] and
          lo[i - 1] > lo[i] and lo[i - 2] > lo[i])
    if p2:
        return True

    # Pattern 3
    if i + 4 >= N:
        return False
    p3 = (lo[i + 4] > lo[i] and lo[i + 3] > lo[i] and lo[i + 2] == lo[i] and
          lo[i + 1] >= lo[i] and lo[i - 1] > lo[i] and lo[i - 2] > lo[i])
    if p3:
        return True

    # Pattern 4
    if i + 5 >= N:
        return False
    p4 = (lo[i + 5] > lo[i] and lo[i + 4] > lo[i] and lo[i + 3] == lo[i] and
          lo[i + 2] == lo[i] and lo[i + 1] >= lo[i] and
          lo[i - 1] > lo[i] and lo[i - 2] > lo[i])
    if p4:
        return True

    # Pattern 5
    if i + 6 >= N:
        return False
    p5 = (lo[i + 6] > lo[i] and lo[i + 5] > lo[i] and lo[i + 4] == lo[i] and
          lo[i + 3] >= lo[i] and lo[i + 2] == lo[i] and lo[i + 1] >= lo[i] and
          lo[i - 1] > lo[i] and lo[i - 2] > lo[i])
    return p5


def compute_fractal_entry(df: pd.DataFrame,
                          fractal_buffer: float = 0.0,
                          buy_signal: np.ndarray = None,
                          sell_signal: np.ndarray = None) -> dict:
    """
    Compute Williams Fractal entry signals matching PineScript V69.1 exactly.

    Parameters:
        df: DataFrame with ['High', 'Low', 'Close']
        fractal_buffer: buffer points added to fractal level (default 0.0)
        buy_signal: PMax buySignalk array — resets high dedup when True
        sell_signal: PMax sellSignallk array — resets low dedup when True

    Returns:
        dict with:
            'frac_high_level': np.ndarray — persisting upper fractal level
            'frac_low_level':  np.ndarray — persisting lower fractal level
            'buy_fractal':     np.ndarray (bool) — close < frac_high_level (armed)
            'sell_fractal':    np.ndarray (bool) — close > frac_low_level (armed)
            'entry_long':      np.ndarray (bool) — deduped long entry signals
            'entry_short':     np.ndarray (bool) — deduped short entry signals
    """
    high = df['High'].values.astype(float)
    low = df['Low'].values.astype(float)
    close = df['Close'].values.astype(float)
    n = len(df)

    # --- Detect fractals at each bar ---
    # PineScript: fractal is detected at bar[n] where n=2
    # So in PineScript, upFractal_def at current bar looks at high[n+2]..high[n-2]
    # which means high[4]..high[0] relative to current.
    # The fractal center is at bar (current - 2), confirmed when bar (current) is available.
    # In our loop: fractal at bar i is detected when we reach bar i+2.
    up_fractal = np.zeros(n, dtype=bool)
    dn_fractal = np.zeros(n, dtype=bool)

    for i in range(2, n - 2):  # center bar must have >=2 bars on each side minimum
        up_fractal[i] = _is_up_fractal(high, i)
        dn_fractal[i] = _is_dn_fractal(low, i)

    # --- Persist fractal levels ---
    # In PineScript: frac_high_level := upFractal_def ? high[n] + buffer : nz(frac_high_level[1])
    # The fractal detection at current bar refers to the high at bar[n] = bar[current-2]
    # But the level is set from high[n] which IS the center bar's high.
    frac_high_level = np.full(n, np.nan)
    frac_low_level = np.full(n, np.nan)

    for i in range(n):
        if up_fractal[i]:
            frac_high_level[i] = high[i] + fractal_buffer
        else:
            frac_high_level[i] = frac_high_level[i - 1] if i > 0 and not np.isnan(frac_high_level[i - 1]) else np.nan

        if dn_fractal[i]:
            frac_low_level[i] = low[i] - fractal_buffer
        else:
            frac_low_level[i] = frac_low_level[i - 1] if i > 0 and not np.isnan(frac_low_level[i - 1]) else np.nan

    # --- Entry signals ---
    # PineScript: buy_fractal = close < frac_high_level
    # This means price is BELOW the breakout level → arms a stop order ABOVE at frac_high_level
    # sell_fractal = close > frac_low_level → arms a stop order BELOW at frac_low_level
    buy_fractal = np.zeros(n, dtype=bool)
    sell_fractal = np.zeros(n, dtype=bool)

    for i in range(n):
        if not np.isnan(frac_high_level[i]):
            buy_fractal[i] = close[i] < frac_high_level[i]
        if not np.isnan(frac_low_level[i]):
            sell_fractal[i] = close[i] > frac_low_level[i]

    # --- Deduplication: 1 entry per unique fractal level ---
    # PineScript logic (line 1692):
    #   fractal_high_order_created := buy_fractal and not fractal_high_order_created[1]
    #       ? true
    #       : frac_high_level != frac_high_level[1] or buySignalk
    #           ? false
    #           : fractal_high_order_created[1]
    #
    # entry_long = buy_fractal AND NOT fractal_high_order_created[1]
    #
    # Reset conditions: fractal level changed OR PMax buySignalk fires
    entry_long = np.zeros(n, dtype=bool)
    entry_short = np.zeros(n, dtype=bool)
    high_order_created = False
    low_order_created = False
    prev_frac_high = np.nan
    prev_frac_low = np.nan

    for i in range(n):
        cur_frac_high = frac_high_level[i]
        cur_frac_low = frac_low_level[i]

        # Reset dedup if fractal level changed OR PMax signal fires
        # PineScript: frac_high_level != frac_high_level[1] or buySignalk ? false
        level_changed_high = False
        if not np.isnan(cur_frac_high) and not np.isnan(prev_frac_high):
            if cur_frac_high != prev_frac_high:
                level_changed_high = True
        elif np.isnan(prev_frac_high) and not np.isnan(cur_frac_high):
            level_changed_high = True

        pmax_buy = buy_signal[i] if buy_signal is not None and i < len(buy_signal) else False
        if level_changed_high or pmax_buy:
            high_order_created = False

        level_changed_low = False
        if not np.isnan(cur_frac_low) and not np.isnan(prev_frac_low):
            if cur_frac_low != prev_frac_low:
                level_changed_low = True
        elif np.isnan(prev_frac_low) and not np.isnan(cur_frac_low):
            level_changed_low = True

        pmax_sell = sell_signal[i] if sell_signal is not None and i < len(sell_signal) else False
        if level_changed_low or pmax_sell:
            low_order_created = False

        # Entry signal: armed and not yet created for this level
        if buy_fractal[i] and not high_order_created:
            entry_long[i] = True
            high_order_created = True

        if sell_fractal[i] and not low_order_created:
            entry_short[i] = True
            low_order_created = True

        prev_frac_high = cur_frac_high
        prev_frac_low = cur_frac_low

    return {
        'frac_high_level': frac_high_level,
        'frac_low_level': frac_low_level,
        'buy_fractal': buy_fractal,
        'sell_fractal': sell_fractal,
        'entry_long': entry_long,
        'entry_short': entry_short,
    }
