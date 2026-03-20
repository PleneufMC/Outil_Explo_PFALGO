"""
PF AI Lab 5.0 — Bollinger Bands + SMA Entry (calibrated from PineScript V69.1)

PineScript logic (lines 1699-1708):
    // Boll+SMA entry
    basis_bb_ma = ta.sma(close, length_bb_ma)           // default 20
    dev_bb_ma   = mult_bb_ma * ta.stdev(close, length_bb_ma)  // default 2.0
    upper_bb_ma = basis_bb_ma + dev_bb_ma
    lower_bb_ma = basis_bb_ma - dev_bb_ma

    sma_50 = ta.sma(close, length_bb_sma)               // default 50

    buy_boll_ma  = sma_50 <= lower_bb_ma
    sell_boll_ma = sma_50 >= upper_bb_ma

    // PineScript stdev uses ddof=0 (population std)
    // This matches ta.stdev() in Pine Script V5

Entry logic (lines 2056-2062):
    Boll+SMA long:
        buy_limit_price = close > sma_50 ? sma_50 : 0   (limit order at sma_50)
        buy_stop_price  = close < sma_50 ? sma_50 : 0    (stop order at sma_50)
    Boll+SMA short:
        sell_limit_price = close < sma_50 ? sma_50 : 0
        sell_stop_price  = close > sma_50 ? sma_50 : 0

    In Python backtest: we use market execution at open[i+1] (same as other entries).
    The signal (buy_boll_ma / sell_boll_ma) determines when to enter.
"""

import numpy as np
import pandas as pd


def compute_boll_sma_entry(df: pd.DataFrame,
                           length_bb_ma: int = 20,
                           mult_bb_ma: float = 2.0,
                           length_bb_sma: int = 50) -> dict:
    """
    Compute Bollinger Bands + SMA entry signals matching PineScript V69.1.

    PineScript ta.stdev uses ddof=0 (population standard deviation).

    Parameters:
        df: DataFrame with ['Close']
        length_bb_ma: Bollinger band MA period (default 20)
        mult_bb_ma: Bollinger band StdDev multiplier (default 2.0)
        length_bb_sma: SMA period for signal line (default 50)

    Returns:
        dict with:
            'basis_bb':    np.ndarray — BB basis (SMA of close)
            'upper_bb':    np.ndarray — BB upper band
            'lower_bb':    np.ndarray — BB lower band
            'sma_signal':  np.ndarray — SMA(close, length_bb_sma)
            'buy_boll_ma': np.ndarray (bool) — long signal (sma <= lower)
            'sell_boll_ma':np.ndarray (bool) — short signal (sma >= upper)
    """
    close = df['Close'].values.astype(float)
    n = len(df)

    close_series = pd.Series(close)

    # Bollinger Band basis = SMA(close, length_bb_ma)
    basis_bb = close_series.rolling(window=length_bb_ma, min_periods=length_bb_ma).mean().values

    # PineScript ta.stdev uses ddof=0 (population std)
    std_bb = close_series.rolling(window=length_bb_ma, min_periods=length_bb_ma).std(ddof=0).values

    # Upper and lower bands
    dev_bb = mult_bb_ma * std_bb
    upper_bb = basis_bb + dev_bb
    lower_bb = basis_bb - dev_bb

    # SMA signal line
    sma_signal = close_series.rolling(window=length_bb_sma, min_periods=length_bb_sma).mean().values

    # Entry signals
    # buy_boll_ma  = sma_50 <= lower_bb_ma
    # sell_boll_ma = sma_50 >= upper_bb_ma
    buy_boll_ma = np.zeros(n, dtype=bool)
    sell_boll_ma = np.zeros(n, dtype=bool)

    for i in range(n):
        if not np.isnan(sma_signal[i]) and not np.isnan(lower_bb[i]):
            buy_boll_ma[i] = sma_signal[i] <= lower_bb[i]
        if not np.isnan(sma_signal[i]) and not np.isnan(upper_bb[i]):
            sell_boll_ma[i] = sma_signal[i] >= upper_bb[i]

    return {
        'basis_bb': basis_bb,
        'upper_bb': upper_bb,
        'lower_bb': lower_bb,
        'sma_signal': sma_signal,
        'buy_boll_ma': buy_boll_ma,
        'sell_boll_ma': sell_boll_ma,
    }
