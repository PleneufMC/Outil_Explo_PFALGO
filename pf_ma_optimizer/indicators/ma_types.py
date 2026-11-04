"""
PF AI Lab 5.0 — 9 Moving Average Types
Exact replication from Pine Script V69.1 getMA() function.
"""

import numpy as np
import pandas as pd


def sma(src: np.ndarray, length: int) -> np.ndarray:
    """Simple Moving Average."""
    s = pd.Series(src)
    return s.rolling(window=length, min_periods=length).mean().values


def ema(src: np.ndarray, length: int) -> np.ndarray:
    """Exponential Moving Average (Pine Script ta.ema)."""
    result = np.full_like(src, np.nan, dtype=float)
    alpha = 2.0 / (length + 1)

    # Seed with SMA of first `length` values
    if len(src) < length:
        return result

    first_valid = np.nan
    for i in range(len(src)):
        if not np.isnan(src[i]):
            start_idx = i
            break
    else:
        return result

    if start_idx + length > len(src):
        return result

    seed = np.mean(src[start_idx:start_idx + length])
    result[start_idx + length - 1] = seed

    for i in range(start_idx + length, len(src)):
        if np.isnan(src[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result


def wma(src: np.ndarray, length: int) -> np.ndarray:
    """Weighted Moving Average."""
    result = np.full_like(src, np.nan, dtype=float)
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()

    for i in range(length - 1, len(src)):
        window = src[i - length + 1:i + 1]
        if np.any(np.isnan(window)):
            result[i] = np.nan
        else:
            result[i] = np.dot(window, weights) / weight_sum

    return result


def tma(src: np.ndarray, length: int) -> np.ndarray:
    """Triangular Moving Average = SMA(SMA(src, ceil(len/2)), floor(len/2)+1)."""
    import math
    first_len = math.ceil(length / 2)
    second_len = math.floor(length / 2) + 1
    first = sma(src, first_len)
    return sma(first, second_len)


def trima(src: np.ndarray, length: int) -> np.ndarray:
    """TRIMA — True Triangular Moving Average = SMA(SMA(src, ceil(N/2)), floor(N/2)+1).
    Fixed in V69.2: was incorrectly using WMA, now uses correct double SMA.
    """
    import math
    first_len = math.ceil(length / 2)
    second_len = math.floor(length / 2) + 1
    first = sma(src, first_len)
    return sma(first, second_len)


def var_func(src: np.ndarray, length: int) -> np.ndarray:
    """
    VAR (Variable Index Dynamic Average) — Pine Script Var_Func.
    valpha = 2 / (length + 1)
    vCMO = (sum_up - sum_dn) / (sum_up + sum_dn)
    VAR = valpha * |vCMO| * src + (1 - valpha * |vCMO|) * VAR[1]
    """
    n = len(src)
    result = np.full(n, np.nan, dtype=float)
    valpha = 2.0 / (length + 1)

    for i in range(1, n):
        if np.isnan(src[i]) or np.isnan(src[i - 1]):
            continue

        # Calculate sum of up/down changes over window of 9
        vUD = 0.0
        vDD = 0.0
        for j in range(max(1, i - 8), i + 1):
            if j >= 1 and not np.isnan(src[j]) and not np.isnan(src[j - 1]):
                diff = src[j] - src[j - 1]
                if diff > 0:
                    vUD += diff
                elif diff < 0:
                    vDD += abs(diff)

        denom = vUD + vDD
        vCMO = (vUD - vDD) / denom if denom != 0 else 0.0

        if np.isnan(result[i - 1]):
            result[i] = src[i]
        else:
            k = valpha * abs(vCMO)
            result[i] = k * src[i] + (1 - k) * result[i - 1]

    return result


def wwma(src: np.ndarray, length: int) -> np.ndarray:
    """Welles Wilder Moving Average (WWMA). alpha = 1/length."""
    n = len(src)
    result = np.full(n, np.nan, dtype=float)
    alpha = 1.0 / length

    for i in range(len(src)):
        if np.isnan(src[i]):
            continue
        if np.isnan(result[i - 1]) if i > 0 else True:
            result[i] = src[i]
        else:
            result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result


def zlema(src: np.ndarray, length: int) -> np.ndarray:
    """Zero-Lag EMA."""
    lag = length // 2 if length % 2 == 0 else (length - 1) // 2
    n = len(src)
    adjusted = np.full(n, np.nan, dtype=float)
    for i in range(lag, n):
        if not np.isnan(src[i]) and not np.isnan(src[i - lag]):
            adjusted[i] = src[i] + (src[i] - src[i - lag])
    return ema(adjusted, length)


def tsf(src: np.ndarray, length: int) -> np.ndarray:
    """Time Series Forecast = linreg(src, length, 0) + slope."""
    result = np.full_like(src, np.nan, dtype=float)
    for i in range(length - 1, len(src)):
        window = src[i - length + 1:i + 1]
        if np.any(np.isnan(window)):
            result[i] = np.nan
            continue
        x = np.arange(length, dtype=float)
        # Linear regression
        mx = x.mean()
        my = window.mean()
        ss_xx = np.sum((x - mx) ** 2)
        if ss_xx == 0:
            result[i] = my
            continue
        slope = np.sum((x - mx) * (window - my)) / ss_xx
        intercept = my - slope * mx
        # lrc = value at x=length-1 (offset 0)
        lrc = intercept + slope * (length - 1)
        # lrc1 = value at x=length (offset 1 forward)
        lrc1 = intercept + slope * length
        lrs = lrc - lrc1
        result[i] = lrc + lrs  # = lrc + (lrc - lrc1) = 2*lrc - lrc1

    return result


def get_ma(src: np.ndarray, length: int, ma_type: str) -> np.ndarray:
    """
    Main dispatcher — matches Pine Script getMA(src, length, mav_type).
    """
    ma_funcs = {
        'SMA': sma,
        'EMA': ema,
        'WMA': wma,
        'TMA': tma,
        'TRIMA': trima,
        'VAR': var_func,
        'WWMA': wwma,
        'ZLEMA': zlema,
        'TSF': tsf,
    }

    if ma_type not in ma_funcs:
        raise ValueError(f"Unknown MA type: {ma_type}. Must be one of {list(ma_funcs.keys())}")

    return ma_funcs[ma_type](src, length)


def rma(src: np.ndarray, length: int) -> np.ndarray:
    """
    Wilder's RMA (ta.rma in Pine) — used for ATR calculation.
    Seed = SMA of first N values, then exponential with alpha=1/N.
    CRITICAL: Pine ta.atr(length) uses ta.rma(ta.tr, length), NOT SMA.
    """
    n = len(src)
    result = np.full(n, np.nan, dtype=float)
    alpha = 1.0 / length

    # Find first valid window for seed
    valid_count = 0
    seed_end = -1
    for i in range(n):
        if not np.isnan(src[i]):
            valid_count += 1
            if valid_count == length:
                seed_end = i
                break

    if seed_end < 0:
        return result

    # Seed = SMA of first `length` valid values
    seed_vals = []
    for i in range(seed_end + 1):
        if not np.isnan(src[i]):
            seed_vals.append(src[i])
    result[seed_end] = np.mean(seed_vals[-length:])

    # Exponential smoothing
    for i in range(seed_end + 1, n):
        if np.isnan(src[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result
