"""
PF AI Lab 5.0 — All Filters Implementation
15+ filters matching Pine Script V69.1 logic.
"""

import numpy as np
import pandas as pd
from ..indicators.pmax import compute_pmax_htf_state, compute_atr_rma
from ..indicators.ma_types import rma
from ..config import HTF_RESAMPLE_MAP


# ==============================================================================
# HTF Resampling Helper
# ==============================================================================

def resample_to_htf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample OHLCV data to a higher timeframe.
    Uses W-FRI for weekly (matching Pine Script).
    """
    rule = HTF_RESAMPLE_MAP.get(tf, tf)

    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }

    df_htf = df.resample(rule).agg(agg).dropna()
    df_htf['hl2'] = (df_htf['High'] + df_htf['Low']) / 2.0
    return df_htf


def map_htf_to_ltf(df_ltf: pd.DataFrame, htf_values: np.ndarray,
                    df_htf: pd.DataFrame) -> np.ndarray:
    """Map HTF indicator values back to LTF bars (forward-fill, no lookahead)."""
    htf_series = pd.Series(htf_values, index=df_htf.index)
    # Reindex to LTF and forward-fill (no lookahead)
    mapped = htf_series.reindex(df_ltf.index, method='ffill')
    return mapped.values


# ==============================================================================
# Filter #1: MA Direction Filter
# ==============================================================================

def ma_direction_filter(mavg: np.ndarray, ma_dir_len: int = 3) -> tuple:
    """
    Long if MA is rising for ma_dir_len consecutive bars.
    Short if MA is falling for ma_dir_len consecutive bars.
    """
    n = len(mavg)
    filt_long = np.ones(n, dtype=bool)
    filt_short = np.ones(n, dtype=bool)

    for i in range(ma_dir_len, n):
        rising = all(mavg[i - j] > mavg[i - j - 1]
                     for j in range(ma_dir_len)
                     if not np.isnan(mavg[i - j]) and not np.isnan(mavg[i - j - 1]))
        falling = all(mavg[i - j] < mavg[i - j - 1]
                      for j in range(ma_dir_len)
                      if not np.isnan(mavg[i - j]) and not np.isnan(mavg[i - j - 1]))
        filt_long[i] = rising
        filt_short[i] = falling

    return filt_long, filt_short


# ==============================================================================
# Filters #2/#3: LT/MT Trend Filters (HTF PMax — V68.8 continuous state)
# ==============================================================================

def lt_mt_trend_filter(df: pd.DataFrame, tf: str, length: int,
                       ma_type: str, multiplier: float) -> tuple:
    """
    LT or MT Trend Filter using PMax on HTF.
    V68.8: Returns CONTINUOUS STATE (not crossover).

    Returns:
        (filt_long, filt_short): np.ndarray (bool) for each bar of df
    """
    # Resample to HTF
    df_htf = resample_to_htf(df, tf)

    if len(df_htf) < length + 5:
        # Not enough HTF data
        n = len(df)
        return np.ones(n, dtype=bool), np.ones(n, dtype=bool)

    # Compute PMax state on HTF
    htf_bullish = compute_pmax_htf_state(df_htf, length, multiplier, ma_type)

    # Map back to LTF
    mapped_bullish = map_htf_to_ltf(df, htf_bullish, df_htf)

    # Convert to float first for safe isnan check, then back to bool
    mapped_float = np.array(mapped_bullish, dtype=float)
    filt_long = np.where(np.isnan(mapped_float), True, mapped_float > 0.5).astype(bool)
    filt_short = ~filt_long

    return filt_long, filt_short


# ==============================================================================
# Filter #4: QQ Estimation Filter
# ==============================================================================

def qq_estimation_filter(close: np.ndarray, length: int = 14) -> tuple:
    """
    QQ Estimation Filter — Pine V69.2 exact port.

    Pine logic:
        SSF = 5
        RSII = ta.ema(ta.rsi(close, length), SSF)
        QQE = ta.ema(RSII, SSF)
        QQE1 = ta.ema(RSII[1], SSF)    (shifted RSII)
        QQE2 = ta.ema(RSII[2], SSF)    (double shifted RSII)
        delta_QQE = QQE - QQE1
        delta_QQE1 = QQE1 - QQE2
        cross_QQE = delta_QQE * delta_QQE1
        cross_QQE2 = RSII - QQE
        Long  = cross_QQE[1] < 0 AND cross_QQE > 0 AND cross_QQE2 > 0
        Short = cross_QQE[1] < 0 AND cross_QQE > 0 AND cross_QQE2 < 0
    """
    from ..indicators.ma_types import ema as ema_func

    ssf = 5
    rsi_vals = _compute_rsi_raw(close, length)
    rsii = ema_func(rsi_vals, ssf)

    # QQE = EMA(RSII, SSF)
    qqe = ema_func(rsii, ssf)

    # QQE1 = EMA(RSII[1], SSF) — shift RSII by 1, then EMA
    rsii_shifted1 = np.roll(rsii, 1)
    rsii_shifted1[0] = np.nan
    qqe1 = ema_func(rsii_shifted1, ssf)

    # QQE2 = EMA(RSII[2], SSF) — shift RSII by 2, then EMA
    rsii_shifted2 = np.roll(rsii, 2)
    rsii_shifted2[:2] = np.nan
    qqe2 = ema_func(rsii_shifted2, ssf)

    # delta_QQE = QQE - QQE1, delta_QQE1 = QQE1 - QQE2
    delta_qqe = qqe - qqe1
    delta_qqe1 = qqe1 - qqe2
    cross_qqe = delta_qqe * delta_qqe1
    cross_qqe2 = rsii - qqe

    n = len(close)
    filt_long = np.ones(n, dtype=bool)
    filt_short = np.ones(n, dtype=bool)

    for i in range(1, n):
        if (np.isnan(cross_qqe[i]) or np.isnan(cross_qqe[i - 1]) or
                np.isnan(cross_qqe2[i])):
            continue
        # Long: cross_QQE[1] < 0 AND cross_QQE > 0 AND cross_QQE2 > 0
        filt_long[i] = (cross_qqe[i - 1] < 0 and cross_qqe[i] > 0 and
                        cross_qqe2[i] > 0)
        # Short: cross_QQE[1] < 0 AND cross_QQE > 0 AND cross_QQE2 < 0
        filt_short[i] = (cross_qqe[i - 1] < 0 and cross_qqe[i] > 0 and
                         cross_qqe2[i] < 0)

    return filt_long, filt_short


# ==============================================================================
# Filter #5: MAvg Position Filter
# ==============================================================================

def mavg_position_filter(direction: np.ndarray) -> tuple:
    """Long if dir==1 (MAvg above PMax), Short if dir==-1."""
    filt_long = direction == 1
    filt_short = direction == -1
    return filt_long, filt_short


# ==============================================================================
# Filter #6: Diff MA/Red Line Filter (V68.5 bugfix: <= not >=)
# ==============================================================================

def diff_ma_red_filter(mavg: np.ndarray, pmax: np.ndarray,
                       close: np.ndarray, max_pct: float) -> tuple:
    """
    Block trades when MA is too far from PMax.
    V68.5 FIX: distance <= max_pct% (CORRECT — blocks distant trades)
    """
    diff = np.abs(mavg - pmax) / close
    threshold = max_pct / 100.0
    filt_long = diff <= threshold
    filt_short = diff <= threshold
    # Handle NaN
    filt_long = np.where(np.isnan(diff), True, filt_long)
    filt_short = np.where(np.isnan(diff), True, filt_short)
    return filt_long.astype(bool), filt_short.astype(bool)


# ==============================================================================
# Filter #7: Diff Price/Red Line Filter
# ==============================================================================

def diff_price_red_filter(close: np.ndarray, pmax: np.ndarray,
                          max_pct: float) -> tuple:
    """Block trades when price is too far from PMax."""
    diff = np.abs(close - pmax) / close
    threshold = max_pct / 100.0
    filt_long = diff <= threshold
    filt_short = diff <= threshold
    filt_long = np.where(np.isnan(diff), True, filt_long)
    filt_short = np.where(np.isnan(diff), True, filt_short)
    return filt_long.astype(bool), filt_short.astype(bool)


# ==============================================================================
# Filter #8: RSI > 50 Filter
# ==============================================================================

def rsi_50_filter(close: np.ndarray, rsi_length: int = 14) -> tuple:
    """Long if RSI > 50, Short if RSI < 50."""
    rsi = _compute_rsi_raw(close, rsi_length)
    filt_long = np.where(np.isnan(rsi), True, rsi > 50)
    filt_short = np.where(np.isnan(rsi), True, rsi < 50)
    return filt_long.astype(bool), filt_short.astype(bool)


# ==============================================================================
# Filter #9: RSI-EMA Directional Filter
# ==============================================================================

def rsi_ema_filter(close: np.ndarray, rsi_length: int = 14,
                   ema_length: int = 12) -> tuple:
    """
    Long if RSI > EMA(RSI), Short if RSI < EMA(RSI).
    Pine V69.1: snab_rsi_ema_filter.
    """
    from ..indicators.ma_types import ema as ema_func
    rsi = _compute_rsi_raw(close, rsi_length)
    rsi_ema = ema_func(rsi, ema_length)

    n = len(close)
    filt_long = np.ones(n, dtype=bool)
    filt_short = np.ones(n, dtype=bool)

    for i in range(n):
        if np.isnan(rsi[i]) or np.isnan(rsi_ema[i]):
            continue
        filt_long[i] = rsi[i] > rsi_ema[i]
        filt_short[i] = rsi[i] < rsi_ema[i]

    return filt_long, filt_short


# ==============================================================================
# Fibonacci Range Filter (always active in Pine V69.1)
# ==============================================================================

def fib_range_filter(df: pd.DataFrame, fib_tf: str = 'D',
                     fib_length: int = 265,
                     zones: list = None) -> np.ndarray:
    """
    Fibonacci Range Filter.
    Returns: bool array — True where entry is allowed.
    When all zones are ON, this is essentially: close within the full fib range.
    """
    if zones is None:
        zones = [True] * 6

    close = df['Close'].values
    n = len(df)

    # Resample to fib TF
    df_htf = resample_to_htf(df, fib_tf)

    if len(df_htf) < fib_length:
        return np.ones(n, dtype=bool)

    # Compute rolling max/min on HTF close
    htf_close = df_htf['Close']
    maxr = htf_close.rolling(window=fib_length, min_periods=fib_length).max()
    minr = htf_close.rolling(window=fib_length, min_periods=fib_length).min()

    # Map back to LTF
    maxr_mapped = maxr.reindex(df.index, method='ffill').values
    minr_mapped = minr.reindex(df.index, method='ffill').values

    ranr = maxr_mapped - minr_mapped

    # Fibonacci levels
    lev_1 = maxr_mapped
    lev_0_764 = maxr_mapped - 0.236 * ranr
    lev_0_618 = maxr_mapped - 0.382 * ranr
    lev_0_50 = maxr_mapped - 0.50 * ranr
    lev_0_382 = minr_mapped + 0.382 * ranr
    lev_0_236 = minr_mapped + 0.236 * ranr
    lev_0 = minr_mapped

    allowed = np.zeros(n, dtype=bool)

    if zones[0]:
        allowed |= (close >= lev_0) & (close <= lev_0_236)
    if zones[1]:
        allowed |= (close >= lev_0_236) & (close <= lev_0_382)
    if zones[2]:
        allowed |= (close >= lev_0_382) & (close <= lev_0_50)
    if zones[3]:
        allowed |= (close >= lev_0_50) & (close <= lev_0_618)
    if zones[4]:
        allowed |= (close >= lev_0_618) & (close <= lev_0_764)
    if zones[5]:
        allowed |= (close >= lev_0_764) & (close <= lev_1)

    # Handle NaN
    allowed = np.where(np.isnan(maxr_mapped), True, allowed)

    return allowed.astype(bool)


# ==============================================================================
# Filter #10: Session / Timing Filter (V5.3 — Pine V69.2 Session days/hours)
# ==============================================================================

def session_timing_filter(df: pd.DataFrame,
                          allowed_days: list = None,
                          session1_start: int = 0,
                          session1_end: int = 2400,
                          session2_start: int = 0,
                          session2_end: int = 0) -> np.ndarray:
    """
    Filter trades by day-of-week and intraday session windows.

    Pine V69.2 equivalent:
      Day_M/Day_T/Day_W/Day_TH/Day_F/Day_SA/Day_S  (boolean per day)
      Session = '0000-1200', Session2 = '1200-2359' (HHMM-HHMM)

    Parameters:
        df: DataFrame with DatetimeIndex
        allowed_days: list of int (0=Mon..6=Sun). None = all allowed.
        session1_start: start hour*100+minute (e.g. 0 = 00:00, 930 = 09:30)
        session1_end: end hour*100+minute (e.g. 1200 = 12:00)
        session2_start: second session start (0 = disabled)
        session2_end: second session end (0 = disabled)

    Returns:
        np.ndarray (bool) — True where trading is allowed.
    """
    n = len(df)
    allowed = np.ones(n, dtype=bool)

    # Require DatetimeIndex
    if not hasattr(df.index, 'dayofweek'):
        try:
            idx = pd.to_datetime(df.index)
        except Exception:
            return allowed
    else:
        idx = df.index

    # Day-of-week filter
    if allowed_days is not None and len(allowed_days) < 7:
        dow = idx.dayofweek  # 0=Mon..6=Sun
        day_ok = np.isin(dow, allowed_days)
        allowed &= day_ok

    # Session filter (HHMM format)
    def _in_session(hhmm_val, start, end):
        """Check if HHMM value is within [start, end)."""
        if start <= end:
            return (hhmm_val >= start) & (hhmm_val < end)
        else:
            # Wraps around midnight
            return (hhmm_val >= start) | (hhmm_val < end)

    has_session1 = session1_end > session1_start or (session1_start > 0 and session1_end > 0)
    has_session2 = session2_end > session2_start or (session2_start > 0 and session2_end > 0)

    if has_session1 or has_session2:
        hhmm = idx.hour * 100 + idx.minute
        session_ok = np.zeros(n, dtype=bool)

        if has_session1:
            session_ok |= _in_session(hhmm, session1_start, session1_end)
        if has_session2:
            session_ok |= _in_session(hhmm, session2_start, session2_end)

        # If no session defined at all, allow everything
        if has_session1 or has_session2:
            allowed &= session_ok

    return allowed


# ==============================================================================
# Filter #11: VIX Filter (Security Layer 0 — Pine V69.1/V69.2)
# ==============================================================================

def vix_filter(df: pd.DataFrame, vix_series: np.ndarray,
               vix_threshold_high: float = 30.0,
               vix_threshold_low: float = 10.0) -> np.ndarray:
    """
    Block trades when VIX > high threshold (Risk-Off).
    Allow trades when VIX < low threshold (Risk-On).
    Between thresholds: neutral (allowed).

    Pine V69.2:
      vix_filter = true, vix_threshold_low = 10, vix_threshold_high = 30

    Parameters:
        df: DataFrame (for length only)
        vix_series: np.ndarray of VIX values aligned to df index.
                    Can contain NaN (treated as neutral = allowed).
        vix_threshold_high: VIX above this = block trades
        vix_threshold_low: VIX below this = explicitly allow (informational)

    Returns:
        np.ndarray (bool) — True where trading is allowed.
    """
    n = len(df)
    if vix_series is None or len(vix_series) != n:
        return np.ones(n, dtype=bool)

    # Block when VIX > high threshold
    vix_arr = np.asarray(vix_series, dtype=float)
    allowed = np.where(np.isnan(vix_arr), True, vix_arr <= vix_threshold_high)
    return allowed.astype(bool)


# ==============================================================================
# Filter #12: Macro Filter Simplifie (V5.3 — reduced RS + Climate proxy)
# ==============================================================================

def macro_filter_simplified(df: pd.DataFrame,
                            macro_scores: np.ndarray = None,
                            macro_min: float = -50.0,
                            macro_max: float = 50.0) -> np.ndarray:
    """
    Simplified Macro Score Filter.

    In Pine V69.2, the full macro uses 20 request.security() calls.
    This simplified version accepts a pre-computed macro score series
    (from external data or a proxy computation) and filters:
      trade allowed if macro_min <= score <= macro_max

    Parameters:
        df: DataFrame (for length)
        macro_scores: np.ndarray of macro score values aligned to df.
                      If None, returns all-True (no filtering).
        macro_min: minimum macro score to allow trades (default -50)
        macro_max: maximum macro score to allow trades (default 50)

    Returns:
        np.ndarray (bool) — True where trading is allowed.
    """
    n = len(df)
    if macro_scores is None or len(macro_scores) != n:
        return np.ones(n, dtype=bool)

    scores = np.asarray(macro_scores, dtype=float)
    allowed = np.where(
        np.isnan(scores), True,
        (scores >= macro_min) & (scores <= macro_max)
    )
    return allowed.astype(bool)


def compute_macro_proxy_score(df: pd.DataFrame,
                              reference_data: dict = None,
                              sma_fast: int = 10,
                              sma_slow: int = 50) -> np.ndarray:
    """
    Compute a simplified macro score from reference data.

    Approximates Pine V69.2 getRSR_strength() and getRSR_climate():
    - For each reference symbol, compute ratio = instrument_close / ref_close
    - SMA(ratio, fast) > SMA(ratio, slow) → +1, else -1
    - Sum all scores and normalize to [-50, +50] range

    Parameters:
        df: DataFrame with Close column (the instrument being traded)
        reference_data: dict of {symbol_name: pd.Series/np.ndarray of close prices}
                        aligned to df index. If None, returns zeros.
        sma_fast: fast SMA period (default 10, Pine default)
        sma_slow: slow SMA period (default 50, Pine default)

    Returns:
        np.ndarray of macro score values in [-50, +50] range.
    """
    n = len(df)
    if reference_data is None or len(reference_data) == 0:
        return np.zeros(n, dtype=float)

    close = df['Close'].values.astype(float)
    total_score = np.zeros(n, dtype=float)
    n_valid = 0

    for sym_name, ref_close in reference_data.items():
        ref = np.asarray(ref_close, dtype=float)
        if len(ref) != n:
            continue

        # Compute ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where((ref != 0) & ~np.isnan(ref) & ~np.isnan(close),
                             close / ref, np.nan)

        # SMA fast/slow on ratio
        ratio_s = pd.Series(ratio)
        sma_f = ratio_s.rolling(sma_fast, min_periods=sma_fast).mean().values
        sma_s = ratio_s.rolling(sma_slow, min_periods=sma_slow).mean().values

        # Score: +1 if fast > slow, -1 if fast < slow, 0 if NaN
        score = np.where(
            np.isnan(sma_f) | np.isnan(sma_s), 0.0,
            np.where(sma_f > sma_s, 1.0, -1.0)
        )

        # Momentum factor (simplified: 5-bar change in ratio)
        momentum = np.zeros(n, dtype=float)
        for i in range(5, n):
            if not np.isnan(ratio[i]) and not np.isnan(ratio[i - 5]) and ratio[i - 5] != 0:
                momentum[i] = (ratio[i] - ratio[i - 5]) / ratio[i - 5] * 100

        # Weighted score (matching Pine V68.7.1 logic)
        weighted = np.where(score == 0, 0.0,
                            score * (1.0 + np.abs(momentum) / 10.0))
        total_score += weighted
        n_valid += 1

    if n_valid == 0:
        return np.zeros(n, dtype=float)

    # Normalize to [-50, +50] range
    # Each symbol contributes roughly [-1.x, +1.x] so max total ~ n_valid * 2
    max_possible = n_valid * 2.0
    normalized = np.clip(total_score / max_possible * 50.0, -50.0, 50.0)
    return normalized


# ==============================================================================
# Combine All Filters
# ==============================================================================

def compute_all_filters(df: pd.DataFrame, pmax_result: dict,
                        config: dict) -> tuple:
    """
    Compute all enabled filters and return combined filter arrays.

    Returns:
        (filter_all_long, filter_all_short): np.ndarray (bool)
    """
    n = len(df)
    filt_long = np.ones(n, dtype=bool)
    filt_short = np.ones(n, dtype=bool)

    mavg = pmax_result['MAvg']
    pmax_line = pmax_result['PMax']
    direction = pmax_result['dir']
    close = df['Close'].values

    # Filter #1: MA Direction
    if config.get('use_ma_dir_filter', False):
        fl, fs = ma_direction_filter(mavg, config.get('ma_dir_len', 3))
        filt_long &= fl
        filt_short &= fs

    # Filter #2: LT Trend
    if config.get('use_lt_filter', False):
        fl, fs = lt_mt_trend_filter(
            df, config.get('lt_tf', 'W'),
            config.get('lt_length', 20),
            config.get('lt_ma_type', 'EMA'),
            config.get('lt_multiplier', 1.5)
        )
        filt_long &= fl
        filt_short &= fs

    # Filter #3: MT Trend
    if config.get('use_mt_filter', False):
        fl, fs = lt_mt_trend_filter(
            df, config.get('mt_tf', '4H'),
            config.get('mt_length', 10),
            config.get('mt_ma_type', 'EMA'),
            config.get('mt_multiplier', 1.5)
        )
        filt_long &= fl
        filt_short &= fs

    # Filter #5: MAvg Position
    if config.get('use_mavg_filter', False):
        fl, fs = mavg_position_filter(direction)
        filt_long &= fl
        filt_short &= fs

    # Filter #6: Diff MA/Red
    if config.get('use_diff_ma_red', False):
        fl, fs = diff_ma_red_filter(
            mavg, pmax_line, close,
            config.get('diff_ma_red_pct', 1.0)
        )
        filt_long &= fl
        filt_short &= fs

    # Filter #7: Diff Price/Red
    if config.get('use_diff_price_red', False):
        fl, fs = diff_price_red_filter(
            close, pmax_line,
            config.get('diff_price_red_pct', 1.0)
        )
        filt_long &= fl
        filt_short &= fs

    # Filter #8: RSI > 50
    if config.get('use_rsi50_filter', False):
        fl, fs = rsi_50_filter(close, config.get('rsi50_length', 14))
        filt_long &= fl
        filt_short &= fs

    # Filter #9: RSI-EMA Directional
    if config.get('use_rsi_ema_filter', False):
        fl, fs = rsi_ema_filter(
            close,
            config.get('rsi50_length', 14),
            config.get('rsi_ema_length', 12)
        )
        filt_long &= fl
        filt_short &= fs

    # V5.3: Filter #4 — QQ Estimation Filter (Pine V69.2 exact port)
    if config.get('use_qq_filter', False):
        fl, fs = qq_estimation_filter(
            close,
            length=config.get('qq_length', 14)
        )
        filt_long &= fl
        filt_short &= fs

    # Fibonacci Range Filter (always active by default)
    if config.get('use_fib_filter', True):
        fib_allowed = fib_range_filter(
            df,
            fib_tf=config.get('fib_tf', 'D'),
            fib_length=config.get('fib_length', 265),
            zones=config.get('fib_zones', [True] * 6)
        )
        filt_long &= fib_allowed
        filt_short &= fib_allowed

    # V5.3: Session / Timing Filter
    if config.get('use_session_filter', False):
        session_allowed = session_timing_filter(
            df,
            allowed_days=config.get('session_allowed_days', None),
            session1_start=config.get('session1_start', 0),
            session1_end=config.get('session1_end', 2400),
            session2_start=config.get('session2_start', 0),
            session2_end=config.get('session2_end', 0),
        )
        filt_long &= session_allowed
        filt_short &= session_allowed

    # V5.3: VIX Filter (Security Layer 0)
    if config.get('use_vix_filter', False):
        vix_data = config.get('_vix_series', None)
        if vix_data is not None:
            vix_allowed = vix_filter(
                df, vix_data,
                vix_threshold_high=config.get('vix_threshold_high', 30.0),
                vix_threshold_low=config.get('vix_threshold_low', 10.0),
            )
            filt_long &= vix_allowed
            filt_short &= vix_allowed

    # V5.3: Macro Filter Simplifie
    if config.get('use_macro_filter', False):
        macro_scores = config.get('_macro_scores', None)
        if macro_scores is not None:
            macro_allowed = macro_filter_simplified(
                df, macro_scores,
                macro_min=config.get('macro_min', -50.0),
                macro_max=config.get('macro_max', 50.0),
            )
            filt_long &= macro_allowed
            filt_short &= macro_allowed

    return filt_long, filt_short


# ==============================================================================
# Internal helpers
# ==============================================================================

def _compute_rsi_raw(close: np.ndarray, length: int = 14) -> np.ndarray:
    """Compute raw RSI using Wilder's smoothing."""
    from ..entries.rsi_divergence import compute_rsi
    return compute_rsi(close, length)
