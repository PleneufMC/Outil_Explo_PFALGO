"""
ATR (Average True Range) — Pine-equivalent implementation.

CRITICAL: Pine's ta.atr(n) uses RMA (Wilder's smoothing), NOT SMA.
  - Pine: ta.atr(14) = ta.rma(ta.tr, 14)
  - Python: compute_rma(compute_tr(high, low, close), 14)
  - ❌ WRONG: rolling(14).mean() — this is SMA, not RMA

True Range formula (Pine ta.tr):
  tr = max(high - low, |high - close[1]|, |low - close[1]|)
"""

import numpy as np
import pandas as pd

Series = pd.Series


def compute_tr(high: Series, low: Series, close: Series) -> Series:
    """
    Pine: ta.tr — True Range.

    TR = max(
        high - low,                    # Current bar range
        abs(high - close[1]),          # Gap up from previous close
        abs(low - close[1])            # Gap down from previous close
    )

    First bar: TR = high - low (no previous close available).
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.name = 'TR'
    return tr


def compute_atr(high: Series, low: Series, close: Series,
                length: int = 14) -> Series:
    """
    Pine: ta.atr(length) — Average True Range using RMA (Wilder's).

    This is the CORE volatility measure used by PMax for stop calculation:
      nLoss = multiplier * ATR(length)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        length: ATR period (default 14, typical values: 10-20)

    Returns:
        ATR series (RMA of True Range)

    PIÈGE: Using SMA instead of RMA will produce different PMax stops,
    leading to different crossover signals — cascading divergence.
    """
    from pineguard.indicators.moving_averages import compute_rma
    tr = compute_tr(high, low, close)
    atr = compute_rma(tr, length)
    atr.name = f'ATR_{length}'
    return atr


def compute_atr_sma(high: Series, low: Series, close: Series,
                    length: int = 14) -> Series:
    """
    SMA-based ATR — FOR COMPARISON/DEBUGGING ONLY.

    This is the WRONG implementation for Pine compatibility.
    Provided to quantify the RMA vs SMA divergence.
    """
    tr = compute_tr(high, low, close)
    atr_sma = tr.rolling(window=length, min_periods=length).mean()
    atr_sma.name = f'ATR_SMA_{length}'
    return atr_sma
