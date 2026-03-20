"""
RSI (Relative Strength Index) — Pine-equivalent implementation.

Pine: ta.rsi(close, length)
  = 100 - 100 / (1 + rma(gain, length) / rma(loss, length))

Uses RMA (Wilder's) for smoothing gains and losses, NOT SMA.
"""

import numpy as np
import pandas as pd

Series = pd.Series


def compute_rsi(close: Series, length: int = 14) -> Series:
    """
    Pine: ta.rsi(close, length) — Relative Strength Index.

    Implementation:
      1. delta = close - close[1]
      2. gain = max(delta, 0)
      3. loss = max(-delta, 0)    # absolute value of losses
      4. avg_gain = rma(gain, length)
      5. avg_loss = rma(loss, length)
      6. rs = avg_gain / avg_loss
      7. rsi = 100 - 100 / (1 + rs)

    Edge cases:
      - If avg_loss = 0 → RSI = 100
      - If avg_gain = 0 → RSI = 0
      - First `length` bars → NaN (insufficient data)

    Args:
        close: Close prices
        length: RSI period (default 14)

    Returns:
        RSI series [0, 100]
    """
    from pineguard.indicators.moving_averages import compute_rma

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = compute_rma(gain, length)
    avg_loss = compute_rma(loss, length)

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge case: avg_loss = 0 → RSI = 100
    rsi = rsi.fillna(100.0).where(avg_gain.notna(), np.nan)

    # If avg_gain = 0 and avg_loss = 0 → RSI = 50 (neutral)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi[both_zero] = 50.0

    rsi.name = f'RSI_{length}'
    return rsi
