"""
Core indicators module — Pine-equivalent implementations.

All indicators follow Pine Script v6 semantics:
- ATR uses RMA (Wilder), NOT SMA
- EMA uses adjust=False
- Stdev uses ddof=0 (population)
- PMax uses stateful stop carry-over
"""

from pineguard.indicators.moving_averages import (
    compute_sma, compute_ema, compute_rma, compute_wma, compute_vwma,
    compute_dema, compute_tema, compute_hma, compute_zlema, get_ma
)
from pineguard.indicators.atr import compute_tr, compute_atr
from pineguard.indicators.rsi import compute_rsi
from pineguard.indicators.stdev import compute_stdev
from pineguard.indicators.pmax import compute_pmax
from pineguard.indicators.bollinger import compute_bollinger
from pineguard.indicators.donchian import compute_donchian
from pineguard.indicators.adx import compute_adx

__all__ = [
    'compute_sma', 'compute_ema', 'compute_rma', 'compute_wma', 'compute_vwma',
    'compute_dema', 'compute_tema', 'compute_hma', 'compute_zlema', 'get_ma',
    'compute_tr', 'compute_atr', 'compute_rsi', 'compute_stdev',
    'compute_pmax', 'compute_bollinger', 'compute_donchian', 'compute_adx',
]
