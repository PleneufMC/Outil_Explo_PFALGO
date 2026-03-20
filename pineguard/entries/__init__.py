"""
Entry Types module — 6 entry signal generators matching Pine V.69.2.

Entry Types and their Python confidence (V5.3+):
  ✅ Donc+Chikou   — 100% (deterministic, SAFE)
  ✅ RSI Divergence — ~97% (cancel logic improved, SAFE)
  ✅ Boll+SMA      — ~90% (ddof=0 + entry_allowed fixed, SAFE)
  ✅ Fractal        — ~85% (all 5 Williams patterns + dedup, SAFE)
  ✅ MM Cross       — ~99% (V69.2 crossover/crossunder pure math, SAFE)
  ❓ VWAP          — Non audité (session-dependent)
  🔴 ORB           — Signal inversé (BLOQUANT)

SAFE_ENTRY_TYPES imported from pineguard.__init__ (single source of truth)
"""

from pineguard.entries.donchian_chikou import compute_donchian_chikou_signals
from pineguard.entries.rsi_divergence import compute_rsi_divergence_signals
from pineguard.entries.boll_sma import compute_boll_sma_signals
from pineguard.entries.fractal import compute_fractal_signals
from pineguard.entries.vwap import compute_vwap_signals
from pineguard.entries.orb import compute_orb_signals

__all__ = [
    'compute_donchian_chikou_signals',
    'compute_rsi_divergence_signals',
    'compute_boll_sma_signals',
    'compute_fractal_signals',
    'compute_vwap_signals',
    'compute_orb_signals',
]
