"""
PineGuard v1.0 — Expert Cohérence Pine↔Python
Auditeur spécialisé pour P.F.Algo V.69.2 + PF AI Lab 5.0

« Même signal, même barre, même résultat. »
"""

__version__ = "1.0.0"
__author__ = "PineGuard"
__algo_version__ = "P.F.Algo V.69.2"
__lab_version__ = "PF AI Lab 5.4.0"

# V5.4: Single source of truth for SAFE entry types.
# Updated to reflect V5.3 fixes: Fractal (85%), Boll+SMA (90%), MM Cross (99%)
# VWAP and ORB remain excluded (session-dependent / signal inversé)
SAFE_ENTRY_TYPES = ['Donc+Chikou', 'RSI Divergence', 'Fractal', 'Boll+SMA', 'MM Cross']

CONFIDENCE_MATRIX = {
    'Donc+Chikou': {'confidence': 1.00, 'label': '✅ 100%', 'test': 'Quick test suffisant'},
    'RSI Divergence': {'confidence': 0.97, 'label': '✅ ~97%', 'test': 'Cancel logic improved'},
    'Boll+SMA': {'confidence': 0.90, 'label': '✅ ~90%', 'test': 'ddof=0 + entry_allowed fixed'},
    'Fractal': {'confidence': 0.85, 'label': '✅ ~85%', 'test': 'All 5 patterns + dedup implemented'},
    'VWAP': {'confidence': None, 'label': '❓ Non audité', 'test': 'Session-dependent'},
    'ORB': {'confidence': 0.00, 'label': '🔴 Signal inversé', 'test': 'Fix B11 in Pine V.69.1'},
    'MM Cross': {'confidence': 0.99, 'label': '✅ ~99%', 'test': 'V69.2 crossover/crossunder pure math'},
}
