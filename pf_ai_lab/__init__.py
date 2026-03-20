"""
PF AI Lab 5.0.4 — Moteur Python corrige pour P.F.Algo V.69.2

Version corrigee integrant les 13 divergences identifiees par PineGuard.
Corrections appliquees:
  - D7  FIXED   : entry_allowed gate 4 composantes (15-25 phantom trades elimines)
  - D8  FIXED   : 5/5 fractals Williams implementes (89+ signaux recuperes)
  - D9  MITIGATED: deduplication fractale (double entrees eliminees)
  - D10 FIXED   : Bollinger ddof=0 population (2.6% ecart resolu)
  - D11 MITIGATED: RSI cancel logic (seuil 15 barres)
  - D13 DOCUMENTED: ORB signal inverse (mode reproduce_pine_bug disponible)
  - PMax direction: compare avec stops PRECEDENTS (bug critique corrige)
  - Backtest O(n^2) -> O(1) equity incremental

Divergences irreductibles (data source):
  - D1: Bid vs Mid-price (utiliser TV CSV)
  - D2: LP aggregation
  - D3: Timezone/bar timing

Divergences ouvertes (resampling):
  - D4: Weekly bar boundaries (Pine vs Python resample W-FRI)
  - D5: Incomplete bar handling
  - D6: 4H from H1 boundaries
  - D12: HTF resampling validation
"""

__version__ = "5.0.4"
__algo_version__ = "P.F.Algo V.69.2"
__lab_name__ = "PF AI Lab"
__codename__ = "PineGuard-Corrected"

from pf_ai_lab.runner import run_backtest, run_full_pipeline
