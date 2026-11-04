"""
Audit Framework — Divergence tracker, trade comparator, signal matcher.

The core of PineGuard: systematically compare Pine and Python results.

5-step protocol:
  1. ISOLER le composant
  2. QUANTIFIER l'écart
  3. DIAGNOSTIQUER la cause
  4. ÉVALUER la criticité
  5. PROPOSER le fix
"""

from pineguard.audit.divergence_tracker import DivergenceTracker, Divergence
from pineguard.audit.trade_comparator import TradeComparator
from pineguard.audit.signal_matcher import SignalMatcher
from pineguard.audit.report_generator import AuditReportGenerator

__all__ = [
    'DivergenceTracker', 'Divergence',
    'TradeComparator', 'SignalMatcher',
    'AuditReportGenerator',
]
