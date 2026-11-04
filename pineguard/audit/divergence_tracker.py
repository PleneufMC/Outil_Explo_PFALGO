"""
Divergence Tracker — Central registry of all known Pine↔Python divergences.

Each divergence has:
  - ID (D1-D13+)
  - Source (data, resampling, implementation, structural)
  - Severity (BLOQUANT, HAUTE, MOYENNE, BASSE)
  - Status (open, fixed, mitigated, irréductible)
  - Impact estimate (trades affected, PnL impact)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime


class Severity(Enum):
    BLOQUANT = 4   # Results are opposite (profitable ↔ losing)
    HAUTE = 3      # >20% trade count divergence
    MOYENNE = 2    # 10-20% divergence, same direction
    BASSE = 1      # <10% divergence, PnL close


class Category(Enum):
    DATA = 'data'               # Bid/Mid, LP aggregation, timezone
    RESAMPLING = 'resampling'   # HTF boundaries, incomplete bars
    IMPLEMENTATION = 'impl'     # Code bugs, missing logic
    STRUCTURAL = 'structural'   # Irréductible by code


class Status(Enum):
    OPEN = 'open'
    FIXED = 'fixed'
    MITIGATED = 'mitigated'
    IRREDUCTIBLE = 'irréductible'


@dataclass
class Divergence:
    """A single divergence between Pine and Python."""
    id: str
    title: str
    category: Category
    severity: Severity
    status: Status
    description: str
    pine_behavior: str
    python_behavior: str
    impact_trades: Optional[int] = None  # Estimated trades affected
    impact_pnl_pct: Optional[float] = None  # Estimated PnL impact %
    module: str = ''  # Affected Python module
    bug_id: str = ''  # Reference to CHANGELOG bug ID (B1-B12)
    fix_effort: str = ''  # Estimated fix time
    fix_description: str = ''
    last_updated: str = field(default_factory=lambda: datetime.now().strftime('%Y-%m-%d'))


class DivergenceTracker:
    """
    Central registry and analyzer for Pine↔Python divergences.

    Pre-loaded with all known divergences from the PineGuard spec.
    Can be extended with new findings during audit sessions.
    """

    def __init__(self):
        self.divergences: Dict[str, Divergence] = {}
        self._load_known_divergences()

    def _load_known_divergences(self):
        """Load all known divergences from PineGuard v1.0 spec."""

        # === STRUCTURAL DIVERGENCES (irréductibles) ===
        self.add(Divergence(
            id='D1', title='Bid vs Mid-price',
            category=Category.DATA, severity=Severity.MOYENNE,
            status=Status.IRREDUCTIBLE,
            description='MT5 exports bid prices, TradingView uses (bid+ask)/2 mid-price',
            pine_behavior='Uses exchange mid-price from 25+ LPs',
            python_behavior='Uses MT5 CSV which is bid-only',
            impact_trades=15, module='data loading',
            fix_effort='Use TV CSV export or tvdatafeed',
            fix_description='Switch to TradingView data source',
        ))

        self.add(Divergence(
            id='D2', title='LP Aggregation',
            category=Category.DATA, severity=Severity.BASSE,
            status=Status.IRREDUCTIBLE,
            description='MT5 uses 1 LP, TV aggregates 25+ LPs via ECN',
            pine_behavior='ECN aggregated price from 25+ liquidity providers',
            python_behavior='Single LP price from MT5 broker',
            impact_trades=0, impact_pnl_pct=0.5,
            fix_effort='Use same data source',
            fix_description='Use TV data or acknowledge ±0.5-2.0 pts/bar difference',
        ))

        self.add(Divergence(
            id='D3', title='Timezone / Bar Timing',
            category=Category.DATA, severity=Severity.BASSE,
            status=Status.IRREDUCTIBLE,
            description='MT5 server clock differs from TV server clock',
            pine_behavior='TV server time (exchange-native)',
            python_behavior='MT5 server time (broker-specific)',
            impact_trades=5, fix_effort='Correct UTC offset in data loading',
        ))

        # === RESAMPLING DIVERGENCES ===
        self.add(Divergence(
            id='D4', title='Weekly Bar Boundaries',
            category=Category.RESAMPLING, severity=Severity.MOYENNE,
            status=Status.OPEN,
            description='Pine W bars are server-native, Python uses resample("W-FRI")',
            pine_behavior='request.security("W", ...) on native server weekly bars',
            python_behavior='df.resample("W-FRI").agg(...) on H1 data',
            impact_trades=10, module='filters/htf_resampling.py',
            fix_effort='2h — validate boundary alignment bar-by-bar',
        ))

        self.add(Divergence(
            id='D5', title='Incomplete Bar Handling',
            category=Category.RESAMPLING, severity=Severity.BASSE,
            status=Status.OPEN,
            description='Pine [1] on HTF vs Python shift(1) after resample — timing differs',
            pine_behavior='[1] refers to last confirmed HTF bar',
            python_behavior='shift(1) after resample may have different timing',
            impact_trades=3, module='filters/htf_resampling.py',
            fix_effort='1h — align shift logic',
        ))

        self.add(Divergence(
            id='D6', title='4H from H1 Boundaries',
            category=Category.RESAMPLING, severity=Severity.BASSE,
            status=Status.OPEN,
            description='TV computes natively in 4H, Python aggregates 4× H1 bars',
            pine_behavior='Native 4H bar calculation on server',
            python_behavior='4× H1 aggregation — boundaries differ for 23h/day markets',
            impact_trades=2, module='filters/htf_resampling.py',
        ))

        # === IMPLEMENTATION DIVERGENCES (corrigibles) ===
        self.add(Divergence(
            id='D7', title='entry_allowed gate',
            category=Category.IMPLEMENTATION, severity=Severity.HAUTE,
            status=Status.FIXED,
            description='4 components of entry_allowed gate implemented in backtest_engine + filter module',
            pine_behavior='Entry gated by PMax dir + position + LT + MT filters',
            python_behavior='entry_allowed gate with PMax, position, LT, MT, RSI components',
            impact_trades=0, module='backtest_engine.py + filters/entry_allowed.py', bug_id='B9',
            fix_effort='2h', fix_description='Implemented 4-component entry_allowed gate (DONE)',
        ))

        self.add(Divergence(
            id='D8', title='Fractal: all 5 patterns',
            category=Category.IMPLEMENTATION, severity=Severity.HAUTE,
            status=Status.FIXED,
            description='All 5 Williams patterns now implemented (classic + 4 plateau variants)',
            pine_behavior='5 Williams patterns (classic + 4 plateau variants)',
            python_behavior='All 5 patterns: classic, plateau-left, plateau-right, double, extended',
            impact_trades=0, module='entries/fractal.py', bug_id='B5',
            fix_effort='4h', fix_description='Patterns 1-5 all implemented (DONE)',
        ))

        self.add(Divergence(
            id='D9', title='Fractal: dedup implemented',
            category=Category.IMPLEMENTATION, severity=Severity.HAUTE,
            status=Status.MITIGATED,
            description='Dedup logic implemented; cancel logic relies on entry_allowed gate',
            pine_behavior='Dedup prevents double entry, cancel removes stale signals',
            python_behavior='Dedup prevents consecutive same-direction signals; cancel via entry_allowed',
            module='entries/fractal.py', bug_id='B5',
            fix_effort='2h (done for dedup, cancel via entry_allowed gate)',
        ))

        self.add(Divergence(
            id='D10', title='Boll+SMA: ddof fixed, entry_allowed available',
            category=Category.IMPLEMENTATION, severity=Severity.MOYENNE,
            status=Status.FIXED,
            description='stdev ddof=0 implemented in stdev.py; entry_allowed gate available via filter module',
            pine_behavior='entry_allowed gate + ta.stdev with ddof=0',
            python_behavior='ddof=0 via compute_stdev + entry_allowed gate in filters/entry_allowed.py',
            module='entries/boll_sma.py + indicators/stdev.py', bug_id='B10',
            fix_effort='1h', fix_description='ddof=0 and entry_allowed both implemented (DONE)',
        ))

        self.add(Divergence(
            id='D11', title='RSI Div: 3-layer cancel logic',
            category=Category.IMPLEMENTATION, severity=Severity.BASSE,
            status=Status.MITIGATED,
            description='3-layer cancel: price invalidation, RSI invalidation, RSI threshold guard',
            pine_behavior='Signal cancelled if conditions change before Open[i+1] fill',
            python_behavior='Cancel via: (a) price recovery >0.2%, (b) RSI move >=5 pts against, (c) RSI overbought/oversold threshold',
            impact_pnl_pct=1.0, module='entries/rsi_divergence.py',
            fix_effort='2h (DONE — 3-layer cancel approximating Pine behavior)',
            fix_description='3-layer cancel logic: price invalidation + RSI invalidation + threshold guard',
        ))

        self.add(Divergence(
            id='D12', title='HTF resampling not validated',
            category=Category.RESAMPLING, severity=Severity.MOYENNE,
            status=Status.OPEN,
            description='HTF resampling not validated bar-by-bar against Pine',
            pine_behavior='Server-native HTF bars',
            python_behavior='Resampled from base TF — boundary alignment unverified',
            module='filters/all_filters.py',
            fix_effort='3h', fix_description='Bar-by-bar comparison W/4H Python vs Pine',
        ))

        self.add(Divergence(
            id='D13', title='ORB: Signal inversé',
            category=Category.IMPLEMENTATION, severity=Severity.BLOQUANT,
            status=Status.OPEN,
            description='ORB signals are inverted (close > orb_high instead of close < orbHigh)',
            pine_behavior='close < orbHigh → long (WRONG in Pine V.69.1)',
            python_behavior='Implementation depends on reproduce_pine_bug flag',
            module='entries/orb.py', bug_id='B11',
            fix_effort='30min', fix_description='Fix signal direction in Pine or flag in Python',
        ))

    def add(self, div: Divergence):
        """Add or update a divergence."""
        self.divergences[div.id] = div

    def get(self, div_id: str) -> Optional[Divergence]:
        """Get a divergence by ID."""
        return self.divergences.get(div_id)

    def list_by_severity(self, severity: Optional[Severity] = None) -> List[Divergence]:
        """List divergences, optionally filtered by severity."""
        divs = list(self.divergences.values())
        if severity:
            divs = [d for d in divs if d.severity == severity]
        return sorted(divs, key=lambda d: d.severity.value, reverse=True)

    def list_open(self) -> List[Divergence]:
        """List all open (unfixed) divergences."""
        return [d for d in self.divergences.values() if d.status == Status.OPEN]

    def list_by_category(self, category: Category) -> List[Divergence]:
        """List divergences by category."""
        return [d for d in self.divergences.values() if d.category == category]

    def summary(self) -> Dict:
        """Summary statistics of all divergences."""
        divs = list(self.divergences.values())
        return {
            'total': len(divs),
            'by_severity': {
                'BLOQUANT': len([d for d in divs if d.severity == Severity.BLOQUANT]),
                'HAUTE': len([d for d in divs if d.severity == Severity.HAUTE]),
                'MOYENNE': len([d for d in divs if d.severity == Severity.MOYENNE]),
                'BASSE': len([d for d in divs if d.severity == Severity.BASSE]),
            },
            'by_status': {
                'open': len([d for d in divs if d.status == Status.OPEN]),
                'fixed': len([d for d in divs if d.status == Status.FIXED]),
                'mitigated': len([d for d in divs if d.status == Status.MITIGATED]),
                'irréductible': len([d for d in divs if d.status == Status.IRREDUCTIBLE]),
            },
            'by_category': {
                'data': len([d for d in divs if d.category == Category.DATA]),
                'resampling': len([d for d in divs if d.category == Category.RESAMPLING]),
                'implementation': len([d for d in divs if d.category == Category.IMPLEMENTATION]),
                'structural': len([d for d in divs if d.category == Category.STRUCTURAL]),
            },
            'estimated_total_phantom_trades': sum(
                d.impact_trades for d in divs
                if d.impact_trades and d.status == Status.OPEN
            ),
        }

    def format_report(self) -> str:
        """Format a human-readable divergence report."""
        lines = []
        lines.append("=" * 70)
        lines.append("PINEGUARD DIVERGENCE REPORT")
        lines.append("=" * 70)

        summary = self.summary()
        lines.append(f"\nTotal divergences: {summary['total']}")
        lines.append(f"Open: {summary['by_status']['open']} | "
                     f"Fixed: {summary['by_status']['fixed']} | "
                     f"Irréductible: {summary['by_status']['irréductible']}")
        lines.append(f"Estimated phantom trades (open): "
                     f"~{summary['estimated_total_phantom_trades']}")

        for sev in [Severity.BLOQUANT, Severity.HAUTE, Severity.MOYENNE, Severity.BASSE]:
            divs = self.list_by_severity(sev)
            if divs:
                lines.append(f"\n--- {sev.name} ---")
                for d in divs:
                    status_icon = {
                        Status.OPEN: '⬜', Status.FIXED: '✅',
                        Status.MITIGATED: '🟡', Status.IRREDUCTIBLE: '🔒'
                    }.get(d.status, '?')
                    lines.append(
                        f"  {status_icon} {d.id}: {d.title}"
                        f" [{d.category.value}]"
                        f" — {d.fix_effort or 'N/A'}"
                    )
                    if d.impact_trades:
                        lines.append(f"       Impact: ~{d.impact_trades} trades")

        lines.append("\n" + "=" * 70)
        return '\n'.join(lines)
