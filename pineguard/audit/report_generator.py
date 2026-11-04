"""
Audit Report Generator — Structured output for PineGuard audits.

Generates comprehensive reports following the 5-step audit protocol:
  1. ISOLER le composant
  2. QUANTIFIER l'écart
  3. DIAGNOSTIQUER la cause
  4. ÉVALUER la criticité
  5. PROPOSER le fix
"""

import json
from datetime import datetime
from typing import Dict, List, Optional


class AuditReportGenerator:
    """
    Generate structured audit reports.

    Reports include:
      - Executive summary
      - Component-by-component analysis
      - Divergence inventory
      - Prioritized fix recommendations
      - Confidence scores
    """

    def __init__(self, instrument: str = '', period: str = '',
                 entry_type: str = '', config: Optional[Dict] = None):
        self.instrument = instrument
        self.period = period
        self.entry_type = entry_type
        self.config = config or {}
        self.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.sections: List[Dict] = []

    def add_section(self, title: str, content: str,
                    severity: str = '', component: str = ''):
        """Add a section to the report."""
        self.sections.append({
            'title': title,
            'content': content,
            'severity': severity,
            'component': component,
        })

    def add_signal_comparison(self, signal_result: Dict):
        """Add signal comparison results to the report."""
        s = signal_result.get('summary', {})
        content = (
            f"Signal: {signal_result.get('signal_name', 'unknown')}\n"
            f"  Pine signals:    {s.get('pine_signals', 'N/A')}\n"
            f"  Python signals:  {s.get('python_signals', 'N/A')}\n"
            f"  Matched (exact): {s.get('matched_exact', 'N/A')}\n"
            f"  Python-only:     {s.get('python_only_count', 'N/A')}\n"
            f"  Pine-only:       {s.get('pine_only_count', 'N/A')}\n"
            f"  Match rate:      {s.get('match_rate_exact', 'N/A')}%"
        )
        self.add_section(
            title=f"Signal: {signal_result.get('signal_name', '')}",
            content=content,
            component='signals',
        )

    def add_trade_comparison(self, trade_result: Dict):
        """Add trade comparison results to the report."""
        s = trade_result.get('summary', {})
        content = (
            f"Pine trades:   {s.get('pine_trades', 'N/A')}\n"
            f"Python trades: {s.get('python_trades', 'N/A')}\n"
            f"Matched:       {s.get('matched', 'N/A')}\n"
            f"Phantom:       {s.get('python_only_phantom', 'N/A')}\n"
            f"Missed:        {s.get('pine_only_missed', 'N/A')}\n"
            f"Match rate:    {s.get('match_rate_pct', 'N/A')}%\n"
            f"Avg PnL delta: {s.get('avg_pnl_delta_pct', 'N/A')}%\n"
            f"Severity:      {s.get('severity', 'N/A')}"
        )
        self.add_section(
            title='Trade Comparison',
            content=content,
            severity=s.get('severity', ''),
            component='trades',
        )

    def add_divergence_inventory(self, tracker):
        """Add divergence tracker summary."""
        content = tracker.format_report()
        self.add_section(
            title='Divergence Inventory',
            content=content,
            component='divergences',
        )

    def add_recommendation(self, title: str, description: str,
                          effort: str, impact: str, priority: str):
        """Add a prioritized fix recommendation."""
        content = (
            f"Fix: {title}\n"
            f"  Description: {description}\n"
            f"  Effort: {effort}\n"
            f"  Impact: {impact}\n"
            f"  Priority: {priority}"
        )
        self.add_section(
            title=f"Recommendation: {title}",
            content=content,
            severity=priority,
            component='recommendations',
        )

    def generate(self) -> str:
        """Generate the complete audit report."""
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append("PINEGUARD v1.0 — AUDIT REPORT")
        lines.append("« Même signal, même barre, même résultat. »")
        lines.append("=" * 70)
        lines.append(f"Date:       {self.timestamp}")
        lines.append(f"Instrument: {self.instrument or 'N/A'}")
        lines.append(f"Period:     {self.period or 'N/A'}")
        lines.append(f"Entry Type: {self.entry_type or 'N/A'}")

        if self.config:
            lines.append(f"\nConfiguration:")
            for k, v in self.config.items():
                lines.append(f"  {k}: {v}")

        # Sections
        for section in self.sections:
            lines.append(f"\n{'─' * 60}")
            sev = f" [{section['severity']}]" if section['severity'] else ''
            lines.append(f"▸ {section['title']}{sev}")
            lines.append(f"{'─' * 60}")
            lines.append(section['content'])

        # Footer
        lines.append(f"\n{'=' * 70}")
        lines.append("PineGuard v1.0 — P.F.Algo V.69.1 + PF AI Lab 5.0")
        lines.append(f"Report generated: {self.timestamp}")
        lines.append("=" * 70)

        return '\n'.join(lines)

    def generate_json(self) -> Dict:
        """Generate the report as a structured JSON-serializable dict."""
        return {
            'pineguard_version': '1.0',
            'algo_version': 'V.69.1',
            'lab_version': '5.0.3',
            'timestamp': self.timestamp,
            'instrument': self.instrument,
            'period': self.period,
            'entry_type': self.entry_type,
            'config': self.config,
            'sections': self.sections,
        }
