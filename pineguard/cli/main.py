"""
PineGuard CLI — Command-line interface for audits and diagnostics.

Commands:
  audit       Run a full coherence audit
  divergences List all known divergences
  compare     Compare Pine vs Python trade lists
  signals     Compare signal series
  validate    Validate a configuration
  demo        Run a demo with sample data
"""

import argparse
import sys
import json
from pathlib import Path


def cmd_divergences(args):
    """List all known divergences."""
    from pineguard.audit.divergence_tracker import DivergenceTracker
    tracker = DivergenceTracker()

    if args.severity:
        from pineguard.audit.divergence_tracker import Severity
        sev_map = {
            'bloquant': Severity.BLOQUANT, 'haute': Severity.HAUTE,
            'moyenne': Severity.MOYENNE, 'basse': Severity.BASSE,
        }
        severity = sev_map.get(args.severity.lower())
        if severity:
            divs = tracker.list_by_severity(severity)
            for d in divs:
                print(f"  {d.id}: {d.title} [{d.status.value}]")
            return

    print(tracker.format_report())


def cmd_validate(args):
    """Validate a configuration."""
    from pineguard.config import PineGuardConfig, validate_config

    if args.config_file:
        with open(args.config_file) as f:
            config = PineGuardConfig.from_json(f.read())
    else:
        config = PineGuardConfig(
            entry_type=args.entry_type or 'Donc+Chikou',
            pmax_length=args.pmax_length or 10,
            pmax_multiplier=args.pmax_mult or 3.0,
        )

    result = validate_config(config)

    print(f"\nConfiguration validation: {'✅ VALID' if result['valid'] else '❌ INVALID'}")
    print(f"Active parameters: {result['active_params']}")

    if result['errors']:
        print("\n❌ Errors:")
        for e in result['errors']:
            print(f"  - {e}")

    if result['warnings']:
        print("\n⚠️ Warnings:")
        for w in result['warnings']:
            print(f"  - {w}")


def cmd_demo(args):
    """Run a demo audit with sample data."""
    from pineguard.utils.data_loader import generate_sample_data
    from pineguard.indicators.pmax import compute_pmax
    from pineguard.entries.donchian_chikou import compute_donchian_chikou_signals
    from pineguard.engine.backtest_engine import BacktestEngine
    from pineguard.metrics.performance import compute_all_metrics
    from pineguard.audit.divergence_tracker import DivergenceTracker
    from pineguard.audit.report_generator import AuditReportGenerator
    from pineguard.config import PineGuardConfig, validate_config

    print("🔬 PineGuard v1.0 — Demo Audit")
    print("« Même signal, même barre, même résultat. »\n")

    # Generate sample data
    print("📊 Generating sample data (1000 H1 bars)...")
    df = generate_sample_data(n_bars=1000, freq='1h', base_price=5000,
                              volatility=0.001, seed=42)

    # Configuration
    config = PineGuardConfig(
        instrument='US500', timeframe='1H',
        entry_type='Donc+Chikou', don_length=20, chikou_shift=26,
        pmax_length=10, pmax_multiplier=3.0, pmax_ma_type='EMA',
    )

    # Validate config
    validation = validate_config(config)
    print(f"⚙️  Config validation: {'✅' if validation['valid'] else '❌'}")
    for w in validation['warnings']:
        print(f"   {w}")

    # Compute PMax
    print("\n📈 Computing PMax...")
    pmax_result = compute_pmax(
        df['High'], df['Low'], df['Close'],
        length=config.pmax_length,
        multiplier=config.pmax_multiplier,
        ma_type=config.pmax_ma_type,
    )
    print(f"   Buy triggers:  {pmax_result['buy_trigger'].sum()}")
    print(f"   Sell triggers: {pmax_result['sell_trigger'].sum()}")

    # Compute entry signals
    print("\n🎯 Computing Donc+Chikou signals...")
    entry_signals = compute_donchian_chikou_signals(
        df['High'], df['Low'], df['Close'],
        don_length=config.don_length,
        chikou_shift=config.chikou_shift,
    )
    print(f"   Long signals:  {entry_signals['don_long'].sum()}")
    print(f"   Short signals: {entry_signals['don_short'].sum()}")

    # Combine: entry signal AND PMax direction
    combined_long = entry_signals['don_long'] & (pmax_result['dir'] == 1)
    combined_short = entry_signals['don_short'] & (pmax_result['dir'] == -1)
    print(f"   Combined long:  {combined_long.sum()}")
    print(f"   Combined short: {combined_short.sum()}")

    # Run backtest
    print("\n🏦 Running backtest (3-phase engine)...")
    engine = BacktestEngine(
        instrument=config.instrument,
        sl_mode=config.sl_mode,
        sl_pct=config.sl_pct,
    )
    from pineguard.indicators.atr import compute_atr
    atr = compute_atr(df['High'], df['Low'], df['Close'], 14)

    result = engine.run(
        df=df,
        entry_long=combined_long,
        entry_short=combined_short,
        atr=atr,
    )

    # Compute metrics
    print(f"\n📊 Performance Metrics:")
    metrics = compute_all_metrics(result['equity'], result['trades'])
    for key, val in metrics.items():
        if key not in ('dd_start', 'dd_end', 'dd_series'):
            print(f"   {key}: {val}")

    # Divergence report
    print("\n")
    tracker = DivergenceTracker()
    summary = tracker.summary()
    print(f"🔍 Known Divergences: {summary['total']}")
    print(f"   BLOQUANT: {summary['by_severity']['BLOQUANT']}")
    print(f"   HAUTE: {summary['by_severity']['HAUTE']}")
    print(f"   MOYENNE: {summary['by_severity']['MOYENNE']}")
    print(f"   BASSE: {summary['by_severity']['BASSE']}")
    print(f"   Estimated phantom trades: ~{summary['estimated_total_phantom_trades']}")

    # Generate report
    print("\n📋 Generating audit report...")
    report_gen = AuditReportGenerator(
        instrument=config.instrument,
        period='Sample 1000 H1 bars',
        entry_type=config.entry_type,
        config=config.to_dict(),
    )
    report_gen.add_divergence_inventory(tracker)
    report = report_gen.generate()

    print("\n" + "=" * 60)
    print("✅ Demo complete! PineGuard is ready for real audits.")
    print("=" * 60)

    return result


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='pineguard',
        description='PineGuard v1.0 — Expert Cohérence Pine↔Python',
        epilog='« Même signal, même barre, même résultat. » 🔬',
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # divergences
    p_div = subparsers.add_parser('divergences', help='List known divergences')
    p_div.add_argument('--severity', choices=['bloquant', 'haute', 'moyenne', 'basse'])

    # validate
    p_val = subparsers.add_parser('validate', help='Validate a configuration')
    p_val.add_argument('--config-file', help='JSON config file')
    p_val.add_argument('--entry-type', help='Entry type')
    p_val.add_argument('--pmax-length', type=int, help='PMax length')
    p_val.add_argument('--pmax-mult', type=float, help='PMax multiplier')

    # demo
    p_demo = subparsers.add_parser('demo', help='Run demo with sample data')

    args = parser.parse_args()

    if args.command == 'divergences':
        cmd_divergences(args)
    elif args.command == 'validate':
        cmd_validate(args)
    elif args.command == 'demo':
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
