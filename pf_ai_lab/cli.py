"""
PF AI Lab 5.0.4 — Interface ligne de commande unifiee.

Commandes:
  web         Lancer le dashboard web (interface graphique)
  backtest    Lancer un backtest complet
  optimize    Optimisation mono/multi-objectif (NSGA-II)
  wfa         Walk-Forward Analysis
  mc          Monte Carlo simulation
  audit       Audit PineGuard (divergences, phantom trades)
  demo        Demo rapide avec donnees sample
  validate    Valider une configuration
  divergences Lister les divergences connues D1-D13
"""

import argparse
import sys
import json
import time


def cmd_backtest(args):
    """Run a full backtest."""
    from pf_ai_lab.runner import run_full_pipeline
    from pineguard.config import PineGuardConfig
    from pineguard.utils.data_loader import load_csv

    config = PineGuardConfig(
        instrument=args.instrument,
        timeframe=args.timeframe,
        entry_type=args.entry_type,
        pmax_length=args.pmax_length,
        pmax_multiplier=args.pmax_mult,
        pmax_ma_type=args.pmax_ma_type,
        sl_mode=args.sl_mode,
        sl_pct=args.sl_pct,
    )

    df = None
    if args.csv:
        df = load_csv(args.csv)

    result = run_full_pipeline(
        df=df, config=config,
        run_audit=not args.no_audit,
        verbose=True,
    )

    if args.output:
        export_data = {
            'metrics': result['metrics'],
            'config': result['config'].to_dict(),
            'trade_count': result['trade_count'],
        }
        if 'audit' in result:
            export_data['audit'] = {
                'confidence': result['audit']['confidence'],
                'phantom_trades': result['audit']['phantom_trades_estimate'],
            }
        with open(args.output, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\nResultats exportes: {args.output}")


def cmd_optimize(args):
    """Run optimization."""
    from pineguard.config import PineGuardConfig
    from pineguard.utils.data_loader import generate_sample_data, load_csv

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = generate_sample_data(n_bars=args.bars, freq='1h', seed=42)

    config = PineGuardConfig(entry_type=args.entry_type)

    if args.multi:
        from pf_ai_lab.optimizer import optimize_multi_objective
        result = optimize_multi_objective(
            df=df, entry_type=args.entry_type,
            n_trials=args.trials, base_config=config,
        )
        print(f"\nNSGA-II Multi-objectif: {len(result.pareto_front)} solutions Pareto")
        for i, p in enumerate(result.pareto_front[:5]):
            m = p.get('metrics', {})
            print(f"  #{i+1}: PF={m.get('profit_factor', 0):.2f} "
                  f"DD={m.get('max_drawdown_pct', 0):.1f}% "
                  f"Sharpe={m.get('sharpe_ratio', 0):.2f}")
    else:
        from pf_ai_lab.optimizer import optimize_single_objective
        result = optimize_single_objective(
            df=df, objective_metric=args.metric,
            entry_type=args.entry_type,
            n_trials=args.trials, base_config=config,
        )
        print(f"\nMeilleur {args.metric}: {result.best_metrics.get(args.metric, 0):.3f}")
        print(f"Params: {json.dumps(result.best_params, indent=2)}")

    if result.warnings:
        print(f"\nAvertissements:")
        for w in result.warnings:
            print(f"  {w}")

    print(f"\nDuree: {result.duration_seconds}s | Trials: {result.n_trials}")


def cmd_wfa(args):
    """Run Walk-Forward Analysis."""
    from pf_ai_lab.walk_forward import run_walk_forward
    from pineguard.config import PineGuardConfig
    from pineguard.utils.data_loader import generate_sample_data, load_csv

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = generate_sample_data(n_bars=args.bars, freq='1h', seed=42)

    config = PineGuardConfig(entry_type=args.entry_type)

    result = run_walk_forward(
        df=df, config=config,
        n_windows=args.windows,
        n_opt_trials=args.trials,
        objective_metric=args.metric,
    )

    verdict = "PASS" if result.is_valid else "FAIL"
    print(f"\nWFA Verdict: {verdict}")
    print(f"WFE moyen: {result.avg_wfe:.3f} (std: {result.std_wfe:.3f})")


def cmd_mc(args):
    """Run Monte Carlo simulation."""
    from pf_ai_lab.runner import run_backtest
    from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle, format_mc_report
    from pineguard.config import PineGuardConfig
    from pineguard.utils.data_loader import generate_sample_data, load_csv

    if args.csv:
        df = load_csv(args.csv)
    else:
        df = generate_sample_data(n_bars=args.bars, freq='1h', seed=42)

    config = PineGuardConfig(entry_type=args.entry_type)
    bt = run_backtest(df=df, config=config, verbose=False)

    if 'error' in bt or not bt['trades']:
        print("Pas de trades pour Monte Carlo.")
        return

    result = monte_carlo_trade_shuffle(
        trades=bt['trades'],
        initial_capital=config.initial_capital,
        n_simulations=args.sims,
    )

    print(f"\n{format_mc_report(result)}")


def cmd_web(args):
    """Launch the original PF AI Lab web dashboard (Flask 7-page interface)."""
    import os
    import sys
    port = getattr(args, 'port', 5000)

    # Find project root (where app.py lives)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(project_root, 'app.py')

    if not os.path.exists(app_path):
        print(f"[ERREUR] app.py introuvable dans {project_root}")
        print("Assurez-vous que le fichier app.py est present a la racine du projet.")
        sys.exit(1)

    # Change to project root so Flask finds templates/
    os.chdir(project_root)

    print("=" * 60)
    print("  PF AI Lab 5.0.4 — Web Dashboard")
    print(f"  http://localhost:{port}")
    print("  Pages: Home | Data | Explore | Validate | Portfolio | Best | DB")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    # Import and run the original Flask app
    # ── Windows connection-refused fix ─────────────────────────────
    # 1. host='127.0.0.1' on Windows to avoid firewall popup
    # 2. threaded=True prevents single-thread blocking when the browser
    #    opens multiple parallel connections (page + favicon + css + js)
    # 3. use_reloader=False avoids double-start on Windows
    sys.path.insert(0, project_root)
    from app import app
    host = '127.0.0.1' if sys.platform == 'win32' else '0.0.0.0'
    app.run(host=host, port=port, debug=False,
            threaded=True, use_reloader=False)


def cmd_demo(args):
    """Run a quick demo."""
    from pf_ai_lab.runner import run_full_pipeline
    from pineguard.config import PineGuardConfig

    print("=" * 60)
    print("PF AI Lab 5.0.4 — Demo")
    print("P.F.Algo V.69.2 | Moteur Python corrige (PineGuard)")
    print("=" * 60)

    config = PineGuardConfig(
        instrument='US500', timeframe='1H',
        entry_type='Donc+Chikou',
        don_length=20, chikou_shift=26,
        pmax_length=10, pmax_multiplier=3.0,
    )

    result = run_full_pipeline(config=config, run_audit=True, verbose=True)

    if 'error' not in result:
        # Quick Monte Carlo
        from pf_ai_lab.monte_carlo import monte_carlo_trade_shuffle, format_mc_report
        if result['trades']:
            mc = monte_carlo_trade_shuffle(
                result['trades'], n_simulations=500,
                initial_capital=config.initial_capital,
            )
            print(f"\n{format_mc_report(mc)}")

    print(f"\n{'='*60}")
    print("Demo terminee. PF AI Lab 5.0.4 operationnel.")
    print(f"{'='*60}")


def cmd_audit(args):
    """Run PineGuard audit."""
    # Delegate to existing PineGuard CLI
    from pineguard.cli.main import cmd_divergences
    cmd_divergences(args)


def cmd_validate(args):
    """Validate a configuration."""
    from pineguard.cli.main import cmd_validate as pg_validate
    pg_validate(args)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='pf-ai-lab',
        description='PF AI Lab 5.0.4 — Moteur Python corrige pour P.F.Algo V.69.2',
        epilog='Corrections D1-D13 integrees | PineGuard audit | NSGA-II optimizer',
    )
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')

    # Common arguments
    def add_common_args(p):
        p.add_argument('--csv', help='Chemin fichier CSV OHLCV')
        p.add_argument('--entry-type', default='Donc+Chikou',
                        choices=['Donc+Chikou', 'RSI Divergence', 'Boll+SMA',
                                 'Fractal', 'VWAP', 'ORB', 'MM Cross'])
        p.add_argument('--bars', type=int, default=2000,
                        help='Barres sample si pas de CSV')

    # backtest
    p_bt = subparsers.add_parser('backtest', help='Lancer un backtest complet')
    add_common_args(p_bt)
    p_bt.add_argument('--instrument', default='US500')
    p_bt.add_argument('--timeframe', default='1H')
    p_bt.add_argument('--pmax-length', type=int, default=10)
    p_bt.add_argument('--pmax-mult', type=float, default=3.0)
    p_bt.add_argument('--pmax-ma-type', default='EMA')
    p_bt.add_argument('--sl-mode', default='atr14',
                       choices=['atr14', 'atr50', 'fixed_pct', 'none'])
    p_bt.add_argument('--sl-pct', type=float, default=5.0)
    p_bt.add_argument('--no-audit', action='store_true')
    p_bt.add_argument('--output', help='Exporter resultats en JSON')

    # optimize
    p_opt = subparsers.add_parser('optimize', help='Optimisation NSGA-II')
    add_common_args(p_opt)
    p_opt.add_argument('--trials', type=int, default=100)
    p_opt.add_argument('--metric', default='sharpe_ratio')
    p_opt.add_argument('--multi', action='store_true',
                        help='Multi-objectif NSGA-II')

    # wfa
    p_wfa = subparsers.add_parser('wfa', help='Walk-Forward Analysis')
    add_common_args(p_wfa)
    p_wfa.add_argument('--windows', type=int, default=5)
    p_wfa.add_argument('--trials', type=int, default=50)
    p_wfa.add_argument('--metric', default='sharpe_ratio')

    # mc
    p_mc = subparsers.add_parser('mc', help='Monte Carlo simulation')
    add_common_args(p_mc)
    p_mc.add_argument('--sims', type=int, default=1000)

    # web
    p_web = subparsers.add_parser('web', help='Lancer le dashboard web')
    p_web.add_argument('--port', type=int, default=5000, help='Port HTTP (default 5000)')
    p_web.add_argument('--no-browser', action='store_true', help='Ne pas ouvrir le navigateur')

    # demo
    p_demo = subparsers.add_parser('demo', help='Demo rapide')

    # audit / divergences
    p_audit = subparsers.add_parser('audit', help='Audit PineGuard')
    p_audit.add_argument('--severity',
                          choices=['bloquant', 'haute', 'moyenne', 'basse'])

    # validate
    p_val = subparsers.add_parser('validate', help='Valider une configuration')
    p_val.add_argument('--config-file', help='Fichier JSON config')
    p_val.add_argument('--entry-type', help='Type entree')
    p_val.add_argument('--pmax-length', type=int)
    p_val.add_argument('--pmax-mult', type=float)

    args = parser.parse_args()

    commands = {
        'web': cmd_web,
        'backtest': cmd_backtest,
        'optimize': cmd_optimize,
        'wfa': cmd_wfa,
        'mc': cmd_mc,
        'demo': cmd_demo,
        'audit': cmd_audit,
        'validate': cmd_validate,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
