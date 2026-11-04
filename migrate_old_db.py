#!/usr/bin/env python3
"""
PF AI Lab 5.0.4 — Migration de base de donnees

Ce script permet de recuperer les runs et configs d'une ancienne
installation PF AI Lab (version precedente) en:

  1. Lisant directement un fichier backtests.db existant (SQLite)
  2. Exportant son contenu au format JSON portable (pf_ai_lab_db_v1)
  3. Le JSON peut ensuite etre importe dans la nouvelle version via
     la page Database > Import JSON

Usage:
  # Exporter l'ancienne base vers JSON
  python migrate_old_db.py export /chemin/vers/ancien/data/backtests.db

  # Exporter vers un fichier specifique
  python migrate_old_db.py export /chemin/vers/backtests.db -o mon_backup.json

  # Importer un JSON dans la base actuelle
  python migrate_old_db.py import mon_backup.json

  # Afficher les infos de la base actuelle
  python migrate_old_db.py info

  # Afficher les infos d'une base quelconque
  python migrate_old_db.py info /chemin/vers/backtests.db

Compatibilite:
  Fonctionne avec n'importe quelle version de backtests.db qui a les
  tables 'runs' et 'configs'. Les colonnes manquantes sont remplies
  par NULL (le schema 5.0.4 est un sur-ensemble des versions precedentes).
"""

import os
import sys
import json
import sqlite3
import argparse
from datetime import datetime, timezone


def get_db_info(db_path: str) -> dict:
    """Lit les informations d'une base backtests.db quelconque."""
    if not os.path.exists(db_path):
        return {'error': f'Fichier introuvable: {db_path}'}

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        info = {'db_path': db_path, 'size_kb': round(os.path.getsize(db_path) / 1024, 1)}

        # Check tables
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        info['tables'] = tables

        if 'runs' not in tables or 'configs' not in tables:
            info['error'] = f"Tables manquantes. Tables trouvees: {tables}"
            conn.close()
            return info

        # Runs
        info['total_runs'] = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

        # Configs
        info['total_configs'] = conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0]

        # Instruments
        instruments = [r[0] for r in conn.execute(
            "SELECT DISTINCT instrument FROM runs ORDER BY instrument").fetchall()]
        info['instruments'] = instruments

        # OOS pass count (colonne peut ne pas exister dans les vieilles versions)
        try:
            info['oos_pass'] = conn.execute(
                "SELECT COUNT(*) FROM configs WHERE oos_gate_pass = 1").fetchone()[0]
        except sqlite3.OperationalError:
            info['oos_pass'] = '(colonne absente)'

        # Schema configs — lister les colonnes
        cols_info = conn.execute("PRAGMA table_info(configs)").fetchall()
        info['config_columns'] = [c[1] for c in cols_info]
        info['config_column_count'] = len(info['config_columns'])

        # Schema runs — lister les colonnes
        run_cols = conn.execute("PRAGMA table_info(runs)").fetchall()
        info['run_columns'] = [c[1] for c in run_cols]

        # Dates
        try:
            first = conn.execute("SELECT MIN(created_at) FROM runs").fetchone()[0]
            last = conn.execute("SELECT MAX(created_at) FROM runs").fetchone()[0]
            info['date_range'] = f"{first} -> {last}"
        except Exception:
            info['date_range'] = '?'

        # Runs detail
        runs = conn.execute("""
            SELECT r.id, r.instrument, r.mode, r.n_trials, r.created_at, r.n_configs
            FROM runs r ORDER BY r.id
        """).fetchall()
        info['runs_detail'] = [dict(r) for r in runs]

        conn.close()
        return info

    except Exception as e:
        return {'error': str(e), 'db_path': db_path}


def export_db_to_json(db_path: str) -> dict:
    """
    Lit une base backtests.db (n'importe quelle version) et produit
    un dict au format pf_ai_lab_db_v1, compatible avec import_full_json().

    Tolerant aux colonnes manquantes: les colonnes absentes dans l'ancienne
    version seront simplement NULL dans le JSON.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Base introuvable: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Decouvrir les colonnes reelles des tables
    run_cols_actual = [c[1] for c in conn.execute("PRAGMA table_info(runs)").fetchall()]
    cfg_cols_actual = [c[1] for c in conn.execute("PRAGMA table_info(configs)").fetchall()]

    # Lire tous les runs
    runs = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    result_runs = []

    for run in runs:
        run_d = dict(run)
        run_id = run_d['id']

        # Lire les configs de ce run
        configs = conn.execute(
            "SELECT * FROM configs WHERE run_id = ? ORDER BY rank",
            (run_id,)).fetchall()
        run_d['configs'] = [dict(c) for c in configs]
        result_runs.append(run_d)

    conn.close()

    return {
        'format': 'pf_ai_lab_db_v1',
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'source_db': db_path,
        'source_run_columns': run_cols_actual,
        'source_config_columns': cfg_cols_actual,
        'runs': result_runs,
    }


def import_json_to_db(json_path: str, db_path: str = None):
    """
    Importe un fichier JSON (pf_ai_lab_db_v1) dans la base actuelle.
    Utilise BacktestDB.import_full_json() pour la deduplication.
    """
    # Importer BacktestDB de la version actuelle
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pf_ma_optimizer.backtest_db import BacktestDB

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    db = BacktestDB(db_path=db_path) if db_path else BacktestDB()
    result = db.import_full_json(data)
    return result


def main():
    parser = argparse.ArgumentParser(
        description='PF AI Lab 5.0.4 — Migration de base de donnees',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Voir les infos de l'ancienne base
  python migrate_old_db.py info /chemin/ancien/data/backtests.db

  # Exporter l'ancienne base vers JSON
  python migrate_old_db.py export /chemin/ancien/data/backtests.db

  # Importer le JSON dans la nouvelle installation
  python migrate_old_db.py import pf_ai_lab_db_20260210.json

  # Copie directe (si meme schema)
  cp /chemin/ancien/data/backtests.db data/backtests.db
        """)

    subparsers = parser.add_subparsers(dest='command', help='Commande')

    # info
    p_info = subparsers.add_parser('info', help='Afficher les infos de la base')
    p_info.add_argument('db_path', nargs='?', default=None,
                        help='Chemin vers backtests.db (defaut: data/backtests.db)')

    # export
    p_export = subparsers.add_parser('export', help='Exporter la base en JSON')
    p_export.add_argument('db_path', help='Chemin vers backtests.db a exporter')
    p_export.add_argument('-o', '--output', default=None,
                          help='Fichier de sortie (defaut: auto-genere)')

    # import
    p_import = subparsers.add_parser('import', help='Importer un JSON dans la base')
    p_import.add_argument('json_path', help='Fichier JSON a importer')
    p_import.add_argument('--db', default=None,
                          help='Chemin vers backtests.db cible (defaut: data/backtests.db)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # ── INFO ──
    if args.command == 'info':
        db_path = args.db_path
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'backtests.db')

        info = get_db_info(db_path)

        if 'error' in info:
            print(f"ERREUR: {info['error']}")
            sys.exit(1)

        print(f"{'=' * 60}")
        print(f"  PF AI Lab — Database Info")
        print(f"{'=' * 60}")
        print(f"  Fichier:      {info['db_path']}")
        print(f"  Taille:       {info['size_kb']} KB")
        print(f"  Tables:       {', '.join(info['tables'])}")
        print(f"  Colonnes run: {len(info['run_columns'])}")
        print(f"  Colonnes cfg: {info['config_column_count']}")
        print(f"  Runs:         {info['total_runs']}")
        print(f"  Configs:      {info['total_configs']}")
        print(f"  OOS PASS:     {info['oos_pass']}")
        print(f"  Instruments:  {', '.join(info['instruments']) or '(aucun)'}")
        print(f"  Periode:      {info['date_range']}")

        if info.get('runs_detail'):
            print(f"\n  {'─' * 56}")
            print(f"  Runs:")
            for r in info['runs_detail']:
                dt = (r['created_at'] or '?')[:16]
                print(f"    #{r['id']:>3d} | {r['instrument']:<10s} | {r['mode']:<8s} | "
                      f"{r['n_trials']:>4d} trials | {r['n_configs']:>2d} cfgs | {dt}")
        print()

    # ── EXPORT ──
    elif args.command == 'export':
        db_path = args.db_path
        if not os.path.exists(db_path):
            print(f"ERREUR: Fichier introuvable: {db_path}")
            sys.exit(1)

        # Afficher les infos d'abord
        info = get_db_info(db_path)
        if 'error' in info:
            print(f"ERREUR: {info['error']}")
            sys.exit(1)

        print(f"Base source: {db_path}")
        print(f"  {info['total_runs']} runs, {info['total_configs']} configs")
        print(f"  Instruments: {', '.join(info['instruments'])}")
        print(f"  Colonnes configs: {info['config_column_count']}")

        # Exporter
        data = export_db_to_json(db_path)

        # Fichier de sortie
        output = args.output
        if output is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M')
            output = f'pf_ai_lab_db_{ts}.json'

        with open(output, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        size_kb = round(os.path.getsize(output) / 1024, 1)
        print(f"\nExport termine: {output} ({size_kb} KB)")
        print(f"  {len(data['runs'])} runs exportes")
        total_cfgs = sum(len(r.get('configs', [])) for r in data['runs'])
        print(f"  {total_cfgs} configs exportees")
        print(f"\nPour importer dans la nouvelle version:")
        print(f"  python migrate_old_db.py import {output}")
        print(f"  ou: page Database > Import JSON > selectionner {output}")

    # ── IMPORT ──
    elif args.command == 'import':
        json_path = args.json_path
        if not os.path.exists(json_path):
            print(f"ERREUR: Fichier introuvable: {json_path}")
            sys.exit(1)

        print(f"Import de {json_path}...")
        result = import_json_to_db(json_path, db_path=args.db)

        print(f"\nResultat:")
        print(f"  Runs importes:    {result['imported_runs']}")
        print(f"  Configs importees:{result['imported_configs']}")
        print(f"  Doublons ignores: {result['skipped_runs']}")
        if result.get('errors'):
            print(f"  Erreurs:")
            for e in result['errors']:
                print(f"    - {e}")

        if result['imported_runs'] > 0:
            print(f"\nImport reussi! Lancez le dashboard pour verifier:")
            print(f"  python launch.py")
        elif result['skipped_runs'] > 0:
            print(f"\nTous les runs existaient deja (doublons ignores).")
        else:
            print(f"\nAucun run a importer.")


if __name__ == '__main__':
    main()
