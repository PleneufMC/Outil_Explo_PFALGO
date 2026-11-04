#!/usr/bin/env python3
"""
PF AI Lab 5.4 — Database Recalculator (C7)
Recomputes WFE (floor/cap) for all existing configs and updates DB in-place.
Optionally recalculates quality scores.

Usage:
    python recalculate_db.py                  # Dry-run (show changes)
    python recalculate_db.py --apply          # Apply changes
    python recalculate_db.py --apply --stats  # Apply + show summary stats
"""

import argparse
import json
import math
import os
import sqlite3
import sys


DB_DEFAULT = os.path.join(os.path.dirname(__file__), 'data', 'backtests.db')


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        fval = float(val)
        if math.isnan(fval) or math.isinf(fval):
            return default
        return fval
    except (TypeError, ValueError, OverflowError):
        return default


def recalculate_wfe(db_path: str, apply: bool = False, verbose: bool = True):
    """Recalculate WFE for all configs using V5.4 rules.

    V5.4 WFE rules:
      - is_calmar < 0.10 => WFE = NULL (unreliable)
      - oos_calmar <= 0   => WFE = 0.0
      - otherwise: WFE = min(oos_calmar / is_calmar, 3.0)  (cap at 300%)
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return {'error': 'DB not found'}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT id, is_calmar, oos_calmar, wfe
        FROM configs
    """).fetchall()

    changes = []
    null_count = 0
    zero_count = 0
    capped_count = 0
    unchanged_count = 0

    for r in rows:
        config_id = r['id']
        is_calmar = _safe_float(r['is_calmar'], 0) or 0
        oos_calmar = _safe_float(r['oos_calmar'], 0) or 0
        old_wfe = _safe_float(r['wfe'])

        # V5.4 WFE calculation
        if is_calmar < 0.10:
            new_wfe = None  # NULL
            null_count += 1
        elif oos_calmar <= 0:
            new_wfe = 0.0
            zero_count += 1
        else:
            new_wfe = min(oos_calmar / is_calmar, 3.0)
            if oos_calmar / is_calmar > 3.0:
                capped_count += 1

        # Compare
        wfe_changed = False
        if new_wfe is None and old_wfe is not None:
            wfe_changed = True
        elif new_wfe is not None and old_wfe is None:
            wfe_changed = True
        elif new_wfe is not None and old_wfe is not None:
            wfe_changed = abs(new_wfe - old_wfe) > 0.001
        # else: both None -> unchanged

        if wfe_changed:
            changes.append({
                'id': config_id,
                'old_wfe': old_wfe,
                'new_wfe': new_wfe,
                'is_calmar': is_calmar,
                'oos_calmar': oos_calmar,
            })
        else:
            unchanged_count += 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"PF AI Lab 5.4 — WFE Recalculation")
        print(f"{'='*60}")
        print(f"Database: {db_path}")
        print(f"Total configs: {len(rows)}")
        print(f"Changes needed: {len(changes)}")
        print(f"  - Set to NULL (is_calmar < 0.10): {null_count}")
        print(f"  - Set to 0.0 (oos_calmar <= 0):   {zero_count}")
        print(f"  - Capped at 3.0:                  {capped_count}")
        print(f"  - Unchanged:                      {unchanged_count}")
        print()

        if changes and len(changes) <= 20:
            print("Changes preview:")
            for c in changes:
                old = f"{c['old_wfe']:.4f}" if c['old_wfe'] is not None else 'NULL'
                new = f"{c['new_wfe']:.4f}" if c['new_wfe'] is not None else 'NULL'
                print(f"  Config #{c['id']}: WFE {old} -> {new} "
                      f"(IS Calmar={c['is_calmar']:.3f}, OOS Calmar={c['oos_calmar']:.3f})")
            print()
        elif changes:
            print(f"(Showing first 10 of {len(changes)} changes)")
            for c in changes[:10]:
                old = f"{c['old_wfe']:.4f}" if c['old_wfe'] is not None else 'NULL'
                new = f"{c['new_wfe']:.4f}" if c['new_wfe'] is not None else 'NULL'
                print(f"  Config #{c['id']}: WFE {old} -> {new}")
            print(f"  ... and {len(changes) - 10} more")
            print()

    if apply and changes:
        for c in changes:
            conn.execute(
                "UPDATE configs SET wfe = ? WHERE id = ?",
                (c['new_wfe'], c['id']))
        conn.commit()
        if verbose:
            print(f"APPLIED: {len(changes)} configs updated.")
    elif not apply and changes:
        if verbose:
            print("DRY-RUN: No changes applied. Use --apply to update the database.")

    conn.close()

    return {
        'total': len(rows),
        'changes': len(changes),
        'null_count': null_count,
        'zero_count': zero_count,
        'capped_count': capped_count,
        'unchanged': unchanged_count,
        'applied': apply,
    }


def show_stats(db_path: str):
    """Show summary statistics of the database after recalculation."""
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total = conn.execute("SELECT COUNT(*) FROM configs").fetchone()[0]
    null_wfe = conn.execute("SELECT COUNT(*) FROM configs WHERE wfe IS NULL").fetchone()[0]
    zero_wfe = conn.execute("SELECT COUNT(*) FROM configs WHERE wfe = 0").fetchone()[0]
    capped = conn.execute("SELECT COUNT(*) FROM configs WHERE wfe >= 2.99").fetchone()[0]
    oos_pass = conn.execute("SELECT COUNT(*) FROM configs WHERE oos_gate_pass = 1").fetchone()[0]

    print(f"\nDatabase Statistics:")
    print(f"  Total configs:    {total}")
    print(f"  OOS Gate PASS:    {oos_pass}")
    print(f"  WFE = NULL:       {null_wfe}")
    print(f"  WFE = 0:          {zero_wfe}")
    print(f"  WFE capped (~3):  {capped}")
    print(f"  WFE valid:        {total - null_wfe}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description='PF AI Lab 5.4 — Recalculate WFE values in database')
    parser.add_argument('--db', default=DB_DEFAULT,
                        help=f'Database path (default: {DB_DEFAULT})')
    parser.add_argument('--apply', action='store_true',
                        help='Apply changes (default is dry-run)')
    parser.add_argument('--stats', action='store_true',
                        help='Show database stats after recalculation')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    result = recalculate_wfe(
        db_path=args.db,
        apply=args.apply,
        verbose=not args.quiet,
    )

    if args.stats:
        show_stats(args.db)

    # Exit code: 0 = no changes needed, 1 = changes needed (dry-run)
    if not args.apply and result.get('changes', 0) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
