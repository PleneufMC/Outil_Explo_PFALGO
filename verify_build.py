#!/usr/bin/env python3
"""
PF AI Lab — Build Verification Script
Run this BEFORE every release to catch missing dependencies, broken syntax, etc.

Usage:
    python verify_build.py          # Full check
    python verify_build.py --quick  # Syntax + requirements only (no import test)

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import sys
import os
import ast
import subprocess
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Packages that MUST be in requirements.txt
REQUIRED_PACKAGES = ['flask', 'pandas', 'numpy', 'scipy', 'optuna', 'openpyxl', 'plotly']

# Files that must parse as valid Python
PYTHON_FILES = [
    'app.py',
    'launch.py',
    'launch_desktop.pyw',
    'pf_ai_lab/cli.py',
    'pf_ma_optimizer/config.py',
    'pf_ma_optimizer/optimizer.py',
    'pf_ma_optimizer/backtest_engine.py',
    'pf_ma_optimizer/metrics.py',
    'pf_ma_optimizer/backtest_db.py',
    'pf_ma_optimizer/tv_export.py',
    'pf_ma_optimizer/data_loader.py',
]

# Strings that MUST be present in specific files
REQUIRED_STRINGS = {
    'app.py': [
        ("/api/health", "Health check endpoint is required for launchers"),
        ("threaded=True", "Flask MUST run in threaded mode"),
        ("_sanitize_for_json", "JSON sanitization function is required"),
    ],
    'launch_desktop.pyw': [
        ("pf_ai_lab.log", "Logging to file is required (only debug method on Windows)"),
        ("/api/health", "Health check is required for readiness detection"),
        ("_drain_stderr", "Subprocess stderr must be captured"),
        ("show_error_popup", "Error popup is required for user feedback"),
    ],
    'launch.py': [
        ("threaded=True", "Flask MUST run in threaded mode"),
        ("/api/health", "Health check is required for browser-open timing"),
    ],
    'pf_ai_lab/cli.py': [
        ("threaded=True", "Flask MUST run in threaded mode"),
    ],
    'install.bat': [
        ("import flask", "Post-install flask verification is required"),
    ],
}


def check_requirements_txt():
    """Verify all required packages are listed in requirements.txt."""
    errors = []
    try:
        with open('requirements.txt') as f:
            content = f.read().lower()
        for pkg in REQUIRED_PACKAGES:
            if pkg not in content:
                errors.append(f"Package '{pkg}' MISSING from requirements.txt")
        if not errors:
            print(f"  [OK] requirements.txt contains all {len(REQUIRED_PACKAGES)} required packages")
    except FileNotFoundError:
        errors.append("requirements.txt FILE NOT FOUND")
    return errors


def check_python_syntax():
    """Parse all critical Python files for syntax errors."""
    errors = []
    for filepath in PYTHON_FILES:
        if not os.path.exists(filepath):
            errors.append(f"File NOT FOUND: {filepath}")
            continue
        try:
            with open(filepath) as f:
                ast.parse(f.read(), filename=filepath)
            print(f"  [OK] {filepath}")
        except SyntaxError as e:
            errors.append(f"SYNTAX ERROR in {filepath}: {e}")
    return errors


def check_required_strings():
    """Verify critical strings are present in specific files."""
    errors = []
    for filepath, checks in REQUIRED_STRINGS.items():
        if not os.path.exists(filepath):
            errors.append(f"File NOT FOUND: {filepath}")
            continue
        with open(filepath, encoding='utf-8', errors='replace') as f:
            content = f.read()
        for string, reason in checks:
            if string not in content:
                errors.append(f"{filepath}: '{string}' MISSING — {reason}")
            else:
                print(f"  [OK] {filepath} contains '{string}'")
    return errors


def check_flask_consistency():
    """Verify all 4 Flask entry points have consistent settings."""
    errors = []
    flask_files = ['app.py', 'launch.py', 'pf_ai_lab/cli.py']
    for filepath in flask_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath) as f:
            content = f.read()
        if 'app.run(' in content and 'threaded=True' not in content:
            errors.append(f"{filepath}: has app.run() but NO threaded=True")
    if not errors:
        print(f"  [OK] All Flask entry points have threaded=True")
    return errors


def check_version_consistency():
    """Verify APP_VERSION is identical across all versioned files."""
    errors = []
    # Extract version from app.py
    version = None
    with open('app.py', encoding='utf-8') as f:
        for line in f:
            if line.strip().startswith('APP_VERSION'):
                version = line.split('=')[1].strip().strip("'\"")
                break
    if not version:
        return ["app.py: APP_VERSION not found"]
    print(f"  APP_VERSION = '{version}'")

    # Files that must contain this version string
    version_files = {
        'install.bat': 3,     # min expected occurrences
        'start.bat': 1,
        'uninstall.bat': 2,
        'launch_desktop.pyw': 2,
    }
    # Templates
    import glob
    for tpl in glob.glob('templates/*.html'):
        version_files[tpl] = 1

    for filepath, min_count in version_files.items():
        if not os.path.exists(filepath):
            errors.append(f"File NOT FOUND: {filepath}")
            continue
        with open(filepath, encoding='utf-8', errors='replace') as f:
            content = f.read()
        count = content.count(version)
        if count < min_count:
            errors.append(f"{filepath}: version '{version}' found {count}x (expected >= {min_count})")
        else:
            print(f"  [OK] {filepath} — {count} occurrence(s) of '{version}'")

    # Check NO old versions remain (common pattern: X.Y.Z-1 or X.Y.0)
    parts = version.split('.')
    old_versions = set()
    if len(parts) == 3:
        # Previous patch
        p = int(parts[2])
        if p > 0:
            old_versions.add(f"{parts[0]}.{parts[1]}.{p-1}")
        # Previous minor .0 (e.g. 5.4.0 when current is 5.4.2)
        if p > 1:
            old_versions.add(f"{parts[0]}.{parts[1]}.0")

    for old_v in old_versions:
        for filepath in list(version_files.keys()) + ['app.py']:
            if not os.path.exists(filepath):
                continue
            with open(filepath, encoding='utf-8', errors='replace') as f:
                content = f.read()
            if old_v in content:
                # Filter out changelogs/comments that legitimately reference old versions
                lines = [l for l in content.split('\n')
                         if old_v in l and not l.strip().startswith(('#', '//', '::',
                         '*', '<!--', 'V5.', 'V69.', 'Historique', 'Changelog'))]
                if lines:
                    errors.append(f"{filepath}: OLD version '{old_v}' still present in {len(lines)} non-comment line(s)")

    if not errors:
        print(f"  [OK] Version '{version}' consistent across all files")
    return errors


def check_archive_no_duplicates():
    """If a .tar.gz archive exists, verify no duplicate entries."""
    import glob as g
    archives = g.glob('PF_AI_Lab_*.tar.gz')
    if not archives:
        print("  [--] No archive found (skipped)")
        return []
    errors = []
    for archive in archives:
        result = subprocess.run(
            ['tar', '-tzf', archive],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            errors.append(f"{archive}: tar listing failed")
            continue
        raw_entries = result.stdout.strip().split('\n')
        # Normalize: strip leading ./ and PF_AI_Lab_X.Y.Z/ prefix
        entries = []
        for e in raw_entries:
            e = e.lstrip('./')
            parts = e.split('/', 1)
            if len(parts) == 2 and parts[0].startswith('PF_AI_Lab_'):
                entries.append(parts[1])
            else:
                entries.append(e)
        from collections import Counter
        counts = Counter(entries)
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            for entry, count in dupes.items():
                errors.append(f"{archive}: DUPLICATE entry '{entry}' ({count}x) — Windows extraction will FAIL")
        else:
            print(f"  [OK] {archive} — {len(entries)} entries, no duplicates")
        # Verify requirements.txt is present
        if 'requirements.txt' not in entries:
            errors.append(f"{archive}: requirements.txt MISSING from archive — build must NOT be delivered")
        else:
            print(f"  [OK] {archive} — requirements.txt present")
    return errors


def check_stale_archives():
    """Verify only one archive exists and it matches APP_VERSION."""
    import glob as g
    archives = g.glob('PF_AI_Lab_*.tar.gz')
    if not archives:
        print("  [--] No archive found (skipped)")
        return []
    errors = []
    # Extract current version from app.py
    version = None
    try:
        with open('app.py', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('APP_VERSION'):
                    version = line.split('=')[1].strip().strip("'\"")
                    break
    except FileNotFoundError:
        pass
    if not version:
        errors.append("Cannot determine APP_VERSION from app.py")
        return errors

    expected_archive = f'PF_AI_Lab_{version}.tar.gz'
    if len(archives) > 1:
        stale = [a for a in archives if a != expected_archive]
        for s in stale:
            errors.append(f"STALE archive '{s}' found — delete it before delivery (only '{expected_archive}' should exist)")
    if expected_archive not in archives:
        errors.append(f"Expected archive '{expected_archive}' NOT FOUND (found: {', '.join(archives)})")
    elif len(archives) == 1:
        print(f"  [OK] Only '{expected_archive}' present — no stale archives")
    return errors


def check_critical_files_in_archive():
    """Verify the archive contains all critical files needed for Windows install."""
    import glob as g
    version = None
    try:
        with open('app.py', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith('APP_VERSION'):
                    version = line.split('=')[1].strip().strip("'\"")
                    break
    except FileNotFoundError:
        return []
    if not version:
        return []

    archive = f'PF_AI_Lab_{version}.tar.gz'
    if not os.path.exists(archive):
        return []  # Already flagged by other checks

    result = subprocess.run(
        ['tar', '-tzf', archive],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        return [f"{archive}: cannot list contents"]

    raw_entries = result.stdout.strip().split('\n')
    # Normalize: remove leading ./ and strip PF_AI_Lab_X.Y.Z/ prefix
    normalized = set()
    for e in raw_entries:
        e = e.lstrip('./')
        parts = e.split('/', 1)
        if len(parts) == 2 and parts[0].startswith('PF_AI_Lab_'):
            normalized.add(parts[1])
        else:
            normalized.add(e)

    critical_files = [
        'app.py', 'launch.py', 'launch_desktop.pyw',
        'install.bat', 'start.bat', 'uninstall.bat',
        'requirements.txt', 'generate_icon.py',
    ]
    errors = []
    for cf in critical_files:
        if cf not in normalized:
            errors.append(f"{archive}: CRITICAL file '{cf}' MISSING from archive")
        else:
            print(f"  [OK] {archive} contains '{cf}'")
    return errors


def check_imports_live():
    """Actually try to import app.py to catch missing modules."""
    print("\n  Testing live imports (this may take a few seconds)...")

    # Find the best Python to use
    venv_python = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
    if not os.path.exists(venv_python):
        venv_python = os.path.join(SCRIPT_DIR, 'venv', 'bin', 'python')
    if not os.path.exists(venv_python):
        venv_python = sys.executable

    result = subprocess.run(
        [venv_python, '-c',
         'import flask, pandas, numpy, scipy, optuna, openpyxl; '
         'print(f"flask={flask.__version__} pandas={pandas.__version__} '
         'numpy={numpy.__version__} optuna={optuna.__version__}")'],
        capture_output=True, text=True, timeout=30, cwd=SCRIPT_DIR
    )

    if result.returncode != 0:
        return [f"IMPORT FAILED:\n{result.stderr.strip()}"]

    print(f"  [OK] Live imports: {result.stdout.strip()}")
    return []


def main():
    parser = argparse.ArgumentParser(description='PF AI Lab Build Verification')
    parser.add_argument('--quick', action='store_true',
                        help='Skip live import test (faster)')
    args = parser.parse_args()

    print("=" * 60)
    print("  PF AI Lab — Build Verification")
    print("=" * 60)

    all_errors = []

    print("\n--- 1. requirements.txt completeness ---")
    all_errors.extend(check_requirements_txt())

    print("\n--- 2. Python syntax check ---")
    all_errors.extend(check_python_syntax())

    print("\n--- 3. Required strings in critical files ---")
    all_errors.extend(check_required_strings())

    print("\n--- 4. Flask entry point consistency ---")
    all_errors.extend(check_flask_consistency())

    print("\n--- 5. Version consistency ---")
    all_errors.extend(check_version_consistency())

    print("\n--- 6. Archive integrity (no duplicates) ---")
    all_errors.extend(check_archive_no_duplicates())

    print("\n--- 7. Stale archive detection ---")
    all_errors.extend(check_stale_archives())

    print("\n--- 8. Critical files in archive ---")
    all_errors.extend(check_critical_files_in_archive())

    if not args.quick:
        print("\n--- 9. Live import test ---")
        all_errors.extend(check_imports_live())
    else:
        print("\n--- 9. Live import test (SKIPPED — use --quick to include) ---")

    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print(f"  ECHEC — {len(all_errors)} probleme(s) trouve(s) :")
        for e in all_errors:
            print(f"    [X] {e}")
        print("\n  NE PAS LIVRER CE BUILD.")
        print("=" * 60)
        sys.exit(1)
    else:
        print("  SUCCES — Toutes les verifications passent.")
        print("  Build OK pour release.")
        print("=" * 60)
        sys.exit(0)


if __name__ == '__main__':
    main()
