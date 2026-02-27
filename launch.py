#!/usr/bin/env python3
"""
PF AI Lab 5.4.0 — Launcher with auto-diagnostics
Run this script to start the web dashboard.
It checks dependencies, creates required folders, and launches Flask.

Usage:
    python launch.py           # Default: port 5000
    python launch.py --port 8080   # Custom port
    python launch.py --check       # Check only, don't start
"""

import sys
import os
import subprocess
import argparse

REQUIRED_PYTHON = (3, 9)
REQUIRED_PACKAGES = {
    'pandas': 'pandas>=2.0',
    'numpy': 'numpy>=1.24',
    'optuna': 'optuna>=3.0',
    'flask': 'flask>=3.0',
    'plotly': 'plotly>=5.0',
    'openpyxl': 'openpyxl>=3.0',
    'scipy': 'scipy>=1.10',
}

def check_python_version():
    """Check Python version >= 3.9"""
    v = sys.version_info
    if v < REQUIRED_PYTHON:
        print(f"[FAIL] Python {v.major}.{v.minor} detected. Need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}")
        print(f"       Download: https://www.python.org/downloads/")
        return False
    print(f"[OK]   Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_packages():
    """Check all required packages are installed."""
    missing = []
    for pkg, spec in REQUIRED_PACKAGES.items():
        try:
            __import__(pkg)
            print(f"[OK]   {pkg}")
        except ImportError:
            print(f"[MISS] {pkg} — not installed")
            missing.append(spec)
    return missing


def install_packages(missing):
    """Auto-install missing packages."""
    print(f"\nInstalling {len(missing)} missing package(s)...")
    cmd = [sys.executable, '-m', 'pip', 'install'] + missing
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[FAIL] pip install failed:\n{result.stderr}")
        print(f"\nTry manually: pip install -r requirements.txt")
        return False
    print("[OK]   All packages installed successfully")
    return True


def check_project_structure():
    """Verify project files exist."""
    base = os.path.dirname(os.path.abspath(__file__))
    ok = True
    
    # Required files
    required = [
        'app.py',
        'requirements.txt',
        'pf_ma_optimizer/__init__.py',
        'pf_ma_optimizer/config.py',
        'pf_ma_optimizer/data_loader.py',
        'pf_ma_optimizer/metrics.py',
        'pf_ma_optimizer/optimizer.py',
        'pf_ma_optimizer/backtest_engine.py',
        'pf_ma_optimizer/tv_export.py',
        'pf_ma_optimizer/ml/monte_carlo.py',
        'pf_ma_optimizer/ml/walk_forward.py',
        'pf_ma_optimizer/indicators/ma_types.py',
        'pf_ma_optimizer/indicators/pmax.py',
        'pf_ma_optimizer/entries/donchian_chikou.py',
        'pf_ma_optimizer/entries/rsi_divergence.py',
        'pf_ma_optimizer/filters/all_filters.py',
        'templates/index.html',
        'templates/data.html',
        'templates/explore.html',
        'templates/validate.html',
        'templates/portfolio.html',
    ]
    
    for f in required:
        path = os.path.join(base, f)
        if not os.path.exists(path):
            print(f"[MISS] {f}")
            ok = False
    
    if ok:
        print(f"[OK]   All {len(required)} required files present")
    
    # Create data/ if missing
    data_dir = os.path.join(base, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"[OK]   Created data/ directory")
    else:
        print(f"[OK]   data/ directory exists")
    
    return ok


def check_port(port):
    """Check if port is available."""
    import socket
    # Use 127.0.0.1 on Windows to match the host Flask will actually bind to
    bind_host = '127.0.0.1' if sys.platform == 'win32' else '0.0.0.0'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((bind_host, port))
            print(f"[OK]   Port {port} is available")
            return True
        except OSError:
            print(f"[WARN] Port {port} is already in use")
            print(f"       Try: python launch.py --port 8080")
            # On macOS, port 5000 is often used by AirPlay Receiver
            if port == 5000 and sys.platform == 'darwin':
                print(f"       (macOS: AirPlay Receiver often uses port 5000)")
                print(f"       System Preferences > General > AirDrop & Handoff > AirPlay Receiver: OFF")
            return False


def test_imports():
    """Test that all app imports work."""
    try:
        from pf_ma_optimizer.data_loader import load_data
        from pf_ma_optimizer.optimizer import run_optimization
        from pf_ma_optimizer.metrics import compute_metrics, passes_oos_gates
        from pf_ma_optimizer.backtest_engine import run_backtest
        from pf_ma_optimizer.ml.monte_carlo import run_monte_carlo
        from pf_ma_optimizer.ml.walk_forward import run_walk_forward
        from pf_ma_optimizer.tv_export import format_all_configs
        from pf_ma_optimizer.config import TRANSACTION_COSTS, OOS_GATES, OOS_DD_LIMITS
        print(f"[OK]   All imports successful")
        print(f"       OOS Gates: Calmar>={OOS_GATES['min_calmar']}, PF>={OOS_GATES['min_pf']}, "
              f"MaxDD adaptive per class (default {OOS_GATES['max_dd']}%), Trades>={OOS_GATES['min_trades']}")
        print(f"       DD Limits: {', '.join(f'{k}:{v}%' for k, v in OOS_DD_LIMITS.items())}")
        print(f"       Instruments: {', '.join(sorted(TRANSACTION_COSTS.keys()))}")
        return True
    except Exception as e:
        print(f"[FAIL] Import error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='PF AI Lab 5.4.0 — Launcher')
    parser.add_argument('--port', type=int, default=5000, help='Port (default: 5000)')
    parser.add_argument('--check', action='store_true', help='Check only, do not start')
    parser.add_argument('--no-install', action='store_true', help='Skip auto-install of missing packages')
    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 60)
    print("  PF AI Lab 5.4.0 — Pre-launch Diagnostics")
    print("=" * 60)
    print()

    # 1. Python version
    print("--- Python Version ---")
    if not check_python_version():
        sys.exit(1)
    print()

    # 2. Project structure
    print("--- Project Structure ---")
    if not check_project_structure():
        print("\n[FAIL] Missing files. Re-extract the archive and try again.")
        sys.exit(1)
    print()

    # 3. Dependencies
    print("--- Dependencies ---")
    missing = check_packages()
    if missing:
        if args.no_install:
            print(f"\n[FAIL] {len(missing)} packages missing. Run: pip install -r requirements.txt")
            sys.exit(1)
        print()
        if not install_packages(missing):
            sys.exit(1)
        print()
        # Re-check
        missing = check_packages()
        if missing:
            print(f"\n[FAIL] Still missing packages after install. Check pip errors above.")
            sys.exit(1)
    print()

    # 4. Import test
    print("--- Import Test ---")
    if not test_imports():
        print("\n[FAIL] Import errors. Check installation and Python version.")
        sys.exit(1)
    print()

    # 5. Port check
    print("--- Port Check ---")
    port_ok = check_port(args.port)
    print()

    if args.check:
        print("=" * 60)
        if port_ok:
            print("  All checks PASSED. Ready to launch.")
        else:
            print(f"  All checks PASSED except port {args.port} (in use).")
            print(f"  Try: python launch.py --port 8080")
        print("=" * 60)
        sys.exit(0)

    if not port_ok:
        # Try alternative ports
        for alt_port in [8080, 8000, 5001, 3000]:
            if check_port(alt_port):
                args.port = alt_port
                port_ok = True
                print(f"       Using alternative port {alt_port}")
                break
        if not port_ok:
            print("[FAIL] No available port found. Close other applications and retry.")
            sys.exit(1)

    # Launch
    print("=" * 60)
    print(f"  Launching PF AI Lab 5.4.0 on port {args.port}")
    print(f"  Open: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)
    print()

    # Auto-open browser after a short delay (gives Flask time to bind)
    import threading, webbrowser, time, socket as _sock

    def _open_browser_when_ready(port, timeout=15):
        """Wait for Flask /api/health to respond 200, then open browser."""
        import http.client
        start = time.time()
        while time.time() - start < timeout:
            try:
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
                conn.request('GET', '/api/health')
                resp = conn.getresponse()
                conn.close()
                if resp.status == 200:
                    print(f"  Server ready ({time.time()-start:.1f}s)")
                    webbrowser.open(f'http://localhost:{port}')
                    return
            except Exception:
                pass
            time.sleep(0.5)
        # Fallback: open anyway after timeout
        print(f"  WARNING: server did not respond within {timeout}s, opening anyway")
        webbrowser.open(f'http://localhost:{port}')

    threading.Thread(
        target=_open_browser_when_ready, args=(args.port,), daemon=True
    ).start()

    # Import and run Flask
    # ── Windows connection-refused fix ─────────────────────────────
    # 1. host='127.0.0.1' on Windows to avoid firewall popup
    # 2. threaded=True prevents single-thread blocking when the browser
    #    opens multiple parallel connections (page + favicon + css + js)
    # 3. use_reloader=False avoids double-start on Windows
    from app import app
    host = '127.0.0.1' if sys.platform == 'win32' else '0.0.0.0'
    app.run(host=host, port=args.port, debug=False,
            threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
