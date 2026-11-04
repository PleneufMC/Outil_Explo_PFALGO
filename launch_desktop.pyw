#!/usr/bin/env python3
"""
PF AI Lab 6.0.3 — Desktop Launcher (windowless)
================================================
This script:
  1. Logs EVERYTHING to pf_ai_lab.log (critical for debugging on Windows
     where .pyw + pythonw.exe = zero console output)
  2. Starts Flask as a subprocess using python.exe (NOT pythonw.exe)
     and captures its stdout/stderr into the log
  3. Waits for Flask to actually respond to HTTP before opening browser
  4. Shows a system-tray icon or Tkinter fallback window
  5. On failure, shows a messagebox AND writes the error to the log

File extension .pyw = Python runs WITHOUT a console window on Windows.
All diagnostics go to SCRIPT_DIR/pf_ai_lab.log
"""

import sys
import os
import time
import logging
import socket
import subprocess
import threading
import webbrowser
import traceback

# ── Resolve paths ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

LOG_FILE = os.path.join(SCRIPT_DIR, 'pf_ai_lab.log')

APP_NAME = "PF AI Lab 6.0.3"
APP_VERSION = "6.0.3"
DEFAULT_PORT = 5000
FALLBACK_PORTS = [8080, 8000, 5001, 3000]


# ── Logging setup ─────────────────────────────────────────────────────
# This is the MOST important part: .pyw has no console, so without a
# log file we are flying blind when something crashes.
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='a'),
    ]
)
log = logging.getLogger('pf_ai_lab')
log.info("=" * 60)
log.info(f"PF AI Lab {APP_VERSION} — Desktop Launcher starting")
log.info(f"  Python    : {sys.executable} ({sys.version})")
log.info(f"  Script    : {__file__}")
log.info(f"  SCRIPT_DIR: {SCRIPT_DIR}")
log.info(f"  Platform  : {sys.platform}")
log.info(f"  CWD       : {os.getcwd()}")
log.info("=" * 60)


# ── Utilities ──────────────────────────────────────────────────────────

def find_free_port():
    """Find an available port."""
    for port in [DEFAULT_PORT] + FALLBACK_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                log.info(f"Port {port} is available")
                return port
            except OSError:
                log.debug(f"Port {port} is in use, trying next...")
                continue
    log.error("No available port found!")
    return None


def find_python():
    """Find the best Python executable (prefer venv).

    CRITICAL: We use python.exe, NOT pythonw.exe, for the Flask subprocess.
    pythonw.exe suppresses all stdout/stderr which makes debugging impossible.
    We use CREATE_NO_WINDOW flag instead to hide the console.
    """
    # Prefer venv python.exe (NOT pythonw.exe — we need stdout/stderr)
    venv_python = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
    if os.path.exists(venv_python):
        log.info(f"Using venv Python: {venv_python}")
        return venv_python
    # Unix venv
    venv_unix = os.path.join(SCRIPT_DIR, 'venv', 'bin', 'python')
    if os.path.exists(venv_unix):
        log.info(f"Using venv Python (Unix): {venv_unix}")
        return venv_unix
    # System Python — try to find python.exe (not pythonw.exe)
    if sys.executable.lower().endswith('pythonw.exe'):
        # We are running under pythonw.exe — find the corresponding python.exe
        python_exe = sys.executable[:-5] + '.exe'  # pythonw.exe -> python.exe
        if os.path.exists(python_exe):
            log.info(f"Switched from pythonw.exe to python.exe: {python_exe}")
            return python_exe
    log.info(f"Using sys.executable: {sys.executable}")
    return sys.executable


def wait_for_server(port, timeout=30):
    """Wait until the Flask server responds to an actual HTTP request.

    A raw TCP connect is NOT enough — Werkzeug may accept the socket
    before the app is fully initialised. We send a real HTTP GET to
    /api/health and wait for a 200.
    """
    import http.client
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=2)
            conn.request('GET', '/api/health')
            resp = conn.getresponse()
            body = resp.read().decode('utf-8', errors='replace')
            conn.close()
            if resp.status == 200:
                log.info(f"Server ready! (attempt {attempt}, "
                         f"{time.time()-start:.1f}s, status={resp.status})")
                return True
            log.debug(f"  attempt {attempt}: status={resp.status}, body={body[:100]}")
        except ConnectionRefusedError:
            log.debug(f"  attempt {attempt}: connection refused")
        except Exception as e:
            log.debug(f"  attempt {attempt}: {type(e).__name__}: {e}")
        time.sleep(0.5)
    log.error(f"Server did NOT respond within {timeout}s ({attempt} attempts)")
    return False


# ── Flask server process ──────────────────────────────────────────────

class FlaskServer:
    def __init__(self):
        self.process = None
        self.port = None
        self._stderr_lines = []

    def start(self):
        log.info("--- FlaskServer.start() ---")

        self.port = find_free_port()
        if not self.port:
            msg = "No available port found (tried 5000, 8080, 8000, 5001, 3000)"
            log.error(msg)
            return False, msg

        python_exe = find_python()
        log.info(f"Python exe: {python_exe}")
        log.info(f"Port: {self.port}")

        # Verify app.py exists
        app_py = os.path.join(SCRIPT_DIR, 'app.py')
        if not os.path.exists(app_py):
            msg = f"app.py not found at {app_py}"
            log.error(msg)
            return False, msg

        # Pre-check: verify critical modules are importable in the target Python
        log.info("Pre-checking critical imports...")
        check_cmd = [python_exe, '-c',
                     'import flask, pandas, numpy, optuna; '
                     'print(f"flask={flask.__version__} '
                     'pandas={pandas.__version__} '
                     'numpy={numpy.__version__} '
                     'optuna={optuna.__version__}")']
        try:
            check_result = subprocess.run(
                check_cmd, capture_output=True, text=True, timeout=15,
                cwd=SCRIPT_DIR)
            if check_result.returncode != 0:
                # Missing module — give a clear, actionable error
                stderr = check_result.stderr.strip()
                msg = (f"Modules manquants dans le venv!\n\n"
                       f"{stderr}\n\n"
                       f"Solution: ouvrez un terminal dans le dossier d'installation et lancez:\n"
                       f"  venv\\Scripts\\pip install -r requirements.txt\n\n"
                       f"Ou relancez install.bat")
                log.error(msg)
                return False, msg
            log.info(f"Import check OK: {check_result.stdout.strip()}")
        except Exception as e:
            log.warning(f"Import pre-check failed ({e}), continuing anyway...")

        # Set environment
        env = os.environ.copy()
        env['FLASK_PORT'] = str(self.port)
        env['FLASK_HOST'] = '127.0.0.1'
        env['PYTHONPATH'] = SCRIPT_DIR
        # Force UTF-8 to avoid encoding errors in subprocess
        env['PYTHONIOENCODING'] = 'utf-8'

        # Build the command — use app.py directly (not -c) for better tracebacks
        cmd = [python_exe, app_py, '--port', str(self.port)]

        log.info(f"Command: {cmd}")
        log.info(f"CWD: {SCRIPT_DIR}")

        # On Windows, use CREATE_NO_WINDOW to hide console (but still get stdout/stderr)
        kwargs = {}
        if sys.platform == 'win32':
            CREATE_NO_WINDOW = 0x08000000
            kwargs['creationflags'] = CREATE_NO_WINDOW
            log.info("Windows: using CREATE_NO_WINDOW flag")

        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=SCRIPT_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **kwargs
            )
            log.info(f"Subprocess started (PID={self.process.pid})")
        except Exception as e:
            msg = f"Failed to start subprocess: {e}\n{traceback.format_exc()}"
            log.error(msg)
            return False, msg

        # Start a background thread to continuously read and log stderr
        def _drain_stderr():
            try:
                for line in iter(self.process.stderr.readline, b''):
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    self._stderr_lines.append(decoded)
                    log.info(f"[Flask stderr] {decoded}")
            except Exception:
                pass

        threading.Thread(target=_drain_stderr, daemon=True).start()

        # Also drain stdout
        def _drain_stdout():
            try:
                for line in iter(self.process.stdout.readline, b''):
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    log.info(f"[Flask stdout] {decoded}")
            except Exception:
                pass

        threading.Thread(target=_drain_stdout, daemon=True).start()

        # Give the process 2 seconds to potentially crash on import
        time.sleep(2)

        # Check if the process crashed immediately
        retcode = self.process.poll()
        if retcode is not None:
            # Process already exited — read remaining stderr
            time.sleep(0.5)  # let drain threads catch up
            stderr_text = '\n'.join(self._stderr_lines[-30:])  # last 30 lines
            msg = (f"Flask process exited immediately with code {retcode}.\n"
                   f"--- stderr (last 30 lines) ---\n{stderr_text}")
            log.error(msg)
            return False, msg

        log.info("Process still running after 2s, waiting for HTTP readiness...")

        # Wait for HTTP readiness
        if wait_for_server(self.port, timeout=30):
            log.info(f"SUCCESS: Flask server ready at http://127.0.0.1:{self.port}")
            return True, f"Server running on port {self.port}"
        else:
            # Server didn't respond — check if process is still alive
            retcode = self.process.poll()
            time.sleep(0.5)  # let drain threads catch up
            stderr_text = '\n'.join(self._stderr_lines[-30:])

            if retcode is not None:
                msg = (f"Flask process died (exit code {retcode}) during startup.\n"
                       f"--- stderr ---\n{stderr_text}")
            else:
                msg = (f"Flask process is running (PID={self.process.pid}) but did "
                       f"not respond to HTTP within 30s.\n"
                       f"--- stderr ---\n{stderr_text}")
            log.error(msg)
            return False, msg

    def stop(self):
        if self.process:
            log.info(f"Stopping Flask process (PID={self.process.pid})...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
                log.info("Flask process terminated cleanly")
            except Exception:
                try:
                    self.process.kill()
                    log.info("Flask process killed (force)")
                except Exception as e:
                    log.error(f"Failed to kill Flask process: {e}")
            self.process = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}" if self.port else None


# ── Error display ─────────────────────────────────────────────────────

def show_error_popup(title, message):
    """Show an error popup that works even without a console."""
    log.error(f"POPUP: {title} — {message}")
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            title,
            f"{message}\n\nLog file: {LOG_FILE}"
        )
        root.destroy()
    except Exception:
        pass  # If even Tkinter fails, the log file is our only hope


# ── GUI (Tkinter fallback — works everywhere) ────────────────────────

def run_tkinter_gui(server):
    """Minimal Tkinter window with status + Quit button."""
    log.info("Starting Tkinter GUI")
    import tkinter as tk

    root = tk.Tk()
    root.title(APP_NAME)
    root.geometry("480x280")
    root.resizable(False, False)

    # Try to set icon
    ico_path = os.path.join(SCRIPT_DIR, 'pf_algo.ico')
    if os.path.exists(ico_path):
        try:
            root.iconbitmap(ico_path)
        except Exception:
            pass

    # PleneufTrading.com brand theme
    bg_color = '#002646'
    fg_color = '#e0e0e0'
    accent_color = '#00B4D8'
    green_color = '#00D084'
    navy_color = '#004C8C'
    root.configure(bg=bg_color)

    # Title
    tk.Label(
        root, text=f"PF AI Lab {APP_VERSION}", font=('Segoe UI', 16, 'bold'),
        bg=bg_color, fg=accent_color
    ).pack(pady=(15, 2))

    tk.Label(
        root, text="PleneufTrading", font=('Segoe UI', 9),
        bg=bg_color, fg=green_color
    ).pack(pady=(0, 5))

    # Status
    status_var = tk.StringVar(value="Demarrage du serveur...")
    status_label = tk.Label(
        root, textvariable=status_var, font=('Segoe UI', 10),
        bg=bg_color, fg=fg_color, wraplength=400
    )
    status_label.pack(pady=(5, 5))

    # URL (clickable)
    url_var = tk.StringVar(value="")
    url_label = tk.Label(
        root, textvariable=url_var, font=('Segoe UI', 10, 'underline'),
        bg=bg_color, fg='#66ccff', cursor='hand2'
    )
    url_label.pack(pady=(0, 5))
    url_label.bind('<Button-1>',
                   lambda e: webbrowser.open(server.url) if server.url else None)

    # Hint
    tk.Label(
        root, text="Fermez cette fenetre pour arreter le serveur.",
        font=('Segoe UI', 8, 'italic'), bg=bg_color, fg='#666666'
    ).pack(pady=(0, 2))

    # Log file link
    log_var = tk.StringVar(value=f"Log: {LOG_FILE}")
    tk.Label(
        root, textvariable=log_var, font=('Segoe UI', 8),
        bg=bg_color, fg='#888888'
    ).pack(pady=(0, 5))

    # Buttons
    btn_frame = tk.Frame(root, bg=bg_color)
    btn_frame.pack(pady=(5, 10))

    def open_browser():
        if server.url:
            webbrowser.open(server.url)

    def quit_app():
        status_var.set("Arret en cours...")
        root.update()
        server.stop()
        root.destroy()

    tk.Button(
        btn_frame, text="Ouvrir Navigateur", font=('Segoe UI', 10),
        command=open_browser, bg=navy_color, fg='white',
        activebackground=accent_color, activeforeground='white',
        relief='flat', padx=15, pady=5, cursor='hand2'
    ).pack(side='left', padx=10)

    tk.Button(
        btn_frame, text="Quitter", font=('Segoe UI', 10),
        command=quit_app, bg='#c0392b', fg='white',
        activebackground='#e74c3c', activeforeground='white',
        relief='flat', padx=15, pady=5, cursor='hand2'
    ).pack(side='left', padx=10)

    # Start server in background thread
    def start_server():
        ok, msg = server.start()
        if ok:
            status_var.set(f"Serveur actif sur le port {server.port}")
            url_var.set(server.url)
            log.info(f"Opening browser: {server.url}")
            webbrowser.open(server.url)
        else:
            error_short = msg.split('\n')[0][:80]
            status_var.set(f"ERREUR: {error_short}")
            url_var.set(f"Voir le log: {LOG_FILE}")
            log.error(f"Server failed to start: {msg}")

    threading.Thread(target=start_server, daemon=True).start()

    root.protocol("WM_DELETE_WINDOW", quit_app)

    # Center on screen
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f'+{x}+{y}')

    root.mainloop()
    log.info("Tkinter mainloop exited")


# ── System tray (if pystray available) ────────────────────────────────

def run_systray_gui(server):
    """System tray icon with menu. Requires pystray + Pillow."""
    log.info("Starting system tray GUI")
    try:
        import pystray
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        log.info(f"pystray/Pillow not available ({e}), falling back to Tkinter")
        return run_tkinter_gui(server)

    # Create tray icon image
    ico_path = os.path.join(SCRIPT_DIR, 'pf_algo.ico')
    try:
        icon_image = Image.open(ico_path)
    except Exception:
        icon_image = Image.new('RGBA', (64, 64), (26, 26, 46, 255))
        draw = ImageDraw.Draw(icon_image)
        draw.ellipse([8, 8, 56, 56], fill=(0, 210, 255, 255))
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.text((18, 16), "PF", fill=(26, 26, 46, 255), font=font)

    def on_open(icon, item):
        if server.url:
            webbrowser.open(server.url)

    def on_quit(icon, item):
        server.stop()
        icon.stop()

    def on_show_log(icon, item):
        """Open the log file in the default text editor."""
        if sys.platform == 'win32':
            os.startfile(LOG_FILE)
        else:
            webbrowser.open(f'file://{LOG_FILE}')

    # Start server
    ok, msg = server.start()
    if ok:
        log.info(f"Opening browser: {server.url}")
        webbrowser.open(server.url)
        status_text = f"Port {server.port} — OK"
    else:
        status_text = f"ERREUR (voir log)"
        log.error(f"Server failed: {msg}")
        # Show error popup since systray may not be visible yet
        show_error_popup("PF AI Lab - Erreur", msg)

    menu = pystray.Menu(
        pystray.MenuItem(f'{APP_NAME}', None, enabled=False),
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('Ouvrir Navigateur', on_open),
        pystray.MenuItem('Voir le Log', on_show_log),
        pystray.MenuItem('Quitter', on_quit),
    )

    icon = pystray.Icon(APP_NAME, icon_image, APP_NAME, menu)
    icon.run()
    log.info("System tray exited")


# ── Desktop shortcut auto-repair ──────────────────────────────────────

def ensure_desktop_shortcut():
    """Create desktop shortcut if it doesn't exist (Windows only).
    
    This is the MANDATORY desktop icon requirement:
    Every launch must verify the shortcut exists and recreate it if missing.
    This ensures the user always has a desktop icon regardless of how they
    installed or updated the application.
    """
    if sys.platform != 'win32':
        return
    
    try:
        log.info("Checking desktop shortcut...")
        
        # Detect Desktop folder (handles French "Bureau", OneDrive, etc.)
        desktop = None
        
        # Method 1: PowerShell (most reliable)
        try:
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 "[Environment]::GetFolderPath('Desktop')"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                desktop = result.stdout.strip()
                log.info(f"Desktop path (PowerShell): {desktop}")
        except Exception as e:
            log.debug(f"PowerShell desktop detection failed: {e}")
        
        # Method 2: Environment variable fallback
        if not desktop or not os.path.isdir(desktop):
            desktop = os.path.join(os.environ.get('USERPROFILE', ''), 'Desktop')
            if not os.path.isdir(desktop):
                desktop = os.path.join(os.environ.get('USERPROFILE', ''), 'Bureau')
        
        if not desktop or not os.path.isdir(desktop):
            log.warning(f"Could not find Desktop folder, tried: {desktop}")
            return
        
        shortcut_path = os.path.join(desktop, f'PF AI Lab {APP_VERSION}.lnk')
        
        if os.path.exists(shortcut_path):
            log.info(f"Desktop shortcut already exists: {shortcut_path}")
            return
        
        log.info(f"Desktop shortcut MISSING — creating: {shortcut_path}")
        
        # Find pythonw.exe (for the shortcut target)
        pythonw_exe = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'pythonw.exe')
        if not os.path.exists(pythonw_exe):
            pythonw_exe = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
        if not os.path.exists(pythonw_exe):
            # System Python fallback
            pythonw_exe = sys.executable
        
        launcher_path = os.path.join(SCRIPT_DIR, 'launch_desktop.pyw')
        icon_path = os.path.join(SCRIPT_DIR, 'pf_algo.ico')
        
        # Generate icon if missing
        if not os.path.exists(icon_path):
            gen_icon = os.path.join(SCRIPT_DIR, 'generate_icon.py')
            if os.path.exists(gen_icon):
                try:
                    log.info("Icon missing, generating...")
                    python_exe = os.path.join(SCRIPT_DIR, 'venv', 'Scripts', 'python.exe')
                    if not os.path.exists(python_exe):
                        python_exe = sys.executable
                    subprocess.run([python_exe, gen_icon], cwd=SCRIPT_DIR,
                                   capture_output=True, timeout=15)
                except Exception as e:
                    log.warning(f"Icon generation failed: {e}")
        
        # Create shortcut using VBScript (same method as install.bat)
        vbs_content = f'''Set WShell = CreateObject("WScript.Shell")
Set Shortcut = WShell.CreateShortcut("{shortcut_path}")
Shortcut.TargetPath = "{pythonw_exe}"
Shortcut.Arguments = """{launcher_path}"""
Shortcut.WorkingDirectory = "{SCRIPT_DIR}"
Shortcut.Description = "PF AI Lab {APP_VERSION} - Backtest Optimizer"
Shortcut.WindowStyle = 7
'''
        if os.path.exists(icon_path):
            vbs_content += f'Shortcut.IconLocation = "{icon_path},0"\n'
        vbs_content += 'Shortcut.Save\n'
        
        # Write temp VBS and execute
        vbs_path = os.path.join(os.environ.get('TEMP', SCRIPT_DIR), 'pf_create_shortcut.vbs')
        with open(vbs_path, 'w', encoding='utf-8') as f:
            f.write(vbs_content)
        
        result = subprocess.run(
            ['cscript', '//nologo', vbs_path],
            capture_output=True, text=True, timeout=10
        )
        
        # Cleanup VBS
        try:
            os.remove(vbs_path)
        except Exception:
            pass
        
        if os.path.exists(shortcut_path):
            log.info(f"[OK] Desktop shortcut created: {shortcut_path}")
        else:
            log.warning(f"Desktop shortcut creation may have failed. "
                       f"VBS stderr: {result.stderr.strip()}")
        
        # Also create/update Start Menu shortcut
        start_menu = os.path.join(
            os.environ.get('APPDATA', ''),
            'Microsoft', 'Windows', 'Start Menu', 'Programs'
        )
        if os.path.isdir(start_menu):
            sm_shortcut = os.path.join(start_menu, f'PF AI Lab {APP_VERSION}.lnk')
            if not os.path.exists(sm_shortcut):
                vbs_sm = f'''Set WShell = CreateObject("WScript.Shell")
Set Shortcut = WShell.CreateShortcut("{sm_shortcut}")
Shortcut.TargetPath = "{pythonw_exe}"
Shortcut.Arguments = """{launcher_path}"""
Shortcut.WorkingDirectory = "{SCRIPT_DIR}"
Shortcut.Description = "PF AI Lab {APP_VERSION} - Backtest Optimizer"
Shortcut.WindowStyle = 7
'''
                if os.path.exists(icon_path):
                    vbs_sm += f'Shortcut.IconLocation = "{icon_path},0"\n'
                vbs_sm += 'Shortcut.Save\n'
                
                vbs_path2 = os.path.join(os.environ.get('TEMP', SCRIPT_DIR), 'pf_create_sm_shortcut.vbs')
                with open(vbs_path2, 'w', encoding='utf-8') as f:
                    f.write(vbs_sm)
                subprocess.run(['cscript', '//nologo', vbs_path2],
                              capture_output=True, timeout=10)
                try:
                    os.remove(vbs_path2)
                except Exception:
                    pass
                if os.path.exists(sm_shortcut):
                    log.info(f"[OK] Start Menu shortcut created: {sm_shortcut}")
        
    except Exception as e:
        log.warning(f"Desktop shortcut auto-repair failed: {e}\n{traceback.format_exc()}")
        # Non-fatal: don't prevent the app from starting


# ── Main ──────────────────────────────────────────────────────────────

def main():
    try:
        # MANDATORY: ensure desktop shortcut exists before anything else
        ensure_desktop_shortcut()
        
        server = FlaskServer()

        # Try system tray first, fallback to Tkinter
        try:
            import pystray
            from PIL import Image
            run_systray_gui(server)
        except ImportError:
            run_tkinter_gui(server)
    except Exception as e:
        log.critical(f"FATAL ERROR in main(): {e}\n{traceback.format_exc()}")
        show_error_popup(
            "PF AI Lab - Erreur Fatale",
            f"{e}\n\nVoir le log pour les details."
        )


if __name__ == '__main__':
    main()
