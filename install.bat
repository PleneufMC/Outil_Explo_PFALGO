@echo off
setlocal EnableDelayedExpansion
title PF AI Lab 5.4.8 - Installation

:: Change to script directory (fixes admin mode)
cd /d "%~dp0"

echo ============================================================
echo   PF AI Lab 5.4.8 - Installation Windows
echo ============================================================
echo.
echo   Dossier : %CD%
echo.

:: ---- Check Python ----
echo [1/6] Verification de Python...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERREUR] Python n est pas installe ou pas dans le PATH.
    echo.
    echo Telechargez Python 3.9+ depuis : https://www.python.org/downloads/
    echo IMPORTANT : Cochez Add Python to PATH pendant l installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK] %PYVER%

python -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Python 3.9 ou superieur requis. Version actuelle : %PYVER%
    pause
    exit /b 1
)
echo.

:: ---- Set install directory ----
set "INSTALL_DIR=%~dp0"
if "%INSTALL_DIR:~-1%"=="\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"
echo [2/6] Repertoire : %INSTALL_DIR%
echo.

:: ---- Create virtual environment ----
echo [3/6] Creation de l environnement virtuel...
if exist "%INSTALL_DIR%\venv" (
    echo      Environnement virtuel deja existant, etape ignoree.
) else (
    python -m venv "%INSTALL_DIR%\venv"
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR] Impossible de creer l environnement virtuel.
        echo Essayez : python -m pip install --upgrade pip virtualenv
        pause
        exit /b 1
    )
    echo [OK] Environnement virtuel cree
)
echo.

:: ---- Install dependencies ----
echo [4/6] Installation des dependances (1-2 minutes)...
call "%INSTALL_DIR%\venv\Scripts\activate.bat"

python -m pip install --upgrade pip -q 2>nul
echo      pip mis a jour

pip install -r "%INSTALL_DIR%\requirements.txt" -q
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Echec de l installation des packages.
    echo Essayez : venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK] Toutes les dependances installees

pip install Pillow pystray -q 2>nul
echo [OK] Packages optionnels (Pillow, pystray) installes

:: ---- Verify critical imports ----
python -c "import flask; print(f'      flask {flask.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERREUR] Flask n est pas installe correctement.
    echo          Tentative de reinstallation...
    pip install flask>=3.0 openpyxl>=3.0 plotly>=5.0 -q
    python -c "import flask" 2>nul
    if %ERRORLEVEL% neq 0 (
        echo [ERREUR CRITIQUE] Impossible d installer Flask.
        echo Essayez manuellement : venv\Scripts\pip install flask
        pause
        exit /b 1
    )
)
python -c "import openpyxl; print(f'      openpyxl {openpyxl.__version__}')" 2>nul
python -c "import optuna; print(f'      optuna {optuna.__version__}')" 2>nul
echo [OK] Verification des imports critiques reussie
echo.

:: ---- Generate icon ----
echo [5/6] Generation de l icone...
if exist "%INSTALL_DIR%\generate_icon.py" (
    python "%INSTALL_DIR%\generate_icon.py" 2>nul
    if not exist "%INSTALL_DIR%\pf_algo.ico" (
        echo [WARN] Generation de l icone echouee, icone par defaut.
    )
) else (
    echo [WARN] generate_icon.py introuvable.
)
echo.

:: ---- Create desktop shortcut ----
echo [6/6] Creation du raccourci bureau...

:: Detect real Desktop path (handles French "Bureau", OneDrive, custom paths)
:: Method 1: PowerShell (most reliable — works with all locales)
for /f "tokens=*" %%D in ('powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')" 2^>nul') do set "DESKTOP=%%D"
:: Method 2: fallback to registry
if not defined DESKTOP (
    for /f "tokens=2*" %%A in ('reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders" /v Desktop 2^>nul ^| findstr Desktop') do (
        call set "DESKTOP=%%B"
    )
)
:: Method 3: last resort
if not defined DESKTOP set "DESKTOP=%USERPROFILE%\Desktop"
if not exist "%DESKTOP%" set "DESKTOP=%USERPROFILE%\Bureau"
if not exist "%DESKTOP%" (
    echo [WARN] Impossible de trouver le dossier Bureau.
    echo        Chemin tente : %DESKTOP%
    set "DESKTOP=%USERPROFILE%\Desktop"
)

echo      Bureau detecte : %DESKTOP%

set "SHORTCUT=%DESKTOP%\PF AI Lab 5.4.8.lnk"
set "VBS_TEMP=%TEMP%\create_shortcut.vbs"
:: Use pythonw.exe for the shortcut — it hides the console window.
:: launch_desktop.pyw handles error display via Tkinter/messagebox.
set "PYTHONEXE=%INSTALL_DIR%\venv\Scripts\pythonw.exe"
if not exist "%PYTHONEXE%" (
    :: Fallback to python.exe if pythonw.exe not found (unusual)
    set "PYTHONEXE=%INSTALL_DIR%\venv\Scripts\python.exe"
)
set "LAUNCHER=%INSTALL_DIR%\launch_desktop.pyw"
set "ICON=%INSTALL_DIR%\pf_algo.ico"

> "%VBS_TEMP%" (
    echo Set WShell = CreateObject^("WScript.Shell"^)
    echo Set Shortcut = WShell.CreateShortcut^("%SHORTCUT%"^)
    echo Shortcut.TargetPath = "%PYTHONEXE%"
    echo Shortcut.Arguments = """%LAUNCHER%"""
    echo Shortcut.WorkingDirectory = "%INSTALL_DIR%"
    echo Shortcut.Description = "PF AI Lab 5.4.8 - Backtest Optimizer"
    echo Shortcut.WindowStyle = 7
)

if exist "%ICON%" (
    >> "%VBS_TEMP%" echo Shortcut.IconLocation = "%ICON%,0"
)

>> "%VBS_TEMP%" echo Shortcut.Save

cscript //nologo "%VBS_TEMP%" 2>nul
del "%VBS_TEMP%" 2>nul

if exist "%SHORTCUT%" (
    echo [OK] Raccourci bureau cree : %SHORTCUT%
) else (
    echo [ERREUR] Raccourci bureau non cree.
    echo          Chemin tente : %SHORTCUT%
    echo          Vous pouvez creer manuellement un raccourci vers :
    echo            Cible : "%PYTHONEXE%" "%LAUNCHER%"
    echo            Demarrer dans : "%INSTALL_DIR%"
)
echo.

:: ---- Start Menu shortcut ----
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
set "SM_SHORTCUT=%STARTMENU%\PF AI Lab 5.4.8.lnk"
set "VBS_TEMP2=%TEMP%\create_sm_shortcut.vbs"

> "%VBS_TEMP2%" (
    echo Set WShell = CreateObject^("WScript.Shell"^)
    echo Set Shortcut = WShell.CreateShortcut^("%SM_SHORTCUT%"^)
    echo Shortcut.TargetPath = "%PYTHONEXE%"
    echo Shortcut.Arguments = """%LAUNCHER%"""
    echo Shortcut.WorkingDirectory = "%INSTALL_DIR%"
    echo Shortcut.Description = "PF AI Lab 5.4.8 - Backtest Optimizer"
    echo Shortcut.WindowStyle = 7
)

if exist "%ICON%" (
    >> "%VBS_TEMP2%" echo Shortcut.IconLocation = "%ICON%,0"
)

>> "%VBS_TEMP2%" echo Shortcut.Save

cscript //nologo "%VBS_TEMP2%" 2>nul
del "%VBS_TEMP2%" 2>nul

if exist "%SM_SHORTCUT%" (
    echo [OK] Raccourci menu Demarrer cree
)

:: ---- Done ----
echo.
echo ============================================================
echo   Installation terminee !
echo ============================================================
echo.
echo   3 facons de lancer PF AI Lab :
echo.
echo   1. Raccourci bureau    : "PF AI Lab 5.4.8" (icone sur le bureau)
echo   2. Menu Demarrer       : cherchez "PF AI Lab 5.4.8"
echo   3. Fichier start.bat   : double-cliquez dans le dossier
echo.
echo   Le navigateur s ouvrira automatiquement a chaque lancement.
echo   Pour desinstaller : lancez uninstall.bat
echo.
echo ============================================================
echo.

:: ---- Create start.bat convenience launcher ----
echo.
echo [BONUS] Creation du lanceur start.bat...
> "%INSTALL_DIR%\start.bat" (
    echo @echo off
    echo cd /d "%%~dp0"
    echo echo Lancement de PF AI Lab 5.4.8...
    echo echo Log : %%CD%%\pf_ai_lab.log
    echo echo.
    echo start "" "%%CD%%\venv\Scripts\pythonw.exe" "%%CD%%\launch_desktop.pyw"
    echo echo Le navigateur va s ouvrir automatiquement.
    echo timeout /t 5 /nobreak ^>nul
)
echo [OK] start.bat cree (lanceur rapide sans reinstallation)
echo.

set /p LAUNCH="Lancer PF AI Lab maintenant ? (O/N) : "
if /i "%LAUNCH%"=="O" (
    echo.
    echo Lancement...
    echo Log de demarrage : %INSTALL_DIR%\pf_ai_lab.log
    start "" "%PYTHONEXE%" "%LAUNCHER%"
    echo.
    echo Le navigateur va s ouvrir automatiquement.
    echo Si ce n est pas le cas apres 15 secondes :
    echo   1. Ouvrez le fichier pf_ai_lab.log pour voir l erreur
    echo   2. Ou double-cliquez sur start.bat
)

echo.
pause
