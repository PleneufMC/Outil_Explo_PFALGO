@echo off
setlocal EnableDelayedExpansion
title PF AI Lab 6.0.3 - Desinstallation

cd /d "%~dp0"

echo ============================================================
echo   PF AI Lab 6.0.3 - Desinstallation
echo ============================================================
echo.
echo ATTENTION : Ceci va supprimer :
echo   - Le raccourci du bureau
echo   - Le raccourci du menu Demarrer
echo   - L environnement virtuel Python (venv/)
echo   - Les fichiers generes (pf_algo.ico, logs)
echo.
echo Les fichiers de donnees (data/) et le code source
echo seront CONSERVES.
echo.

set /p CONFIRM="Continuer la desinstallation ? (O/N) : "
if /i not "%CONFIRM%"=="O" (
    echo.
    echo Desinstallation annulee.
    pause
    exit /b 0
)

echo.

:: ---- Kill running instances ----
echo [1/5] Arret des instances en cours...
taskkill /f /fi "WINDOWTITLE eq PF AI Lab 6.0.3" 2>nul
echo [OK] Fait
echo.

:: ---- Remove desktop shortcut ----
echo [2/5] Suppression du raccourci bureau...

:: Detect real Desktop path (same logic as install.bat)
for /f "tokens=*" %%D in ('powershell -NoProfile -Command "[Environment]::GetFolderPath('Desktop')" 2^>nul') do set "DESKTOP=%%D"
if not defined DESKTOP set "DESKTOP=%USERPROFILE%\Desktop"
if not exist "%DESKTOP%" set "DESKTOP=%USERPROFILE%\Bureau"

set "SHORTCUT=%DESKTOP%\PF AI Lab 6.0.3.lnk"
if exist "%SHORTCUT%" (
    del "%SHORTCUT%"
    echo [OK] Raccourci bureau supprime : %SHORTCUT%
) else (
    echo [--] Pas de raccourci bureau trouve (%SHORTCUT%)
)
echo.

:: ---- Remove Start Menu shortcut ----
echo [3/5] Suppression du raccourci menu Demarrer...
set "STARTMENU=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
set "SM_SHORTCUT=%STARTMENU%\PF AI Lab 6.0.3.lnk"
if exist "%SM_SHORTCUT%" (
    del "%SM_SHORTCUT%"
    echo [OK] Raccourci menu Demarrer supprime
) else (
    echo [--] Pas de raccourci menu Demarrer trouve
)
echo.

:: ---- Remove virtual environment ----
echo [4/5] Suppression de l environnement virtuel...
set "INSTALL_DIR=%~dp0"
if "%INSTALL_DIR:~-1%"=="\" set "INSTALL_DIR=%INSTALL_DIR:~0,-1%"

if exist "%INSTALL_DIR%\venv" (
    echo      Cela peut prendre quelques secondes...
    rmdir /s /q "%INSTALL_DIR%\venv"
    echo [OK] Environnement virtuel supprime
) else (
    echo [--] Pas d environnement virtuel trouve
)
echo.

:: ---- Remove generated files ----
echo [5/5] Nettoyage des fichiers generes...
if exist "%INSTALL_DIR%\pf_algo.ico" del "%INSTALL_DIR%\pf_algo.ico"
if exist "%INSTALL_DIR%\pf_ai_lab.log" del "%INSTALL_DIR%\pf_ai_lab.log"
if exist "%INSTALL_DIR%\__pycache__" rmdir /s /q "%INSTALL_DIR%\__pycache__"
if exist "%INSTALL_DIR%\pf_ma_optimizer\__pycache__" rmdir /s /q "%INSTALL_DIR%\pf_ma_optimizer\__pycache__"
echo [OK] Fichiers generes nettoyes
echo.

:: ---- Done ----
echo ============================================================
echo   Desinstallation terminee.
echo ============================================================
echo.
echo   Le code source et les donnees sont conserves dans :
echo   %INSTALL_DIR%
echo.
echo   Pour supprimer completement, supprimez ce dossier.
echo.
echo ============================================================
echo.
pause
