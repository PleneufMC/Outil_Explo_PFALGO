@echo off
:: ============================================================
:: PF AI Lab 6.0.3 — Quick Launcher
:: ============================================================
:: Double-cliquez sur ce fichier pour lancer l'application.
:: Pas besoin de relancer install.bat a chaque fois !
::
:: Si l'application ne demarre pas :
::   1. Verifiez que install.bat a ete lance au moins une fois
::   2. Consultez le fichier pf_ai_lab.log pour les erreurs
:: ============================================================

cd /d "%~dp0"
title PF AI Lab 6.0.3

:: Check if venv exists (install.bat must have been run at least once)
if not exist "%CD%\venv\Scripts\pythonw.exe" (
    if not exist "%CD%\venv\Scripts\python.exe" (
        echo.
        echo [ERREUR] L'environnement virtuel n'existe pas.
        echo.
        echo Lancez d'abord install.bat pour installer les dependances.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo   PF AI Lab 6.0.3 — Demarrage...
echo   Log : %CD%\pf_ai_lab.log
echo.

:: Prefer pythonw.exe (no console window) over python.exe
set "PYTHONEXE=%CD%\venv\Scripts\pythonw.exe"
if not exist "%PYTHONEXE%" set "PYTHONEXE=%CD%\venv\Scripts\python.exe"

:: Launch the desktop launcher (starts Flask + opens browser + shows tray icon)
start "" "%PYTHONEXE%" "%CD%\launch_desktop.pyw"

echo   Le navigateur va s'ouvrir automatiquement.
echo   (Fenetre de controle dans la barre des taches)
echo.
echo   Pour arreter : clic droit sur l'icone PF dans la barre
echo   des taches ^> Quitter
echo.

:: Wait a few seconds so the user sees the message, then auto-close
timeout /t 8 /nobreak >nul
