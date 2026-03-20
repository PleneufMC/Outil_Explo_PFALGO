# PF AI Lab 6.0.3 - Guide d'installation

## Prerequis

- **Python 3.9+** ([python.org/downloads](https://www.python.org/downloads/))
- ~200 Mo d'espace disque (avec les dependances)

---

## Installation Windows

### Methode rapide (recommandee)

1. Extraire l'archive `PF_AI_Lab_v6.0.3.tar.gz` (ou `.zip`) dans le dossier de votre choix
2. Double-cliquer sur **`install.bat`**
3. L'installateur va :
   - Verifier Python
   - Creer un environnement virtuel (`venv/`)
   - Installer toutes les dependances
   - Creer un raccourci bureau
4. Accepter le lancement a la fin

### Lancement quotidien

- **Raccourci bureau** : `PF AI Lab 6.0.3`
- **Ou** : double-cliquer sur `start.bat`
- **Ou** : ligne de commande :
  ```
  venv\Scripts\activate
  python launch.py --port 5000
  ```

### Desinstallation

Double-cliquer sur `uninstall.bat` — supprime le venv et les raccourcis, conserve vos donnees.

---

## Installation Linux / macOS

### Methode rapide

```bash
# Extraire l'archive
tar xzf PF_AI_Lab_v6.0.3.tar.gz
cd PF_AI_Lab_v6.0.3/

# Rendre executable et installer
chmod +x install.sh
./install.sh
```

### Lancement quotidien

```bash
./start.sh                  # Port par defaut (5000)
./start.sh --port 8080      # Port personnalise
./start.sh --check          # Diagnostic uniquement
```

### Ou manuellement

```bash
source venv/bin/activate
python launch.py --port 5000
```

---

## Installation manuelle (tous systemes)

Si les scripts automatiques ne fonctionnent pas :

```bash
# 1. Creer le venv
python3 -m venv venv

# 2. Activer
#    Windows : venv\Scripts\activate
#    Linux/Mac : source venv/bin/activate

# 3. Installer les dependances
pip install -r requirements.txt

# 4. Lancer
python launch.py
```

---

## Structure des fichiers

```
PF_AI_Lab_v6.0.3/
  install.bat          # Installation Windows (double-cliquer)
  install.sh           # Installation Linux/macOS (chmod +x && ./install.sh)
  start.bat            # Lanceur rapide Windows
  uninstall.bat        # Desinstallation Windows
  launch.py            # Lanceur cross-platform avec diagnostics
  launch_desktop.pyw   # Lanceur Windows sans console (system tray)
  requirements.txt     # Dependances Python
  pyproject.toml       # Configuration package Python
  app.py               # Serveur web Flask (dashboard)
  pf_ma_optimizer/     # Moteur d'optimisation (metrics, config, DB, etc.)
  pf_ai_lab/           # Librairie backtesting
  pineguard/           # Module d'audit PineScript
  templates/           # Pages HTML du dashboard
  tests/               # Tests unitaires
  data/                # Donnees (cree automatiquement)
```

---

## Nouveautes v6.0.3 (Risk Management Audit)

- **Export Integrity** : validation round-trip JSON, SHA-256 checksum, mode strict
- **FTMO Transparency** : max daily DD, worst consecutive loss, recovery time, best day rule
- **FTMO Hard Constraints** : disqualification automatique si MaxDD > 10% ou daily DD > 5%
- **Feed Discount** : PF live estime = 1 + (PF_oos - 1) x 0.70 (haircut 30%)
- **Bootstrap CI** : intervalles de confiance PF/WR/Sharpe (p5/p50/p95)
- **Stress Test** : sensibilite WR/avg_win, break-even WR, distance au break-even
- **Overfitting Indicator** : ratio PF IS/OOS avec severite (LOW/MODERATE/SEVERE)
- **Live Readiness Score** : score composite 0-100 (DEPLOY/MONITOR/PAPER_TRADE/REJECT)
- **Version alignment** : toutes les references coherentes a 6.0.3

---

## Depannage

| Probleme | Solution |
|----------|----------|
| `Python n'est pas installe` | Installer Python 3.9+ et cocher "Add to PATH" |
| `ModuleNotFoundError` | `pip install -r requirements.txt` dans le venv |
| Port 5000 occupe | `python launch.py --port 8080` |
| macOS port 5000 | Desactiver AirPlay Receiver dans Preferences Systeme |
| Venv corrompu | Supprimer `venv/` et relancer `install.bat` ou `install.sh` |
