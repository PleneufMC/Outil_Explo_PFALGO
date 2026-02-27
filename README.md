# PF AI Lab 5.1.2

**Moteur Python corrige pour P.F.Algo V.69.2 + Interface Web 8 pages**
*Integre les corrections D1-D13 identifiees par PineGuard*

> *"Meme signal, meme barre, meme resultat."*

---

## Table des matieres

1. [Presentation](#presentation)
2. [Architecture du projet](#architecture-du-projet)
3. [Installation](#installation)
4. [Demarrage rapide](#demarrage-rapide)
5. [Interface Web (8 pages)](#interface-web-8-pages)
6. [Fonctionnalites implementees](#fonctionnalites-implementees)
7. [Pipeline d'optimisation](#pipeline-doptimisation)
8. [Systeme de base de donnees](#systeme-de-base-de-donnees)
9. [OOS Gates (validation Out-of-Sample)](#oos-gates-validation-out-of-sample)
10. [Backup et restauration de la base](#backup-et-restauration-de-la-base)
11. [Exclusion des configs DB (deduplication)](#exclusion-des-configs-db-deduplication)
12. [Corrections du moteur (PineGuard v1.1)](#corrections-du-moteur-pineguard-v11)
13. [Tests](#tests)
14. [API REST](#api-rest)
15. [Workflow hybride Python/TradingView](#workflow-hybride-pythontradingview)
16. [Implementations Pine-exactes](#implementations-pine-exactes)
17. [Matrice de confiance](#matrice-de-confiance)
18. [Instruments supportes](#instruments-supportes)

---

## Presentation

PF AI Lab 5.1.2 est le moteur Python de trading algorithmique **corrige et operationnel** pour P.F.Algo V.69.2 (Pine Script ~2687 lignes). Il offre :

- **Backtest 3 phases** (Pine-equivalent : Open execute, H/L SL, Close signal)
- **13 divergences corrigees/documentees** (D1-D13) par PineGuard
- **Optimiseur NSGA-II** multi-objectifs (Optuna) avec post-traitement parallele
- **Walk-Forward Analysis** anti-overfitting (5 fenetres, 60/40 IS/OOS)
- **Monte Carlo** 10 000 simulations (trade shuffle, classification ROBUST/STABLE/FRAGILE/REJECT)
- **Interface Web Flask** 8 pages avec palette sombre PleneufTrading
- **Base de donnees SQLite** pour stocker tous les runs et configs
- **Export/Import JSON** portable pour sauvegarder et restaurer la base entre installations
- **Exclusion des configs DB** pour forcer l'exploration de nouvelles regions parametriques

### Chiffres cles

| Metrique | Valeur |
|---|---|
| Fichiers Python | 101 |
| Lignes de code Python | ~15 100 |
| Templates HTML | 8 |
| Tests unitaires | 160/160 (1.3s) |
| Routes HTTP | 10 (toutes 200 OK) |
| Endpoints API | 14 (dont /api/health) |
| Instruments configures | 60+ |

---

## Architecture du projet

```
PF AI Lab 5.1.2/
|
|-- app.py                      # Serveur Flask principal (33 routes/API)
|-- launch.py                   # Lanceur avec auto-diagnostic des dependances
|-- launch_desktop.pyw          # Lanceur Windows avec ouverture navigateur
|-- migrate_old_db.py           # Migration DB: info/export/import (CLI standalone)
|
|-- pf_ma_optimizer/            # Couche optimisation web
|   |-- config.py               # Constantes: espaces de recherche, OOS gates,
|   |                             couts de transaction, groupes d'instruments
|   |-- optimizer.py            # Optuna NSGA-II + exclusion DB + perturbation
|   |-- backtest_engine.py      # Moteur 3 phases (Open/HL/Close)
|   |-- backtest_db.py          # SQLite: save/query/export/import runs+configs
|   |-- metrics.py              # Calmar, Sharpe, Sortino, PF, WR, OOS gates
|   |-- data_loader.py          # Chargement CSV/XLSX, parseur TradingView
|   |-- tv_export.py            # Formatage config pour copier dans TradingView
|   |-- entries/                # Signaux d'entree (4 types)
|   |   |-- donchian_chikou.py
|   |   |-- rsi_divergence.py
|   |   |-- fractal.py
|   |   |-- boll_sma.py
|   |-- filters/
|   |   |-- all_filters.py      # LT/MT trend, RSI50, DiffMA, MA dir, MAVG
|   |-- indicators/
|   |   |-- pmax.py             # PMax (direction corrigee)
|   |   |-- ma_types.py         # 9 types MA (SMA, EMA, WMA, HMA, DEMA...)
|   |-- ml/
|       |-- monte_carlo.py      # 10k sims, P(ruin), classification
|       |-- walk_forward.py     # WFA 5 fenetres, stabilite, degradation
|
|-- pf_ai_lab/                  # Couche operationnelle (CLI)
|   |-- cli.py                  # CLI: backtest, optimize, wfa, mc, audit, demo
|   |-- runner.py               # Pipeline complet data->backtest->metriques
|   |-- optimizer.py            # Optuna TPE/NSGA-II (CLI)
|   |-- walk_forward.py         # WFA (CLI)
|   |-- monte_carlo.py          # MC (CLI)
|
|-- pineguard/                  # Moteur de calcul Pine-equivalent
|   |-- indicators/             # ATR(RMA), RSI, PMax, Bollinger, Donchian,
|   |                             ADX, Stdev(ddof=0), 9 MA types
|   |-- entries/                # 6 types: Donc+Chikou, RSI Div, Boll+SMA,
|   |                             Fractal(5/5), VWAP, ORB
|   |-- filters/                # LT/MT trend continu, HTF resampling,
|   |                             entry_allowed (4 composantes), ADX regime
|   |-- engine/                 # Backtest 3 phases + couts de transaction
|   |-- metrics/                # Performance metrics Pine-equivalent
|   |-- audit/                  # Divergence tracker D1-D13, signal matcher
|   |-- config/                 # Parametres V.69.1 complets
|   |-- tests/                  # 160 tests Pine<->Python equivalence
|
|-- templates/                  # 8 pages HTML (Jinja2, dark theme)
|   |-- index.html              # Home: workflow en 5 etapes
|   |-- data.html               # Upload et previsualisation CSV/XLSX
|   |-- explore.html            # Optimisation Optuna (formulaire + progress)
|   |-- test.html               # Test de Robustesse (config manuelle)
|   |-- validate.html           # Monte Carlo + Walk-Forward sur trades TV
|   |-- portfolio.html          # Analyse multi-instruments (correlation)
|   |-- best.html               # Meilleurs configs par instrument (quality score)
|   |-- database.html           # Navigateur DB, filtres, export/import, suppression
|
|-- data/                       # Donnees (exclu de git)
|   |-- backtests.db            # Base SQLite (runs + configs)
|
|-- pyproject.toml              # Configuration projet + dependances
|-- requirements.txt            # Dependances minimales
|-- .gitignore                  # Exclusions: *.db, *.csv, __pycache__, etc.
```

---

## Installation

### Prerequis

- Python >= 3.9
- pip

### Installation rapide

```bash
# Cloner le depot
git clone https://github.com/PleneufMC/Outil_Explo_PFALGO.git
cd Outil_Explo_PFALGO

# Installer avec toutes les dependances
pip install -e ".[full]"

# Ou dependances minimales seulement
pip install pandas numpy scipy optuna flask openpyxl
```

### Dependances principales

| Package | Version min. | Role |
|---|---|---|
| pandas | >= 2.0 | DataFrames, resampling HTF |
| numpy | >= 1.24 | Calcul numerique |
| scipy | >= 1.10 | Fonctions statistiques |
| optuna | >= 3.0 | Optimiseur NSGA-II |
| flask | >= 3.0 | Serveur web |
| openpyxl | >= 3.0 | Lecture fichiers Excel TradingView |
| plotly | >= 5.0 | Graphiques (validate, portfolio) |

---

## Demarrage rapide

### Interface Web (recommande)

```bash
# Methode 1: Lanceur avec auto-diagnostic
python launch.py

# Methode 2: Lancement direct
python app.py --port 5000

# Methode 3: Sur Windows (ouvre le navigateur automatiquement)
# Double-cliquer launch_desktop.pyw
```

Le dashboard est accessible a `http://localhost:5000`.

### Ligne de commande (CLI)

```bash
# Demo complete
python -m pf_ai_lab.cli demo

# Backtest simple
python -m pf_ai_lab.cli backtest --entry-type "Donc+Chikou" --pmax-length 15

# Backtest sur vos donnees
python -m pf_ai_lab.cli backtest --csv data/US500_H1.csv --instrument US500

# Optimisation NSGA-II
python -m pf_ai_lab.cli optimize --trials 200 --metric sharpe_ratio

# Walk-Forward Analysis
python -m pf_ai_lab.cli wfa --windows 5 --trials 50

# Monte Carlo
python -m pf_ai_lab.cli mc --sims 1000

# Audit divergences
python -m pf_ai_lab.cli audit --severity haute
```

---

## Interface Web (8 pages)

> **Ordre du menu de navigation** (v5.1.2) :
> Home — 1. Data — 2. Explore — 3. Test — 4. Validate — 5. Portfolio — Best Configs — DB
>
> Les pages numerotees suivent le workflow naturel :
> chargement des donnees → exploration automatique → test de robustesse → validation Monte-Carlo → analyse portefeuille.

### 1. Home (`/`)

Page d'accueil presentant le workflow en 5 etapes avec liens directs.

### 2. Data (`/data`)

- Upload de fichiers CSV ou XLSX (max 50 MB)
- Detection automatique du format (OHLCV, TradingView export)
- Previsualisation: nombre de barres, plage de dates, apercu des 10 premieres lignes
- Le fichier uploade est conserve en session pour l'etape suivante

### 3. Explore (`/explore`)

- **Formulaire de lancement**: instrument, nombre de trials (50-5000), mode (Focused ~12 params / Full ~27 params), ratio IS (0.50-0.90)
- **Signaux d'entree** selectionnables: Donc+Chikou, RSI Divergence, Fractal, Boll+SMA, **MM Cross**
- **Checkbox "Exclude DB Configs"**: force l'exploration de nouvelles regions en penalisant les configs deja en base (empreinte + voisinage PMax +-1)
- **Panneau "Verrouiller des Parametres"** (repliable): PMax length, PMax multiplier, MA mode, Entry type, LT filter ON/OFF, LT ATR multiplier, MT filter, RSI > 50 filter, Diff MA/Red line filter
- **Barre de progression** en temps reel (polling toutes les 2s)
- **Tableau des top 20** configs avec colonnes IS/OOS, OOS Gate (PASS/FAIL), WFE badge (EXCELLENT/VERY GOOD/GOOD/ACCEPTABLE/WEAK), stabilite perturbation, Walk-Forward score
- **Modal TradingView**: copie du texte TV ou du JSON config pour validation
- **Auto-save** en base de donnees a la fin de l'optimisation

### 4. Test de Robustesse (`/test`)

- **PMax Settings** : type de MA (9 types), longueur, multiplicateur ATR, mode (Long/Short/Both)
  - Badges TradingView `Inp XX` sur chaque champ pour retrouver le parametre dans PineScript
- **Signal d'Entree** : menu deroulant avec 5 choix (Donc+Chikou, RSI Divergence, Fractal, Boll+SMA, **MM Cross**)
  - Parametres dynamiques selon le signal selectionne (ex. : don_length, rsi_div_length, MM1/MM2 type et longueur)
  - Le croisement de moyennes mobiles ("MM Cross") est traite comme un signal d'entree standard, pas un filtre
- **Filtres de tendance** :
  - **Filtre Long Terme (LT)** : toggle ON/OFF + timeframe (D, W, M), type MA, longueur MA, multiplicateur ATR — tous editables
  - **Filtre Moyen Terme (MT)** : identique au LT avec parametres independants
  - RSI > 50, MA Direction, MAvg Position — toggles independants
- **Diff MA / Red Line** : seuil en pourcentage (0.1 a 10 %), visible quand active
- **Diff Price / Red Line** : seuil en pourcentage (0.1 a 10 %), visible quand active
- **Gestion du risque (SL)** : 4 modes (Static %, Red Line, ATR-Based, None)
  - En mode "ATR-Based" : champ multiplicateur ATR visible (recommande 1.5 - 3.0)
- **Tooltips francais** et tags `Inp XX` (PineScript) sur tous les champs
- Upload de donnees OHLCV + choix instrument + ratio IS

### 5. Validate (`/validate`)

- Upload d'un export de trades TradingView (XLSX)
- **Monte Carlo**: 10 000 simulations, histogrammes drawdown/return, classification (ROBUST/STABLE/FRAGILE/REJECT), P(ruine)
- **Walk-Forward** optionnel (necessite OHLCV + config JSON)
- **Verdict combine** MC + WF: DEPLOY / MONITOR / REJECT (WF ne peut que degrader, jamais ameliorer)

### 6. Portfolio (`/portfolio`)

- Upload de 2+ fichiers de trades TV
- Matrice de correlation inter-instruments
- Courbe d'equity combinee (poids egal)
- Statistiques par instrument: trades, win rate, PnL, max DD

### 7. Best Configs (`/best`)

- Classement des meilleures configs par instrument
- **Quality Score v5.0.7** composite:
  - OOS Performance (max ~7 pts): Calmar cappe a 2.0 + bonus gate +5
  - Robustesse (max ~5.5 pts): WFE x2.5 + WF stabilite x2.0 + perturbation +0.5 + consistance +0.5
  - Penalite WFE graduee: 100-120% = -0.3, 120-150% = -0.8, >150% = -1.5
  - IS Score (tiebreaker): x0.2

### 8. Database (`/database`)

- Navigateur complet de la base avec filtres: instrument, run, OOS Gate (all/PASS only), score minimum
- Deux modes d'affichage : **Cartes** (grid) et **Tableau** (21 colonnes triables)
- Detail modal avec copie TV/JSON
- **Export CSV**: toutes les configs filtrees
- **Export JSON**: backup complet de toute la base (format `pf_ai_lab_db_v1`)
- **Import JSON**: restauration avec deduplication automatique
- **Suppression de resultats** (v5.1.2) :
  - Supprimer une **config individuelle** : bouton `×` rouge au survol de chaque carte ou en fin de ligne du tableau
  - Supprimer un **run entier** (et toutes ses configs) : selectionner un run dans le filtre, cliquer "Supprimer le Run"
  - Dialogue de confirmation en francais avant chaque suppression ("Cette action est irreversible")
  - Rafraichissement automatique de l'interface apres suppression

---

## Fonctionnalites implementees

### Moteur de backtest 3 phases

Le moteur reproduit fidelement le comportement de Pine Script `process_orders_on_close = false` :

1. **Phase 1 (Open)**: Execute les ordres en attente au prix d'ouverture
2. **Phase 2 (High/Low)**: Verifie les stop-loss sur les extremes intra-barre
3. **Phase 3 (Close)**: Evalue les signaux d'entree pour la prochaine barre

### Optimiseur NSGA-II (Optuna)

- **Mode Focused** (~12 parametres): LT/MT filtres verrouilles aux valeurs V69.1
- **Mode Full** (~27 parametres): LT/MT timeframes, MA types, longueurs deverrouilles
- **Post-traitement parallele** (ProcessPoolExecutor) sur les 20 meilleures configs:
  1. Backtest OOS (out-of-sample)
  2. Test de perturbation +-10% (5 runs, seuil 60% survie)
  3. Walk-Forward Analysis (5 fenetres, 60/40 IS/OOS, 50% overlap)

### Score composite IS

Combinaison ponderee pour le classement Optuna:
- Calmar: 35%
- Sortino: 25%
- Trade score: 20% (penalite <60 trades, bonus >200)
- DD score: 20%
- Bonus robustesse: x1.10 si PMax length >= 15 ET multiplier >= 3.0

### Monte Carlo

- 10 000 simulations par defaut
- Shuffle de l'ordre des trades (preservant les PnL reels)
- Seuil de ruine adaptatif: 1.5 x median DD si le seuil fixe est inatteignable
- Classification: ROBUST (P(ruine) <= 5%), STABLE (<= 15%), FRAGILE (<= 30%), REJECT
- Verdict final: DEPLOY, MONITOR, REJECT

### Walk-Forward Analysis

- 5 fenetres glissantes, 60% IS / 40% OOS, 50% overlap
- Score de stabilite (0-100) base sur:
  - Calmar moyen OOS, Profit Factor moyen OOS
  - Consistance (% fenetres profitables)
  - Degradation (tendance baissiere)

---

## Systeme de base de donnees

### Schema SQLite

La base `data/backtests.db` contient deux tables:

**Table `runs`** (1 ligne par optimisation):

| Colonne | Type | Description |
|---|---|---|
| id | INTEGER PK | Identifiant auto-increment |
| created_at | TEXT | Date UTC de creation |
| instrument | TEXT | Instrument (ex: UK100, EURUSD) |
| mode | TEXT | focused ou full |
| n_trials | INTEGER | Nombre de trials Optuna |
| is_ratio | REAL | Ratio IS (ex: 0.70) |
| is_start, is_end | TEXT | Plage de dates In-Sample |
| oos_start, oos_end | TEXT | Plage de dates Out-of-Sample |
| duration_sec | REAL | Duree de l'optimisation |
| n_configs | INTEGER | Nombre de configs enregistrees |
| data_file | TEXT | Chemin du fichier de donnees |
| notes | TEXT | Notes (ex: "Auto-saved from web UI") |

**Table `configs`** (1 ligne par config dans le top 20):

60+ colonnes couvrant :
- Parametres de strategie: entry_type, PMax, LT/MT filtres, entry-specific params
- Gestion du risque: sl_mode, sl_pct, order_pause, long_side, short_side
- Metriques IS: trades, win_rate, calmar, sharpe, sortino, PF, max_dd, avg_win/loss
- Metriques OOS: trades, win_rate, calmar, sharpe, sortino, PF, max_dd
- OOS Gate: oos_gate_pass (0/1)
- WFE: ratio OOS Calmar / IS Calmar
- Perturbation: stable (0/1), degradation
- Walk-Forward: stability (0-100), consistency, avg_calmar, avg_pf, degradation
- Full config: config_json (JSON dump complet), tv_export (texte TradingView)

### Requetes disponibles

```python
from pf_ma_optimizer.backtest_db import BacktestDB
db = BacktestDB()

# Statistiques
stats = db.get_stats()
# {'total_runs': 5, 'total_configs': 100, 'instruments': [...], 'oos_pass_count': 42}

# Lister les runs
runs = db.get_runs(instrument='UK100', limit=50)

# Filtrer les configs
configs = db.get_configs(instrument='UK100', oos_gate_only=True, min_score=0.5)

# Meilleures configs par instrument (quality score v5.0.7)
best = db.get_best_per_instrument(top_n=5)

# Configs explorees (pour deduplication)
explored = db.get_explored_configs('UK100')

# Export CSV
csv_string = db.export_csv(instrument='UK100', oos_gate_only=True)
```

---

## OOS Gates (validation Out-of-Sample)

Les configs doivent passer **4 conditions simultanees** pour obtenir le badge PASS:

| Condition | Seuil | Source |
|---|---|---|
| Calmar OOS | >= 0.25 | Pardo WFE: tolere ~50-60% degradation vs IS |
| Profit Factor OOS | >= 1.20 | Standard industrie |
| Max Drawdown OOS | >= seuil adaptatif | Par classe d'actif (voir ci-dessous) |
| Trades OOS | >= 15 | Significance statistique minimale |

### Seuils adaptatifs de drawdown par classe d'actif

| Classe | DD max | Exemples |
|---|---|---|
| Indices | -15% | US30, US100, UK100, GER40, JP225 |
| Crypto Majors | -18% | BTCUSD, ETHUSD, SOLUSD, BNBUSD |
| Crypto Altcoins | -20% | LTCUSD, DOGEUSD, AVAXUSD, LINKUSD |
| Metals | -12% | XAUUSD, XAGUSD |
| Energy | -12% | USOIL, UKOIL, NATGAS |
| Forex Majors | -8% | EURUSD, GBPUSD, USDJPY |
| Forex Minors | -10% | EURGBP, EURJPY, GBPJPY, AUDJPY |
| Forex Exotics | -15% | EURTRY, USDMXN, USDZAR |

### WFE (Walk-Forward Efficiency) grades

| Grade | Seuil | Description |
|---|---|---|
| EXCELLENT | >= 85% | Degradation minimale |
| VERY GOOD | >= 70% | Faible degradation |
| GOOD | >= 50% | Acceptable (minimum Pardo) |
| ACCEPTABLE | >= 40% | Degradation moderee |
| WEAK | < 40% | Forte degradation |

---

## Backup et restauration de la base

### Probleme

Le fichier `data/backtests.db` est exclu de git (`.gitignore` contient `*.db` et `data/`). Chaque nouvelle installation ou mise a jour repart avec une base vierge. Les runs d'optimisation precedents sont perdus.

### Migration depuis une version precedente

Si vous avez une installation existante avec des runs dans `data/backtests.db`, **3 methodes** sont disponibles pour recuperer vos donnees:

#### Methode 1: Copie directe du fichier (la plus simple)

Si vous mettez a jour en ecrasant les fichiers Python/HTML, le schema SQLite est **compatible** (`CREATE TABLE IF NOT EXISTS` ne touche pas aux donnees existantes). Il suffit de ne pas supprimer le dossier `data/`.

```bash
# Sur l'ancienne installation: copier la base
cp /chemin/ancien/PF-AI-Lab/data/backtests.db ./sauvegarde_backtests.db

# Sur la nouvelle installation: la remettre
cp ./sauvegarde_backtests.db data/backtests.db

# Verifier
python migrate_old_db.py info
```

**Important**: Fermez l'application avant de copier le fichier `.db` (sinon les fichiers `-shm` et `-wal` peuvent contenir des donnees non encore ecrites).

#### Methode 2: Script de migration (recommande)

Un script `migrate_old_db.py` est fourni pour exporter l'ancienne base en JSON, puis l'importer dans la nouvelle :

```bash
# Etape 1: Inspecter l'ancienne base
python migrate_old_db.py info /chemin/ancien/data/backtests.db

# Etape 2: Exporter en JSON
python migrate_old_db.py export /chemin/ancien/data/backtests.db
# -> cree pf_ai_lab_db_20260213_0942.json

# Etape 3: Importer dans la nouvelle installation
python migrate_old_db.py import pf_ai_lab_db_20260213_0942.json
# -> "5 runs importes, 100 configs"
```

#### Methode 3: Procedure de mise a jour sans perte

Pour les mises a jour futures (quand l'Export JSON est deja disponible) :

```bash
# AVANT la mise a jour: exporter via l'interface web
# Page Database > Export JSON > telecharger le fichier

# Faire la mise a jour (ecraser les fichiers)
# ...

# APRES la mise a jour: importer via l'interface web
# Page Database > Import JSON > selectionner le fichier
```

### Comment sauvegarder (nouvelle version)

#### Via l'interface web

1. Aller sur la page **Database** (`/database`)
2. Dans la barre de filtres, section **"Database Backup"**
3. Cliquer **Export JSON** -- telecharge un fichier `pf_ai_lab_db_YYYYMMDD_HHMM.json`

#### Via l'API

```bash
curl http://localhost:5000/api/db/export_json -o backup.json
```

#### Via Python

```python
from pf_ma_optimizer.backtest_db import BacktestDB
import json

db = BacktestDB()
data = db.export_full_json()
with open('backup.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
```

### Comment restaurer

#### Via l'interface web

1. Sur la nouvelle installation, aller sur **Database** (`/database`)
2. Cliquer **Import JSON** et selectionner le fichier de backup
3. Le feedback affiche: "X runs imported, Y configs, Z duplicates skipped"
4. La page se recharge automatiquement

#### Via l'API

```bash
curl -X POST -F "db_file=@backup.json" http://localhost:5000/api/db/import_json
```

#### Via Python

```python
from pf_ma_optimizer.backtest_db import BacktestDB
import json

db = BacktestDB()
with open('backup.json') as f:
    data = json.load(f)
result = db.import_full_json(data)
print(result)
# {'imported_runs': 5, 'imported_configs': 100, 'skipped_runs': 0, 'errors': []}
```

### Format du fichier JSON

```json
{
  "format": "pf_ai_lab_db_v1",
  "exported_at": "2026-02-13T08:55:00+00:00",
  "runs": [
    {
      "id": 1,
      "created_at": "2026-02-10T14:30:00+00:00",
      "instrument": "UK100",
      "mode": "focused",
      "n_trials": 500,
      "is_ratio": 0.7,
      "is_start": "2023-01-01", "is_end": "2024-06-30",
      "oos_start": "2024-07-01", "oos_end": "2025-01-01",
      "duration_sec": 42.5,
      "n_configs": 20,
      "configs": [
        {
          "rank": 1, "score": 0.875,
          "entry_type": "Donc+Chikou",
          "pmax_length": 10, "pmax_multiplier": 3.0,
          "is_calmar": 1.5, "oos_calmar": 0.8,
          "oos_gate_pass": 1, "wfe": 0.533,
          "config_json": "{...}",
          "tv_export": "..."
        }
      ]
    }
  ]
}
```

### Deduplication

L'import utilise une cle composite `(created_at, instrument, n_configs)` pour eviter les doublons. Si un run avec les memes valeurs existe deja, il est ignore (compteur `skipped_runs` incremente). Cela permet de re-importer le meme fichier sans creer de doublons.

---

## Exclusion des configs DB (deduplication)

### Probleme

Quand on relance une optimisation sur le meme instrument, Optuna peut re-explorer les memes regions parametriques et trouver des configs identiques ou quasi-identiques a celles deja en base.

### Solution: option "Exclude DB Configs"

Sur la page **Explore**, une checkbox **"Exclude DB Configs"** permet de :

1. **Charger les empreintes** de toutes les configs existantes en base pour l'instrument selectionne
2. **Penaliser les doublons exacts**: meme entry_type + PMax + filtres -> score = -998
3. **Penaliser les quasi-doublons**: meme entry_type, PMax length +-1, multiplier +-0.2 -> score = -998
4. Le nombre de configs exclues est affiche dans le compteur dynamique et dans le resume des resultats

### Implementation technique

```python
# Empreinte d'une config (18 cles normalisees)
_FINGERPRINT_KEYS = (
    'entry_type', 'pmax_length', 'pmax_multiplier', 'pmax_ma_type',
    'don_length', 'rsi_div_length', 'rsi_div_pivot', 'rsi_div_hidden',
    'fractal_buffer', 'length_bb_ma', 'mult_bb_ma', 'length_bb_sma',
    'use_lt_filter', 'lt_multiplier', 'use_mt_filter',
    'use_rsi50_filter', 'use_diff_ma_red', 'diff_ma_red_pct',
)

# Comparaison: empreinte exacte (string) + voisinage PMax
def _is_too_similar(config, explored_fps):
    fp = _config_fingerprint(config)
    if fp in explored_fps:
        return True  # exact match
    # Near match: same entry_type, PMax length +-1, multiplier +-0.2
    ...
```

### API associe

```
GET /api/db/instrument_config_count?instrument=UK100
-> {"count": 42, "instrument": "UK100"}
```

---

## Corrections du moteur (PineGuard v1.1)

### Divergences corrigees (D1-D13)

| ID | Severite | Statut | Description | Impact |
|---|---|---|---|---|
| D1 | MOYENNE | IRRED. | Bid vs Mid-price (data source) | ~15 trades |
| D2 | BASSE | IRRED. | LP aggregation | 0.5% PnL |
| D3 | BASSE | IRRED. | Timezone/bar timing | ~5 trades |
| D4 | MOYENNE | OUVERTE | Weekly bar boundaries (resample) | ~10 trades |
| D5 | BASSE | OUVERTE | Incomplete bar handling | ~3 trades |
| D6 | BASSE | OUVERTE | 4H from H1 boundaries | ~2 trades |
| D7 | **HAUTE** | **FIXE** | entry_allowed gate 4 composantes | 25 phantom |
| D8 | **HAUTE** | **FIXE** | 5/5 fractals Williams | 89 signaux |
| D9 | HAUTE | MITIGE | Deduplication fractale | double entrees |
| D10 | MOYENNE | **FIXE** | Bollinger ddof=0 (population) | 2.6% ecart |
| D11 | BASSE | MITIGE | RSI cancel logic (seuil 15 barres) | ~2 trades |
| D12 | MOYENNE | OUVERTE | HTF resampling validation | variable |
| D13 | BLOQUANT | OUVERTE | ORB signal inverse (Pine-side bug) | total |

### Corrections critiques supplementaires

- **PMax direction**: comparaison avec les stops **precedents** (`shortStopPrev`/`longStopPrev`), pas les courants
- **Bollinger std**: `ddof=0` (population) au lieu de `ddof=1` (echantillon)
- **Fractal patterns**: 5/5 patterns Williams complets (pas 3/3 tronque)
- **entry_allowed**: gate a 4 composantes corrigee (eliminait ~25 phantom trades)
- **Equity O(1)**: calcul d'equity optimise, plus de recalcul O(n) a chaque barre
- **Same-bar reversal**: gestion correcte des reversals sur la meme barre
- **Phantom trades**: ~129 -> ~15 (reduction 88%)

---

## Tests

```bash
# Tous les tests PineGuard (160 tests)
python -m pytest pineguard/tests/ -v

# Resultats attendus: 160 passed, 4 warnings (FutureWarning pandas) en ~1.3s

# Avec couverture
python -m pytest --cov=pineguard --cov=pf_ai_lab --cov-report=term-missing

# Categories de tests:
#   test_indicators/    - ATR, RSI, Stdev, PMax, MA types (35 tests)
#   test_entries/       - Donchian, RSI Div, Fractal, Boll+SMA, VWAP, ORB (37 tests)
#   test_filters/       - ADX regime, entry_allowed, HTF, trend (29 tests)
#   test_engine/        - Backtest 3 phases, transaction costs (19 tests)
#   test_metrics/       - Performance metrics (7 tests)
#   test_audit/         - Divergence tracker, signal matcher, trade comparator (24 tests)
#   test_utils/         - Crossovers, data loader (13 tests)
```

---

## API REST

### Pages HTML

| Route | Methode | Description |
|---|---|---|
| `/` | GET | Page d'accueil |
| `/data` | GET/POST | Upload et previsualisation de donnees |
| `/explore` | GET/POST | Lancement d'optimisation |
| `/test` | GET/POST | Test de Robustesse (config manuelle) |
| `/validate` | GET/POST | Validation Monte Carlo + Walk-Forward |
| `/portfolio` | GET/POST | Analyse multi-instruments |
| `/best` | GET | Meilleures configs par instrument |
| `/database` | GET | Navigateur de la base de donnees |
| `/audit` | GET | **V5.4.1** Quality Audit Dashboard |

### Endpoints API JSON

| Route | Methode | Description |
|---|---|---|
| `/api/optimization_status` | GET | Progression de l'optimisation en cours |
| `/api/optimization_result` | GET | Resultats de la derniere optimisation |
| `/api/db/runs` | GET | Liste des runs (`?instrument=UK100`) |
| `/api/db/configs` | GET | Configs filtrees (`?run_id=&instrument=&oos_gate_only=&min_score=`) |
| `/api/db/config/<id>` | GET | Detail d'une config (JSON + TV export) |
| `/api/db/best` | GET | Meilleures par instrument (`?top_n=5&instrument=`) |
| `/api/db/instrument_config_count` | GET | Nombre de configs uniques (`?instrument=`) |
| `/api/db/export_csv` | GET | Export CSV (`?instrument=&oos_gate_only=`) |
| `/api/db/export_json` | GET | Export JSON complet (backup portable) |
| `/api/db/import_json` | POST | Import JSON (multipart form, champ `db_file`) |
| `/api/db/save_current` | POST | Sauvegarde manuelle des resultats courants |
| `/api/db/delete_run/<id>` | DELETE | Suppression d'un run et ses configs |
| `/api/db/delete_config/<id>` | DELETE | Suppression d'une config individuelle |

---

## Workflow hybride Python/TradingView

```
Python EXPLORER (NSGA-II, SAFE entries, +/-10-30% disclaimer)
    |
    v Top 5-10 configs (copier via modal TradingView)
TradingView VALIDATOR (Strategy Tester = source de verite)
    |
    v Export real trades (XLSX)
Python ANALYZER (Monte Carlo 10k sims + Walk-Forward sur VRAIS trades TV)
    |
    v DEPLOY / MONITOR / REJECT verdict
```

**Important**: Python est un explorateur. TradingView reste la source de verite. Le workflow hybride compare les metriques Python vs TV reelles pour valider la fiabilite.

---

## Implementations Pine-exactes

| Composant | Regle Pine | Python |
|---|---|---|
| ATR | `ta.atr` = RMA (Wilder) | `ewm(alpha=1/n, adjust=False)` pas SMA |
| EMA | `ta.ema` = `adjust=false` | `ewm(span=n, adjust=False)` |
| Stdev | `ta.stdev` = population (ddof=0) | `rolling().std(ddof=0)` pas ddof=1 |
| PMax stops | Stateful carry-over (`shortStopPrev`) | Boucle avec `prev_ss`/`prev_ls` |
| PMax direction | Compare MAvg vs stops **precedents** | `curr_mavg > prev_ss` (pas `short_stop[i]`) |
| Crossover | `a > b AND a[1] <= b[1]` | Strict `<=`/`>=` sur barre precedente |
| Backtest | `process_orders_on_close = false` | 3 phases: Open exec, H/L SL, Close signal |
| HTF | `request.security(tf, expr)[1]` | `shift(1)` en HTF puis `ffill` vers base |
| Trend filter | Etat continu (V.68.8+) | MAvg_HTF > PMax_HTF (boolean continu) |

---

## Matrice de confiance

| Type entree | Confiance | Notes |
|---|---|---|
| Donc+Chikou | 100% | Deterministe, reproductible |
| RSI Divergence | ~97% | Cancel logic amelioree (seuil 15 barres) |
| Boll+SMA | ~90% | ddof=0 fixe, entry_allowed disponible |
| Fractal | ~85% | 5/5 patterns Williams + dedup |
| VWAP | ? | Dependant de la session, non audite |
| ORB | 0% | Signal inverse dans Pine V.69.1 |

### Matrice de sensibilite

| Parametre | Zone sensible | Zone robuste | Risque |
|---|---|---|---|
| pmax_length | < 10 | >= 15 | EMA courte = reactive aux micro-ecarts |
| multiplier | < 2.5 | >= 3.0 | PMax proche du prix = crossovers sur fil |
| don_length | < 15 | >= 20 | Canal Donchian volatil |
| sl_pct | < 2% | 3-8% | SL serre touche par micro-differences |

**Regle**: Si l'optimiseur converge vers EMA(6)x2.2, ca marche en Python mais diverge dans TradingView. Preferer EMA(15+)x3.0+ pour la reproductibilite.

---

## Instruments supportes

### Avec couts de transaction configures

| Classe | Instruments | Spread/Cout |
|---|---|---|
| Indices | US500, US100, US30, UK100, DE40, FR40, JP225 | 0.3 - 5.0 pts |
| Forex Majors | EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD | 0.00008 - 0.00025 |
| Forex Minors | EURGBP, EURJPY, GBPJPY, AUDJPY, etc. (20+) | 0.00015 - 0.0004 |
| Forex Exotics | EURTRY, USDMXN, USDZAR, etc. (16+) | 0.0003 - 0.0015 |
| Metals | XAUUSD, XAGUSD, XPTUSD, XPDUSD | 0.35 - 5.0 |
| Energy | USOIL, UKOIL, NATGAS | 0.03 - 0.015 |
| Crypto Majors | BTCUSD, ETHUSD, SOLUSD, BNBUSD, XRPUSD, ADAUSD | 15.0 - 50.0 |
| Crypto Altcoins | LTCUSD, DOGEUSD, AVAXUSD, LINKUSD, etc. (20+) | 0.1 - 5.0 |
| **Custom** | N'importe quel nom (AAPL, TSLA, MY_STRAT...) | Configurable par l'utilisateur |

### Instruments custom

Vous pouvez tester **n'importe quel instrument** (actions individuelles, ETFs, portefeuilles custom) :

1. Dans le menu deroulant "Instrument" (pages **Explore** ou **Test**), selectionnez **"Instrument custom..."**
2. Saisissez le nom de votre instrument (ex. `AAPL`, `TSLA`, `SP500_CUSTOM`)
3. Optionnel : indiquez le cout de transaction (spread) en points/unites
4. Si le cout n'est pas renseigne, le fallback de 1.0 est utilise

L'instrument custom est enregistre pour la session en cours et apparait dans les statistiques de la DB.

### Timeframes supportes

Les donnees OHLCV uploadees peuvent etre dans **n'importe quel timeframe**. Le systeme resampliera automatiquement pour les filtres LT/MT.

| Timeframe | Alias Pine | Usage recommande |
|---|---|---|
| M (mensuel) | — | Filtre LT uniquement |
| W (hebdo) | — | Filtre LT (defaut) |
| D (journalier) | — | Filtre LT ou MT |
| 4H | 240 | Filtre MT (defaut) |
| 3H | 180 | Filtre MT |
| 2H | 120 | Filtre MT |
| 1H | 60 | Filtre MT ou base |
| 45min | 45 | Filtre MT ou base |
| 30min | 30 | Filtre MT ou base |
| 15min | 15 | Filtre MT ou base |
| 10min | 10 | Base uniquement |
| 5min | 5 | Base uniquement |

---

## Depannage

### Le navigateur ne s'ouvre pas / rien ne se passe

1. **Ouvrir `pf_ai_lab.log`** dans le dossier d'installation — toutes les etapes y sont horodatees
2. Chercher `ModuleNotFoundError` → solution : `venv\Scripts\pip install -r requirements.txt`
3. Chercher `Port ... is in use` → fermer l'autre application ou `python launch.py --port 8080`
4. Si le fichier log n'existe pas → Python n'est pas installe ou le venv est casse
5. En dernier recours : `venv\Scripts\python launch.py` (console visible)

### ERR_CONNECTION_REFUSED dans le navigateur

Le navigateur s'est ouvert avant que Flask soit pret. Attendre 5 secondes et rafraichir (F5).
Si le probleme persiste, verifier `pf_ai_lab.log`.

### Erreur lors de l'optimisation

Verifier la console Flask pour les tracebacks. Les erreurs courantes :
- Division par zero → corrige par `_safe_pnl_pct()` dans `backtest_engine.py`
- NaN/Inf dans JSON → corrige par `_sanitize_for_json()` dans `app.py`
- ProcessPool crash → fallback automatique en mode sequentiel

### Verification du build

```bash
python verify_build.py          # Verification complete
python verify_build.py --quick  # Syntaxe + requirements seulement
```

Voir `BUILD_CHECKLIST.md` pour la checklist complete pre-release.

---

## Historique des versions

### v5.1.2 / P.F.Algo V.69.2 (fevrier 2026) - Version actuelle

**Instruments custom (actions, portefeuilles, etc.) :**
- Nouveau choix "Instrument custom..." dans le menu deroulant (pages Explore et Test)
- Permet de saisir un nom libre (ex. `AAPL`, `TSLA`, `MY_PORTFOLIO`)
- Champ optionnel pour le cout de transaction (spread en points/unites)
- Si non renseigne, le cout par defaut (1.0) est utilise
- L'instrument custom est enregistre dynamiquement pour la session (apparait dans les dropdowns)

**Timeframes exotiques (M45, M30, M15, H2, H3, etc.) :**
- Les filtres LT et MT supportent maintenant des timeframes etendus :
  - **LT** : M (mensuel), W, D, 4H, 2H, 1H
  - **MT** : W, D, 4H, 2H, 1H, 45min, 30min, 15min
- Les donnees de base (OHLCV) peuvent etre dans n'importe quel timeframe (M5, M15, M30, M45, H1, H2, etc.)
- Le resampling HTF supporte : M, W, D, 4H, 3H, 2H, 1H, 45min, 30min, 15min, 10min, 5min
- Aliases numeriques Pine Script (240, 180, 120, 60, 45, 30, 15, 10, 5) supportes

**Refonte UX de l'interface (page Test de Robustesse) :**
- Descriptions claires en francais avec tooltips pour chaque champ
- Badges TradingView (`Inp XX`) sur tous les champs pour identifier rapidement les parametres PineScript
- Filtres LT et MT entierement editables : timeframe (D/W/M), type MA, longueur MA, multiplicateur ATR
- Champs pourcentage (0.1 - 10 %) pour "Diff MA / Red Line" et "Diff Price / Red Line"
- Mode SL "ATR-Based" : champ multiplicateur visible quand selectionne (recommande 1.5 - 3.0)
- Suppression de la section "Signal Independant — MA Cross Signal" (source de confusion)
  - Le croisement de moyennes est desormais uniquement un signal d'entree ("MM Cross" dans la liste deroulante)

**Reorganisation du menu de navigation :**
- Ancien ordre : Home, Data, Explore, Test, Validate, Portfolio, Best Configs, DB
- Nouvel ordre : Home, **1. Data**, **2. Explore**, **3. Test**, **4. Validate**, **5. Portfolio**, Best Configs, DB
- Le workflow suit maintenant un parcours naturel : donnees → exploration → test → validation → portefeuille
- Applique de maniere coherente dans les 8 templates HTML

**Suppression de resultats dans la DB :**
- Suppression d'une config individuelle : bouton `×` au survol de chaque carte ou en fin de ligne du tableau
- Suppression d'un run entier : bouton "Supprimer le Run" (rouge) quand un run est selectionne
- Dialogue de confirmation en francais avant chaque suppression
- Nouvelles API : `DELETE /api/db/delete_config/<id>` et methode `BacktestDB.delete_config()`
- Rafraichissement automatique de l'interface apres suppression

**Panneau "Verrouiller des Parametres" (page Explore) :**
- Parametres verrouillables : PMax length, PMax multiplier, MA mode, Entry type, LT filter, LT ATR multiplier, MT filter, RSI > 50 filter, Diff MA/Red line filter
- Labels mis a jour (suppression du lock "Signal Croisement MA" devenu inutile)

**Nouveau signal d'entree :**
- **MM Cross** : croisement de moyennes mobiles (EMA/TRIMA configurable, defaut EMA 9 / EMA 21)
- Integre dans optimizer, backtest engine, TV export, config

**Correction TRIMA :**
- Implementation corrigee : double SMA `SMA(SMA(src, ceil(N/2)), floor(N/2)+1)` (matching Pine V69.2)

**Defensive coding (NaN/Inf/None guards) :**
- `backtest_db.py` : `_safe_int()` / `_safe_float()` pour l'insertion DB
- `metrics.py` : guard `_finite()` sur les 19 metriques
- `backtest_engine.py` : `_safe_pnl_pct()` (division par zero)
- `optimizer.py` : try/except ProcessPoolExecutor avec fallback sequentiel
- `app.py` : `_sanitize_for_json()` pour toutes les reponses API

**Fix lancement Windows :**
- `requirements.txt` : ajout de `flask>=3.0`, `openpyxl>=3.0`, `plotly>=5.0` (manquaient !)
- `app.py` : `threaded=True` + `host='127.0.0.1'` sur Windows + endpoint `/api/health`
- `launch_desktop.pyw` : reecrit completement — logging vers `pf_ai_lab.log`, capture stderr Flask, pre-check imports, popup d'erreur
- `install.bat` : verification post-install `import flask`
- `launch.py` : browser-open uniquement apres `/api/health` OK

**Documentation :**
- `BUILD_CHECKLIST.md` : checklist pre-release obligatoire
- `CLAUDE.md` : instructions pour les assistants IA
- `verify_build.py` : script de verification automatique du build

### v5.4.1 / P.F.Algo V.69.3 (26 fevrier 2026)

**PF_Algo V69.3 (Pine Script V6, ~2 954 lignes) :**
- Export automatique du JSON de configuration (67 parametres) dans chaque alerte TradingView
- Input `export_config_in_alerts` (ON par defaut) dans le groupe "Debug & Export"
- Construction `cfg_json` sur chaque barre (scope global, pas `barstate.islast`)
- Suffixe `PFLAB_CONFIG:{...}` ajoute a tous les alert_message (buy, sell, close)
- Export copiable via `log.info()` dans les Pine Logs (copie immediate sans attendre de trade)
- Rappel compact en bas a droite du graphique (placeholder alerte + nb parametres)
- Placeholder alerte : `{{strategy.order.alert_message}}` a utiliser dans le message
- Compatible PineConnector (commande sur la 1ere ligne, JSON sur la 2e)
- Correction : `var table` declare au scope global (pas dans un `if`)
- Aucun impact sur le fonctionnement normal de la strategie

**PF AI Lab 5.4.1 -- Import automatique depuis TradingView :**
- Nouveau module `tv_settings_parser.py` avec mapping complet Pine <-> JSON (67+ variables)
- Parser intelligent a 4 methodes : alerte enrichie, JSON brut, webhook, copie multi-ligne
- Transformation automatique des modes (SL/PT/BE display -> internal)
- Coercion booleenne (`"true"`/`"false"` -> `bool`)
- Correction des artefacts PineScript (`"3."` -> `"3.0"`)
- Nouvel onglet "Importer depuis TV" sur la page Test (3 onglets au total)
- 3 nouveaux endpoints API : `/api/parse_tv_settings`, `/api/pine_export_code`, `/api/tv_mapping_guide`

**Quality Score V2 + Audit Dashboard :**
- Nouvelle page `/audit` : dashboard de qualite avec grades A/B/C/D
- API `/api/db/audit` : donnees d'audit filtrables (instrument, grade, OOS gate)
- Quality Score V2 : performance OOS + robustesse WFE/WF + perturbation + floor rule
- Compteurs par grade et par type d'entree
- Detail modal avec flags de qualite

**Tests :**
- 175 tests (15 nouveaux pour tv_settings_parser)
- Couverture : parser alert, JSON brut, webhook, boolean coercion, float artifacts, backward compat

### v5.0.4 (fevrier 2026)

**Nouvelles fonctionnalites:**
- Export/Import JSON de la base de donnees (backup portable, deduplication automatique)
- Script de migration `migrate_old_db.py` (info/export/import pour recuperer une ancienne DB)
- Exclusion des configs DB lors de l'optimisation (empreinte + voisinage PMax)
- OOS Gate adaptatif dans le JavaScript de la page Explore (remplace -8% code en dur par DD_LIMITS par instrument)
- API `/api/db/instrument_config_count` pour le compteur dynamique
- API `/api/db/export_json` et `/api/db/import_json`
- Boutons "Export JSON" / "Import JSON" sur la page Database

**Corrections moteur (PineGuard v1.1):**
- PMax direction: comparaison avec stops precedents
- Bollinger std ddof=0
- Fractal patterns 5/5 complets
- entry_allowed gate 4 composantes
- Equity O(1) optimise
- Same-bar reversal handling
- RSI cancel logic revisee

**Infrastructure:**
- 160/160 tests passent
- 9/9 routes HTTP retournent 200
- Quality Score v5.0.7 pour le classement Best Configs

---

*PF AI Lab 5.4.1 | P.F.Algo V.69.3 (Pine Script V6) | PineGuard v1.1*
*"Meme signal, meme barre, meme resultat."*
