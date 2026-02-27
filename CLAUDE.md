# CLAUDE.md — Instructions pour les assistants IA

**Projet :** PF AI Lab 5.4.3 — Moteur Python pour P.F.Algo V.69.3
**Proprieatire :** PleneufMC (PleneufTrading.com)
**Repo :** https://github.com/PleneufMC/Outil_Explo_PFALGO

---

## Ce que fait ce projet

C'est un outil d'exploration et de validation de strategies de trading algorithmique.
- **Backtest engine** reproduisant le comportement de Pine Script V69.2
- **Optimiseur NSGA-II** (Optuna) avec Walk-Forward et Monte Carlo
- **Dashboard web Flask** (8 pages) pour l'interface utilisateur
- **Installateur Windows** (`install.bat` + raccourci bureau)

L'utilisateur final est sur **Windows 10/11** et n'est pas developpeur.
Le produit doit fonctionner en double-cliquant un raccourci bureau.

---

## V5.3.0 Changelog (Tier 3 — Advanced Filters + Partial TP)

### New Features (Tier 3 Roadmap)
- **Partial Profit Taking** (Tier 3, Impact 3/5): Close TP1 size at TP1%, TP2 size
  at TP2%, remainder rides with TSL. Matches Pine V69.2 partial_tp/tp1_pct/tp2_pct.
  New exit reasons: `partial_tp1`, `partial_tp2`.
- **Session/Timing Filter** (Tier 3, Impact 3/5): Filter trades by day-of-week
  (Mon-Sun toggle) and intraday session windows (HHMM format). Matches Pine V69.2
  Session days and Session hours group.
- **VIX Filter — Security Layer 0** (Tier 3, Impact 3/5): Block trades when external
  VIX series > threshold. Risk-Off when VIX high, Risk-On when VIX low. Matches
  Pine V69.1/V69.2 Security Layer 0.
- **QQ Estimation Filter** (Tier 3, Impact 2/5): Exact port of Pine V69.2 QQEF
  oscillator (fixed from previous stub). Uses shifted EMA(RSI) cross detection.
- **Macro Filter Simplifie** (Tier 3, Impact 4/5): Reduced RS + Climate score
  using pre-computed external data. Includes compute_macro_proxy_score() for
  building RS/Climate scores from reference data (matches Pine V68.4 getRSR logic).

### Search Space Growth
- Focused: 47 -> 62 parameters (+15 Tier 3 params)
- Full: 61 -> 76 parameters (+15 Tier 3 params)
- Default config: 69 -> 90 keys (+21 including internal keys)
- Fingerprint keys: +12 new Tier 3 keys for trial exclusion
- Perturbation: +10 new numeric keys for robustness checking

### New Exit Reasons (V5.3)
- `partial_tp1` — first partial take-profit hit
- `partial_tp2` — second partial take-profit hit

### Modified Files
- `pf_ma_optimizer/backtest_engine.py` — V5.3 docstring, partial TP logic
  (position_size tracking, tp1/tp2 price levels, partial close in _close_trade)
- `pf_ma_optimizer/config.py` — V5.3 docstring, 15 new search space params,
  15 new default config entries
- `pf_ma_optimizer/optimizer.py` — suggest_params() handles all V5.3 params,
  fingerprint and perturbation include Tier 3 keys
- `pf_ma_optimizer/filters/all_filters.py` — 5 new filter functions:
  session_timing_filter(), vix_filter(), macro_filter_simplified(),
  compute_macro_proxy_score(), qq_estimation_filter() (fixed from stub);
  compute_all_filters() wires all new filters via config

---

## V5.2.0 Changelog (Risk Management Revolution)

### New Features (Sprint 1-3 Roadmap)
- **SL ATR-Based + Dynamic SL** (Tier 1, Impact 5/5): ATR(14)-based stop loss with
  dynamic ATR14/ATR50 vol ratio adjustment, matching Pine V69.2 exactly
- **SL Variable per Asset Class**: sl_pct is now optimizable (1-20%) instead of
  fixed 30%; adaptive ranges per asset class (Forex: 0.5-5%, Indices: 2-15%, Crypto: 3-20%)
- **TP Modes** (Tier 1, Impact 5/5): No / Static % / SL Ratio take profit modes
- **Break-Even + Trailing Stop** (Tier 2, Impact 4/5): BE moves SL to entry after
  X% profit; TSL follows price at distance %
- **Max Hold Days** (Tier 2, Impact 4/5): Security Layer 3 forced exit after N calendar days
- **ADX Regime Filter** (Tier 1, Impact 4/5): Gate entries by trending/ranging market using
  ADX indicator + entry type compatibility matrix
- **Volatility ATR Percentile Filter** (Tier 2, Impact 3/5): Security Layer 1 blocks
  entries when ATR is in top X percentile
- **Long/Short Direction** (Tier 1, Impact 3/5): long_side/short_side now optimizable
  (was fixed True/True)

### New Files
- `pf_ma_optimizer/indicators/atr.py` — ATR, ATR14/50, ATR percentile
- `pf_ma_optimizer/indicators/adx.py` — ADX/DMI indicator for regime filter

### Modified Files
- `pf_ma_optimizer/backtest_engine.py` — Full rewrite with 7 new exit types
- `pf_ma_optimizer/config.py` — 19 new search space params, ADX_REGIME_COMPATIBILITY,
  SL_DEFAULTS_BY_CLASS, get_sl_range_for_instrument()
- `pf_ma_optimizer/optimizer.py` — suggest_params() handles all V5.2 params,
  fingerprint includes risk params

### Search Space Growth
- Focused mode: 28 -> 47 parameters (19 new, most conditional)
- Full mode: 34 -> 61 parameters
- Active free params per trial: ~15-20 (conditional params reduce effective count)

### Pine V69.2 Coverage
- Before V5.2: ~35% of Pine parameter space explored
- After V5.2: ~75% coverage (Risk Management + Security Layers 1,3 + ADX Regime)

---

## Architecture critique

### 4 points d'entree Flask (DOIVENT etre synchronises)

```
app.py              ← execution directe (python app.py --port 5000)
launch.py           ← lanceur CLI avec diagnostics
launch_desktop.pyw  ← lanceur Windows sans console (raccourci bureau)
pf_ai_lab/cli.py    ← CLI (pf-ai-lab web)
```

**Tous les 4 DOIVENT avoir :** `threaded=True`, `host='127.0.0.1'` sur Windows, `use_reloader=False`.
**Si tu modifies un parametre Flask dans l'un, modifie-le dans les 4.**

### Chaine de lancement Windows

```
install.bat
  → cree venv + pip install -r requirements.txt
  → cree raccourci bureau vers pythonw.exe + launch_desktop.pyw

Raccourci bureau
  → pythonw.exe launch_desktop.pyw    (AUCUNE console)
    → subprocess: python.exe app.py --port 5000
    → wait_for_server(/api/health)
    → webbrowser.open(http://127.0.0.1:5000)
    → log tout dans pf_ai_lab.log
```

### Fichier de log

`pf_ai_lab.log` dans le dossier d'installation. C'est la SEULE source de diagnostic sur Windows.

---

## ERREURS A NE JAMAIS REFAIRE

### ERREUR #1 (CRITIQUE) : Oublier un package dans requirements.txt

**Historique :** Le 13/02/2026, `flask`, `openpyxl` et `plotly` etaient absents de `requirements.txt`.
`install.bat` fait `pip install -r requirements.txt`, donc ces packages n'etaient jamais installes.
`pythonw.exe` a avale l'erreur silencieusement. 3 sessions de debug pour trouver une ligne manquante.

**REGLE :** Si tu ajoutes un `import xxx` DANS N'IMPORTE QUEL FICHIER, ajoute `xxx` dans `requirements.txt` **dans le meme commit**.

**Verification :**
```bash
grep -rn "^from\|^import" app.py pf_ma_optimizer/*.py pf_ma_optimizer/**/*.py \
  | grep -v "from \.\|from pf_\|import pf_\|from pineguard\|import pineguard" \
  | grep -oP "(?:from |import )\K[a-zA-Z_]+" | sort -u
```
Comparer avec le contenu de `requirements.txt`.

### ERREUR #2 : Oublier threaded=True dans Flask

**Historique :** Werkzeug est mono-thread par defaut. Le navigateur envoie 3-4 requetes paralleles (HTML + favicon + CSS + JS). En mono-thread, la premiere bloque toutes les autres → ERR_CONNECTION_REFUSED.

**REGLE :** Toujours `app.run(threaded=True)` dans les 4 points d'entree.

### ERREUR #3 : Utiliser host='0.0.0.0' sur Windows

**Historique :** `0.0.0.0` declenche un popup Windows Defender qui peut bloquer la connexion.

**REGLE :** `host='127.0.0.1'` si `sys.platform == 'win32'`, sinon `'0.0.0.0'`.

### ERREUR #4 : Faire confiance a pythonw.exe pour le subprocess

**Historique :** `pythonw.exe` supprime stdout et stderr. Si Flask crash, l'erreur disparait.

**REGLE :** Le subprocess Flask utilise `python.exe` (pas `pythonw.exe`) + `CREATE_NO_WINDOW` pour cacher la console. `stdout` et `stderr` sont draines vers `pf_ai_lab.log`.

### ERREUR #5 : NaN/Inf/numpy types dans json.dumps

**Historique :** `json.dumps(numpy.float64('nan'))` produit `{"v": NaN}` qui est du JSON invalide.

**REGLE :** Toujours passer par `_sanitize_for_json()` dans `app.py` avant `jsonify()`.

### ERREUR #6 : Division par zero dans backtest_engine.py

**Historique :** `pnl_pct = (exit - entry) / entry * 100` crash si `entry_price == 0`.

**REGLE :** Utiliser `_safe_pnl_pct()` pour tout calcul de PnL pourcentage.

### ERREUR #7 (CRITIQUE) : Oublier de bumper la version a chaque modification

**Historique :** Le 26/02/2026, le modal Best Configs a ete completement refait mais la version est restee 5.4.1. L'utilisateur a recu un build sans pouvoir distinguer l'ancien du nouveau.

**REGLE :** Chaque modification de code livree = **nouvelle version** (patch minimum).
Modifier `APP_VERSION` dans `app.py` ET les references dans :
- 9 templates HTML (`<title>` tags)
- `install.bat` (titre, raccourci, messages)
- `start.bat` (titre, messages)
- `uninstall.bat` (titre, raccourci, messages)
- `launch_desktop.pyw` (`APP_NAME`, `APP_VERSION`, docstring)
- Commentaire d'en-tete de `app.py`

**Verification :**
```bash
grep -rn "5\.X\.Y" app.py templates/ install.bat start.bat uninstall.bat launch_desktop.pyw
# Doit retourner UNIQUEMENT la nouvelle version, zero occurrence de l'ancienne
```

### ERREUR #8 (CRITIQUE) : Doublons dans l'archive tar.gz

**Historique :** Le 26/02/2026, `requirements.txt` etait liste 2 fois dans le tar (via `*.txt` glob + explicitement). Windows ne sait pas extraire un tar avec des entrees en double → extraction echoue → `install.bat` ne trouve pas `requirements.txt` → erreur.

**REGLE :** Ne JAMAIS mixer un glob (`*.txt`) avec une liste explicite de fichiers. Choisir l'un ou l'autre.

**Verification post-build :**
```bash
# Verifier ZERO doublon dans l'archive
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | sort | uniq -d
# Doit retourner RIEN. Si une ligne apparait, l'archive est cassee.
```

### ERREUR #9 : Afficher des parametres internes dans l'UI

**Historique :** Le 26/02/2026, le modal "Config Detail" affichait `PMax MA Factor` et `PMax MA Distance` — des parametres internes de l'optimiseur Python (`pmax_ma_mode`, `pmax_ma_factor`, `pmax_ma_distance`) qui n'existent PAS dans le Pine Script. L'utilisateur ne pouvait pas les retrouver dans TradingView.

**REGLE :** L'UI ne doit afficher QUE les parametres qui existent dans le Pine Script (PF_Algo_69.X.txt). Toute valeur interne a l'optimiseur (modes internes, facteurs de transformation) ne doit JAMAIS apparaitre dans un modal ou export utilisateur.

**Verification :** Comparer chaque champ affiche dans `buildTvConfig()` (best.html) avec les `input.xxx()` du Pine Script.

### ERREUR #10 : Scripts .bat/.pyw jamais mis a jour

**Historique :** Les fichiers `install.bat`, `start.bat`, `uninstall.bat`, `launch_desktop.pyw` sont restes en version 5.4.0 pendant que `app.py` et les templates passaient a 5.4.1 puis 5.4.2. Resultat : le titre d'installation affiche "5.4.0" pour un build 5.4.2.

**REGLE :** Voir ERREUR #7. Utiliser `grep` pour traquer TOUTES les occurrences de l'ancienne version avant de committer.

### ERREUR #11 : Archives de versions precedentes non supprimees

**Historique :** Le 26/02/2026, `PF_AI_Lab_5.3.0.tar.gz` (581 KB) etait encore present dans le dossier a cote de `PF_AI_Lab_5.4.2.tar.gz`. `verify_build.py` a detecte l'archive et reporte une erreur (requirements.txt manquant dans l'ancien format). L'utilisateur peut telecharger le mauvais fichier.

**REGLE :** Avant de construire une nouvelle archive, supprimer TOUTES les anciennes archives `PF_AI_Lab_*.tar.gz`. La commande de build doit inclure un `rm -f PF_AI_Lab_*.tar.gz` en premiere etape.

**Verification :**
```bash
# Verifier qu'il n'y a QU'UNE seule archive et qu'elle correspond a APP_VERSION
ls PF_AI_Lab_*.tar.gz
# Doit retourner EXACTEMENT un fichier : PF_AI_Lab_X.Y.Z.tar.gz
```

### ERREUR #12 : Doublons dans la commande tar (glob + fichier explicite)

**Historique :** Le 26/02/2026, la commande tar utilisait `*.txt requirements.txt` — le glob `*.txt` attrapait deja `requirements.txt`, creant un doublon. Sous Windows, le decompresseur (7-Zip, WinRAR, Explorer) echoue silencieusement ou ecrase le fichier, rendant `install.bat` incapable de trouver `requirements.txt`.

**REGLE :** Utiliser une commande tar qui collecte les fichiers via `find` avec deduplication :
```bash
find . -maxdepth 1 \( -name '*.py' -o -name '*.pyw' -o -name '*.bat' -o -name '*.txt' ... \) \
  | sort -u | tar -czf archive.tar.gz -T -
```
Ou lister CHAQUE fichier explicitement sans aucun glob.

**Verification post-build OBLIGATOIRE :**
```bash
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | sort | uniq -d
# Doit retourner RIEN
```

---

## Quand tu modifies du code

### Avant de committer

1. **Syntax check :** `python -c "import ast; ast.parse(open('FICHIER').read())"`
2. **requirements.txt :** Tout nouveau `import` tiers = nouvelle ligne dans requirements.txt
3. **4 points d'entree :** Si tu touches a Flask, synchronise les 4 fichiers
4. **Tests :** `python -m pytest pineguard/tests/ -q` (si disponible)
5. **Version :** Bumper `APP_VERSION` dans app.py + sed dans TOUS les fichiers (voir ERREUR #7)
6. **Archives obsoletes :** Supprimer `PF_AI_Lab_*.tar.gz` avant de reconstruire (voir ERREUR #11)

### Avant de livrer un build

Lire et suivre `BUILD_CHECKLIST.md`. Pas de raccourcis.

### Commande de build standard

```bash
# 1. Supprimer les vieilles archives
rm -f PF_AI_Lab_*.tar.gz

# 2. Construire (sans doublons)
tar -czf PF_AI_Lab_X.Y.Z.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
  --exclude='node_modules' --exclude='PF_AI_Lab_*.tar.gz' \
  --exclude='*.db' --exclude='pf_ai_lab.log' \
  -T <(find . -maxdepth 1 \( -name '*.py' -o -name '*.pyw' -o -name '*.bat' -o -name '*.txt' -o -name '*.toml' -o -name '*.md' \) -print \
       && find templates pf_ma_optimizer pineguard pf_ai_lab tests -type f 2>/dev/null | sort -u)

# 3. Verifier
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | sort | uniq -d  # ZERO doublon
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | grep -c requirements.txt  # exactement 1
python verify_build.py --quick
```

---

## Navigation et APIs (v5.1.2)

### Ordre du menu de navigation

L'ordre dans les 9 templates est :
`Home | 1. Data | 2. Explore | 3. Test | 4. Validate | 5. Portfolio | Best Configs | Audit | DB`

**Si tu ajoutes ou reordonnes un lien nav, modifie les 9 templates.**

### APIs de suppression DB

- `DELETE /api/db/delete_run/<run_id>` — supprime un run et tous ses configs (cascade)
- `DELETE /api/db/delete_config/<config_id>` — supprime une config individuelle
- Methodes correspondantes dans `backtest_db.py` : `delete_run()` et `delete_config()`

### Section "MA Cross Signal" — SUPPRIMEE

La v5.1.2 a supprime la section "Signal Independant — Croisement de Moyennes Mobiles" de `test.html`.
Le croisement de moyennes est uniquement un **signal d'entree** ("MM Cross" dans le dropdown `entry_type`).
**Ne JAMAIS re-creer une section filtre separee pour le croisement MA.**

### Import TV / Parser — IMPORTANT (V5.4.1)

Le module `tv_settings_parser.py` contient le mapping AUTHORITATIVE entre Pine et PF AI Lab.
**Si un nouveau parametre est ajoute dans PF_Algo, il DOIT etre ajoute dans :**
1. `PINE_TO_PFLAB` (dict de mapping)
2. `_EXPORT_PARAMS` (liste pour generation du code Pine)
3. `generate_tv_mapping_guide()` (table de reference)

**Les 4 formats de parsing sont testes automatiquement (11 tests).**
Ne jamais casser la retro-compatibilite du parser (`parse_pine_json` = alias de `parse_alert_with_config`).

---

## Structure des fichiers importants

```
requirements.txt          ← LISTE COMPLETE des dependances pip
install.bat               ← Installateur Windows (venv + pip + raccourci)
launch_desktop.pyw        ← Lanceur bureau (logging, subprocess, GUI)
launch.py                 ← Lanceur CLI (diagnostics, auto-install)
app.py                    ← Dashboard Flask (routes, API, /api/health)
pf_ai_lab/cli.py          ← CLI unifie
pf_ai_lab.log             ← Log de diagnostic (genere au runtime)
BUILD_CHECKLIST.md         ← Checklist pre-release (LIRE AVANT RELEASE)
verify_build.py           ← Script de verification automatique
PF_Algo_69.3.txt          ← Script Pine V6 avec export JSON dans alertes
```

---

## Packages tiers et leurs roles

| Package | Utilise par | Role |
|---|---|---|
| `flask` | app.py | Serveur web dashboard |
| `pandas` | partout | DataFrames, resampling |
| `numpy` | partout | Calcul numerique |
| `scipy` | metrics.py, filters | Statistiques (argrelextrema, etc.) |
| `optuna` | optimizer.py | Optimisation NSGA-II |
| `openpyxl` | data_loader.py | Lecture fichiers Excel TradingView |
| `plotly` | templates | Graphiques interactifs |
| `Pillow` | launch_desktop.pyw | Generation icone systray (optionnel) |
| `pystray` | launch_desktop.pyw | Icone systray Windows (optionnel) |

**Pillow et pystray sont optionnels** — le launcher tombe en fallback Tkinter s'ils manquent.
**Tous les autres sont obligatoires.**

---

## Conventions de code

- Python 3.9+ (f-strings, type hints legeres)
- Pas de type: ignore, pas de noqa sauf justifie
- Docstrings en anglais, commentaires en francais ou anglais
- Noms de variables/fonctions en anglais, messages utilisateur en francais
- Pas d'emoji dans le code (sauf dans les labels GUI Tkinter)
- Commit messages : `type(scope): description` en anglais
