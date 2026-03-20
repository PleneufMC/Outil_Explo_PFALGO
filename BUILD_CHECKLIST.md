# BUILD CHECKLIST — PF AI Lab

**Ce fichier existe parce qu'on a perdu des heures a debugger un `ModuleNotFoundError: No module named 'flask'` invisible.**
**Lisez-le AVANT chaque release. Pas apres.**

---

## Contexte : pourquoi cette checklist existe

Le 13 fevrier 2026, le build V69.2 a ete livre avec un bug critique :
- `requirements.txt` ne contenait pas `flask`, `openpyxl`, ni `plotly`
- `install.bat` installe uniquement ce qui est dans `requirements.txt`
- `launch_desktop.pyw` lance Flask via `pythonw.exe` (extension `.pyw`) qui n'a **AUCUNE sortie console**
- Resultat : Flask crash a l'import, l'erreur est avalee en silence, l'utilisateur voit une fenetre vide

**3 sessions de debug ont ete necessaires** pour trouver un bug qui se resume a une ligne manquante dans un fichier texte. La chaine d'erreurs silencieuses (`pythonw.exe` + `.pyw` + `subprocess.PIPE` non lu) a rendu le diagnostic impossible sans le systeme de logging ajoute en v2.

---

## Checklist pre-release (OBLIGATOIRE)

### 1. requirements.txt — Completude des dependances

**REGLE ABSOLUE : tout module `import`e dans le code DOIT etre dans `requirements.txt`.**

```bash
# Lister tous les imports tiers (hors stdlib et modules internes)
grep -rn "^from\|^import" app.py pf_ma_optimizer/*.py pf_ma_optimizer/**/*.py \
  | grep -v "from \.\|from pf_\|import pf_\|from pineguard\|import pineguard" \
  | grep -oP "(?:from |import )\K[a-zA-Z_]+" \
  | sort -u
```

**Verifier que chaque resultat a une ligne correspondante dans `requirements.txt` :**

| Import Python | Package pip | Ligne requise dans requirements.txt |
|---|---|---|
| `flask` | flask | `flask>=3.0` |
| `pandas` | pandas | `pandas>=1.5.0` |
| `numpy` | numpy | `numpy>=1.23.0` |
| `scipy` | scipy | `scipy>=1.9.0` |
| `optuna` | optuna | `optuna>=3.0` |
| `openpyxl` | openpyxl | `openpyxl>=3.0` |
| `plotly` | plotly | `plotly>=5.0` |

**Modules stdlib (PAS besoin d'etre dans requirements.txt) :**
`os`, `sys`, `json`, `time`, `math`, `datetime`, `threading`, `socket`,
`subprocess`, `sqlite3`, `io`, `traceback`, `warnings`, `typing`,
`concurrent.futures`, `argparse`, `signal`, `webbrowser`, `logging`,
`http.client`, `collections`, `functools`, `pathlib`

**Test automatise :**
```bash
# Depuis le venv, verifier que app.py s'importe sans erreur
venv\Scripts\python -c "from app import app; print('OK')"

# Ou sous Linux/macOS
venv/bin/python -c "from app import app; print('OK')"
```

### 2. install.bat — Verification post-install

`install.bat` DOIT verifier que les imports critiques fonctionnent apres `pip install`.
Actuellement il fait :
```batch
python -c "import flask; print(f'      flask {flask.__version__}')" 2>nul
if %ERRORLEVEL% neq 0 ( ... reinstallation ... )
```

**Si vous ajoutez un nouveau package critique, ajoutez une ligne de verification.**

### 3. launch_desktop.pyw — Logging obligatoire

Le launcher desktop ecrit dans `pf_ai_lab.log`. Ce fichier est la SEULE source de diagnostic quand l'app ne demarre pas.

**REGLES :**
- Ne JAMAIS supprimer le `logging.basicConfig` au debut du fichier
- Ne JAMAIS utiliser `pythonw.exe` pour le subprocess Flask (utiliser `python.exe` + `CREATE_NO_WINDOW`)
- Toujours drainer `stdout` et `stderr` du subprocess dans le log
- Toujours verifier si le process est mort avec `self.process.poll()` avant d'attendre HTTP
- Toujours faire un pre-check des imports avant de lancer le subprocess

### 4. Test de l'archive complete

**AVANT de livrer une archive `.tar.gz` ou `.zip`, tester le flux complet :**

```bash
# 1. Extraire l'archive dans un dossier VIERGE
mkdir /tmp/test_build && cd /tmp/test_build
tar -xzf pf_algo_v69.2_complete.tar.gz

# 2. Creer un venv propre
python -m venv venv

# 3. Installer depuis requirements.txt
venv/bin/pip install -r requirements.txt   # Linux/macOS
# ou
venv\Scripts\pip install -r requirements.txt  # Windows

# 4. Verifier les imports
venv/bin/python -c "from app import app; print('OK')"

# 5. Tester /api/health
venv/bin/python -c "
import threading, time, json, http.client
import sys; sys.path.insert(0, '.')
from app import app
t = threading.Thread(target=lambda: app.run(port=9999, threaded=True), daemon=True)
t.start()
time.sleep(3)
conn = http.client.HTTPConnection('127.0.0.1', 9999, timeout=5)
conn.request('GET', '/api/health')
resp = conn.getresponse()
body = json.loads(resp.read())
assert resp.status == 200 and body['status'] == 'ok', f'FAIL: {resp.status} {body}'
print(f'HEALTH CHECK OK: {body}')
"

# 6. Nettoyer
rm -rf /tmp/test_build
```

### 5. Coherence des 4 points d'entree Flask

Il y a **4 fichiers** qui demarrent Flask. Ils doivent TOUS avoir les memes parametres :

| Fichier | host (Windows) | threaded | use_reloader |
|---|---|---|---|
| `app.py` (`__main__`) | `127.0.0.1` | `True` | selon `--debug` |
| `launch.py` | `127.0.0.1` | `True` | `False` |
| `launch_desktop.pyw` | `127.0.0.1` | `True` (via app.py) | `False` (via app.py) |
| `pf_ai_lab/cli.py` (`cmd_web`) | `127.0.0.1` | `True` | `False` |

**Si vous modifiez un parametre Flask dans l'un, modifiez-le dans les 4.**

**Note V5.5 :** Le projet a maintenant **9 templates HTML** (ajout de `audit.html`).
Si vous modifiez la navigation, mettez a jour les 9 templates.

```bash
# Verifier la coherence :
grep -n "app.run\|threaded\|use_reloader" app.py launch.py launch_desktop.pyw pf_ai_lab/cli.py
```

### 6. Coherence de version (V5.4.3+)

**REGLE : Chaque modification livree = nouvelle version.**

La version DOIT etre identique dans TOUS ces fichiers :

| Fichier | Emplacement |
|---|---|
| `app.py` | `APP_VERSION = 'X.Y.Z'` + commentaire docstring ligne 3 |
| 9 templates HTML | `<title>PF AI Lab X.Y.Z - ...` |
| `install.bat` | Titre, messages, raccourcis (~9 occurrences) |
| `start.bat` | Titre, messages (~3 occurrences) |
| `uninstall.bat` | Titre, messages, raccourcis (~5 occurrences) |
| `launch_desktop.pyw` | `APP_NAME`, `APP_VERSION`, docstring (~3 occurrences) |

**Procedure de bump :**
```bash
# 1. Mettre a jour app.py
APP_VERSION = 'X.Y.Z'

# 2. Remplacer dans TOUS les fichiers
sed -i 's/OLD_VERSION/NEW_VERSION/g' app.py templates/*.html install.bat start.bat uninstall.bat launch_desktop.pyw

# 3. Verifier ZERO reste de l'ancienne version
grep -rn 'OLD_VERSION' app.py templates/ *.bat *.pyw
# Doit retourner RIEN
```

### 7. Archive tar.gz sans doublons (V5.4.3+)

**REGLE : Ne JAMAIS mixer glob (`*.txt`) et fichiers explicites dans la commande tar.**

```bash
# MAUVAIS (requirements.txt liste 2 fois : via *.txt ET explicitement)
tar -czf archive.tar.gz *.txt requirements.txt  # DOUBLON!

# BON (lister chaque fichier explicitement, sans glob)
tar -czf archive.tar.gz \
  requirements.txt pyproject.toml \
  PF_Algo_69.2.txt PF_Algo_69.3.txt \
  README.md BUILD_CHECKLIST.md ...
```

**Verification post-build OBLIGATOIRE :**
```bash
# Verifier ZERO doublon
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | sort | uniq -d
# Doit retourner RIEN

# Verifier que requirements.txt est present exactement 1 fois
tar -tzf PF_AI_Lab_X.Y.Z.tar.gz | grep -c requirements.txt
# Doit retourner 1
```

Doublons dans un tar.gz = extraction Windows echoue silencieusement.

### 8. Suppression des archives obsoletes (V5.4.3+)

**REGLE : Supprimer TOUTES les anciennes archives avant de construire la nouvelle.**

Historique : le 26/02/2026, `PF_AI_Lab_5.3.0.tar.gz` trainait encore a cote de `PF_AI_Lab_5.4.2.tar.gz`. `verify_build.py` a detecte l'ancien fichier et reporte une fausse erreur. L'utilisateur peut aussi telecharger le mauvais fichier.

```bash
# OBLIGATOIRE avant de reconstruire
rm -f PF_AI_Lab_*.tar.gz

# Puis construire
tar -czf PF_AI_Lab_X.Y.Z.tar.gz ...

# Verifier qu'il n'y a QU'UNE archive
ls PF_AI_Lab_*.tar.gz  # doit lister exactement 1 fichier
```

### 9. Commande de build complete et sure (V5.4.3+)

**Procedure complete en une seule etape :**

```bash
# 0. Verifier la version
VERSION=$(python3 -c "import re; print(re.search(r\"APP_VERSION = '(.+?)'\", open('app.py').read()).group(1))")
echo "Building PF_AI_Lab_${VERSION}.tar.gz"

# 1. Supprimer les anciennes archives
rm -f PF_AI_Lab_*.tar.gz

# 2. Construire SANS doublons (find + sort -u + tar -T)
find . \( -name '__pycache__' -o -name '.git' -o -name 'node_modules' -o -name '*.db' -o -name 'pf_ai_lab.log' \) -prune -o \
  -type f ! -name '*.tar.gz' ! -name '*.pyc' -print \
  | sort -u \
  | tar -czf "PF_AI_Lab_${VERSION}.tar.gz" -T -

# 3. Verifications
tar -tzf "PF_AI_Lab_${VERSION}.tar.gz" | sort | uniq -d  # ZERO
tar -tzf "PF_AI_Lab_${VERSION}.tar.gz" | grep -c requirements.txt  # 1
python verify_build.py --quick
```

---

## Erreurs connues et leurs causes

### BUG-001 : "Rien ne se passe au lancement" (Windows)

**Symptome :** L'utilisateur clique sur le raccourci, rien ne se passe. Pas de navigateur, pas d'erreur.

**Causes possibles (par ordre de probabilite) :**

| # | Cause | Diagnostic | Fix |
|---|---|---|---|
| 1 | Module manquant dans le venv | Ouvrir `pf_ai_lab.log` → chercher `ModuleNotFoundError` | `venv\Scripts\pip install -r requirements.txt` |
| 2 | `requirements.txt` incomplet | Comparer imports vs requirements.txt | Ajouter le package manquant |
| 3 | Port 5000 occupe | `pf_ai_lab.log` → "Port 5000 is in use" | Fermer l'autre app ou `launch.py --port 8080` |
| 4 | Firewall bloque | `pf_ai_lab.log` → timeout HTTP | Ajouter exception Windows Defender |
| 5 | Python pas dans PATH | `pf_ai_lab.log` n'existe meme pas | Reinstaller Python avec "Add to PATH" |
| 6 | Antivirus bloque pythonw.exe | Process tue immediatement | Ajouter exception antivirus |

**Procedure de debug :**
1. Ouvrir `pf_ai_lab.log` dans le dossier d'installation
2. Si le fichier n'existe pas → Python n'est pas installe ou le venv est casse
3. Si le fichier existe → lire les dernieres lignes, l'erreur y est
4. En dernier recours : `venv\Scripts\python launch.py` (console visible)

### BUG-002 : "ERR_CONNECTION_REFUSED" dans le navigateur

**Symptome :** Le navigateur s'ouvre mais affiche une erreur de connexion.

**Causes :**
- Flask est mono-thread (manque `threaded=True`)
- Le navigateur s'ouvre avant que Flask soit pret (manque wait_for_server)
- `host='0.0.0.0'` declenche le firewall Windows

**Fix :** Deja applique dans tous les points d'entree (voir tableau ci-dessus).

### BUG-003 : Erreurs JSON (NaN, Inf, numpy types)

**Symptome :** Page blanche ou erreur 500 apres une optimisation.

**Cause :** `json.dumps` ne gere pas `numpy.float64`, `NaN`, `Inf`.

**Fix :** Fonction `_sanitize_for_json()` dans `app.py`. TOUJOURS passer les resultats par cette fonction avant `jsonify()`.

### BUG-004 : ProcessPoolExecutor crash (OSError Errno 22)

**Symptome :** Optimisation crash avec "Invalid argument" ou "too many open files".

**Cause :** Multiprocessing + numpy types dans les arguments.

**Fix :** try/except avec fallback sequentiel dans `optimizer.py`.

### BUG-005 : Archives de versions precedentes dans le dossier

**Symptome :** `verify_build.py` echoue en inspectant une vieille archive qui ne contient pas les fichiers attendus. L'utilisateur peut telecharger le mauvais fichier.

**Cause :** La commande `rm -f PF_AI_Lab_*.tar.gz` n'est pas executee avant de construire la nouvelle archive.

**Fix :** Toujours commencer la procedure de build par `rm -f PF_AI_Lab_*.tar.gz`. `verify_build.py` verifie maintenant qu'il n'y a qu'une seule archive et qu'elle correspond a `APP_VERSION`.

### BUG-006 : Version desynchronisee entre app.py et scripts Windows

**Symptome :** L'installateur affiche "PF AI Lab 5.4.0" alors que l'app est en 5.4.2. Le raccourci bureau affiche le mauvais nom.

**Cause :** Les fichiers `.bat` et `.pyw` ne sont pas mis a jour lors du version bump. Seuls `app.py` et les templates HTML sont modifies.

**Fix :** Procedure de version bump exhaustive (voir section 6 ci-dessus). `verify_build.py` verifie maintenant la coherence de version dans tous les fichiers. `grep -rn 'OLD_VERSION' app.py templates/ *.bat *.pyw` doit retourner zero resultat.

### BUG-007 : Doublons dans l'archive tar.gz

**Symptome :** `requirements.txt` apparait 2 fois dans l'archive. Sous Windows, l'extraction echoue silencieusement → `install.bat` ne trouve pas le fichier.

**Cause :** La commande tar melange un glob (`*.txt`) avec une liste explicite (`requirements.txt`), creant un doublon.

**Fix :** Ne JAMAIS mixer glob et fichiers explicites. Utiliser `find | sort -u | tar -T -` pour garantir l'unicite. `verify_build.py` verifie zero doublon dans l'archive.

---

## Regles pour les futurs developpeurs (et IA)

### REGLE 1 : Si tu ajoutes un `import xxx`, ajoute `xxx` dans `requirements.txt`
Pas demain. Pas dans le prochain commit. **Maintenant.**

### REGLE 2 : Teste dans un venv VIERGE avant de livrer
`pip install -r requirements.txt` dans un nouveau venv. Si ca casse, le build est casse.

### REGLE 3 : Ne fais JAMAIS confiance a `pythonw.exe`
Il avale tout. Utilise `python.exe` + `CREATE_NO_WINDOW` si tu veux cacher la console.

### REGLE 4 : Les 4 points d'entree Flask doivent etre synchronises
`app.py`, `launch.py`, `launch_desktop.pyw`, `pf_ai_lab/cli.py` — meme host, meme threaded, meme use_reloader.

### REGLE 5 : `pf_ai_lab.log` est sacre
C'est la seule facon de debugger sur Windows. Ne le supprime pas, ne desactive pas le logging.

### REGLE 6 : Avant de dire "bug de connexion Flask", verifie d'abord les imports
90% du temps, "le serveur ne demarre pas" = un module manquant. 10% = probleme de port/firewall.

### REGLE 7 : Supprime les archives obsoletes avant de builder
`rm -f PF_AI_Lab_*.tar.gz` en premiere commande du build. Une seule archive doit exister.

### REGLE 8 : N'affiche JAMAIS un parametre interne dans l'UI
Si un champ n'existe pas dans `PF_Algo_69.X.txt` (le Pine Script), il ne doit PAS apparaitre dans l'interface utilisateur. Les parametres internes de l'optimiseur Python (comme `pmax_ma_factor`, `pmax_ma_distance`) sont invisibles pour l'utilisateur.

### REGLE 9 : Le version bump couvre TOUS les fichiers
`app.py` + 9 templates HTML + `install.bat` + `start.bat` + `uninstall.bat` + `launch_desktop.pyw`. Verifier avec grep qu'aucune ancienne version ne subsiste.

---

## Script de verification rapide

Copier-coller dans un terminal pour verifier un build :

```python
#!/usr/bin/env python3
"""Verification rapide du build PF AI Lab."""
import sys, os, ast

def check():
    errors = []

    # 1. requirements.txt contient les packages critiques
    with open('requirements.txt') as f:
        req = f.read().lower()
    for pkg in ['flask', 'pandas', 'numpy', 'scipy', 'optuna', 'openpyxl', 'plotly']:
        if pkg not in req:
            errors.append(f"MANQUANT dans requirements.txt: {pkg}")

    # 2. Tous les fichiers Python parsent
    for f in ['app.py', 'launch.py', 'launch_desktop.pyw', 'pf_ai_lab/cli.py']:
        try:
            with open(f) as fh:
                ast.parse(fh.read())
        except SyntaxError as e:
            errors.append(f"ERREUR SYNTAXE {f}: {e}")

    # 3. app.py contient /api/health et threaded=True
    with open('app.py') as f:
        app_src = f.read()
    if '/api/health' not in app_src:
        errors.append("app.py: endpoint /api/health MANQUANT")
    if 'threaded=True' not in app_src:
        errors.append("app.py: threaded=True MANQUANT")

    # 4. launch_desktop.pyw a le logging
    with open('launch_desktop.pyw') as f:
        desktop_src = f.read()
    if 'pf_ai_lab.log' not in desktop_src:
        errors.append("launch_desktop.pyw: logging vers pf_ai_lab.log MANQUANT")
    if '/api/health' not in desktop_src:
        errors.append("launch_desktop.pyw: health check MANQUANT")

    # 5. install.bat verifie flask
    with open('install.bat') as f:
        bat_src = f.read()
    if 'import flask' not in bat_src:
        errors.append("install.bat: verification 'import flask' MANQUANTE")

    if errors:
        print("ECHEC — Problemes trouves :")
        for e in errors:
            print(f"  [X] {e}")
        sys.exit(1)
    else:
        print("OK — Toutes les verifications passent.")
        sys.exit(0)

if __name__ == '__main__':
    check()
```

Sauvegardez dans `verify_build.py` et lancez avant chaque release.
