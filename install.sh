#!/bin/bash
# ============================================================
#   PF AI Lab 6.0.3 - Installation Linux / macOS
# ============================================================
# Usage:
#   chmod +x install.sh
#   ./install.sh
# ============================================================

set -e

VERSION="6.0.3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "  PF AI Lab $VERSION - Installation"
echo "============================================================"
echo ""
echo "  Dossier : $SCRIPT_DIR"
echo ""

cd "$SCRIPT_DIR"

# ---- Check Python ----
echo "[1/5] Verification de Python..."
if ! command -v python3 &>/dev/null; then
    echo "[ERREUR] Python 3 n'est pas installe."
    echo ""
    echo "  Ubuntu/Debian : sudo apt install python3 python3-venv python3-pip"
    echo "  macOS          : brew install python3"
    echo "  Fedora         : sudo dnf install python3"
    echo ""
    exit 1
fi

PYVER=$(python3 --version 2>&1)
echo "[OK] $PYVER"

python3 -c "import sys; exit(0 if sys.version_info >= (3,9) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[ERREUR] Python 3.9 ou superieur requis. Version actuelle : $PYVER"
    exit 1
fi
echo ""

# ---- Create virtual environment ----
echo "[2/5] Creation de l'environnement virtuel..."
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo "      Environnement virtuel deja existant, etape ignoree."
else
    python3 -m venv "$SCRIPT_DIR/venv"
    if [ $? -ne 0 ]; then
        echo "[ERREUR] Impossible de creer l'environnement virtuel."
        echo "  Essayez : sudo apt install python3-venv  (Ubuntu/Debian)"
        exit 1
    fi
    echo "[OK] Environnement virtuel cree"
fi
echo ""

# ---- Activate venv ----
source "$SCRIPT_DIR/venv/bin/activate"

# ---- Install dependencies ----
echo "[3/5] Installation des dependances (1-2 minutes)..."
pip install --upgrade pip -q 2>/dev/null
echo "      pip mis a jour"

pip install -r "$SCRIPT_DIR/requirements.txt" -q
if [ $? -ne 0 ]; then
    echo "[ERREUR] Echec de l'installation des packages."
    echo "  Essayez : venv/bin/pip install -r requirements.txt"
    exit 1
fi
echo "[OK] Toutes les dependances installees"
echo ""

# ---- Verify critical imports ----
echo "[4/5] Verification des imports..."
python3 -c "
import flask, pandas, numpy, optuna, scipy
print(f'      flask {flask.__version__}')
print(f'      pandas {pandas.__version__}')
print(f'      numpy {numpy.__version__}')
print(f'      optuna {optuna.__version__}')
print(f'      scipy {scipy.__version__}')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "[ERREUR] Certains packages ne sont pas installes correctement."
    echo "  Essayez : venv/bin/pip install -r requirements.txt"
    exit 1
fi
echo "[OK] Tous les imports critiques valides"
echo ""

# ---- Create data directory ----
echo "[5/5] Preparation des dossiers..."
mkdir -p "$SCRIPT_DIR/data"
echo "[OK] Dossier data/ pret"
echo ""

# ---- Create start.sh convenience script ----
cat > "$SCRIPT_DIR/start.sh" << 'STARTEOF'
#!/bin/bash
# PF AI Lab 6.0.3 - Quick launcher
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$SCRIPT_DIR/venv" ]; then
    echo "[ERREUR] Environnement virtuel introuvable."
    echo "  Lancez d'abord : ./install.sh"
    exit 1
fi

source "$SCRIPT_DIR/venv/bin/activate"
echo ""
echo "  PF AI Lab 6.0.3 - Demarrage..."
echo "  Ctrl+C pour arreter"
echo ""
python3 launch.py "$@"
STARTEOF
chmod +x "$SCRIPT_DIR/start.sh"
echo "[OK] start.sh cree (lanceur rapide)"
echo ""

# ---- Done ----
echo "============================================================"
echo "  Installation terminee !"
echo "============================================================"
echo ""
echo "  Pour lancer PF AI Lab :"
echo ""
echo "    ./start.sh                  # Port par defaut (5000)"
echo "    ./start.sh --port 8080      # Port personnalise"
echo "    ./start.sh --check          # Diagnostic uniquement"
echo ""
echo "  Ou manuellement :"
echo ""
echo "    source venv/bin/activate"
echo "    python launch.py"
echo ""
echo "============================================================"
echo ""

read -p "Lancer PF AI Lab maintenant ? (o/n) : " LAUNCH
if [ "$LAUNCH" = "o" ] || [ "$LAUNCH" = "O" ]; then
    echo ""
    echo "Lancement..."
    python3 launch.py
fi
