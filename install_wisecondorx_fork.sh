#!/bin/bash
# Script à exécuter sur le cluster pour installer le fork fcliquet de WisecondorX
set -euo pipefail

module load Python/3.10.13

# Vérifier que python3.10 est bien utilisé
echo "Python version: $(python3 --version)"
echo "Python path: $(which python3)"

VENV_DIR="/pasteur/helix/projects/ghfc_wgs/tools/wisecondorx_v2_venv"

# Supprimer le venv précédent s'il existe
rm -rf "${VENV_DIR}"

echo "Creating virtual environment at ${VENV_DIR} ..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

# Vérifier que le venv utilise bien Python 3.10
echo "Venv Python: $(python --version)"

echo "Upgrading pip ..."
python -m pip install --upgrade pip

echo "Installing WisecondorX fork (fcliquet v2.0.0) ..."
pip install git+https://github.com/fcliquet/WisecondorX.git

echo "Verifying installation ..."
WisecondorX --version

echo "Done. Activate with: source ${VENV_DIR}/bin/activate"
