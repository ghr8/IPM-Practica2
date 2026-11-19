#!/usr/bin/env bash
# Instalacion en Linux / macOS: crea el entorno virtual e instala dependencias.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python download_model.py
echo
echo "Instalacion completada. Ejecuta: ./run.sh"
