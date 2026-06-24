#!/usr/bin/env bash
# Setup del entorno en Linux/Mac
# Ejecutar desde la raíz del proyecto:  bash scripts/setup.sh
set -e

echo "== Creando entorno virtual (venv) =="
python3 -m venv venv
source venv/bin/activate

echo "== Actualizando pip e instalando dependencias =="
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env desde la plantilla. EDÍTALO con tus credenciales."
fi

echo ""
echo "Listo. Para conectar a Azure necesitas 'ODBC Driver 18 for SQL Server' en el SO."
echo "Prueba rápida (datos sintéticos):  python -m src.train && python -m src.predict"
