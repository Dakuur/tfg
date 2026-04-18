#!/bin/bash
# HistoGAT Explorer — arranca el frontend web
# Uso: ./start_frontend.sh [port]
#
# Usa `venv` si existe (servidor CVC), si no usa `.venv` (local).

set -e
PORT=${1:-8000}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Detectar entorno virtual
if [ -d "venv" ]; then
  source venv/bin/activate
elif [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "[WARN] No se encontró ningún entorno virtual (venv / .venv)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  HistoGAT Explorer"
echo "  http://localhost:${PORT}"
echo "  (Ctrl+C para parar)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Instalar deps del frontend si faltan
pip install -q fastapi "uvicorn[standard]" 2>/dev/null || true

uvicorn frontend.main:app --host 0.0.0.0 --port "$PORT" --reload
