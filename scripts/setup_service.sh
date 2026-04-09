#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_service.sh — instala HistoGAT Frontend como servicio systemd
#
# Ejecutar UNA SOLA VEZ con sudo desde el directorio del proyecto:
#   sudo bash scripts/setup_service.sh
#
# Qué hace:
#   1. Genera /etc/systemd/system/histogat-frontend.service con las rutas reales
#   2. Añade regla sudoers para que el runner pueda reiniciarlo sin contraseña
#   3. Activa el servicio (enable + start)
# ─────────────────────────────────────────────────────────────────────────────
set -e

SERVICE_NAME="histogat-frontend"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
SUDOERS_FILE="/etc/sudoers.d/${SERVICE_NAME}"

# ── Detectar rutas y usuario ──────────────────────────────────────────────────
DEPLOY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_USER="${SUDO_USER:-$(whoami)}"
DEPLOY_GROUP="$(id -gn "$DEPLOY_USER")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Instalando servicio: ${SERVICE_NAME}"
echo "  Directorio: ${DEPLOY_DIR}"
echo "  Usuario:    ${DEPLOY_USER} (grupo: ${DEPLOY_GROUP})"
echo "  Unit file:  ${UNIT_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Comprobar que existe el venv ───────────────────────────────────────────────
if [ ! -f "${DEPLOY_DIR}/venv/bin/uvicorn" ]; then
  echo "[ERROR] No se encuentra venv/bin/uvicorn en ${DEPLOY_DIR}"
  echo "        Ejecuta el pipeline de CI primero para crear el venv."
  exit 1
fi

# ── Generar unit file desde la plantilla ─────────────────────────────────────
sed \
  -e "s|DEPLOY_USER|${DEPLOY_USER}|g" \
  -e "s|DEPLOY_GROUP|${DEPLOY_GROUP}|g" \
  -e "s|DEPLOY_DIR|${DEPLOY_DIR}|g" \
  "${DEPLOY_DIR}/deploy/histogat-frontend.service" \
  > "${UNIT_FILE}"

echo "[OK] Creado ${UNIT_FILE}"

# ── Regla sudoers para el runner ──────────────────────────────────────────────
# Permite: sudo systemctl restart/start/stop/status histogat-frontend
SYSTEMCTL_BIN="$(command -v systemctl)"
cat > "${SUDOERS_FILE}" <<EOF
# Permite al usuario ${DEPLOY_USER} gestionar el servicio ${SERVICE_NAME} sin contraseña
${DEPLOY_USER} ALL=(ALL) NOPASSWD: ${SYSTEMCTL_BIN} restart ${SERVICE_NAME}
${DEPLOY_USER} ALL=(ALL) NOPASSWD: ${SYSTEMCTL_BIN} start ${SERVICE_NAME}
${DEPLOY_USER} ALL=(ALL) NOPASSWD: ${SYSTEMCTL_BIN} stop ${SERVICE_NAME}
${DEPLOY_USER} ALL=(ALL) NOPASSWD: ${SYSTEMCTL_BIN} status ${SERVICE_NAME}
${DEPLOY_USER} ALL=(ALL) NOPASSWD: ${SYSTEMCTL_BIN} is-active ${SERVICE_NAME}
EOF
chmod 440 "${SUDOERS_FILE}"
echo "[OK] Creado ${SUDOERS_FILE}"

# ── Activar el servicio ───────────────────────────────────────────────────────
systemctl daemon-reload
systemctl enable "${SERVICE_NAME}"
systemctl restart "${SERVICE_NAME}"

sleep 2
echo ""
if systemctl is-active --quiet "${SERVICE_NAME}"; then
  echo "✅ Servicio activo y en ejecución"
  echo ""
  systemctl status "${SERVICE_NAME}" --no-pager -l
else
  echo "❌ El servicio no ha arrancado. Revisa los logs:"
  journalctl -u "${SERVICE_NAME}" -n 30 --no-pager
  exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Comandos útiles:"
echo "    sudo systemctl status ${SERVICE_NAME}"
echo "    sudo systemctl restart ${SERVICE_NAME}"
echo "    journalctl -u ${SERVICE_NAME} -f"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
