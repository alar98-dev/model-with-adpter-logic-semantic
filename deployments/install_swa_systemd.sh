#!/usr/bin/env bash
# Instala units SWA em /etc/systemd/system e habilita boot automático.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${ROOT}/deployments/systemd"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Execute com sudo: sudo bash $0"
  exit 1
fi

echo "📦 Instalando units de ${UNIT_DIR} → /etc/systemd/system/"
for unit in swa-mcp swa-subcortex swa-manifold swa-auto swa-auditory swa-obelisk; do
  install -m 0644 "${UNIT_DIR}/${unit}.service" "/etc/systemd/system/${unit}.service"
  echo "  ✓ ${unit}.service"
done
install -m 0644 "${UNIT_DIR}/swa.target" "/etc/systemd/system/swa.target"
echo "  ✓ swa.target"
# Remover artefato de instalações antigas (copiava swa.target como swa.target.service)
rm -f /etc/systemd/system/swa.target.service

systemctl daemon-reload

echo "🔗 Habilitando no boot..."
systemctl enable swa.target
systemctl enable swa-mcp.service swa-subcortex.service swa-manifold.service
systemctl enable swa-auto.service swa-auditory.service swa-obelisk.service

echo ""
echo "✅ Units instaladas. Para aplicar agora:"
echo "   sudo systemctl start swa-manifold.service"
echo "   sudo systemctl restart swa-auto.service"
echo ""
systemctl is-enabled swa-manifold.service swa-auto.service 2>/dev/null || true
