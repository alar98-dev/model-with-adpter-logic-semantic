#!/bin/bash
set -euo pipefail

# Backup do swa_matrix (PostgreSQL) — pg_dump custom + checksum sha256.
# Segurança de estrutura válida:
#   1. pg_restore -l valida o TOC do dump (só aceita se estruturalmente válido)
#   2. sha256 gravado em sidecar <dump>.sha256
#   3. rotação automática mantém os KEEP mais recentes
#
# Uso: bash scripts/backup_db.sh
# Timer: systemctl list-timers | grep swa-backup

DB="${SWA_BACKUP_DB:-swa_matrix}"
BACKUP_ROOT="/var/lib/swa/backups/pg"
KEEP="${SWA_BACKUP_KEEP:-7}"
STAMP=$(date +%Y%m%d_%H%M%S)
DUMP="$BACKUP_ROOT/${DB}_${STAMP}.dump"
ERR_LOG="/tmp/swa_backup_err.log"

mkdir -p "$BACKUP_ROOT"

echo "[backup_db] dump de '$DB' -> $DUMP ..."

if pg_dump -Fc -d "$DB" -f "$DUMP" 2>"$ERR_LOG"; then
    sha256sum "$DUMP" | awk '{print $1}' > "$DUMP.sha256"
    if pg_restore -l "$DUMP" >/dev/null 2>&1; then
        echo "[backup_db] OK: $DUMP ($(du -h "$DUMP" | cut -f1)) sha256=$(cat "$DUMP.sha256" | head -c 16)..."
    else
        echo "[backup_db] ERRO: dump estruturalmente INVALIDO — removendo $DUMP"
        rm -f "$DUMP" "$DUMP.sha256"
        exit 1
    fi
    ls -1t "$BACKUP_ROOT"/*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
    ls -1t "$BACKUP_ROOT"/*.dump.sha256 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f
    echo "[backup_db] rotação: mantendo os $KEEP dumps mais recentes."
else
    echo "[backup_db] ERRO no pg_dump:"
    cat "$ERR_LOG"
    rm -f "$DUMP"
    exit 1
fi
