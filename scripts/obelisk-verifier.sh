#!/bin/bash
set -euo pipefail

# Obelisk Verifier — Watchdog do Watchdog
# Monitora a integridade do obelisco por fora (bash puro).
# Se detectar mutação ou falha de heartbeat → restart do serviço.
#
# Uso:
#   sudo bash scripts/obelisk-verifier.sh              # execução única
#   sudo bash scripts/obelisk-verifier.sh --watch       # loop infinito (5s)
#   systemctl start swa-obelisk-verifier.service        # via systemd

SWA_ROOT="/services/model-with-adpter-logic-semantic"
BASELINE="${OBELISK_BASELINE_SHA256:-/etc/swa/obelisk_baseline.sha256.json}"
HEARTBEAT_FILE="${OBELISK_HEARTBEAT_FILE:-/var/run/aeon/obelisk.heartbeat}"
SERVICE="swa-obelisk.service"
SELF_HASH=""

SELF=$(readlink -f "$0" 2>/dev/null || echo "$0")
if [ -f "$SELF" ]; then
    SELF_HASH=$(sha256sum "$SELF" | cut -d' ' -f1)
fi

log()  { echo "[verifier] $(date '+%H:%M:%S') $*"; }
warn() { log "⚠️  $*"; }
crit() { log "🔴 $*"; }

check_baseline() {
    if [ ! -f "$BASELINE" ]; then
        crit "Baseline não encontrado em $BASELINE"
        return 1
    fi
    return 0
}

check_self() {
    if [ -n "$SELF_HASH" ]; then
        CURRENT=$(sha256sum "$SELF" 2>/dev/null | cut -d' ' -f1)
        if [ "$CURRENT" != "$SELF_HASH" ] && [ -n "$CURRENT" ]; then
            crit "Verificador mutado! Hash mudou: $SELF_HASH → $CURRENT"
            return 1
        fi
    fi
    return 0
}

check_files() {
    local mismatches=0 total=0
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        total=$((total + 1))
        NAME=$(echo "$line" | cut -d: -f1 | tr -d ' "')
        REF_HASH=$(echo "$line" | cut -d: -f2- | tr -d ' ",')

        if [ -z "$REF_HASH" ]; then
            continue
        fi

        FULL_PATH="$SWA_ROOT/$NAME"
        if [ ! -f "$FULL_PATH" ]; then
            warn "Arquivo faltando: $NAME"
            mismatches=$((mismatches + 1))
            continue
        fi

        CUR_HASH=$(sha256sum "$FULL_PATH" 2>/dev/null | cut -d' ' -f1)
        if [ "$CUR_HASH" != "$REF_HASH" ] && [ -n "$CUR_HASH" ]; then
            warn "Mutação detectada: $NAME"
            warn "  referência: $REF_HASH"
            warn "  atual:      $CUR_HASH"
            mismatches=$((mismatches + 1))
        fi
    done < <(grep ':' "$BASELINE" | head -n -1)

    if [ "$mismatches" -gt 0 ]; then
        crit "$mismatches/$total arquivos com mutação"
        return 1
    fi
    log "OK: $total arquivos íntegros"
    return 0
}

check_heartbeat() {
    if [ ! -f "$HEARTBEAT_FILE" ]; then
        crit "Heartbeat não encontrado"
        return 1
    fi
    local now mtime diff
    now=$(date +%s)
    mtime=$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || echo 0)
    diff=$((now - mtime))
    if [ "$diff" -gt 90 ]; then
        crit "Heartbeat parado há ${diff}s (limite: 90s)"
        return 1
    fi
    log "Heartbeat: ${diff}s atrás"
    return 0
}

check_pid() {
    local pid_file="/dev/shm/aeon/obelisk.pid"
    if [ ! -f "$pid_file" ]; then
        crit "PID file não encontrado"
        return 1
    fi
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [ -z "$pid" ]; then
        crit "PID file vazio"
        return 1
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
        crit "Processo $pid não está rodando"
        return 1
    fi
    log "PID $pid: alive"
    return 0
}

check_core_heartbeat() {
    local file="/dev/shm/aeon/core.heartbeat"
    if [ ! -f "$file" ]; then
        log "Core heartbeat ainda não iniciado (boot em andamento)"
        return 0
    fi
    local hb_pid
    hb_pid=$(/home/alifer/.pyenv/versions/3.12.0/bin/python3 -c "import json;print(json.load(open('$file')).get('pid',''))" 2>/dev/null || echo "")
    if [ -n "$hb_pid" ] && ! kill -0 "$hb_pid" 2>/dev/null; then
        log "Heartbeat de processo morto (resto de instância anterior — boot/restart em andamento)"
        return 0
    fi
    local now mtime diff
    now=$(date +%s)
    mtime=$(stat -c %Y "$file" 2>/dev/null || echo 0)
    diff=$((now - mtime))
    if [ "$diff" -gt 180 ]; then
        warn "Heartbeat do núcleo parado há ${diff}s (limite: 180s)"
        return 1
    fi
    log "Core heartbeat: ${diff}s atrás"
    return 0
}

check_core_pid() {
    # O pid é informativo: durante o boot (load+compile ~3min) o core.pid fica
    # antigo e o processo vivo ainda não o regravou — reiniciar por isso
    # mataria o boot em loop. A decisão de restart é só por heartbeat obsoleto.
    local pid_file="/dev/shm/aeon/core.pid"
    if [ ! -f "$pid_file" ]; then
        return 0
    fi
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [ -z "$pid" ]; then
        return 0
    fi
    if kill -0 "$pid" 2>/dev/null; then
        log "Core PID $pid: alive"
    fi
    return 0
}

check_tensor_signature() {
    # Verifica a assinatura BLAKE3 da injeção tensorial: recomputa o hash do
    # core.tensor.bin e compara com a assinatura gravada no core.tensor.json.
    # Apenas informativo — não derruba o núcleo (restart regeneraria o tensor).
    local bin_file="/dev/shm/aeon/core.tensor.bin"
    local json_file="/dev/shm/aeon/core.tensor.json"
    if [ ! -f "$bin_file" ] || [ ! -f "$json_file" ]; then
        log "Tensor de injeção ausente (tensor sharing off)"
        return 0
    fi
    local result
    result=$(/home/alifer/.pyenv/versions/3.12.0/bin/python3 - "$bin_file" "$json_file" <<'PY'
import json, sys, hashlib
try:
    import blake3
except Exception:
    blake3 = None
bin_file, json_file = sys.argv[1], sys.argv[2]
with open(json_file, encoding="utf-8") as f:
    meta = json.load(f)
sig = meta.get("signature", "")
with open(bin_file, "rb") as f:
    data = f.read()
expected, alg = None, "?"
if sig.startswith("blake3:") and blake3 is not None:
    expected, alg = blake3.blake3(data).hexdigest(), "blake3"
    sig = sig.split(":", 1)[1]
elif sig.startswith("sha256:"):
    expected, alg = hashlib.sha256(data).hexdigest(), "sha256"
    sig = sig.split(":", 1)[1]
if expected is None:
    print("no-signature")
elif expected == sig:
    print(f"ok:{alg}")
else:
    print("mismatch")
PY
)
    case "$result" in
        ok:*)
            log "Assinatura do tensor: íntegra (${result#ok:})"
            ;;
        mismatch)
            warn "Assinatura do tensor DIVERGENTE — injeção suspeita"
            ;;
        no-signature)
            warn "Tensor sem assinatura verificável"
            ;;
        *)
            warn "Verificação de tensor indisponível"
            ;;
    esac
    return 0
}

check_ooda() {
    # Trace OODA da resposta final do núcleo (informacional, não derruba nada).
    local file="/dev/shm/aeon/core.ooda.jsonl"
    if [ ! -f "$file" ]; then
        log "Trace OODA ainda não emitido (aguardando primeira resposta)"
        return 0
    fi
    local mtime now diff last
    mtime=$(stat -c %Y "$file" 2>/dev/null || echo 0)
    now=$(date +%s)
    diff=$((now - mtime))
    last=$(tail -1 "$file" 2>/dev/null | head -c 160)
    log "Trace OODA: ${diff}s atrás | ${last}"
    return 0
}

restart_obelisk() {
    log "Reiniciando $SERVICE..."
    systemctl restart "$SERVICE" 2>/dev/null || {
        crit "Falha ao reiniciar via systemctl, tentando start..."
        systemctl start "$SERVICE" 2>/dev/null || true
    }
}

restart_core() {
    log "Reiniciando swa-aeonc.service..."
    systemctl restart swa-aeonc.service 2>/dev/null || true
}

run_checks() {
    local failed=0 core_failed=0
    check_self || failed=$((failed + 1))
    check_baseline || failed=$((failed + 1))
    check_files || failed=$((failed + 1))
    check_heartbeat || failed=$((failed + 1))
    check_pid || failed=$((failed + 1))
    check_core_heartbeat || core_failed=$((core_failed + 1))
    check_core_pid || core_failed=$((core_failed + 1))
    check_tensor_signature
    check_ooda

    if [ "$failed" -gt 0 ]; then
        restart_obelisk
        return 1
    fi
    if [ "$core_failed" -gt 0 ]; then
        restart_core
        return 1
    fi
    return 0
}

if [ "${1:-}" = "--watch" ]; then
    log "Modo watch iniciado (intervalo: 5s)"
    while true; do
        run_checks || true
        sleep 5
    done
else
    run_checks
fi
