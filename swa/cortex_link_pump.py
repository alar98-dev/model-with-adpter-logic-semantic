"""Processo dedicado: bomba de ligação córtex↔subcortex (GAPs 1-3).

Roda em processo separado (GIL próprio) para não competir com a geração
Gemma no MainThread. Lê o estado tensorial compartilhado em
/dev/shm/aeon_latent_chaos.dat (64x64 float32) e publica:
  - ContextBus.broadcast("cortex_cycle", ...)   (GAP 1)
  - core.intent.jsonl                            (GAP 2)
  - core.ooda.jsonl                              (GAP 3)

O loop é controlado por wall-clock para garantir execução periódica
mesmo sob contenção de CPU (multiprocessing com GIL independente).
"""

import json
import os
import struct
import sys
import time

sys.path.insert(0, "/home/alifer/swa-aeon")
sys.path.insert(0, "/services/model-with-adpter-logic-semantic")

CONTEXT_BUS_PATH = "/dev/shm/aeon_context_bus.dat"
LATENT_CHAOS_PATH = "/dev/shm/aeon_latent_chaos.dat"
SHM_DIR = "/dev/shm/aeon"
INTENT_PATH = os.path.join(SHM_DIR, "core.intent.jsonl")
OODA_PATH = os.path.join(SHM_DIR, "core.ooda.jsonl")

AGENT = "aeon_cortex"

# ─── CALIBRAÇÃO DE PULSES (Teorema de Nyquist na malha córtex↔subcortex) ───
# O latent_chaos muda a cada ~60s (ciclo do subcortex). Para capturar sem
# perder estado, o pump amostra a 2x (Nyquist): T_latent/2 = 30s.
# Qualquer escrita mais rápida gera intent REDUNDANTE (delta=0 no subcortex)
# e esgota o ContextBus (8192 slots) prematuramente.
INTERVAL = 30.0          # período de amostragem do latent_chaos (ContextBus)
INTENT_PERIOD = 30.0     # core.intent.jsonl — alinhado ao pulso subcortex
OODA_PERIOD = 30.0       # core.ooda.jsonl — traço observável por ciclo
# Delta adaptativo: tolerância relativa de variação do latent_chaos.
# Se o estado não mudou mais que DELTA_EPS, o intent é considerado redundante
# e NÃO é reemitido (evita falso-pulso sináptico).
DELTA_EPS = 1e-3


def _read_float32_mmap(path: str, size: int = 16384):
    """Lê os bytes brutos do tensor 64x64 float32 compartilhado."""
    try:
        with open(path, "rb") as f:
            raw = f.read(size)
        if len(raw) != size:
            return None
        return struct.unpack(f"{size // 4}f", raw)
    except Exception:
        return None


def _metrics_from_bytes(values):
    """Deriva métricas tensoriais (entropia/deriva) dos 4096 floats 64x64."""
    if not values:
        return 0.0, 0.0, False
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    center = values[32 * 64 + 32] if n > 32 * 64 + 32 else mean
    entropy = min(1.2, max(0.0, abs(center) / 10.0 + (0.02 if variance > 5.0 else 0.0)))
    drift = abs(mean)
    return float(entropy), float(variance), variance > 5.0


def _fingerprint(values):
    """Hash leve do estado tensorial (L2-norm do sub-amostrado) para delta-check."""
    if not values:
        return 0.0
    step = max(1, len(values) // 64)
    sel = values[::step]
    return sum(v * v for v in sel) ** 0.5


def _describe_intent(entropy, variance, quantum_leap):
    """Intenção descritiva derivada das métricas, consumida pelo subcortex."""
    if quantum_leap:
        return "singularidade topológica - salto quântico cortical detectado"
    if variance > 5.0:
        return "alta tensão tensorial - redistribuição de energia entre hemisférios"
    if entropy > 0.8:
        return "estado de vigília ativa - exploração cognitiva"
    if entropy < 0.2:
        return "homeostase basal - manutenção de equilíbrio tensorial"
    return "fluxo tensorial estável - integração cortical contínua"


def _pump_loop():
    from swa.context_bus import ContextBus

    os.makedirs(SHM_DIR, exist_ok=True)
    ctx = None
    try:
        ctx = ContextBus(path=CONTEXT_BUS_PATH)
        ctx.register_agent(AGENT)
    except Exception as e:
        print(f"[CORTEX LINK] ContextBus indisponível: {e}")

    seq = 0
    last_intent = 0.0
    last_ooda = 0.0
    last_fingerprint = None
    metrics = (0.0, 0.0, False)
    _redis = None
    try:
        import redis as _r
        _redis = _r.Redis(host="127.0.0.1", port=6379, decode_responses=True)
        _redis.ping()
    except Exception:
        _redis = None

    while True:
        t = time.time()

        # Amostragem periódica do estado tensorial compartilhado
        vals = _read_float32_mmap(LATENT_CHAOS_PATH)
        if vals is not None:
            metrics = _metrics_from_bytes(vals)

        entropy, variance, quantum_leap = metrics
        current_fp = (round(entropy, 5), round(variance, 5), bool(quantum_leap),
                      round(_fingerprint(vals), 8) if vals else None)

        # ── Identidade/presença observada (quanto mais te vê, mais te nomeia) ──
        vision_name = None
        vision_conf = None
        if _redis is not None:
            try:
                raw = _redis.get("swa:vision:current_presence")
                if raw:
                    vd = json.loads(raw)
                    ids = vd.get("identities", [])
                    if ids and vd.get("faces_detected", 0) > 0:
                        vision_name = ids[0].get("name") or None
            except Exception:
                pass
        # Confiança do reconhecimento via ORM (fonte canônica: PostgreSQL)
        if vision_name:
            try:
                from swa.db.models import session_scope, Identity, RecognitionSession
                with session_scope() as db:
                    latest = (
                        db.query(RecognitionSession.confidence)
                        .join(Identity, Identity.id == RecognitionSession.identity_id)
                        .order_by(RecognitionSession.detected_at.desc())
                        .first()
                    )
                    if latest:
                        vision_conf = float(latest[0])
            except Exception:
                vision_conf = None

        # Detecção de delta sináptico: só há um "pulso" real quando o estado
        # tensorial do córtex mudou além de DELTA_EPS. Intent idêntico gera
        # delta=0 no subcortex (que o ignora) — é I/O residual sem informação.
        changed = (last_fingerprint is None) or (
            last_fingerprint[2] != current_fp[2] or
            (current_fp[3] is not None and last_fingerprint[3] is not None and
             abs(current_fp[3] - last_fingerprint[3]) > DELTA_EPS)
        )

        # GAP 3: trace OODA a cada ciclo (wal-clock) — registro observável
        if t - last_ooda >= OODA_PERIOD:
            last_ooda = t
            try:
                ooda_entry = {
                    "seq": seq + 1,
                    "ts": t,
                    "entropy": entropy,
                    "variance": variance,
                    "drift": entropy,
                    "quantum_leap": quantum_leap,
                    "phase": "observe",
                    "context": "autonomous_cycle",
                    "delta": bool(changed),
                    "source": "cortex_link_pump",
                }
                os.makedirs(os.path.dirname(OODA_PATH), exist_ok=True)
                with open(OODA_PATH, "a", encoding="utf-8") as fo:
                    fo.write(json.dumps(ooda_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        # GAP 1 + GAP 2: só emitem quando houve delta real (pulso sináptico)
        if changed:
            last_fingerprint = current_fp
            seq += 1
            # Intent nomeado quando o rosto observado é reconhecido (nome vindo
            # do ORM/respectivo reconhecimento — sem hardcode). Quando ninguém
            # é observado, cai no intent tensorial puro.
            base_intent = _describe_intent(entropy, variance, quantum_leap)
            if vision_name:
                conf_txt = f" ({vision_conf:.0%} confianca)" if vision_conf is not None else ""
                intent = (f"{vision_name} presente e reconhecido"
                          f"{conf_txt} - {base_intent}")
            else:
                intent = base_intent
            cycle = {
                "seq": seq,
                "ts": t,
                "entropy": entropy,
                "variance": variance,
                "drift": entropy,
                "quantum_leap": quantum_leap,
                "intent": intent,
                "identity": vision_name,
                "identity_confidence": vision_conf,
                "source": "cortex_link_pump",
            }
            # GAP 1: ContextBus broadcast — apenas mudança de estado
            if ctx is not None:
                try:
                    ctx.broadcast("cortex_cycle", cycle)
                except Exception:
                    pass
            # GAP 2: core.intent.jsonl — sincronizado ao pulso
            try:
                entry = dict(cycle)
                entry["sleeping"] = False
                os.makedirs(os.path.dirname(INTENT_PATH), exist_ok=True)
                with open(INTENT_PATH, "a", encoding="utf-8") as fi:
                    fi.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

        time.sleep(INTERVAL)


def main():
    try:
        _pump_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
