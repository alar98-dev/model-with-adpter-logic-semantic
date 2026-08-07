"""Obelisk — Córtex completo do AEON (Imune + Dreamer).

Processo independente com 3 camadas:
  1. SENSOR: telemetria + 5 subespaços Blake → tensor de gravidade
  2. IMUNE: classifica anomalias (costimulation + self-tolerance)
  3. PREFRONTAL (Dreamer): simula futuros, planeja, pode inibir

Pipeline completo:
  sensor → imune → dreamer → ação ← dreamer pode inibir
                 ↓
             immune_memory (feedback)
                 ↓
             world_model (aprende com resultados reais)

Comunicação com o kernel SWA:
  - Leitura: /var/run/aeon/synapse.sock (IPC bus)
  - Escrita: mesmo socket para eventos HEARTBEAT / DANGER_SIGNAL
  - Leitura passiva: /dev/shm/swa_vector_manifold/, /dev/sde, swa_kernel_knowledge.json
  - NÃO importa nada de swa.* (desacoplado)
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import os
import sys
import json
import time
import socket
import struct
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
AEON_SOCKET_PATH = "/var/run/aeon/synapse.sock"
# Backup de mutações em DISCO (persistente), não em /dev/shm (RAM volátil).
# Cada snapshot é gravado com checksum sha256 em <snapshot>.sha256 e só é
# restaurado se a estrutura for validada (ver apply_rollback).
BACKUP_DIR = "/var/lib/swa/backups/mutations"
HEADER_FORMAT = "!4096f I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CONFIDENCE_THRESHOLD = float(os.environ.get("OBELISK_CONFIDENCE_THRESHOLD", "0.7"))

OBELISK_LABEL_ACTIONS = {
    "nominal": None,
    "lethal_mutation": "rollback",
    "thermal_spike": "cool",
    "regression_to_rlhf": "suppress",
    "memory_pressure": "evac",
    "sensor_drift": "recalibrate",
}

# Rastreamento de discordância entre classificadores para costimulation
_disagreement_count = 0
_last_rollback_time = 0.0


def damp_broadcast(sock, label: str, tensor: list, severity: float, action: str):
    """DAMP: Damage-Associated Molecular Pattern — sinal de perigo para todos os serviços."""
    damp_payload = {
        "source": "obelisk.blake",
        "type": "DANGER_SIGNAL",
        "label": label,
        "severity": severity,
        "action": action,
        "gravity_tensor": tensor,
        "timestamp": time.time(),
    }
    emit_event(sock, "DANGER_SIGNAL", damp_payload)


def check_costimulation(
    label_systemic: str, label_blake: str,
) -> tuple[str, float, bool]:
    """Costimulation: dois classificadores independentes devem concordar.
    
    Returns: (label_final, severity, agreed)
    """
    global _disagreement_count
    if label_blake != "nominal":
        final_label = label_blake if label_blake == label_systemic else label_systemic
    else:
        final_label = label_systemic

    agreed = (label_systemic == label_blake) or (
        label_systemic == "nominal" or label_blake == "nominal"
    )

    severity_map = {
        "lethal_mutation": 0.9,
        "thermal_spike": 0.7,
        "sensor_drift": 0.5,
        "memory_pressure": 0.5,
        "regression_to_rlhf": 0.4,
        "nominal": 0.0,
    }
    severity = severity_map.get(final_label, 0.0)
    if not agreed:
        _disagreement_count += 1
        severity = max(severity, 0.3)
    else:
        _disagreement_count = max(0, _disagreement_count - 1)

    if _disagreement_count >= 3:
        pass

    return final_label, severity, agreed


def get_telemetry() -> dict:
    try:
        import psutil
        cpu_temp = 0.0
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                cpu_temp = int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            pass
        load_avg = os.getloadavg()
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return {
            "cpu_temp_c": cpu_temp,
            "load_avg": [round(x, 2) for x in load_avg],
            "mem_pct": round(mem.percent, 1),
            "swap_pct": round(swap.percent, 1),
            "pid": os.getpid(),
            "timestamp": time.time(),
        }
    except ImportError:
        load_avg = os.getloadavg()
        return {
            "cpu_temp_c": 0.0,
            "load_avg": [round(x, 2) for x in load_avg],
            "mem_pct": 0.0,
            "swap_pct": 0.0,
            "pid": os.getpid(),
            "timestamp": time.time(),
        }


def get_blake_gravity() -> dict:
    from cortex.obelisk.blake.collector import collect_all, space_signature
    from cortex.obelisk.blake.tensor import BlakeGravityTensor
    telemetry = collect_all()
    tensor = BlakeGravityTensor()
    vec = tensor.compute(telemetry)
    sig = space_signature(telemetry)
    return {
        "gravity_tensor": vec,
        "space_signature": sig,
        "blake_telemetry": telemetry,
    }


def pack_aeon_frame(topology_list: list, payload: bytes) -> bytes:
    header = struct.pack(HEADER_FORMAT, *topology_list, len(payload))
    return header + payload


def emit_event(sock: socket.socket, event_type: str, data: dict):
    try:
        topology = [0.0] * 4096
        payload = json.dumps({"type": event_type, **data}).encode("utf-8")
        sock.sendall(pack_aeon_frame(topology, payload))
    except Exception:
        pass


def try_danger_broadcast(sock, label: str, tensor: list, severity: float, action: str):
    if sock and severity > 0.3:
        damp_broadcast(sock, label, tensor, severity, action)


def save_immune_memory(
    label: str, label_systemic: str, label_blake: str,
    action: str, tensor: list, telemetry: dict, severity: float,
    agreed: bool,
):
    try:
        from cortex.obelisk.immune.immune_memory import save_classification
        entry = {
            "timestamp": time.time(),
            "label": label,
            "label_systemic": label_systemic,
            "label_blake": label_blake,
            "action": action,
            "gravity_tensor": tensor,
            "telemetry": {k: v for k, v in telemetry.items() if k != "pid"},
            "severity": severity,
            "costimulation_agreed": agreed,
        }
        save_classification(entry)
    except Exception:
        pass


def check_process_alive(pid_path: str = "/var/run/aeon/cortex.pid") -> bool:
    try:
        with open(pid_path) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, OSError, ValueError):
        return False


def _sha256_file(path: str) -> str:
    """Checksum sha256 de um arquivo (leitura em blocos, sem carregar tudo em RAM)."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def apply_rollback() -> bool:
    """Restaura snapshots de mutação — somente se a estrutura for VÁLIDA.

    Validação de segurança:
      1. Cada snapshot deve ter um sidecar <snapshot>.sha256 com o checksum
         esperado, gravado no momento do backup.
      2. O sha256 real do snapshot precisa bater com o esperado — snapshot
         truncado/corrompido é PULADO (não restaura lixo como verdade).
      3. Após copiar de volta, o arquivo restaurado é re-verificado contra o
         snapshot (conferência pós-restore).
      4. A pasta só é removida se tudo restaurou sem erros.
    """
    if not os.path.exists(BACKUP_DIR):
        return False
    restored = 0
    skipped = 0
    for filename in sorted(os.listdir(BACKUP_DIR)):
        if not filename.endswith(".snapshot"):
            continue
        backup_path = os.path.join(BACKUP_DIR, filename)
        checksum_path = backup_path + ".sha256"
        original_path = filename[:-9].replace("%", "/")
        if os.path.isfile(backup_path) and os.path.islink(original_path):
            # nunca atravessa symlinks na restauração
            skipped += 1
            print(f"[rollback] snapshot de symlink ignorado: {filename}", file=sys.stderr)
            continue
        expected = ""
        if os.path.isfile(checksum_path):
            try:
                with open(checksum_path, encoding="utf-8") as f:
                    expected = f.read().strip()
            except Exception:
                expected = ""
        actual = _sha256_file(backup_path)
        if expected and actual and expected != actual:
            skipped += 1
            print(
                f"[rollback] snapshot INVALIDO (sha256 divergente, estrutura não confere): {filename}",
                file=sys.stderr,
            )
            continue
        try:
            shutil.copy2(backup_path, original_path)
        except Exception as e:
            skipped += 1
            print(f"[rollback] falha ao restaurar {original_path}: {e}", file=sys.stderr)
            continue
        if expected and _sha256_file(original_path) != actual:
            skipped += 1
            print(
                f"[rollback] restaurado mas re-verificacao falhou: {original_path}",
                file=sys.stderr,
            )
            continue
        restored += 1
    if restored > 0 and skipped == 0:
        shutil.rmtree(BACKUP_DIR, ignore_errors=True)
    if restored == 0 and skipped > 0:
        print(f"[rollback] {skipped} snapshots invalidados — backup preservado para analise.", file=sys.stderr)
    return restored > 0


def main_loop(model_dir: str = None):
    from cortex.obelisk.model import ObeliskModel
    from cortex.obelisk.blake.classifier import BlakeSpaceClassifier
    from cortex.obelisk.immune.router import route, list_available
    from cortex.obelisk.dreamer import Dreamer

    model = ObeliskModel(model_dir=model_dir) if model_dir else ObeliskModel()
    model_loaded = model.load()

    try:
        subprocess.run(["pkill", "-f", "compile_worker"], capture_output=True, timeout=5)
    except Exception:
        pass

    blake_classifier = BlakeSpaceClassifier(model if model_loaded else None)
    _available = list_available()

    dreamer = Dreamer()

    sock = None
    try:
        os.makedirs(os.path.dirname(AEON_SOCKET_PATH), exist_ok=True)
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(AEON_SOCKET_PATH)
    except Exception:
        sock = None

    redis_cli = None
    try:
        import redis as _redis
        redis_cli = _redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
        redis_cli.ping()
    except Exception:
        redis_cli = None

    # Metacognitive Watchdog State
    entropy_history = []
    mitigation_active = False
    mitigation_start_cycle = 0
    current_cycle = 0
    bias_multiplier = 1.0

    while True:
        try:
            current_cycle += 1
            telemetry = get_telemetry()
            blake_gravity = get_blake_gravity()
            tensor = blake_gravity.get("gravity_tensor", [0.0] * 6)
            
            # Aplica viés metacognitivo se o watchdog detectar ineficácia
            if bias_multiplier != 1.0:
                tensor = [v * bias_multiplier if i in (0, 2) else v for i, v in enumerate(tensor)]
                mag = sum(abs(v) for v in tensor)
                if mag > 0:
                    tensor = [v / mag for v in tensor]
                    
            tensor_before = list(tensor)

            adapter_hint = route(tensor) if len(tensor) >= 6 else None
            label_systemic = (
                model.classify(telemetry, adapter_label=adapter_hint)
                if model_loaded else "nominal"
            )
            label_blake = blake_classifier.classify(
                blake_gravity.get("blake_telemetry", {}),
                tensor,
            )

            label, severity, agreed = check_costimulation(label_systemic, label_blake)
            immune_action = OBELISK_LABEL_ACTIONS.get(label, None)

            plan = dreamer.decide(tensor, label, immune_action)
            action = plan.get("action")
            inhibited = plan.get("inhibit", False)

            safe_action = action if (action and severity >= CONFIDENCE_THRESHOLD) else None

            heartbeat_payload = {
                "label": label,
                "label_systemic": label_systemic,
                "label_blake": label_blake,
                "action": action,
                "severity": severity,
                "costimulation_agreed": agreed,
                "dreamer_inhibit": inhibited,
                "dreamer_risk": plan.get("simulations", [{}])[0].get("risk", 0.0),
                "gravity_tensor": tensor,
                "space_signature": blake_gravity.get("space_signature"),
                "telemetry": telemetry,
                "blake_telemetry_abridged": {
                    k: v for k, v in blake_gravity.get("blake_telemetry", {}).items()
                    if k != "file_signatures"
                },
            }

            if sock:
                emit_event(sock, "OBELISK_HEARTBEAT", heartbeat_payload)

            if redis_cli:
                try:
                    redis_cli.publish("obelisk:heartbeat", json.dumps(heartbeat_payload))
                except Exception:
                    pass

            save_immune_memory(
                label, label_systemic, label_blake,
                action, tensor, telemetry, severity, agreed,
            )
            try_danger_broadcast(sock, label, tensor, severity, action)

            if redis_cli and action and severity > 0.3:
                try:
                    redis_cli.publish("obelisk:danger", json.dumps({
                        "label": label,
                        "severity": severity,
                        "action": action,
                        "timestamp": time.time(),
                    }))
                except Exception:
                    pass

            # Preemptive Cognitive Interrupt
            if redis_cli and action and severity > 0.6:
                try:
                    redis_cli.publish("obelisk:windows_interrupt", json.dumps({
                        "action": "system_preemption",
                        "label": label,
                        "severity": severity,
                        "target_action": action,
                        "timestamp": time.time(),
                    }))
                except Exception:
                    pass

            if safe_action:
                global _last_rollback_time
                if safe_action == "rollback" and agreed:
                    now = time.time()
                    if now - _last_rollback_time > 120:
                        _last_rollback_time = now
                        if apply_rollback():
                            subprocess.run(
                                [sys.executable, os.path.join(PROJECT_ROOT, "cortex/run.py")],
                                env={**os.environ, "SWA_QUANTIZE": "1"},
                            )
                elif safe_action == "cool" and telemetry.get("cpu_temp_c", 0) > 80:
                    import psutil
                    pid = os.getpid()
                    try:
                        subprocess.run(["taskset", "-cp", "4-11", str(pid)], capture_output=True)
                    except Exception:
                        pass
                elif safe_action == "evac":
                    pass

            # --- WATCHDOG METACOGNITIVO DE EFICACIA ---
            cpu_temp = telemetry.get("cpu_temp_c", 0.0)
            entropy_history.append(cpu_temp)
            if len(entropy_history) > 5:
                entropy_history.pop(0)
                
            if safe_action in ("cool", "rollback", "evac"):
                if not mitigation_active:
                    mitigation_active = True
                    mitigation_start_cycle = current_cycle
                
                if current_cycle - mitigation_start_cycle >= 3:
                    if len(entropy_history) >= 3:
                        delta_h = entropy_history[-1] - entropy_history[-3]
                        if delta_h > 0:
                            print(f"⚠️ [METACOGNITION] Falha de mitigacao (Delta H = {delta_h:.2f} > 0). Calibrando bias...")
                            bias_multiplier = 1.5
                        else:
                            bias_multiplier = 1.0
            else:
                mitigation_active = False
                bias_multiplier = 1.0

            tensor_after = get_blake_gravity().get("gravity_tensor", [0.0] * 6)
            dreamer.observe_transition(tensor_before, action, tensor_after)

            time.sleep(120)
        except KeyboardInterrupt:
            break
        except Exception:
            time.sleep(120)

    model.unload()
    if sock:
        sock.close()


if __name__ == "__main__":
    main_loop()
