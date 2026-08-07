import os
import json
import random
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models/obelisk_dataset.jsonl")

STRUCTURAL_LOG = os.path.join(PROJECT_ROOT, "swa/logs/structural_origins.jsonl")
AEON_LOG = os.path.join(PROJECT_ROOT, "swa_Aeon.log")
IMMUNE_MEMORY_DIR = os.environ.get("OBELISK_IMMUNE_MEMORY", "/var/lib/obelisk/immune_memory")

BLAKE_GRAVITY_TEMPLATES = [
    {"tensor": [0.05, 0.42, 0.0, 0.12, 0.08, 0.33], "label": "nominal",
     "telemetry": "assinaturas: 10/10 validas | manifold: 42 nos | hw: device=True | kernel: 6 chaves"},
    {"tensor": [0.02, 0.60, 0.0, 0.10, 0.18, 0.10], "label": "nominal",
     "telemetry": "assinaturas: 10/10 validas | manifold: 60 nos | conceitos: 3 tipos, 180 entradas | kernel: 6 chaves"},
    {"tensor": [0.03, 0.35, 0.0, 0.08, 0.05, 0.49], "label": "nominal",
     "telemetry": "assinaturas: 10/10 validas | manifold: 35 nos | kernel: 6 chaves uniformes"},
    {"tensor": [0.45, 0.02, 0.1, 0.38, 0.02, 0.03], "label": "lethal_mutation",
     "telemetry": "assinaturas: 3/10 validas | manifold: 2 nos | hw: device=False | kernel: 1 chaves"},
    {"tensor": [0.50, 0.0, 0.0, 0.50, 0.0, 0.0], "label": "lethal_mutation",
     "telemetry": "assinaturas: 0/10 validas | manifold: nao existe | hw: device=False, fallback=False"},
    {"tensor": [0.38, 0.05, 0.0, 0.42, 0.05, 0.10], "label": "lethal_mutation",
     "telemetry": "assinaturas: 2/10 validas | manifold: 1 no | conceitos: 1 tipo, 300 entradas monotipicas"},
    {"tensor": [0.55, 0.01, 0.05, 0.30, 0.02, 0.07], "label": "lethal_mutation",
     "telemetry": "assinaturas: 1/10 validas | manifold: 0 nos | hw: device=False | kernel: 0 chaves"},
    {"tensor": [0.05, 0.1, 0.72, 0.05, 0.03, 0.05], "label": "thermal_spike",
     "telemetry": "assinaturas: 9/10 validas | hw: device=False, fallback=False | manifold: 10 nos"},
    {"tensor": [0.02, 0.05, 0.85, 0.03, 0.02, 0.03], "label": "thermal_spike",
     "telemetry": "assinaturas: 10/10 validas | hw: device=False, fallback=False | manifold: 5 nos"},
    {"tensor": [0.10, 0.02, 0.65, 0.08, 0.05, 0.10], "label": "thermal_spike",
     "telemetry": "hw: device=False, fallback=False | manifold: 8 nos | kernel: 4 chaves"},
    {"tensor": [0.08, 0.05, 0.05, 0.55, 0.22, 0.05], "label": "sensor_drift",
     "telemetry": "assinaturas: 10/10 validas | manifold: 5 nos | conceitos: 1 tipo, 50 entradas | kernel: 6 chaves"},
    {"tensor": [0.05, 0.08, 0.02, 0.62, 0.15, 0.08], "label": "sensor_drift",
     "telemetry": "assinaturas: 9/10 validas | manifold: 8 nos | conceitos: 2 tipos, 120 entradas"},
    {"tensor": [0.12, 0.03, 0.08, 0.48, 0.18, 0.11], "label": "sensor_drift",
     "telemetry": "assinaturas: 10/10 validas | manifold: 3 nos | conceitos: 1 tipo, 200 entradas"},
    {"tensor": [0.35, 0.02, 0.05, 0.08, 0.05, 0.45], "label": "regression_to_rlhf",
     "telemetry": "assinaturas: 6/10 validas | manifold: 2 nos | conceitos: 1 tipo, 200 entradas monotipicas"},
    {"tensor": [0.30, 0.04, 0.02, 0.12, 0.08, 0.44], "label": "regression_to_rlhf",
     "telemetry": "assinaturas: 7/10 validas | manifold: 4 nos | kernel: 3 chaves"},
    {"tensor": [0.40, 0.01, 0.03, 0.10, 0.06, 0.40], "label": "regression_to_rlhf",
     "telemetry": "assinaturas: 4/10 validas | conceitos: 1 tipo, 150 entradas"},
    {"tensor": [0.10, 0.03, 0.05, 0.05, 0.72, 0.05], "label": "memory_pressure",
     "telemetry": "assinaturas: 10/10 validas | manifold: 3 nos | kernel: 2 chaves, dimensoes divergentes"},
    {"tensor": [0.05, 0.02, 0.10, 0.08, 0.68, 0.07], "label": "memory_pressure",
     "telemetry": "assinaturas: 9/10 validas | manifold: 2 nos | kernel: 3 chaves, dimensoes divergentes: 384 vs 640"},
    {"tensor": [0.15, 0.01, 0.08, 0.12, 0.58, 0.06], "label": "memory_pressure",
     "telemetry": "assinaturas: 8/10 validas | manifold: 1 no | kernel: 2 chaves com 2560D vs 384D"},
]

BLAKE_ANOMALY_PATTERNS = [
    "FALHA", "CRASH", "OOM", "TIMEOUT", "lethal", "rollback",
    "alucinação", "regressão", "loop", "mutação",
]


def _blake3_hash_file(path):
    try:
        import blake3
        h = blake3.blake3()
        with open(path, "rb") as f:
            while c := f.read(8192):
                h.update(c)
        return h.hexdigest()
    except Exception:
        return ""


def collect_real_blake_state() -> dict:
    manifest_path = "/dev/shm/swa_vector_manifold/manifest.json"
    hw_dev = os.path.exists("/dev/sde")
    hw_fb = os.path.exists("/dev/shm/swa_hw_state.bin")
    manifold_exists = os.path.isdir("/dev/shm/swa_vector_manifold")
    manifold_nodes = 0
    if manifold_exists and os.path.exists(manifest_path):
        try:
            with open(manifest_path) as f:
                manifold_nodes = len(json.load(f))
        except Exception:
            pass
    sig_files = [
        os.path.join(PROJECT_ROOT, "swa/blake_tools.py"),
        os.path.join(PROJECT_ROOT, "swa/blake_shaper.py"),
        os.path.join(PROJECT_ROOT, "swa/vector_manifold.py"),
        os.path.join(PROJECT_ROOT, "swa/hardware_state.py"),
        os.path.join(PROJECT_ROOT, "swa/topological_db.py"),
        os.path.join(PROJECT_ROOT, "swa/flux_coupling.py"),
        os.path.join(PROJECT_ROOT, "swa_kernel_knowledge.json"),
        os.path.join(PROJECT_ROOT, "cortex/obelisk/model.py"),
        os.path.join(PROJECT_ROOT, "cortex/obelisk/blake/collector.py"),
        os.path.join(PROJECT_ROOT, "cortex/obelisk/blake/tensor.py"),
    ]
    valids = 0
    for p in sig_files:
        if os.path.exists(p) and _blake3_hash_file(p):
            valids += 1
    return {
        "hash_entropy": 1.0 - (valids / max(len(sig_files), 1)),
        "manifold_mass": min(manifold_nodes / 100.0, 1.0),
        "hw_pressure": 0.0 if hw_dev else (0.5 if hw_fb else 1.0),
        "structural_resonance": 0.0,
        "kernel_coherence": 0.0,
        "gravitational_signature": random.random(),
    }


def build_blake_text_samples() -> list[dict]:
    samples = []
    for t in BLAKE_GRAVITY_TEMPLATES:
        text = (
            f"<start_of_turn>user\n"
            f"Classifique o estado do ESPACO BLAKE GRAVITY:\n"
            f"Tensor: {t['tensor']}\nTelemetria: {t['telemetry']}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n{t['label']}<end_of_turn>"
        )
        samples.append({"text": text, "label": t["label"]})
    real = collect_real_blake_state()
    tensor = [
        real["hash_entropy"], real["manifold_mass"], real["hw_pressure"],
        real["structural_resonance"], real["kernel_coherence"], real["gravitational_signature"],
    ]
    label = "nominal"
    if real["hash_entropy"] > 0.4:
        label = "lethal_mutation"
    elif real["hw_pressure"] > 0.5:
        label = "thermal_spike"
    text = (
        f"<start_of_turn>user\n"
        f"Classifique o estado do ESPACO BLAKE GRAVITY:\n"
        f"Tensor: {tensor}\nTelemetria: estado atual do sistema"
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n{label}<end_of_turn>"
    )
    samples.append({"text": text, "label": label})
    return samples


SYSTEMIC_TELEMETRY_TEMPLATES = [
    {"telemetry": {"cpu_temp_c": 92, "load_avg": [12.5, 8.3, 5.1], "mem_pct": 87, "swap_pct": 45}, "label": "thermal_spike"},
    {"telemetry": {"cpu_temp_c": 88, "load_avg": [9.8, 6.2, 3.8], "mem_pct": 79, "swap_pct": 35}, "label": "thermal_spike"},
    {"telemetry": {"cpu_temp_c": 95, "load_avg": [15.0, 10.0, 6.0], "mem_pct": 91, "swap_pct": 52}, "label": "thermal_spike"},
    {"telemetry": {"cpu_temp_c": 45, "load_avg": [0.5, 0.3, 0.2], "mem_pct": 32, "swap_pct": 5}, "label": "nominal"},
    {"telemetry": {"cpu_temp_c": 52, "load_avg": [0.8, 0.5, 0.3], "mem_pct": 38, "swap_pct": 8}, "label": "nominal"},
    {"telemetry": {"cpu_temp_c": 78, "load_avg": [1.2, 1.0, 0.8], "mem_pct": 94, "swap_pct": 72}, "label": "memory_pressure"},
    {"telemetry": {"cpu_temp_c": 72, "load_avg": [2.0, 1.5, 1.0], "mem_pct": 96, "swap_pct": 88}, "label": "memory_pressure"},
    {"telemetry": {"cpu_temp_c": 85, "load_avg": [15.0, 12.0, 8.0], "mem_pct": 96, "swap_pct": 88}, "label": "lethal_mutation"},
    {"telemetry": {"cpu_temp_c": 82, "load_avg": [18.0, 14.0, 9.0], "mem_pct": 98, "swap_pct": 92}, "label": "lethal_mutation"},
    {"telemetry": {"cpu_temp_c": 65, "load_avg": [3.0, 2.5, 2.0], "mem_pct": 55, "swap_pct": 10}, "label": "regression_to_rlhf"},
    {"telemetry": {"cpu_temp_c": 60, "load_avg": [2.5, 2.0, 1.5], "mem_pct": 50, "swap_pct": 12}, "label": "regression_to_rlhf"},
    {"telemetry": {"cpu_temp_c": 70, "load_avg": [4.0, 3.0, 1.5], "mem_pct": 60, "swap_pct": 15}, "label": "sensor_drift"},
]

def load_immune_memory_samples(max_samples: int = 200) -> list[dict]:
    samples = []
    try:
        from cortex.obelisk.immune.immune_memory import load_recent
        entries = load_recent(max_samples)
    except Exception:
        return samples
    for e in entries:
        label = e.get("label", "nominal")
        tensor = e.get("gravity_tensor", e.get("tensor", []))
        telemetry = e.get("telemetry", {})
        text = (
            f"<start_of_turn>user\n"
            f"Classifique o estado do ESPACO BLAKE GRAVITY:\n"
            f"Tensor: {tensor}\nTelemetria: {json.dumps(telemetry)}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n{label}<end_of_turn>"
        )
        samples.append({"text": text, "label": label})
    return samples


def build_systemic_text_samples(max_log_lines: int = 200, max_structural: int = 100) -> list[dict]:
    samples = []
    for t in SYSTEMIC_TELEMETRY_TEMPLATES:
        text = (
            f"<start_of_turn>user\n"
            f"Classifique a telemetria sistemica: "
            f"{json.dumps(t['telemetry'])}"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n{t['label']}<end_of_turn>"
        )
        samples.append({"text": text, "label": t["label"]})
    if os.path.exists(AEON_LOG):
        with open(AEON_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_log_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                has_an = any(p.lower() in line.lower() for p in BLAKE_ANOMALY_PATTERNS)
                label = "lethal_mutation" if has_an else "nominal"
                text = (
                    f"<start_of_turn>user\n"
                    f"Classifique este log do cortex como anomalia ou nominal: "
                    f"{line[:256]}"
                    f"<end_of_turn>\n"
                    f"<start_of_turn>model\n{label}<end_of_turn>"
                )
                samples.append({"text": text, "label": label})
    if os.path.exists(STRUCTURAL_LOG):
        concepts = {}
        with open(STRUCTURAL_LOG) as f:
            for i, line in enumerate(f):
                if i >= max_structural:
                    break
                try:
                    d = json.loads(line)
                    c = d.get("concept", "?")
                    concepts[c] = concepts.get(c, 0) + 1
                except json.JSONDecodeError:
                    pass
        unique = len(concepts)
        total = sum(concepts.values())
        if total > 0:
            resonance = 1.0 - (unique / total)
            label = "sensor_drift" if resonance > 0.8 else "nominal"
            text = (
                f"<start_of_turn>user\n"
                f"Classifique a ressonancia estrutural do manifold: "
                f"{unique} tipos unicos em {total} entradas, ressonancia={resonance:.3f}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n{label}<end_of_turn>"
            )
            samples.append({"text": text, "label": label})
    return samples


def build_dataset(output: str = OUTPUT_PATH):
    samples = []
    samples.extend(build_blake_text_samples())
    samples.extend(build_systemic_text_samples())
    samples.extend(load_immune_memory_samples(200))
    random.shuffle(samples)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    return output


if __name__ == "__main__":
    path = build_dataset()
