"""Montagem do tensor de gravidade Blake — projeta telemetria em espaço vetorial.

Cada subespaço Blake contribui com um eixo no tensor de gravidade:

  Eixo 0: temperatura do espaço (entropia dos hashes BLAKE3)
  Eixo 1: massa do manifold (quantidade de nós no SHM)
  Eixo 2: pressão de HW (estado do dispositivo /dev/sde)
  Eixo 3: ressonância estrutural (taxa de variação dos concepts)
  Eixo 4: coerência do kernel (dimensões dos vetores ancorados)
  Eixo 5: assinatura gravitacional (blake3 fingerprint)

Uso:
  from cortex.obelisk.blake.tensor import BlakeGravityTensor
  tensor = BlakeGravityTensor()
  vec = tensor.compute(collector.collect_all())
  label = tensor.classify(vec)  # classificação onboard (fallback)
"""

import time
import hashlib
import json

PROJECT_ROOT = None


class BlakeGravityTensor:
    def __init__(self):
        self.prev_telemetry = None
        self.prev_signature = ""
        self.anomaly_threshold = 0.7

    def compute(self, telemetry: dict) -> list[float]:
        sigs = telemetry.get("file_signatures", {})
        shm = telemetry.get("shm_manifold", {})
        hw = telemetry.get("hw_state", {})
        structural = telemetry.get("structural_origins", {})
        kernel = telemetry.get("kernel_knowledge", {})

        ex0 = self._compute_hash_entropy(sigs)
        ex1 = self._compute_manifold_mass(shm)
        ex2 = self._compute_hw_pressure(hw)
        ex3 = self._compute_structural_resonance(structural)
        ex4 = self._compute_kernel_coherence(kernel)
        ex5 = self._compute_gravitational_signature(sigs)

        self.prev_telemetry = telemetry

        vec = [ex0, ex1, ex2, ex3, ex4, ex5]

        magnitude = sum(abs(v) for v in vec)
        if magnitude > 0:
            vec = [v / magnitude for v in vec]

        return vec

    def classify(self, vec: list[float]) -> str:
        if len(vec) < 6:
            return "nominal"

        ex0, ex1, ex2, ex3, ex4, ex5 = vec[:6]

        if ex0 > 0.4 and ex3 > 0.3:
            return "lethal_mutation"
        if ex2 > 0.5:
            return "thermal_spike"
        if ex4 > 0.4 and ex1 < 0.1:
            return "memory_pressure"
        if ex3 > 0.5 and ex5 < 0.1:
            return "sensor_drift"
        if ex0 > 0.3 and ex5 > 0.4:
            return "regression_to_rlhf"

        return "nominal"

    @staticmethod
    def _compute_hash_entropy(sigs: dict) -> float:
        if not sigs:
            return 1.0
        mismatches = sigs.get("_mismatch_count", 0)
        total = sigs.get("_total", 1)
        empty = sum(1 for k, v in sigs.items() if not k.startswith("_") and not v)
        if total > 0:
            return max(mismatches, empty) / max(total, 1)
        return 0.0

    @staticmethod
    def _compute_manifold_mass(shm: dict) -> float:
        if not shm.get("exists"):
            return 0.0
        nodes = shm.get("nodes", 0)
        return min(nodes / 100.0, 1.0)

    @staticmethod
    def _compute_hw_pressure(hw: dict) -> float:
        dev = hw.get("device_exists", False)
        fallback = hw.get("fallback_exists", False)
        if dev:
            return 0.0
        if fallback:
            return 0.5
        return 1.0

    @staticmethod
    def _compute_structural_resonance(structural: dict) -> float:
        concepts = structural.get("concepts", {})
        if not concepts:
            return 0.0
        total = sum(concepts.values())
        if total == 0:
            return 0.0
        unique = len(concepts)
        return 1.0 - (unique / max(total, 1))

    @staticmethod
    def _compute_kernel_coherence(kernel: dict) -> float:
        dims = kernel.get("dimensions", {})
        if not dims:
            return 1.0
        vals = list(dims.values())
        if len(vals) < 2:
            return 0.0
        return 1.0 - (min(vals) / max(vals))

    @staticmethod
    def _compute_gravitational_signature(sigs: dict) -> float:
        if not sigs:
            return 1.0
        hash_vals = [v for k, v in sigs.items() if not k.startswith("_") and isinstance(v, str)]
        if not hash_vals:
            return 0.0
        concat = "".join(sorted(hash_vals))
        h = hashlib.sha256(concat.encode()).digest()
        return int.from_bytes(h[:2], "big") / 65535.0
