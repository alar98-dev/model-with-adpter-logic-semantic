"""Classificador do espaço Blake — integra o tensor de gravidade ao modelo obelisco.

Pipeline:
  collect_all() → BlakeGravityTensor.compute() → ObeliskModel.classify()

Se o modelo LLM não estiver carregado (ex: sem GPU/RAM), usa o classificador
onboard do tensor como fallback determinístico.
"""

import os
import json
import time

BLAKE_LABELS = [
    "nominal",
    "lethal_mutation",
    "thermal_spike",
    "regression_to_rlhf",
    "memory_pressure",
    "sensor_drift",
]

CLASSIFICATION_PROMPT = """INPUT_TENSOR:{tensor} TELEMETRY:{telemetry} CLASSIFICATION_LABELS:{labels} OUTPUT:"""

CONFIDENCE_THRESHOLD = float(os.environ.get("OBELISK_CONFIDENCE_THRESHOLD", "0.7"))


class BlakeSpaceClassifier:
    def __init__(self, obelisk_model=None):
        self.model = obelisk_model
        self._labels_str = ", ".join(BLAKE_LABELS)
        self._cached_tensor = [0.0] * 6
        self._cached_signature = ""
        self._last_classification = "nominal"
        self._last_confidence = 1.0
        self._last_update = 0.0
        
        # Hysteresis state tracking
        self.last_state = "nominal"
        self.state_entry_time = time.time()
        self.min_state_duration = 30.0 # 30 seconds minimum duration for critical states

    def classify(self, blake_telemetry: dict, gravity_tensor: list[float]) -> str:
        label, _ = self.classify_with_confidence(blake_telemetry, gravity_tensor)
        return label

    def classify_with_confidence(self, blake_telemetry: dict, gravity_tensor: list[float]) -> tuple[str, float]:
        if not gravity_tensor or len(gravity_tensor) < 6:
            return ("nominal", 1.0)

        input_sig = str(hash(json.dumps(blake_telemetry, sort_keys=True)))
        if input_sig == self._cached_signature:
            return (self._last_classification, self._last_confidence)

        self._cached_tensor = gravity_tensor
        self._cached_signature = input_sig

        import math
        import psycopg2
        import hashlib

        from cortex.obelisk.blake.tensor import BlakeGravityTensor
        fallback = BlakeGravityTensor()
        raw_label = fallback.classify(gravity_tensor)
        
        # 1. Dynamic Symbol Grounding
        is_anomaly = False
        l2_dist_to_nominal = math.sqrt(sum((v - 0.166)**2 for v in gravity_tensor))
        
        if raw_label == "nominal" and l2_dist_to_nominal > 0.35:
            is_anomaly = True
            
        if is_anomaly:
            h_val = hashlib.sha256(json.dumps(gravity_tensor).encode()).hexdigest()[:4]
            raw_label = f"DIR:ANOMALY_{h_val}"
            try:
                dsn = "dbname=swa_matrix user=alifer"
                password = os.environ.get("SWA_DB_PASSWORD")
                if password:
                    dsn += f" password={password} host=localhost"
                
                with psycopg2.connect(dsn) as conn:
                    with conn.cursor() as cur:
                        # Insere no banco com vetor padronizado de 2560D
                        emb_2560 = gravity_tensor + [0.0]*(2560-6)
                        cur.execute("""
                            INSERT INTO blake_semantic_anchors 
                            (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                            VALUES (%s, 'ANOMALY_DYNAMIC', %s, %s, %s, 1.0)
                            ON CONFLICT (file_path) DO NOTHING;
                        """, (f"ANOMALY://{raw_label}", h_val, f"Dynamic anomaly tensor detected: {gravity_tensor}", emb_2560))
            except Exception:
                pass

        # 2. Hysteresis Damping
        now = time.time()
        if raw_label != self.last_state:
            # Permite transição imediata se saindo de nominal para crítico,
            # caso contrário exige que tenha decorrido o tempo mínimo.
            if self.last_state == "nominal" or (now - self.state_entry_time >= self.min_state_duration):
                self.last_state = raw_label
                self.state_entry_time = now
            else:
                raw_label = self.last_state

        self._last_classification = raw_label
        self._last_confidence = 1.0
        self._last_update = now
        return (raw_label, 1.0)

    @staticmethod
    def _summarize(telemetry: dict) -> str:
        parts = []
        sigs = telemetry.get("file_signatures", {})
        if sigs:
            valids = sum(1 for v in sigs.values() if v)
            parts.append(f"assinaturas: {valids}/{len(sigs)} validas")
        shm = telemetry.get("shm_manifold", {})
        if shm:
            parts.append(f"manifold: {shm.get('nodes', 0)} nos, existe={shm.get('exists', False)}")
        hw = telemetry.get("hw_state", {})
        if hw:
            parts.append(f"hw: device={hw.get('device_exists', False)}, fallback={hw.get('fallback_exists', False)}")
        structural = telemetry.get("structural_origins", {})
        if structural:
            concepts = structural.get("concepts", {})
            parts.append(f"conceitos estruturais: {len(concepts)} tipos, {structural.get('entries', 0)} entradas")
        kernel = telemetry.get("kernel_knowledge", {})
        if kernel:
            parts.append(f"kernel: {kernel.get('keys', 0)} chaves, {len(kernel.get('dimensions', {}))} dimensoes")
        return " | ".join(parts) if parts else "sem dados"
