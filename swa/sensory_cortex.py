"""SWA Sensory Cortex — hub de sensores ricos do mundo real → latent_chaos.

Converte sinais sensoriais reais (visão, presença, dispositivos, sistema físico)
em um vetor tensorial denso e o injeta no latent_chaos (64x64 float32) em uma
região dedicada que o cortex_link_pump amostra no delta-detection.

Isso quebra o "ponto fixo" do latent_chaos: cada mudança real no mundo cria um
delta tensorial → o pump emite um verdadeiro pulso sináptico (ContextBus/intent),
mesmo quando CPU/RAM permanecem estáveis.

Região reservada: índices amostrados 20..63 (do fingerprint step=64), que ficam
na faixa "livre" do tensor (ZERO no boot) e são TODOS lidos pelo pump via
_fingerprint (values[::64]).

Fontes consumidas (Redis Pub/Sub), cada uma mapeada a uma janela do vetor:
  - swa:vision:current_presence     -> janela 0..15  (presença/identidades rostos)
  - swa:vision:identity_recognized  -> janela 16..23 (eventos de reconhecimento)
  - sensores físicos psutil          -> janela 24..43 (CPU/RAM/termal/disco)
"""

import json
import os
import time
import threading
import numpy as np
import psutil

LATENT_CHAOS_PATH = "/dev/shm/aeon_latent_chaos.dat"
TENSOR_DIM = 4096            # 64x64
STEP = 64                    # fingerprint step do pump
SAMPLED_IDX = list(range(0, TENSOR_DIM, STEP))
SENSOR_SLOT_START = 20       # primeiro slot livre dos amostrados
SENSOR_SLOT_END = 64         # exclusivo

PREVIEW_PATH = "/dev/shm/aeon_sensory_state.json"


class SensoryCortex:
    """Agrega sensores ricos e injeta o vetor sensorial no latent_chaos."""

    def __init__(self, redis_host="127.0.0.1", redis_port=6379):
        self.r = None
        self.lock = threading.Lock()
        self.running = False
        self.signal = np.zeros(SENSOR_SLOT_END - SENSOR_SLOT_START, dtype=np.float32)
        self.last_fingerprint = 0.0
        import redis
        self.r = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.r.ping()
        print("[SENSORY CORTEX] Redis conectado.", flush=True)

    def _inject(self, vector: np.ndarray):
        """Grava o vetor sensorial nos slots dedicados do latent_chaos."""
        n = len(vector)
        if n == 0:
            return
        try:
            data = open(LATENT_CHAOS_PATH, "rb").read()
            if len(data) < TENSOR_DIM * 4:
                arr = np.zeros(TENSOR_DIM, dtype=np.float32)
            else:
                arr = np.frombuffer(data, dtype=np.float32).copy()
            for i, v in enumerate(vector):
                slot = SENSOR_SLOT_START + i
                if slot < SENSOR_SLOT_END and slot < len(SAMPLED_IDX):
                    arr[SAMPLED_IDX[slot]] = float(v)
            with open(LATENT_CHAOS_PATH, "wb") as f:
                f.write(arr.tobytes())
        except Exception as e:
            print(f"[SENSORY CORTEX] falha injeção: {e}", flush=True)

    MASTER_NAMES = {"master", "alifer", "alifer ramos", "aliferramos", "alifer.ramos", "alifer_01", "alifer_ramos"}

    def _is_master(self, name: str) -> bool:
        """True se o nome reconhecido é o usuário principal (Master/ALifer)."""
        if not name:
            return False
        return name.strip().lower() in self.MASTER_NAMES

    def _consume_vision(self):
        """Lê o estado atual de visão (presença/rostos) do Redis.

        Codificação do vetor de presença (16 dims):
          [0]  nº rostos presente (norm /4)
          [1]  nº identidades reconhecidas (norm /4)
          [2]  flag PRESENÇA MASTER (voce = alifer/master) -> 1.0
          [3]  flag presença de outra identidade conhecida -> 1.0
          [4]  flag rosto DESCONHECIDO -> 1.0
          [5-8] nomes normalizados (1º ao 4º) como fingerprint léxico
          [9]  confiança média do reconhecimento atual
          [15] presença ativa (frame de visão chegando) -> 1.0
        """
        vec = np.zeros(16, dtype=np.float32)
        if self.r is None:
            return vec
        try:
            raw = self.r.get("swa:vision:current_presence")
            if raw:
                d = json.loads(raw.decode("utf-8"))
                n_faces = int(d.get("faces_detected", 0))
                vec[0] = min(float(n_faces), 1.0) / 4.0
                ids = d.get("identities", [])
                vec[1] = min(float(len(ids)), 1.0) / 4.0
                master = self._is_master(ids[0].get("name", "")) if ids else False
                vec[2] = 1.0 if master else 0.0
                vec[3] = 1.0 if (ids and not master) else 0.0
                vec[4] = 1.0 if (n_faces > len(ids)) else 0.0
                for k, i in enumerate(ids[:4]):
                    nm = (i.get("name") or "").strip().lower()
                    # fingerprint léxico do nome nos slots 5..8 (determinístico)
                    if nm:
                        vec[5 + k] = (sum(ord(c) for c in nm) % 97) / 97.0
                # busca a última confiança publicada pelo face_organ
                try:
                    lates = self.r.get("swa:vision:latest_for_calibrate")
                    if lates:
                        cal = json.loads(lates.decode("utf-8"))
                        vec[9] = min(max(float(cal.get("confidence", 0.0)) / 100.0, 0.0), 1.0)
                except Exception:
                    pass
                vec[15] = 1.0  # presença ativa
        except Exception:
            pass
        return vec

    def _consume_identity_events(self):
        """Janela de eventos recentes de reconhecimento (Redis consistent-probable).

        Usa a lista swa:vision:recent_recognitions caso exista; senão consulta o
        PostgreSQL (swa_recognition_sessions) das últimas N sessões e empilha
        confianças. Quem for Master recebe reforço (valor elevado).
        """
        vec = np.zeros(8, dtype=np.float32)
        if self.r is None:
            return vec
        # 1) Tenta lista Redis alimentada (se existir)
        try:
            key = "swa:vision:recent_recognitions"
            recent = self.r.lrange(key, 0, 7) or []
            if recent:
                for i, item in enumerate(recent[:8]):
                    try:
                        ev = json.loads(item.decode("utf-8"))
                        conf = float(ev.get("confidence", 0.5)) / 100.0
                        nm = ev.get("name", "")
                        if self._is_master(nm):
                            conf = min(conf + 0.15, 1.0)  # reforço Master
                        vec[i] = min(max(conf, 0.0), 1.0)
                    except Exception:
                        vec[i] = 0.5
                return vec
        except Exception:
            pass
        # 2) Fallback: leitura direta do PostgreSQL (reconhecimentos do Master)
        try:
            import psycopg2
            conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
            cur = conn.cursor()
            cur.execute(
                """SELECT i.nome, s.confidence
                   FROM swa_recognition_sessions s
                   JOIN swa_identities i ON i.id = s.identity_id
                   WHERE i.papel = 'admin' AND i.ativo = TRUE
                   ORDER BY s.detected_at DESC LIMIT 8;"""
            )
            rows = cur.fetchall()
            cur.close(); conn.close()
            for i, (nome, conf) in enumerate(rows[:8]):
                v = min(float(conf), 1.0)
                if self._is_master(nome):
                    v = min(v + 0.15, 1.0)
                vec[i] = v
        except Exception:
            pass
        return vec

    def _consume_physical(self):
        """Sensores físicos do mundo (CPU/RAM/termal/disco/IO)."""
        vec = np.zeros(20, dtype=np.float32)
        try:
            vec[0] = min(psutil.cpu_percent(interval=None) / 100.0, 1.0)
            mem = psutil.virtual_memory()
            vec[1] = mem.percent / 100.0
            try:
                vec[2] = psutil.sensors_temperatures().get("coretemp", [{}])[0].get("current", 0) / 100.0
            except Exception:
                vec[2] = 0.0
            sw = psutil.swap_memory()
            vec[3] = sw.percent / 100.0
            disk = psutil.disk_usage("/")
            vec[4] = disk.percent / 100.0
            net = psutil.net_io_counters()
            vec[5] = min(net.bytes_sent / 1e9, 1.0)
            vec[6] = min(net.bytes_recv / 1e9, 1.0)
            # carga/nº processos
            vec[7] = min(len(psutil.pids()) / 1000.0, 1.0)
            vec[8] = min(psutil.cpu_count() or 8, 64) / 64.0
            vec[9] = min(mem.available / (2 ** 40), 1.0)
        except Exception:
            pass
        return vec

    def _compose(self):
        """Junta todas as fontes num único vetor sensorial rico."""
        parts = [
            self._consume_vision(),
            self._consume_identity_events(),
            self._consume_physical(),
        ]
        return np.concatenate(parts)[: SENSOR_SLOT_END - SENSOR_SLOT_START]

    def run(self):
        self.running = True
        print("[SENSORY CORTEX] Ciclo sensorial ativo "
              f"(injetando nos slots {SENSOR_SLOT_START}..{SENSOR_SLOT_END}).", flush=True)
        while self.running:
            try:
                vec = self._compose()
                with self.lock:
                    self.signal[:] = vec
                    self._inject(vec)
            except Exception as e:
                print(f"[SENSORY CORTEX] erro: {e}", flush=True)
            time.sleep(5.0)

    def stop(self):
        self.running = False


def main():
    cortex = SensoryCortex()
    cortex.run()


if __name__ == "__main__":
    main()
