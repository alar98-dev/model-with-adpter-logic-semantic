"""Barramento UDP tensorial entre AeonCell, SWA-Matriz e o ciclo OODA.

Este processo é deliberadamente pequeno e seguro:

* recebe somente datagramas loopback em ``127.0.0.1:20011``;
* valida framing, dimensão, tamanho, finitude e BLAKE3 antes de aceitar o vetor;
* projeta o estado de entrada para o espaço da ``swa_matriz`` quando a matriz
  1152x640 está disponível;
* publica um resumo operacional no Redis, nunca bytes arbitrários como comando;
* entrega ao AeonCell apenas estado visual/afetivo e uma decisão allowlisted.

``:20001`` permanece reservado ao transporte textual legado. O v2 tensorial
usa ``:20011`` para que os dois contratos não disputem a mesma porta.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import socket
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from swa.tensor_protocol import TensorPacketError, UdpTensorPacket, UdpTensorProtocol


LOG = logging.getLogger("swa.matriz_ooda")
DEFAULT_HOST = os.environ.get("SWA_TENSOR_UDP_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("SWA_TENSOR_UDP_PORT", str(UdpTensorProtocol.UDP_PORT)))
DEFAULT_REDIS_CHANNEL = os.environ.get("SWA_MASCOT_REDIS_CHANNEL", "swa:mascot:state")
DEFAULT_REDIS_STREAM = os.environ.get("SWA_MATRIX_REDIS_STREAM", "aeon:matrix:ooda")
DEFAULT_MATRIX_PATH = os.environ.get(
    "SWA_MATRIX_WEIGHTS",
    "/dev/shm/aeon/swa_matriz_wa_1152x640_f16.bin",
)
STATE_PATH = Path(os.environ.get("SWA_MATRIZ_OODA_STATE", "/dev/shm/aeon/matriz_ooda_state.json"))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _cosine(left: np.ndarray, right: Optional[np.ndarray]) -> Optional[float]:
    if right is None or left.shape != right.shape:
        return None
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return None
    return float(np.clip(np.dot(left, right) / (left_norm * right_norm), -1.0, 1.0))


def _resize_vector(vector: np.ndarray, size: int) -> np.ndarray:
    """Reamostra um vetor sensorial para o domínio de entrada da matriz."""
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    if flat.size == size:
        return flat.copy()
    if flat.size == 0:
        return np.zeros(size, dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    return np.interp(target_x, source_x, flat).astype(np.float32)


class SwaMatriz:
    """Leitor read-only da matriz física 1152x640 usada pelo gateway."""

    INPUT_DIM = 640
    OUTPUT_DIM = 1152

    def __init__(self, path: str = DEFAULT_MATRIX_PATH):
        self.path = Path(path)
        self._matrix: Optional[np.ndarray] = None
        self._load_error = ""
        self._load()

    def _load(self) -> None:
        try:
            expected = self.INPUT_DIM * self.OUTPUT_DIM
            if not self.path.is_file() or self.path.stat().st_size < expected * 2:
                self._load_error = "arquivo ausente ou menor que 1152x640 float16"
                return
            matrix = np.fromfile(self.path, dtype="<f2", count=expected)
            if matrix.size != expected:
                self._load_error = "quantidade de pesos incompatível"
                return
            self._matrix = matrix.astype(np.float32).reshape(self.OUTPUT_DIM, self.INPUT_DIM)
        except (OSError, ValueError) as exc:
            self._load_error = str(exc)

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def project(self, vector: np.ndarray) -> np.ndarray:
        source = _resize_vector(vector, self.INPUT_DIM)
        if self._matrix is None:
            # Fallback determinístico: o contrato continua 1152D sem fingir que
            # pesos da matriz foram aplicados.
            return _resize_vector(source, self.OUTPUT_DIM)
        projected = self._matrix @ source
        norm = float(np.linalg.norm(projected))
        return (projected / norm).astype(np.float32) if norm > 1e-12 else projected.astype(np.float32)


@dataclass
class OodaResult:
    cycle_id: int
    phase: str
    decision: str
    tensor: dict[str, Any]
    metadata: dict[str, Any]
    mascot: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": "swa-matriz-ooda/v1",
            "cycle_id": self.cycle_id,
            "phase": self.phase,
            "decision": self.decision,
            "tensor": self.tensor,
            "metadata": self.metadata,
            "mascot": self.mascot,
            "timestamp": time.time(),
        }


class MatrizOodaEngine:
    """Transforma um pacote tensorial em um ciclo OODA auditável."""

    ALLOWED_DECISIONS = frozenset({"mascot_reflect", "mascot_alert", "mascot_idle"})

    def __init__(self, matrix: Optional[SwaMatriz] = None, state_path: Path = STATE_PATH):
        self.matrix = matrix or SwaMatriz()
        self.state_path = Path(state_path)
        self._previous: Optional[np.ndarray] = None
        self._cycle = 0
        self._lock = threading.Lock()
        self.last_state: dict[str, Any] = {"state": "standby", "cycles": 0}

    def process(self, packet: UdpTensorPacket) -> OodaResult:
        with self._lock:
            self._cycle += 1
            vector = np.asarray(packet.tensor, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vector))
            mean = float(np.mean(vector)) if vector.size else 0.0
            std = float(np.std(vector)) if vector.size else 0.0
            cosine = _cosine(vector, self._previous)
            novelty = 1.0 if cosine is None else float(np.clip(1.0 - cosine, 0.0, 2.0))
            projected = self.matrix.project(vector)
            metadata = dict(packet.metadata)
            event_type = str(metadata.get("event_type", "tensor_observation"))[:80]
            energy = float(np.clip(norm / max(1.0, math.sqrt(vector.size)), 0.0, 1.0))

            # OBSERVE → ORIENT → DECIDE → ACT. A phase supplied by a sensor is
            # evidence only; the bridge still closes all four stages itself.
            if novelty >= 0.35 or event_type in {"mouse_click", "voice", "typing"}:
                decision = "mascot_alert"
            elif energy <= 0.05:
                decision = "mascot_idle"
            else:
                decision = "mascot_reflect"
            assert decision in self.ALLOWED_DECISIONS

            tensor = {
                "input_dim": int(vector.size),
                "matrix_input_dim": SwaMatriz.INPUT_DIM,
                "matrix_output_dim": SwaMatriz.OUTPUT_DIM,
                "norm": round(norm, 6),
                "mean": round(mean, 6),
                "std": round(std, 6),
                "energy": round(energy, 6),
                "cosine_previous": None if cosine is None else round(cosine, 6),
                "novelty": round(novelty, 6),
                "projection_applied": self.matrix.ready,
                "projection_norm": round(float(np.linalg.norm(projected)), 6),
            }
            mascot = {
                "entity": "aeon_cell",
                "role": "mascot_autonomo_receptaculo",
                "autonomous": True,
                "safe_action": decision,
                "energy": round(energy, 6),
                "phase": "ACT",
                "thought_hint": self._thought_hint(decision, event_type, novelty),
            }
            result = OodaResult(
                cycle_id=self._cycle,
                phase="ACT",
                decision=decision,
                tensor=tensor,
                metadata={
                    "sequence": packet.sequence,
                    "event_type": event_type,
                    "input_phase": str(metadata.get("phase", ""))[:32],
                    "source": str(metadata.get("source", "aeon_cell"))[:64],
                },
                mascot=mascot,
            )
            self._previous = vector.copy()
            self.last_state = result.as_dict()
            self._write_state(self.last_state)
            return result

    @staticmethod
    def _thought_hint(decision: str, event_type: str, novelty: float) -> str:
        if decision == "mascot_alert":
            return f"Percebi {event_type}; novidade tensorial {novelty:.2f}."
        if decision == "mascot_idle":
            return "Estado estável; permaneço em observação."
        return f"Integrei {event_type} à matriz e continuo acompanhando."

    def _write_state(self, state: dict[str, Any]) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(prefix="matriz_ooda.", dir=self.state_path.parent)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.state_path)
        except OSError:
            LOG.exception("não foi possível persistir o estado OODA")


class MatrizOodaBridge:
    """Servidor UDP + adaptador Redis do ciclo OODA tensorial."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        redis_client: Any = None,
        publisher: Optional[Callable[[str, dict[str, Any]], None]] = None,
        engine: Optional[MatrizOodaEngine] = None,
    ):
        self.host = host
        self.port = int(port)
        self.redis = redis_client
        self.publisher = publisher
        self.engine = engine or MatrizOodaEngine()
        self.stop_event = threading.Event()
        self.sock: Optional[socket.socket] = None
        self.accepted = 0
        self.rejected = 0

    def _publish(self, result: OodaResult) -> None:
        payload = result.as_dict()
        if self.publisher:
            self.publisher(DEFAULT_REDIS_CHANNEL, payload)
            self.publisher(DEFAULT_REDIS_STREAM, payload)
        if self.redis is not None:
            encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            try:
                self.redis.publish(DEFAULT_REDIS_CHANNEL, encoded)
                self.redis.xadd(DEFAULT_REDIS_STREAM, {"payload": encoded}, maxlen=4096, approximate=True)
            except Exception:
                LOG.exception("falha ao publicar ciclo OODA no Redis")

    def handle_datagram(self, datagram: bytes, address: tuple[str, int]) -> bytes:
        try:
            packet = UdpTensorProtocol.decode(datagram)
            result = self.engine.process(packet)
            self._publish(result)
            self.accepted += 1
            return json.dumps({
                "protocol": "swa-matriz-ooda/v1",
                "status": "accepted",
                "sequence": packet.sequence,
                "cycle_id": result.cycle_id,
                "phase": result.phase,
                "decision": result.decision,
            }, separators=(",", ":")).encode("utf-8")
        except (TensorPacketError, ValueError) as exc:
            self.rejected += 1
            LOG.warning("pacote UDP rejeitado de %s: %s", address, exc)
            return json.dumps({"protocol": "swa-matriz-ooda/v1", "status": "rejected", "error": str(exc)}).encode("utf-8")
        except Exception:
            self.rejected += 1
            LOG.exception("falha processando datagrama de %s", address)
            return b'{"protocol":"swa-matriz-ooda/v1","status":"error"}'

    def serve_forever(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(1.0)
        LOG.info("SWA-Matriz OODA UDP ativo em %s:%s; matriz=%s", self.host, self.port, self.engine.matrix.ready)
        while not self.stop_event.is_set():
            try:
                datagram, address = self.sock.recvfrom(UdpTensorProtocol.MAX_DATAGRAM)
            except socket.timeout:
                continue
            except OSError:
                if self.stop_event.is_set():
                    break
                raise
            reply = self.handle_datagram(datagram, address)
            try:
                self.sock.sendto(reply, address)
            except OSError:
                LOG.debug("não foi possível enviar ACK UDP", exc_info=True)

    def close(self) -> None:
        self.stop_event.set()
        if self.sock:
            self.sock.close()


def _redis_from_env() -> Any:
    try:
        import redis
        client = redis.Redis(
            host=os.environ.get("REDIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            decode_responses=True,
            socket_connect_timeout=1,
        )
        client.ping()
        return client
    except Exception as exc:
        LOG.warning("Redis indisponível; bridge continuará somente com state file: %s", exc)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-redis", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        packet = UdpTensorProtocol.encode(np.linspace(0, 1, 64, dtype=np.float32), sequence=1, metadata={"event_type": "selftest"})
        result = MatrizOodaEngine().process(UdpTensorProtocol.decode(packet))
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0
    bridge = MatrizOodaBridge(args.host, args.port, None if args.no_redis else _redis_from_env())
    try:
        bridge.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
