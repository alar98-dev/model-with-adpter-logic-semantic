"""
Órgão Metarealidade — esteira de realidade do organismo.

Sintetiza o estado interno (EM bridge, metamatéria, gravidade, energia, fissão)
com a memória externa (REPIO/k-base) numa leitura coesa da "realidade
corrente" do organismo. Cada onda pode ser ancorada nessa leitura:

  - organism_context:  trecho de telemetria adicionado ao contexto do prompt.
  - build_steering:    texto usado como `context` do polvo (reforça a âncora
                       k-base na direção da memória corrente).
  - v_prefix (FOLLOW):       vetor 1152 que condiciona os embeddings na
                       realidade corrente (bruto da metamatéria + snapshot).
  - OODA:              snapshot publicado em state.metareality a cada ciclo.
  - observe:           ao fim de cada turno, registra o episódio (RealityStore
                       + REPIO ingest concept=REALITY) na memória da realidade.

ORIGEM ÚNICA E REAL: a metamatéria vem apenas do subcórtex (push via
set_metamatter ou binário /tmp/swa_metamatter.npy). Nunca é sintetizada —
sem sinal real, vector() responde None (não há âncora confiável).

Tudo é best-effort e nunca lança exceção (o organismo segue vivo sem o órgão).
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np
from swa.tensor_protocol import SWATensorProtocol, TensorIntegrityError


# metamateria acima desta magnitude é lida como "alto estresse" pelo subcórtex
_METAMATTER_STRESS = 50.0

# amostragem do fingerprint do embedding na âncora (resumo compacto da Tábua)
_EMB_SAMPLES = 8


@dataclass
class RealitySnapshot:
    """Leitura instantânea da realidade corrente do organismo."""

    ts: float = field(default_factory=time.time)
    emotion_label: str = "neutral"
    emotion_load: float = 0.5
    energy: float = 0.0
    field_coherence: float = 1.0
    matrixial_stability: float = 1.0
    shield_throttle: float = 0.0
    microgravity: float = 0.0
    gravity_modulation: float = 0.0
    metamatter_magnitude: float = 0.0
    metamatter_variance: float = 0.0
    internal_energy: float = 0.0
    will: float = 0.5
    memory_pull: float = 0.0
    memory_hits: int = 0
    repio_available: bool = False
    reality_index: float = 0.5
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "emotion_label": self.emotion_label,
            "emotion_load": round(self.emotion_load, 3),
            "energy": round(self.energy, 2),
            "field_coherence": round(self.field_coherence, 3),
            "matrixial_stability": round(self.matrixial_stability, 3),
            "shield_throttle": round(self.shield_throttle, 3),
            "microgravity": round(self.microgravity, 4),
            "gravity_modulation": round(self.gravity_modulation, 4),
            "metamatter_magnitude": round(self.metamatter_magnitude, 2),
            "metamatter_variance": round(self.metamatter_variance, 3),
            "internal_energy": round(self.internal_energy, 2),
            "will": round(self.will, 3),
            "memory_pull": round(self.memory_pull, 3),
            "memory_hits": self.memory_hits,
            "repio_available": self.repio_available,
            "reality_index": round(self.reality_index, 3),
            "components": {k: round(v, 3) for k, v in self.components.items()},
        }


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _reality_drift(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Distância de cosseno normalizada [0,1] entre dois vetores de realidade.

    Retorna None quando não há sinal comparável (norma ~0 ou dimensões
    incompatíveis) — a ausência é mais honesta do que um 0.5 fabricado.
    """
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape[0] != b.shape[0]:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return None
    cos = float(np.dot(a, b) / (na * nb))
    cos = max(-1.0, min(1.0, cos))
    return max(0.0, min(1.0, (1.0 - cos) / 2.0))


def _compact_anchor(snap: RealitySnapshot, emb) -> Dict[str, Any]:
    """Âncora persistida da onda: 7 escalares do snapshot + embedding semântico REAL.

    Formato: {"keys": [...7..., "emb"], "vals": [7 escalares, <emb 640D>],
    "summary": {dim, norm, sampled, mini}}. A onda seguinte do mesmo reinado
    usa <emb> para alinhamento cosseno real — substitui o vetor escalar 7D raso
    que produzia flats (0.0/0.5) sem sinal comparável.
    """
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = int(vec.size)
    step = max(1, n // _EMB_SAMPLES) if n > 0 else 1
    mini = [float(x) for x in vec[::step][:_EMB_SAMPLES]]
    scalar_keys = [
        "reality_index",
        "load",
        "energy",
        "coherence",
        "will",
        "metamatter",
        "memory_pull",
    ]
    vals = [
        round(float(snap.reality_index), 3),
        round(float(snap.emotion_load), 3),
        round(float(snap.energy) / 100.0, 4),
        round(float(snap.field_coherence), 3),
        round(float(snap.will), 3),
        round(float(snap.metamatter_magnitude) / 1000.0, 6),
        round(float(snap.memory_pull), 3),
    ]
    return {
        "keys": scalar_keys + ["emb"],
        "vals": vals + [[round(float(x), 4) for x in vec]],
        "summary": {
            "dim": n,
            "norm": round(float(np.linalg.norm(vec)) if n else 0.0, 3),
            "sampled": _EMB_SAMPLES,
            "mini": [round(x, 4) for x in mini],
        },
    }


def _anchor_emb(anchor_vec) -> Optional[np.ndarray]:
    """Embedding semântico real da âncora de uma onda persistida (best-effort)."""
    if not isinstance(anchor_vec, dict):
        return None
    try:
        vals = anchor_vec.get("vals")
        if isinstance(vals, list) and vals and isinstance(vals[-1], list):
            vec = np.asarray(vals[-1], dtype=np.float32).reshape(-1)
            if vec.size >= 2:
                return vec
        if isinstance(vals, dict):
            emb = vals.get("emb")
            if isinstance(emb, (list, tuple)):
                vec = np.asarray(emb, dtype=np.float32).reshape(-1)
                if vec.size >= 2:
                    return vec
    except Exception:
        pass
    return None


class MetarealityOrgan:
    """Esteira de realidade: estado interno × memória externa → leitura coerente."""

    METAMATTER_PATH = "/tmp/swa_metamatter.npy"
    METAMATTER_TENSOR_PATH = "/tmp/swa_metamatter_tensor.json"

    def export_metamatter_tensor_packet(self, dtype: str = "float32") -> Optional[dict]:
        """Exporta o vetor de metamatéria corrente como um pacote assinado SWA-TensorProtocol."""
        meta_vec = self.metamatter()
        if meta_vec is None:
            return None
        snap = self._last_snapshot or RealitySnapshot()
        metadata = {
            "source": "metareality_organ",
            "ts": snap.ts,
            "reality_index": snap.reality_index,
            "will": snap.will
        }
        return SWATensorProtocol.encode(meta_vec, metadata=metadata, dtype=dtype)

    def import_metamatter_tensor_packet(self, packet: dict) -> bool:
        """Importa e verifica a assinatura BLAKE3 de um pacote tensorial de metamatéria."""
        try:
            arr = SWATensorProtocol.decode(packet, return_type="numpy")
            return self.set_metamatter(arr)
        except TensorIntegrityError as tie:
            self._logger.error(f"🚨 [METAREALITY] Rejeição imunológica do pacote tensorial: {tie}")
            return False
        except Exception as e:
            self._logger.error(f"⚠️ [METAREALITY] Falha ao importar pacote tensorial: {e}")
            return False


    def __init__(self, core, max_episodes: int = 24):
        self._core = core
        self._logger = logging.getLogger("swa.metareality")
        self._episodes: Deque[str] = deque(maxlen=int(max_episodes))
        self._last_snapshot: Optional[RealitySnapshot] = None
        self._observations = 0
        # -- origem única da metamatéria -----------------------------------------
        # O órgão NUNCA fabrica metamatéria sintética: ou tem a real (push do
        # subcórtex via set_metamatter, ou o binário bruto do subcórtex no
        # METAMATTER_PATH), ou reporta ausência. Sem fallback de tile.
        self._metamatter: Optional[np.ndarray] = None
        self._metamatter_ts: float = 0.0
        # deriva da realidade entre ciclos (vetor escalar do último snapshot)
        self._last_reality_vec: Optional[np.ndarray] = None
        self._last_reality_key: Optional[str] = None

    # ------------------------------------------------------------ metamatéria
    def set_metamatter(self, vector) -> bool:
        """Origem única real: subcórtex/orquestrador empurra o vetor (640,)."""
        try:
            arr = np.asarray(vector, dtype=np.float32).reshape(-1)
            if arr.size == 0:
                return False
            self._metamatter = arr
            self._metamatter_ts = time.time()
            return True
        except Exception:
            return False

    def clear_metamatter(self) -> None:
        """Força ausência de sinal (útil quando o subcórtex é desligado)."""
        self._metamatter = None
        self._metamatter_ts = 0.0

    def metamatter(self, max_age: float = 4.0) -> Optional[np.ndarray]:
        """Vetor real da metamatéria (cache único). Recarrega do binário real
        quando o push está velho ou ausente — nunca sintetiza."""
        now = time.time()
        if self._metamatter is not None and (now - self._metamatter_ts) <= max_age:
            return self._metamatter
        # recarrega da única origem persistida real (subcórtex)
        try:
            if os.path.exists(self.METAMATTER_TENSOR_PATH):
                import json
                try:
                    with open(self.METAMATTER_TENSOR_PATH, "r", encoding="utf-8") as tf:
                        pkt = json.load(tf)
                    arr = SWATensorProtocol.decode(pkt, return_type="numpy")
                    if arr.size:
                        self._metamatter = arr
                        self._metamatter_ts = now
                        return self._metamatter
                except TensorIntegrityError as tie:
                    self._logger.warning(f"🚨 Assinatura BLAKE3 inválida no disco: {tie}")
                    return None
                except Exception:
                    pass
            if os.path.exists(self.METAMATTER_PATH):
                arr = np.asarray(
                    np.load(self.METAMATTER_PATH), dtype=np.float32
                ).reshape(-1)
                if arr.size:
                    self._metamatter = arr
                    self._metamatter_ts = now
        except Exception:
            pass
        return self._metamatter if self._metamatter is not None else None

    # ------------------------------------------------------------ sinais internos
    def _internal_state(self, snap: RealitySnapshot) -> None:
        core = self._core
        try:
            em_bridge = getattr(core, "em_bridge", None)
            em_core = getattr(em_bridge, "em", None)
            if em_bridge is not None and em_core is not None:
                em_state = em_core.status()
                snap.emotion_label = str(
                    (em_state.get("emotion", {}) or {}).get("label", "neutral")
                ).lower()
                snap.emotion_load = float(
                    (em_state.get("emotion", {}) or {}).get("load", 0.5)
                )
                snap.field_coherence = float(em_state.get("field_coherence", 1.0))
                snap.matrixial_stability = float(
                    em_state.get("matrixial_stability", 1.0)
                )
                snap.energy = float(em_state.get("field_energy", 0.0))
                snap.shield_throttle = float(
                    (em_state.get("shield_status", {}) or {}).get(
                        "throttle_level", 0.0
                    )
                )
        except Exception:
            pass

        try:
            fission = getattr(core, "fission", None)
            if fission is not None:
                snap.microgravity = float(getattr(fission, "microgravity", 0.0))
        except Exception:
            pass
        try:
            gravity = getattr(core, "gravity", None)
            if gravity is not None:
                snap.gravity_modulation = float(getattr(gravity, "modulation", 0.0))
        except Exception:
            pass
        snap.internal_energy = float(getattr(core, "internal_energy", 0.0))

        try:
            mm = self.metamatter()
            if mm is not None:
                snap.metamatter_magnitude = float(np.linalg.norm(mm))
                snap.metamatter_variance = float(np.var(mm))
        except Exception:
            pass

        if not snap.energy:
            snap.energy = (
                snap.internal_energy + snap.metamatter_magnitude * 0.1
            )

    # ------------------------------------------------------------ sinais externos
    def _external_state(self, snap: RealitySnapshot, user_msg: str) -> None:
        repio = getattr(self._core, "_repio", None)
        if repio is None:
            return
        try:
            if not repio.pg_available():
                return
            snap.repio_available = True
        except Exception:
            return
        if not (user_msg and user_msg.strip()):
            return
        try:
            hits = repio.recall(user_msg, k=3)
            snap.memory_hits = len(hits)
            pulls = [
                float(h.get("pull", 0.0))
                for h in hits
                if h.get("pull") is not None
            ]
            if pulls:
                snap.memory_pull = _clamp(sum(pulls) / len(pulls))
        except Exception:
            pass

    # ------------------------------------------------------------ leitura coesa
    def integrate(
        self,
        user_msg: str = "",
        *,
        semantic_will: Optional[float] = None,
    ) -> RealitySnapshot:
        """Constrói o snapshot da realidade corrente (nunca lança)."""
        snap = RealitySnapshot()
        self._internal_state(snap)
        self._external_state(snap, user_msg)
        if semantic_will is not None:
            snap.will = _clamp(semantic_will)

        calm = 1.0 - _clamp(
            snap.metamatter_magnitude / _METAMATTER_STRESS
        )
        state_quality = (
            0.30 * _clamp(snap.field_coherence)
            + 0.25 * _clamp(snap.matrixial_stability)
            + 0.15 * _clamp(1.0 - snap.emotion_load)
            + 0.15 * _clamp(1.0 - snap.shield_throttle)
            + 0.15 * calm
        )
        memory_coupling = 0.5 + 0.5 * _clamp(snap.memory_pull)
        base_index = 0.70 * state_quality + 0.30 * memory_coupling
        snap.reality_index = _clamp(base_index)

        snap.components = {
            "state_quality": state_quality,
            "memory_coupling": memory_coupling,
            "calm": calm,
            "metamatter_state": 1.0 - calm,
        }
        self._last_snapshot = snap
        return snap

    def status(self) -> Dict[str, Any]:
        last = self._last_snapshot
        mm = self.metamatter()
        return {
            "organ": "metareality",
            "observations": self._observations,
            "episodes": len(self._episodes),
            "metamatter_path": self.METAMATTER_PATH,
            "metamatter_source": "real"
            if self._metamatter is not None
            else "offline",
            "metamatter_dims": int(mm.size) if mm is not None else 0,
            "last_snapshot": last.to_dict() if last is not None else None,
        }

    def _reality_vector(self, snap: RealitySnapshot) -> np.ndarray:
        """Vetor escalar de realidade [reality_index, emoção, energia, campo,
        vontade, metamateria, memória] — base da deriva entre ciclos."""
        return np.array(
            [
                float(snap.reality_index),
                float(snap.emotion_load),
                float(snap.energy) / 100.0,
                float(snap.field_coherence),
                float(snap.will),
                float(snap.metamatter_magnitude) / 1000.0,
                float(snap.memory_pull),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------ alinhamento tensorial
    def _semantic_reality_metrics(
        self,
        snippet: str,
        snap: RealitySnapshot,
        *,
        conversation_key: Optional[str] = None,
    ) -> tuple[Optional[float], Optional[Dict[str, Any]]]:
        """Estratégia sênior: deriva/cos via embedding semântico REAL do turno.

        Compara o embedding (serviço local :21000 → 640D) do snippet atual com o
        âncora persistida da última onda do mesmo reinado. Retorna (drift, anchor):
        o âncora persiste sempre que o embedding existe (mesmo na 1ª onda, sem
        vetor de comparação) para a 2ª onda alinhar de verdade. Sem serviço de
        embedding → (None, None) e o chamador cai no legado.
        """
        embed_client = getattr(getattr(self._core, "kbase", None), "client", None)
        if embed_client is None or not hasattr(embed_client, "embed"):
            return None, None
        try:
            raw = embed_client.embed(snippet)
        except Exception:
            return None, None
        if not raw:
            return None, None
        cur_vec = np.asarray(raw, dtype=np.float32).reshape(-1)
        if cur_vec.size < 2:
            return None, None
        prev_vec: Optional[np.ndarray] = None
        reality = getattr(self._core, "reality_store", None)
        if reality is not None and hasattr(reality, "latest_wave_metrics"):
            try:
                anchor = reality.latest_wave_metrics(
                    conversation_key=conversation_key
                )
                if isinstance(anchor, dict):
                    prev_vec = _anchor_emb(anchor.get("anchor_vec"))
            except Exception:
                prev_vec = None
        drift = _reality_drift(prev_vec, cur_vec) if prev_vec is not None else None
        anchor_vec = _compact_anchor(snap, cur_vec)
        return drift, anchor_vec

    def _legacy_reality_metrics(
        self,
        snap: RealitySnapshot,
        *,
        conversation_key: Optional[str] = None,
    ) -> Optional[float]:
        """Fallback legado: deriva escalar 7D × última onda persistida (ou último
        observe em memória). Sem histórico comparável → None, nunca 0.5 fabricado."""
        try:
            cur_vec = self._reality_vector(snap)
            prev_metrics: Optional[Dict[str, Any]] = None
            reality = getattr(self._core, "reality_store", None)
            if reality is not None and hasattr(reality, "latest_wave_metrics"):
                try:
                    prev_metrics = reality.latest_wave_metrics(
                        conversation_key=conversation_key
                    )
                except Exception:
                    prev_metrics = None
            if prev_metrics:
                prev_vec = np.array(
                    [
                        float(prev_metrics.get("reality_index", 0.5)),
                        float(prev_metrics.get("load", 0.5)),
                        float(prev_metrics.get("energy", 0.0) or 0.0) / 100.0,
                        float(prev_metrics.get("coherence", 1.0)),
                        float(prev_metrics.get("will", 0.5)),
                        float(prev_metrics.get("metamatter", 0.0) or 0.0) / 1000.0,
                        float(prev_metrics.get("memory_pull", 0.0)),
                    ],
                    dtype=np.float32,
                )
                drift = _reality_drift(prev_vec, cur_vec)
            elif (
                self._last_reality_vec is not None
                and self._last_reality_key == conversation_key
            ):
                drift = _reality_drift(self._last_reality_vec, cur_vec)
            else:
                drift = None
            self._last_reality_vec = cur_vec
            self._last_reality_key = conversation_key
            return drift
        except Exception:
            return None

    # ------------------------------------------------------------ tábua dos reinados
    def cycle_history(
        self,
        *,
        conversation_key: Optional[str] = None,
        limit: int = 40,
    ) -> Dict[str, Any]:
        """Cada ciclo = um passo. Série das ondas (reinados) + totais + eco.

        Best-effort: sem RealityStore disponível, retorna a estrutura vazia.
        """
        reality = getattr(self._core, "reality_store", None)
        series: List[Dict[str, Any]] = []
        if reality is not None and hasattr(reality, "wave_series"):
            try:
                rows = reality.wave_series(conversation_key=conversation_key, limit=limit)
                for row in rows:
                    metrics = row.get("metrics") or {}
                    series.append(
                        {
                            "id": row.get("id"),
                            "conversation_key": row.get("conversation_key"),
                            "wave_index": row.get("wave_index"),
                            "phase": row.get("phase"),
                            "created_at": str(row.get("created_at") or ""),
                            "reality_index": metrics.get("reality_index"),
                            "drift": metrics.get("drift"),
                            "will": metrics.get("will"),
                            "decision": metrics.get("decision"),
                            "emotion": metrics.get("emotion"),
                            "metamatter": metrics.get("metamatter"),
                            "budget": metrics.get("budget"),
                        }
                    )
            except Exception:
                pass
        return {
            "cycles": series,
            "count": len(series),
            "totals": self._cycle_totals(series),
            "quality": self.session_quality(conversation_key=conversation_key, limit=limit),
            "echo": self.last_reality_index(conversation_key=conversation_key),
            "anchor": round(self.cycle_weight(conversation_key=conversation_key), 3),
        }

    def session_quality(
        self,
        *,
        conversation_key: Optional[str] = None,
        limit: int = 40,
    ) -> Dict[str, Any]:
        """Qualidade da fissão na sessão: cos_sim médio + razão de ondas completas."""
        reality = getattr(self._core, "reality_store", None)
        if reality is not None and hasattr(reality, "session_quality"):
            try:
                return reality.session_quality(
                    conversation_key=conversation_key, limit=limit
                )
            except Exception:
                pass
        return {
            "conversation_key": conversation_key,
            "waves": 0,
            "terminal_waves": 0,
            "complete_ratio": 0.0,
            "mean_cos_sim": None,
        }

    @staticmethod
    def _cycle_totals(series: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not series:
            return {
                "mean_reality_index": None,
                "mean_drift": None,
                "decision_entropy": None,
            }
        import math

        ris = [
            c["reality_index"]
            for c in series
            if isinstance(c["reality_index"], (int, float))
        ]
        drifts = [
            c["drift"]
            for c in series
            if isinstance(c["drift"], (int, float))
        ]
        decisions = [c["decision"] for c in series if c["decision"]]
        mean_ri = (sum(ris) / len(ris)) if ris else None
        mean_drift = (sum(drifts) / len(drifts)) if drifts else None
        entropy = None
        if decisions:
            from collections import Counter

            counts = Counter(decisions)
            total = len(decisions)
            entropy = -sum((n / total) * math.log2(n / total) for n in counts.values())
        return {
            "mean_reality_index": round(mean_ri, 3) if mean_ri is not None else None,
            "mean_drift": round(mean_drift, 3) if mean_drift is not None else None,
            "decision_entropy": round(entropy, 3) if entropy is not None else None,
        }

    def last_reality_index(self, *, conversation_key: Optional[str] = None) -> Optional[float]:
        """Eco do ciclo anterior: reality_index da onda de realidade mais recente."""
        reality = getattr(self._core, "reality_store", None)
        if reality is None or not hasattr(reality, "latest_wave_score"):
            return None
        try:
            return reality.latest_wave_score(
                conversation_key=conversation_key, metric_key="reality_index"
            )
        except Exception:
            return None

    def cycle_weight(self, *, conversation_key: Optional[str] = None) -> float:
        """Âncora adaptativa: drift alto do último ciclo → mais peso da realidade.

        Neutro em 1.0; varia ±drift·2·; clampado em [0.6, 1.5].
        """
        reality = getattr(self._core, "reality_store", None)
        if reality is None or not hasattr(reality, "latest_wave_score"):
            return 1.0
        try:
            drift = reality.latest_wave_score(
                conversation_key=conversation_key, metric_key="drift"
            )
            if drift is None:
                return 1.0
            weight = 1.0 + 2.0 * max(-0.3, min(0.5, float(drift)))
            return max(0.6, min(1.5, weight))
        except Exception:
            return 1.0

    # ------------------------------------------------------------ contexto injetável
    def context(self, user_msg: str = "", max_chars: int = 600) -> str:
        """Trecho de telemetria injetável (organism_context / build_steering)."""
        snap = self.integrate(user_msg)
        recent = self.recent_episodes(1)
        recent_txt = ""
        if recent:
            snippet = recent[0].replace('"', "'")[:120]
            recent_txt = f', "recent": "{snippet}"'
        eco = self.last_reality_index()
        eco_txt = f"{eco:.2f}" if eco is not None else "null"
        anchor = round(self.cycle_weight(), 2)
        label = "alerta" if snap.metamatter_magnitude > _METAMATTER_STRESS else "estável"
        memory = (
            f"{snap.memory_pull:.2f} ({snap.memory_hits} hits)"
            if snap.repio_available
            else "offline"
        )
        body = (
            '{"metareality": {'
            f'"emotion": "{snap.emotion_label}", '
            f'"load": {snap.emotion_load:.2f}, '
            f'"energy": {snap.energy:.1f}, '
            f'"coherence": {snap.field_coherence:.2f}, '
            f'"stability": {snap.matrixial_stability:.2f}, '
            f'"metamatter": "{label}", '
            f'"memory_pull": "{memory}", '
            f'"will": {snap.will:.2f}, '
            f'"reality_index": {snap.reality_index:.2f}, '
            f'"cycle_echo": {eco_txt}, '
            f'"cycles": {self._observations}, '
            f'"anchor": {anchor}'
            f"{recent_txt}"
            "}}"
        )
        if len(body) > max(80, max_chars):
            body = body[: max(80, max_chars)] + "...}"
        return body

    # ------------------------------------------------------------ vetor de realidade
    def vector(self, model_dim: int = 1152) -> Optional[Any]:
        """Representação vetorial da realidade (1,1,model_dim).

        Origem única e real: metamatéria (640) do subcórtex concatenada aos
        componentes escalares do snapshot. Sem metamatéria real → None
        (nunca sintetiza: sem sinal do subcórtex não há âncora confiável).
        """
        try:
            mm = self.metamatter()
            if mm is None:
                return None
            import torch

            snap = self.integrate()
            parts = [mm]
            if snap.components:
                parts.append(
                    np.asarray(list(snap.components.values()), dtype=np.float32)
                )
            parts.append(
                np.asarray(
                    [
                        snap.reality_index,
                        snap.will,
                        snap.memory_pull,
                        1.0 - snap.emotion_load,
                    ],
                    dtype=np.float32,
                )
            )
            base = np.concatenate(parts).astype(np.float32)
            base = np.resize(base, (model_dim,)).astype(np.float32)
            norm = float(np.linalg.norm(base))
            if norm > 0:
                base = base / norm
            t = torch.as_tensor(base, dtype=torch.float32)
            return t.view(1, 1, model_dim)
        except Exception:
            return None

    # ------------------------------------------------------------ observação
    def observe(
        self,
        user_msg: str,
        reply: str,
        *,
        wave: int = 0,
        concept: str = "REALITY",
        semantic_will: Optional[float] = None,
        conversation_key: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Registra o episódio de realidade ao fim de um turno."""
        try:
            snap = self.integrate(user_msg, semantic_will=semantic_will)
        except Exception:
            return None

        # ------- alinhamento tensorial entre ondas do mesmo reinado ------------
        # Estratégia sênior: cos REAL entre o embedding semântico do turno
        # (serviço local :21000 → 640D) e a âncora da última onda persistida.
        # Fallback: deriva escalar legada (7D). Sem sinal comparável → None,
        # nunca um 0.5 fabricado (a 1ª onda de um reinado fica sem cos até a 2ª).
        snippet = (reply or "").strip().replace("<end_of_turn>", "")[:200]
        if not snippet:
            return None

        # ---- idempotência: um turno só vira UM evento/onda --------------------
        # `hybrid_generate`, `stream` (nativo/fallback) e `_observe_turn` podem
        # observar o mesmo turno em camadas diferentes → digest do turno para a
        # Tábua nunca gravar ping-pong de observações dentro da janela.
        ckey = str(conversation_key) if conversation_key else ""
        turn_digest = hashlib.sha256(
            f"{ckey}|{snippet}".encode("utf-8")
        ).hexdigest()[:16]
        reality = getattr(self._core, "reality_store", None)
        if reality is not None and hasattr(reality, "metareality_seen"):
            try:
                if reality.metareality_seen(turn_digest):
                    return None
            except Exception:
                pass

        # ---- auto-avanço da Tábua: cada turno do reinado sobe a onda ----------
        # O caminho híbrido chega com `wave=0` fixo; a fissão do `stream_messages`
        # recém-gravou a onda do turno, então reutilizamos o índice recente
        # (janela de segundos) — 1 onda por turno — e só avançamos (máx+1) quando
        # não há onda recente duplicada do mesmo reinado.
        if conversation_key and reality is not None and hasattr(
            reality, "latest_wave_index_if_recent"
        ):
            try:
                recent = reality.latest_wave_index_if_recent(
                    conversation_key=conversation_key, window_s=45
                )
                if recent is not None:
                    wave = int(recent)
                elif hasattr(reality, "next_wave_index"):
                    wave = int(
                        reality.next_wave_index(conversation_key=conversation_key)
                    )
            except Exception:
                wave = int(wave)
        else:
            wave = int(wave)

        drift: Optional[float] = None
        anchor_vec: Optional[Dict[str, Any]] = None
        sem_drift, sem_anchor = self._semantic_reality_metrics(
            snippet, snap, conversation_key=conversation_key
        )
        if sem_drift is not None:
            drift = sem_drift
        else:
            drift = self._legacy_reality_metrics(
                snap, conversation_key=conversation_key
            )
        # âncora real persiste sempre que o embedding existe (mesmo na 1ª onda,
        # sem vetor de comparação) para a 2ª onda do reinado alinhar de verdade.
        anchor_vec = sem_anchor

        # 1) RealityStore: evento de realidade com os escalares do snapshot
        event_id: Optional[int] = None
        if reality is not None:
            try:
                payload: Dict[str, Any] = {
                    "reality_index": round(snap.reality_index, 3),
                    "emotion": snap.emotion_label,
                    "load": round(snap.emotion_load, 3),
                    "energy": round(snap.energy, 2),
                    "coherence": round(snap.field_coherence, 3),
                    "metamatter": round(snap.metamatter_magnitude, 2),
                    "memory_pull": round(snap.memory_pull, 3),
                    "reply": snippet,
                    "turn_digest": turn_digest,
                }
                # relógio de eras: carimba a pressão de sistema (pacing do ciclo)
                try:
                    sys_load = float(os.getloadavg()[0])
                    payload["sys_load"] = round(sys_load, 2)
                except Exception:
                    pass
                reality.append_event(
                    "metareality",
                    payload,
                    wave=int(wave),
                )
            except Exception:
                pass

        # 2) Onda relacional: cada turno vira uma onda de realidade
        # (aeon_reality_waves) — a metarealidade interna fica restaurável e
        # consultável só pelo PostgreSQL, sem depender do journal binário.
        if reality is not None and conversation_key:
            try:
                metrics = {
                    "reality_index": round(snap.reality_index, 3),
                    "emotion": snap.emotion_label,
                    "load": round(snap.emotion_load, 3),
                    "coherence": round(snap.field_coherence, 3),
                    "metamatter": round(snap.metamatter_magnitude, 2),
                    "memory_pull": round(snap.memory_pull, 3),
                    "will": round(snap.will, 3),
                    "turn_digest": turn_digest,
                }
                if drift is not None:
                    metrics["drift"] = round(drift, 3)
                    metrics["cos_sim"] = round(max(-1.0, min(1.0, 1.0 - 2.0 * drift)), 3)
                if anchor_vec is not None:
                    metrics["anchor_vec"] = anchor_vec
                try:
                    sys_load = float(os.getloadavg()[0])
                    metrics["sys_load"] = round(sys_load, 2)
                except Exception:
                    pass
                event_id = reality.append_wave_state(
                    conversation_key,
                    int(wave),
                    phase="fim",
                    metrics=metrics,
                    merge_metrics=True,
                )
            except Exception:
                pass

        # 3) REPIO: persiste o episódio como âncora da realidade (k-base)
        ingested = False
        repio = getattr(self._core, "_repio", None)
        if repio is not None:
            try:
                ingested = bool(
                    repio.ingest(snippet, concept=concept, source="metareality")
                )
            except Exception:
                ingested = False

        episode = f"{snap.reality_index:.2f} | {snap.emotion_label} | {snippet}"
        self._episodes.append(episode)
        self._observations += 1
        return {
            "reality_index": round(snap.reality_index, 3),
            "ingested": ingested,
            "wave": int(wave),
            "wave_id": event_id,
        }

    def recent_episodes(self, n: int = 3) -> List[str]:
        items = list(self._episodes)
        return items[-int(n):]

    # ------------------------------------------------------------ OODA
    def register_ooda(self, ooda_flow) -> bool:
        """Placa o snapshot da realidade no estado do ciclo OODA (não-bloqueante)."""
        try:
            if ooda_flow is None or not hasattr(ooda_flow, "on_cycle"):
                return False

            def _inject(state):
                try:
                    snap = self.integrate()
                    setattr(state, "metareality", snap.to_dict())
                except Exception:
                    pass

            ooda_flow.on_cycle(_inject)
            return True
        except Exception:
            return False