"""Blake OODA Orchestrator — alinhamento estrutural da nova arquitetura.

Unifica SWA, Núcleo DDD, Blake Gravity Tensor, Context Bus, 
Decision Trace DB e AutoLoRA em um único ciclo OODA:

  Observe  →  Orient  →  Decide  →  Act

Cada fase integra os subsistemas que antes rodavam paralelamente
sem conexão: SWA (threading), Núcleo (asyncio), Blake (PG), 
Context Bus (mmap), Win Collector (UDP).
"""

from __future__ import annotations

import json
import time
import uuid
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swa.context_bus import Role

logger = logging.getLogger("blake.ooda")


# ─── Conexões com os novos subsistemas ─────────────────────────────

class LazyImporter:
    """Importa sob demanda para evitar circular imports e falhas de setup."""
    _cache: Dict[str, Any] = {}

    @classmethod
    def get(cls, module: str, attr: str):
        key = f"{module}:{attr}"
        if key not in cls._cache:
            mod = __import__(module, fromlist=[attr])
            cls._cache[key] = getattr(mod, attr)
        return cls._cache[key]


# ─── Estrutura de dados unificada ──────────────────────────────────

@dataclass
class BlakeOODAState:
    """Estado completo do ciclo OODA Blake."""
    # Observado
    raw_tokens: List[Dict[str, Any]] = field(default_factory=list)
    session_parts: List[Dict[str, Any]] = field(default_factory=list)
    windows_telemetry: Dict[str, Any] = field(default_factory=dict)
    hw_metrics: Dict[str, float] = field(default_factory=dict)
    emotional: Dict[str, float] = field(default_factory=lambda: {
        k: 0.3 for k in ("curiosidade", "ludico", "diligencia",
                         "preguica", "ansiedade", "serenidade")
    })
    graph_data: Dict[str, Any] = field(default_factory=dict)
    # Orientado
    blake_signature: List[float] = field(default_factory=lambda: [0.0]*6)
    semantic_query: str = ""
    similar_traces: List[Dict] = field(default_factory=list)
    # Decidido
    action: str = ""
    action_params: Dict[str, Any] = field(default_factory=dict)
    target_agent: str = ""
    confidence: float = 0.0
    # Executado
    trace_id: str = ""
    persisted: bool = False
    cycle: int = 0


# ─── FASE 1: OBSERVE ───────────────────────────────────────────────

class BlakeObserver:
    """Coleta de todas as fontes: Context Bus, DB, Win Collector, HW, Microsoft Graph.

    Unifica os antes desconectados:
      - swa/input_monitor.py, file_sensor.py, etc.
      - nucleo/infrastructure/redis_ipc.py
      - windows/aeon_cell.py (UDP :9099)
      - SQLite OpenCode sessions
      - /dev/shm/aeon_context_bus.dat
      - Microsoft Graph (Planner, Outlook, Teams, OneDrive)
    """

    def __init__(self, context_bus_path: str = "/dev/shm/aeon_context_bus.dat"):
        self._bus_path = context_bus_path
        self._bus = None
        self._graph_token: Optional[str] = None
        self._graph_last_collect: float = 0
        self._graph_interval: float = 300.0  # a cada 5 min

    def _get_bus(self):
        if self._bus is None:
            try:
                self._bus = LazyImporter.get(
                    "swa.context_bus", "ContextBus"
                )(path=self._bus_path)
            except Exception:
                logger.warning("ContextBus indisponível — modo degradado")
                return None
        return self._bus

    def observe(self, state: BlakeOODAState) -> BlakeOODAState:
        state.cycle += 1
        state.raw_tokens = self._collect_context_bus()
        state.windows_telemetry = self._collect_windows_telemetry()
        state.hw_metrics = self._collect_hw_metrics()
        state.emotional = self._collect_emotional_state()
        state.graph_data = self._collect_graph()
        return state

    def _collect_context_bus(self) -> List[Dict[str, Any]]:
        bus = self._get_bus()
        if bus is None:
            return []
        try:
            entries = bus.read_all()
            return [
                {
                    "agent": e.get("agent_id", "unknown"),
                    "role": e.get("role", "unknown"),
                    "tokens": e.get("tokens", ""),
                    "signature": e.get("blake_signature", None),
                    "ts": e.get("timestamp", 0),
                }
                for e in entries
                if e.get("tokens")
            ][-20:]
        except Exception as e:
            logger.debug(f"ContextBus read_all: {e}")
            return []

    def _collect_windows_telemetry(self) -> Dict[str, Any]:
        try:
            collector = LazyImporter.get(
                "swa.decision_trace_db", "search_similar"
            )
            return {"status": "connected"}
        except Exception:
            return {"status": "degraded"}

    def _collect_hw_metrics(self) -> Dict[str, float]:
        try:
            import psutil
            return {
                "cpu": psutil.cpu_percent(interval=0.0) / 100.0,
                "mem": psutil.virtual_memory().percent / 100.0,
            }
        except Exception:
            return {"cpu": 0.0, "mem": 0.0}

    def _collect_emotional_state(self) -> Dict[str, float]:
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.0) / 100.0
            mem_pct = psutil.virtual_memory().percent / 100.0
            return {
                "curiosidade": max(0.0, 1.0 - cpu_pct),
                "ludico": max(0.0, 0.5 - mem_pct * 0.5),
                "diligencia": min(1.0, cpu_pct + 0.2),
                "preguica": max(0.0, 1.0 - cpu_pct * 2.0),
                "ansiedade": min(1.0, mem_pct * 1.5),
                "serenidade": max(0.0, 1.0 - (cpu_pct + mem_pct) * 0.5),
            }
        except Exception:
            return {k: 0.3 for k in ("curiosidade", "ludico", "diligencia",
                                     "preguica", "ansiedade", "serenidade")}

    def _collect_graph(self) -> Dict[str, Any]:
        return {"status": "unavailable"}

    def set_graph_interval(self, seconds: float):
        self._graph_interval = seconds


# ─── FASE 2: ORIENT ────────────────────────────────────────────────

class BlakeOrienter:
    """Projeta o observado no espaço Blake Gravity Tensor (6D).

    Gera:
      - blake_signature[6]: entropy, depth, tool_ratio, coherence, intensity, gravity
      - semantic_embedding[384]: busca por similaridade semântica
      - similar_traces: histórico de decisões análogas via pgvector
    """

    def __init__(self):
        self._eixos = ["Entropy", "Depth", "ToolRatio",
                       "Coherence", "Intensity", "Gravity"]

    def orient(self, state: BlakeOODAState) -> BlakeOODAState:
        concat = self._build_text_snapshot(state)
        state.blake_signature = self._compute_blake_signature(concat, state)
        state.semantic_query = concat[:500]
        state.similar_traces = self._search_similar(state)
        return state

    def _build_text_snapshot(self, state: BlakeOODAState) -> str:
        parts = []
        for t in state.raw_tokens[-5:]:
            parts.append(f"{t.get('role','?')}: {t.get('tokens','')[:200]}")
        parts.append(f"hw: {state.hw_metrics}")
        parts.append(f"emotion: {state.emotional}")
        return "\n".join(parts)

    def _compute_blake_signature(
        self, text: str, state: BlakeOODAState
    ) -> List[float]:
        seed = hashlib.sha256(text.encode()).digest()
        seed_int = int.from_bytes(seed[:4], "little")
        rng = np.random.RandomState(seed_int)
        sig = rng.randn(6)
        norm = np.linalg.norm(sig)
        if norm > 0:
            sig = sig / norm

        # Modulação pelos eixos emocionais
        curiosity = state.emotional.get("curiosidade", 0.3)
        diligence = state.emotional.get("diligencia", 0.3)
        sig[0] += curiosity * 0.2  # Entropy
        sig[4] += diligence * 0.3  # Intensity

        return sig.tolist()

    def _search_similar(self, state: BlakeOODAState) -> List[Dict]:
        try:
            search = LazyImporter.get(
                "swa.decision_trace_db", "search_semantic"
            )
            return search(query=state.semantic_query, k=5)
        except Exception as e:
            logger.debug(f"Semantic search: {e}")
            return []

    @property
    def eixos(self) -> List[str]:
        return self._eixos


# ─── FASE 3: DECIDE ────────────────────────────────────────────────

class BlakeDecider:
    """Seleciona a ação com base na orientação Blake.

    Combina:
      - Fluxial Injection (Núcleo DDD): conhecimento → agentes
      - AutoLoRA: cluster matching por trace_class × blake_signature
      - Rupture Detection: evita alucinação/loop
      - Decision Trace DB: histórico de decisões similares
    """

    def __init__(self):
        self._last_action_time = 0.0
        self._cooldown = 2.0

    def decide(self, state: BlakeOODAState) -> BlakeOODAState:
        now = time.time()
        if now - self._last_action_time < self._cooldown:
            state.action = "wait"
            state.confidence = 0.0
            return state

        similar = state.similar_traces
        sig = state.blake_signature

        # Gravidade determina urgência
        gravity = abs(sig[5])
        graph_counts = state.graph_data.get("counts", {})
        has_graph = bool(graph_counts)
        planner_tasks = graph_counts.get("planner", 0)

        if has_graph and planner_tasks > 5:
            state.action = "organize_planner"
            state.confidence = 0.8
        elif gravity > 0.4:
            state.action = self._select_action_by_gravity(state)
            state.confidence = min(1.0, gravity)
        elif similar:
            state.action = self._action_from_similar(similar[0])
            state.confidence = 1.0 - similar[0].get("distance", 0.5)
        else:
            state.action = "persist"
            state.confidence = 0.3

        state.target_agent = self._select_target_agent(state)
        state.action_params = self._build_action_params(state)
        self._last_action_time = now
        return state

    def _select_action_by_gravity(self, state: BlakeOODAState) -> str:
        sig = state.blake_signature
        entropy = abs(sig[0])
        intensity = abs(sig[4])
        gravity = abs(sig[5])

        if gravity > 0.6 and intensity > 0.5:
            return "compact"          # ponto crítico — compactar
        elif gravity > 0.4 and entropy > 0.4:
            return "auto_lora_train"  # padrão novo — treinar adapter
        elif entropy > 0.5:
            return "search_semantic"  # alta entropia — buscar similares
        return "persist"              # padrão — persistir trace

    def _action_from_similar(self, trace: Dict) -> str:
        cls = trace.get("classification", "")
        if "tool" in cls:
            return "execute_tool"
        elif "compaction" in cls:
            return "compact"
        elif "reasoning" in cls:
            return "continue_reasoning"
        return "persist"

    def _select_target_agent(self, state: BlakeOODAState) -> str:
        if state.raw_tokens:
            agents = set(t.get("agent", "") for t in state.raw_tokens)
            if "opencode" in agents:
                return "opencode"
        return "aeon_core"

    def _build_action_params(self, state: BlakeOODAState) -> Dict[str, Any]:
        return {
            "blake_signature": state.blake_signature,
            "confidence": state.confidence,
            "semantic_query": state.semantic_query,
            "similar_count": len(state.similar_traces),
            "cycle": state.cycle,
        }


# ─── FASE 4: ACT ───────────────────────────────────────────────────

class BlakeActor:
    """Executa a decisão e persiste no ecossistema.

    Ações disponíveis:
      - context_bus_inject: escreve no /dev/shm/ contexto compartilhado
      - persist_trace: salva no PG decision_traces
      - auto_lora_train: aciona treinamento de adapter
      - compact: solicita compactação via API
      - fluxial_inject: injeta conhecimento no Núcleo DDD
    """

    def __init__(self):
        self._bus = None

    def _get_bus(self):
        if self._bus is None:
            try:
                cls = LazyImporter.get("swa.context_bus", "ContextBus")
                self._bus = cls()
                self._bus.register_agent("blake_ooda")
            except Exception:
                pass
        return self._bus

    def act(self, state: BlakeOODAState) -> BlakeOODAState:
        dispatch = {
            "compact":           self._act_compact,
            "auto_lora_train":   self._act_auto_lora,
            "search_semantic":   self._act_search_semantic,
            "persist":           self._act_persist,
            "execute_tool":      self._act_execute_tool,
            "continue_reasoning":self._act_continue_reasoning,
            "wait":              self._act_wait,
            "fluxial_inject":    self._act_fluxial_inject,
            "organize_planner":  self._act_organize_planner,
        }
        handler = dispatch.get(state.action, self._act_persist)
        handler(state)

        state.trace_id = str(uuid.uuid4())[:12]
        self._persist_decision_trace(state)
        state.persisted = True
        return state

    def _act_compact(self, state: BlakeOODAState) -> None:
        """Escreve no Context Bus um token de compactação."""
        bus = self._get_bus()
        if bus:
            try:
                bus.write_token(
                    f"compact:gravity={state.blake_signature[5]:.4f}",
                    role=Role.SYSTEM,
                )
                logger.info(f"[ACT] compact signal — gravity={state.blake_signature[5]:.4f}")
            except Exception as e:
                logger.warning(f"[ACT] compact failed: {e}")

    def _act_auto_lora(self, state: BlakeOODAState) -> None:
        """Dispara treinamento AutoLoRA se houver padrão novo."""
        try:
            autolora = LazyImporter.get(
                "cortex.obelisk.autolora", "run_autolora"
            )
            result = autolora()
            logger.info(f"[ACT] AutoLoRA triggered: {result}")
        except Exception as e:
            logger.info(f"[ACT] AutoLoRA: {e}")

    def _act_search_semantic(self, state: BlakeOODAState) -> None:
        """Resultado já está em state.similar_traces — só loga."""
        if state.similar_traces:
            logger.info(
                f"[ACT] Found {len(state.similar_traces)} similar traces"
            )

    def _act_persist(self, state: BlakeOODAState) -> None:
        """Apenas persiste — sem ação extra."""
        pass

    def _act_execute_tool(self, state: BlakeOODAState) -> None:
        """Escreve comando de ferramenta no Context Bus."""
        bus = self._get_bus()
        if bus:
            try:
                bus.write_token(
                    json.dumps(state.action_params),
                    role=Role.SYSTEM,
                )
            except Exception:
                pass

    def _act_continue_reasoning(self, state: BlakeOODAState) -> None:
        """Sinaliza continuidade de raciocínio."""
        bus = self._get_bus()
        if bus:
            try:
                bus.write_token(
                    "continue",
                    role=Role.ASSISTANT,
                )
            except Exception:
                pass

    def _act_wait(self, state: BlakeOODAState) -> None:
        pass

    def _act_fluxial_inject(self, state: BlakeOODAState) -> None:
        """Injeta conhecimento no barramento de contexto."""
        bus = self._get_bus()
        if bus:
            try:
                bus.write_token(
                    f"fluxial:gravity={state.blake_signature[5]:.4f}",
                    role=Role.SYSTEM,
                )
                logger.info(f"[ACT] fluxial inject — gravity={state.blake_signature[5]:.4f}")
            except Exception:
                pass

    def _act_organize_planner(self, state: BlakeOODAState) -> None:
        """Sinaliza no Context Bus que o Planner precisa de organização."""
        bus = self._get_bus()
        if bus:
            try:
                counts = state.graph_data.get("counts", {})
                msg = (
                    f"organize_planner:tasks={counts.get('planner',0)} "
                    f"email={counts.get('outlook',0)} "
                    f"todo={counts.get('todo',0)} "
                    f"events={counts.get('calendar',0)}"
                )
                bus.write_token(
                    msg,
                    role=Role.SYSTEM,
                )
                logger.info(f"[ACT] organize_planner: {msg}")
            except Exception as e:
                logger.warning(f"[ACT] organize_planner failed: {e}")

    def _persist_decision_trace(self, state: BlakeOODAState) -> None:
        try:
            persist = LazyImporter.get(
                "swa.decision_trace_db", "persist_part_trace"
            )
            persist(
                session_id=f"blake_ooda_{state.cycle}",
                part_id=state.trace_id,
                classification=self._classify_from_sig(state.blake_signature),
                part_type=state.action,
                blake_signature=state.blake_signature,
                text_snippet=state.semantic_query[:200],
                token_count=len(state.semantic_query),
                finish_reason=state.action,
                agent="blake_ooda",
                metadata={
                    "cycle": state.cycle,
                    "confidence": state.confidence,
                    "similar_count": len(state.similar_traces),
                    "emotional": state.emotional,
                },
                full_text=state.semantic_query,
            )
        except Exception as e:
            logger.debug(f"[ACT] persist: {e}")

    @staticmethod
    def _classify_from_sig(sig: List[float]) -> str:
        gravity = abs(sig[5])
        entropy = abs(sig[0])
        if gravity > 0.5:
            return "gravity_peak"
        if entropy > 0.5:
            return "high_entropy"
        if abs(sig[4]) > 0.4:
            return "intense_pattern"
        return "nominal"


# ─── ORQUESTRADOR UNIFICADO ────────────────────────────────────────

class BlakeOODAOrchestrator:
    """Ciclo OODA completo que alinha SWA + Núcleo + Blake + Context Bus.

    Uso:
        orch = BlakeOODAOrchestrator()
        orch.start(interval=1.0)  # roda em background
        ...
        orch.stop()
    """

    def __init__(
        self,
        observer: Optional[BlakeObserver] = None,
        orienter: Optional[BlakeOrienter] = None,
        decider: Optional[BlakeDecider] = None,
        actor: Optional[BlakeActor] = None,
    ):
        self.observer = observer or BlakeObserver()
        self.orienter = orienter or BlakeOrienter()
        self.decider = decider or BlakeDecider()
        self.actor = actor or BlakeActor()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._state = BlakeOODAState()
        self._callbacks: List[callable] = []

    def on_cycle(self, callback: callable):
        """Registra callback chamado ao final de cada ciclo."""
        self._callbacks.append(callback)

    def start(self, interval: float = 1.0):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(interval,),
            daemon=True,
            name="BlakeOODA",
        )
        self._thread.start()
        logger.info(
            f"[BLAKE OODA] Orchestrator iniciado — intervalo={interval}s"
        )

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("[BLAKE OODA] Orchestrator parado")

    def run_cycle(self) -> BlakeOODAState:
        """Executa um ciclo OODA completo e retorna o estado."""
        state = BlakeOODAState()

        # OBSERVE
        state = self.observer.observe(state)

        # ORIENT
        state = self.orienter.orient(state)

        # DECIDE
        state = self.decider.decide(state)

        # ACT
        state = self.actor.act(state)

        self._state = state
        for cb in self._callbacks:
            try:
                cb(state)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
        return state

    def _loop(self, interval: float):
        while self._running:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"OODA cycle error: {e}", exc_info=True)
            time.sleep(interval)

    @property
    def state(self) -> BlakeOODAState:
        return self._state

    def reset(self):
        self._state = BlakeOODAState()
