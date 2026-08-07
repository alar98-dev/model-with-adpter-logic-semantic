from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator



class ToolParameter(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    required: Optional[bool] = False


class ToolSchema(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    required: Optional[List[str]] = None


class ProcessPayload(BaseModel):
    model_config = {"populate_by_name": True}

    texts: Optional[List[str]] = Field(None, description="List of texts/events to process")
    use_fake: Optional[bool] = Field(False, description="Use fake vectorizer for quick tests")
    validate_data: Optional[bool] = Field(False, description="Run a lightweight self-check instead of normal processing", alias="validate")
    tool_schema: Optional[Dict[str, Any]] = Field(None, description="Optional tool/schema object (will be flattened for embedding)", alias="schema")
    tools: Optional[Any] = Field(None, description="Optional tools metadata (list or dict) to include in context")


class ChatRequest(BaseModel):
    """Entrada unificada para /api/v1/chat: o motor funde schema/tools/contexto automaticamente."""

    model_config = {"populate_by_name": True}

    prompt: Optional[str] = Field(None, description="Pergunta ou instrução principal")
    message: Optional[str] = Field(None, description="Alias de prompt (clientes tipo chat)")
    context: Optional[str] = Field(None, description="Contexto livre incorporado no embedding SWA")
    instruction: Optional[str] = Field(None, description="Instrução de sistema ou tarefa")
    tool_schema: Optional[Dict[str, Any]] = Field(
        None,
        alias="schema",
        description="Schema triplo (claim/…) ou definição de tool (name/parameters)",
    )
    tools: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(
        None,
        description="Lista de tools OpenAPI-like ou dict nome→resultado",
    )
    texts: Optional[List[str]] = Field(None, description="Blocos de contexto anexados ao prompt")
    tool: Optional[str] = Field(None, description="Força tool conhecida (read_file_content, …)")
    max_new_tokens: int = Field(128, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.05, le=2.0)
    reset_context: bool = Field(False, description="Limpa memória SWA antes deste turno")
    use_llama_context: Optional[bool] = Field(
        None,
        description="Llama bruto no embedding; None = auto se USE_DUAL_MODEL e SWA_AUTO_LLAMA_PREFACE",
    )

    @model_validator(mode="after")
    def _require_input(self) -> ChatRequest:
        p = (self.prompt or self.message or "").strip()
        structured = bool(self.tool_schema or self.tools)
        ctx = bool((self.context or "").strip() or (self.instruction or "").strip())
        extra = bool(self.texts)
        if not p and not structured and not ctx and not extra:
            raise ValueError("Forneça prompt/message, texts, ou schema/tools/context/instruction.")
        return self


class ResultItem(BaseModel):
    index: Optional[int]
    shape: List[int]
    norm: float


class ProcessResponse(BaseModel):
    results: Optional[List[ResultItem]]


class SelfCheckDiagnostics(BaseModel):
    ok: bool
    count: Optional[int] = None
    shapes: Optional[List[List[int]]] = None
    norms: Optional[List[float]] = None
    error: Optional[str] = None


class SelfCheckResponse(BaseModel):
    self_check: SelfCheckDiagnostics


# --- Protocolo interno swa.internal.v1 (JSON em /api/v1/internal/*) ---


class LatencyProbeRequest(BaseModel):
    """POST /api/v1/internal/latency-probe — espelha prep do /chat + geração não stream."""

    prompt: str = Field(..., min_length=1, description="Texto igual ao fluxo de chat")
    temperature: float = Field(0.7, ge=0.05, le=2.0)

# --- Eventos do Semantic Bus ---

class SWAEventSchema(BaseModel):
    """Base schema for all events on the Semantic Bus."""
    event_type: str = Field(..., description="O tipo do evento (ex: telemetry:win32, ui_focus)")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Dados do evento")
    origin: str = Field(default="system", description="Origem do evento")

class SystemVitalSchema(BaseModel):
    cpu: float
    mem: float
    swap: float
    net_in: float
    net_out: float
    therm: float
    entropy: float
    blake3_hash: Optional[str] = None

def validate_event(channel: str, data: dict) -> dict:
    if not isinstance(data, dict):
        return data
    if channel == 'grid_events':
        try:
            return SWAEventSchema(**data).model_dump()
        except Exception:
            return data
    return data
