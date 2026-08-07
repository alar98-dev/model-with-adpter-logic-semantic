from __future__ import annotations

from typing import List, Optional, Generator, Any, Dict, Union
from collections import deque
import os
import re
import torch
import torch.nn.functional as F

from .gemma_integration import GemmaGenerator

from swa.vectorizer import Vectorizer
from swa.adapter import SemanticWarpAdapter
from swa.calculations import closed_context_vector
from swa import rest_translator


SYSTEM_CORE = 1

# Tools reconhecidos no fluxo automático (regex + schema OpenAPI-like).
KNOWN_TOOL_NAMES = ("read_file_content", "write_file", "get_file", "list_dir")


class LogicalEngine:

    """Motor lógico que processa eventos/stream, aplica SWA local+macro e mantém memória.

    CPU-only use: instantiate Vectorizer with device='cpu' e adapters no mesmo device.
    """

    def __init__(
        self,
        model_dir: str,
        adapter_local: SemanticWarpAdapter,
        adapter_macro: SemanticWarpAdapter,
        device: torch.device | str = "cpu",
        use_gemma: bool = False,
        closed_agg_method: str = "mean",
        tokenizer: Any = None,
    ):
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model_dir = model_dir
        self.gemma_gen: Optional[GemmaGenerator] = None
        self._tokenizer = tokenizer

        if use_gemma:

            # GemmaGenerator MUST receive tokenizer dependency injected.
            # LogicalEngine itself does not load tokenizer.
            if self._tokenizer is None:
                raise RuntimeError(
                    "LogicalEngine with use_gemma=True must inject tokenizer into GemmaGenerator upstream"
                )
            self.gemma_gen = GemmaGenerator(
                model_dir,
                device=str(self.device),
                tokenizer=self._tokenizer,
            )

        # Resolve target dtype: align Vectorizer and adapters with GemmaGenerator dtype.
        target_dtype = self.gemma_gen.embeddings.weight.dtype if self.gemma_gen is not None else torch.float32

        # ── SEPARAÇÃO DE PAPÉIS: Vectorizer usa 270M (embeddings internos) ──
        # O Gemma-3-1B (model_dir) é reservado para linguagem natural ao usuário.
        # O Gemma-3-270M (VECTORIZER_MODEL_DIR) é o córtex interno de embeddings.
        vec_model_dir = os.environ.get("VECTORIZER_MODEL_DIR", model_dir)
        if vec_model_dir != model_dir:
            print(f"🔀 [SWA DUAL-MODEL] Vectorizer → {vec_model_dir} (270M internal)")
            print(f"🗣️  [SWA DUAL-MODEL] GemmaGen  → {model_dir} (1B natural language)")

        # IMPORTANT: embeddings + generation must share the same tokenizer runtime.
        self.vectorizer = Vectorizer(
            model_dir=vec_model_dir,
            device=str(self.device),
            tokenizer=self._tokenizer,
            dtype=target_dtype,
        )

        if self.gemma_gen is not None and hasattr(self.gemma_gen, "d_model"):
            real_d_model = self.gemma_gen.d_model
            print(f"🌲 [SWA ALIGNMENT] Configurando motor base para R^{real_d_model}")

        

        self.adapter_local = adapter_local.to(device=self.device, dtype=target_dtype)
        self.adapter_macro = adapter_macro.to(device=self.device, dtype=target_dtype)
        self.memory: deque = deque(maxlen=32)
        self.closed_agg_method = closed_agg_method
        self.engines: dict = {}

    def generate_guided_stream(
        self,
        prompt: str,
        v_global: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream com bias inj + entropy check para stream_processor feedback."""
        if self.gemma_gen is None:
            raise ValueError("GemmaGenerator not initialized. Set use_gemma=True in LogicalEngine.__init__.")
        
        # [SWA ALIGNMENT] Ensure v_global matches engine device and dtype (bfloat16/float32)
        target_dtype = self.gemma_gen.embeddings.weight.dtype
        v_global = v_global.to(device=self.device, dtype=target_dtype)
        
        if self.adapter_macro.prefix_projection is not None:
            v_prefix, v_bias = self.adapter_macro.prefix_and_bias(v_global)
            # [SWA DTYPE ALIGNMENT] Cast back to target_dtype (model device/dtype)
            v_prefix = v_prefix.to(dtype=target_dtype)
            v_bias = v_bias.to(dtype=target_dtype)
        else:
            z = torch.zeros(self.gemma_gen.d_model, device=self.device, dtype=target_dtype)
            v_prefix = z
            v_bias = torch.zeros_like(z) # Avoid None bias to prevent crashes in stream_with_bias
            
        for token in self.gemma_gen.stream_with_bias(
            v_prefix, v_bias, prompt, max_new_tokens=max_new_tokens, temperature=temperature
        ):
            # [SWA ROBUSTNESS] Ensure we only yield strings to the SSE/UI layer
            if isinstance(token, tuple):
                yield token[0]
            else:
                yield str(token)

    def entropy_norm(self, text: str) -> float:
        """Métrica entropia para divergência detect (stream feedback)."""
        emb = self.vectorizer.embed([text])
        v = self.adapter_local(emb)
        return float(v.norm().item())

    def generate_response(self, user_query: str) -> str:
        """Non-streaming response generation: update context and generate full text."""
        self.process_event(user_query)
        v_global = self.close_context()
        if v_global is None:
            raise ValueError("No context available. Process events first.")
        full_response = ""
        for token in self.generate_guided_stream(user_query, v_global):
            if isinstance(token, tuple):
                full_response += str(token[0])
            else:
                full_response += str(token)
        return full_response

    def dispatch_with_tool(self, prompt: str) -> str:
        """Tool-aware dispatch: detect tool intent, update context, force JSON response."""
        pat = "(" + "|".join(re.escape(n) for n in KNOWN_TOOL_NAMES) + ")"
        tool_match = re.search(pat, prompt, re.IGNORECASE)
        if tool_match:
            tool_name = tool_match.group(1)
            tool_text = f"Execute tool: {tool_name} to access system resources."
            self.process_event(tool_text)
            json_prefix = f'{{"tool": "{tool_name}", "parameters": {{}}}} # Complete JSON: '
            return self.generate_response(json_prefix + prompt)
        return self.generate_response(prompt)

    def process_event(self, text: str) -> torch.Tensor:
        # 1. Gera o embedding do texto (Nativo do modelo em memória: 1152 ou 2560)
        emb = self.vectorizer.embed(text)
        
        # ── BLINDAGEM ADAPTATIVA DE MANIFOLD (ALINHAMENTO DE DIMENSÃO) ──
        input_dim = emb.shape[-1]
        
        try:
            # Tenta o fluxo normal (Se o modelo for o 1B/2560D, passa direto aqui)
            v_seg = self.adapter_local(emb)
        except RuntimeError as e:
            # Se estourar o erro de tamanho e o input for 1152 (Gemma 270M), aplicamos a expansão
            if "2560" in str(e) and input_dim == 1152:
                # Cria uma matriz de identidade de projeção temporária: [2560, 1152]
                projection_weight = torch.eye(2560, 1152, device=emb.device, dtype=emb.dtype)
                # Projeta linearmente o vetor de 1152 para 2560 dimensões de forma limpa
                emb_projected = F.linear(emb.to(dtype=projection_weight.dtype), projection_weight)
                
                # Executa o adaptador local com o tensor perfeitamente harmonizado
                v_seg = self.adapter_local(emb_projected)
            else:
                # Se for qualquer outro erro, propaga a exceção
                raise e

        # 2. Registra o segmento na memória cíclica do OODA Loop
        self.memory.append(v_seg.detach().cpu())
        
        v_global_in = closed_context_vector(list(self.memory), method=self.closed_agg_method)
        if v_global_in is None:
            v_global_in = v_seg.detach().cpu()
            
        v_global_in = v_global_in.to(self.device)
        
        # 3. Processa o contexto unificado através do adaptador macro
        try:
            v_global = self.adapter_macro(v_global_in)
        except RuntimeError as e:
            # Aplica a mesma blindagem de projeção para o adaptador macro se necessário
            if "2560" in str(e) and v_global_in.shape[-1] == 1152:
                projection_weight = torch.eye(2560, 1152, device=v_global_in.device, dtype=v_global_in.dtype)
                v_global_in_projected = F.linear(v_global_in.to(dtype=projection_weight.dtype), projection_weight)
                v_global = self.adapter_macro(v_global_in_projected)
            else:
                raise e
                
        return v_global

    def process_stream(self, texts: List[str]):
        outs = []
        for t in texts:
            outs.append(self.process_event(t))
        return outs

    def close_context(self):
        if not self.memory:
            return None
        v_global_in = closed_context_vector(list(self.memory), method=self.closed_agg_method)
        if v_global_in is None:
            return None
        return self.adapter_macro(v_global_in.to(self.device))

    def respond_schema(self, schema: dict):
        """Produce adapter-friendly embeddings from a schema without executing tools."""
        if isinstance(schema, dict) and ("parameters" in schema or "properties" in schema or "name" in schema):
            text = rest_translator.tool_schema_to_text(rest_translator.ensure_schema_compat(schema))
            emb = self.adapter_local(self.vectorizer.embed(text).to(self.device))
            return {"tool_text": emb}

        schema = rest_translator.ensure_schema_compat(schema)
        anchor, positive, negative = rest_translator.schema_to_triplet(schema)
        out = {}
        if anchor:
            out["anchor"] = self.adapter_local(self.vectorizer.embed(anchor).to(self.device))
        if positive:
            out["positive"] = self.adapter_local(self.vectorizer.embed(positive).to(self.device))
        if negative:
            out["negative"] = self.adapter_local(self.vectorizer.embed(negative).to(self.device))
        return out

    def process_request(self, req: dict):
        """Process a generic request dict (supports `schema`, `tools`, `texts`)."""
        if "schema" in req and isinstance(req["schema"], dict):
            resp = self.respond_schema(req["schema"])
            out = {}
            for k, v in resp.items():
                if hasattr(v, "detach"):
                    out[k] = v.detach().cpu().tolist()
                else:
                    out[k] = v
            return out

        if "texts" in req and isinstance(req["texts"], (list, tuple)) and req["texts"]:
            outs = self.process_stream(req["texts"])
            result = []
            for v in outs:
                result.append(
                    {"shape": list(v.shape), "norm": float(v.norm().item()), "embedding": v.detach().cpu().tolist()}
                )
            return {"results": result}

        flattened = rest_translator.normalize_and_flatten_request(req)
        if flattened:
            v = self.process_event(flattened)
            return {
                "text": flattened,
                "shape": list(v.shape),
                "norm": float(v.norm().item()),
                "embedding": v.detach().cpu().tolist(),
            }

        return {"error": "empty request"}

    @staticmethod
    def _user_text(payload: Dict[str, Any]) -> str:
        p = (payload.get("prompt") or payload.get("message") or "").strip()
        texts = payload.get("texts")
        if isinstance(texts, (list, tuple)) and texts:
            block = "\n".join(str(t) for t in texts if str(t).strip())
            if block.strip():
                p = f"{p}\n\n[CONTEXT_TEXTS]\n{block}".strip() if p else block.strip()
        return p

    @staticmethod
    def _tool_flatten_subset(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: payload[k]
            for k in ("schema", "tools", "context", "instruction")
            if k in payload and payload[k] not in (None, "", [], {})
        }

    def build_fused_swa_text(
        self,
        user_text: str,
        payload: Dict[str, Any],
        *,
        llama_bruto_preface: Optional[str] = None,
    ) -> str:
        """Texto único para embedding SWA: opc. Llama bruto + schema/tools + utilizador."""
        sub = self._tool_flatten_subset(payload)
        blob = rest_translator.normalize_and_flatten_request(sub) if sub else ""
        parts = []
        if llama_bruto_preface and str(llama_bruto_preface).strip():
            parts.append(f"[LLAMA_BRUTO]\n{str(llama_bruto_preface).strip()}")
        if blob.strip():
            parts.append(f"[SCHEMA_TOOLS]\n{blob.strip()}")
        if user_text.strip():
            parts.append(f"[USER]\n{user_text.strip()}")
        fused = "\n\n".join(parts)
        if not fused.strip():
            raise ValueError("empty_fused_input: falta prompt/message, texts, schema ou tools")
        return fused

    def infer_tool_name(self, user_text: str, payload: Dict[str, Any]) -> Optional[str]:
        ex = payload.get("tool")
        if isinstance(ex, str) and ex.strip() and ex.strip() in KNOWN_TOOL_NAMES:
            return ex.strip()
        for name in KNOWN_TOOL_NAMES:
            if re.search(rf"\b{re.escape(name)}\b", user_text, re.IGNORECASE):
                return name
        tools = payload.get("tools")
        if isinstance(tools, list):
            for item in tools:
                if isinstance(item, dict):
                    n = item.get("name")
                    if isinstance(n, str) and n in KNOWN_TOOL_NAMES:
                        return n
        schema = payload.get("schema")
        if isinstance(schema, dict):
            n = schema.get("name")
            if isinstance(n, str) and n in KNOWN_TOOL_NAMES:
                return n
        return None

    @staticmethod
    def generation_prompt(user_text: str, tool_name: Optional[str]) -> str:
        if not tool_name:
            return user_text if user_text.strip() else "Responde de forma clara e concisa."
        prefix = f'{{"tool": "{tool_name}", "parameters": {{}}}} '
        if user_text.strip():
            return prefix + user_text.strip()
        return prefix + "Completa o JSON dos parâmetros conforme o pedido e o schema."

    def should_use_llama_preface(self, payload: Dict[str, Any]) -> bool:
        v = payload.get("use_llama_context")
        if v is not None:
            return bool(v)
        if os.environ.get("USE_DUAL_MODEL", "false").lower() != "true":
            return False
        return os.environ.get("SWA_AUTO_LLAMA_PREFACE", "true").lower() != "false"

    def run_optimized_stream(
        self,
        payload: Union[str, Dict[str, Any]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Um único turno: funde contexto/schema/tools, SWA, deteção de tool e geração Gemma.

        Aceita string (prompt) ou dict com prompt|message, schema, tools, context, instruction, texts.
        """
        if isinstance(payload, str):
            payload = {"prompt": payload}
        else:
            payload = dict(payload)

        if payload.get("reset_context"):
            self.memory.clear()

        user_text = self._user_text(payload)
        llama_pref: Optional[str] = None
        if self.should_use_llama_preface(payload) and self.engines:
            lb = self.engines.get("llama_bruto")
            if lb is not None and user_text.strip():
                try:
                    llama_pref = lb.process_embedding_context(user_text)
                except Exception:
                    llama_pref = None

        fused = self.build_fused_swa_text(user_text, payload, llama_bruto_preface=llama_pref)
        self.process_event(fused)
        v_global = self.close_context()
        if v_global is None:
            raise ValueError("No context after process_event; memory may be empty.")

        tool_name = self.infer_tool_name(user_text, payload)
        gen_prompt = self.generation_prompt(user_text, tool_name)
        yield from self.generate_guided_stream(
            gen_prompt, v_global, max_new_tokens=max_new_tokens, temperature=temperature
        )
