"""
Embeddings reais da realidade — Gemma-3-270M (640D) projetado para pgvector (2560D).

Substitui vetores hash-seeded quando SWA_REAL_EMBEDDINGS=1 (padrão).
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

import torch
import torch.nn.functional as F
from swa.dimensions import MEMORY_DIM, NATIVE_DIM, project_vector, projection_matrix
from swa.paths import PROJECT_ROOT

GOLD_DIM = MEMORY_DIM
_PROJECTOR: Optional[torch.Tensor] = None
_PROJECTOR_LOCK = threading.Lock()


def _project_root() -> str:
    return str(PROJECT_ROOT)


def _real_embeddings_enabled() -> bool:
    return os.environ.get("SWA_REAL_EMBEDDINGS", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def get_projection_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Matriz fixa 2560×640 do contrato dimensional compartilhado."""
    global _PROJECTOR
    with _PROJECTOR_LOCK:
        if _PROJECTOR is None:
            _PROJECTOR = projection_matrix(NATIVE_DIM, GOLD_DIM)
        return _PROJECTOR.to(device=device, dtype=dtype)


def native_to_gold(v_native: torch.Tensor) -> torch.Tensor:
    v = v_native.detach().float().view(-1)
    if v.numel() > NATIVE_DIM:
        v = v[:NATIVE_DIM]
    elif v.numel() < NATIVE_DIM:
        v = F.pad(v, (0, NATIVE_DIM - v.numel()))
    p = get_projection_matrix(v.device, v.dtype)
    out = p @ v
    n = out.norm()
    if n > 0:
        out = out / n
    return out


def _hash_fallback(text: str) -> List[float]:
    from swa.file_crawler import BlakeNativeCrawler
    return BlakeNativeCrawler().generate_native_2560_vector(text)


class RealityEmbedder:
    """Singleton — carrega o 270M uma vez por processo."""

    _instance: Optional["RealityEmbedder"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "RealityEmbedder":
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._vectorizer = None
                inst._ready = False
                inst._use_hash = not _real_embeddings_enabled()
                inst._native_dim = NATIVE_DIM
                cls._instance = inst
            return cls._instance

    def attach_shared_model(self, model, tokenizer, device):
        """Compartilha a referência em memória do Gemma 270M do AeonCore (Zero RAM duplicada)."""
        if model is not None:
            class SharedVectorizerProxy:
                def __init__(self, m, tok, dev):
                    self.model = m
                    self.tokenizer = tok
                    self.device = dev
                def embed(self, text: str):
                    import torch
                    ids = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(self.device)
                    with torch.no_grad():
                        out = self.model(ids, output_hidden_states=True)
                        h = out.hidden_states[-1] if hasattr(out, "hidden_states") and out.hidden_states is not None else getattr(out, "last_hidden_state", None)
                        if h is None:
                            h = out[0]
                        return h.mean(dim=1).to(torch.float32)
            self._vectorizer = SharedVectorizerProxy(model, tokenizer, device)
            self._native_dim = getattr(model.config, "hidden_size", 640)
            self._ready = True
            print("⚡ [REALITY EMBEDDER] Gemma 270M compartilhado com o AeonCore (Zero cópias adicionais em RAM)!")

    def ensure_loaded(self) -> None:
        if self._ready and self._vectorizer is not None:
            return
        if self._use_hash:
            self._ready = True
            print("⚠️ [REALITY EMBEDDER] Modo hash (SWA_REAL_EMBEDDINGS=0).")
            return

        model_rel = os.environ.get("VECTORIZER_MODEL_DIR", "models/gemma-3-270m")
        model_dir = model_rel if os.path.isabs(model_rel) else os.path.join(_project_root(), model_rel)
        device = os.environ.get("GEMMA_DEVICE", "cpu")

        try:
            from swa.vectorizer import Vectorizer
            print(f"🧬 [REALITY EMBEDDER] Carregando Gemma 270M de {model_dir} ({device})…")
            self._vectorizer = Vectorizer(model_dir, device=device)
            d = int(self._vectorizer.model.config.hidden_size)
            self._native_dim = d
            self._ready = True
            print(f"✅ [REALITY EMBEDDER] Ativo — {d}D nativo → {GOLD_DIM}D pgvector.")
        except Exception as e:
            print(f"⚠️ [REALITY EMBEDDER] Falha ao carregar 270M ({e}). Fallback hash.")
            self._use_hash = True
            self._ready = True

    def embed_text_2560(self, text: str, max_chars: int = 8000) -> List[float]:
        self.ensure_loaded()
        chunk = (text or " ").strip()[:max_chars]
        if not chunk or self._use_hash:
            return _hash_fallback(chunk or " ")

        try:
            emb = self._vectorizer.embed(chunk)
            v2560 = native_to_gold(emb.squeeze(0))
            return v2560.cpu().tolist()
        except Exception as e:
            print(f"⚠️ [REALITY EMBEDDER] Inferência falhou: {e}")
            return _hash_fallback(chunk)

    def embed_text_640(self, text: str, max_chars: int = 8000) -> List[float]:
        """Return the canonical native subcortex representation."""
        self.ensure_loaded()
        chunk = (text or " ").strip()[:max_chars]
        if not chunk:
            return [0.0] * NATIVE_DIM
        if self._use_hash:
            return project_vector(
                torch.as_tensor(_hash_fallback(chunk)), MEMORY_DIM, NATIVE_DIM
            ).cpu().tolist()
        try:
            return F.normalize(
                torch.as_tensor(self._vectorizer.embed(chunk)).reshape(-1)[:NATIVE_DIM],
                p=2, dim=0,
            ).cpu().tolist()
        except Exception:
            return self.embed_text_640(chunk, max_chars=max_chars) if self._use_hash else (
                project_vector(torch.as_tensor(_hash_fallback(chunk)), MEMORY_DIM, NATIVE_DIM).cpu().tolist()
            )

    def embed_file_2560(self, filepath: str, max_chars: int = 8000) -> List[float]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read(max_chars)
        except OSError:
            return _hash_fallback(filepath)
        return self.embed_text_2560(text, max_chars=max_chars)

    def embedding_source_label(self) -> str:
        self.ensure_loaded()
        if self._use_hash:
            return "hash_fallback"
        return "gemma-3-270m_proj2560"


def get_reality_embedder() -> RealityEmbedder:
    return RealityEmbedder()
