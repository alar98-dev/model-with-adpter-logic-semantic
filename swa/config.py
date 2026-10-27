from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from swa.paths import PROJECT_ROOT


class SWAConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── HuggingFace ──────────────────────────────────────────────────────
    hf_token: str = Field(default="", description="HF hub token for gated models")

    # ── Model paths ──────────────────────────────────────────────────────
    model_dir: Path = Field(
        default=Path("models/gemma-3-4-b"),
        description="Path to the main generative model (Gemma 3 1B/4B)",
    )
    vectorizer_model_dir: Path = Field(
        default=Path("models/gemma-3-270m"),
        description="Path to the embedding/vectorizer model (Gemma 3 270M)",
    )
    llama_model_dir: Optional[Path] = Field(
        default=None, description="Optional Llama model path for dual-model mode"
    )

    # ── Device & dtype ───────────────────────────────────────────────────
    gemma_device: Literal["cpu", "cuda", "xpu"] = Field(
        default="cpu", description="Torch device for the generative model"
    )
    llama_device: Literal["cpu", "cuda", "xpu"] = Field(
        default="cuda", description="Torch device for the Llama model"
    )
    gemma_torch_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = Field(
        default=None, description="Force a specific torch dtype"
    )

    # ── CPU topology ─────────────────────────────────────────────────────
    swa_cpu_topology: Literal["single", "dual_xeon"] = Field(
        default="single", description="CPU socket layout for NUMA affinity"
    )
    swa_intraop_threads: Optional[int] = Field(
        default=None,
        description="Max intra-op threads for OpenMP / MKL / torch",
        ge=1,
        le=128,
    )

    # ── Dual-model & steering ────────────────────────────────────────────
    use_dual_model: bool = Field(default=False)
    swa_auto_llama_preface: bool = Field(default=True)
    swa_expose_debug_routes: bool = Field(default=False)
    use_fake: bool = Field(default=False)

    # ── KV-cache / decode ────────────────────────────────────────────────
    swa_kv_use_cache: bool = Field(default=True)
    swa_max_seq_len: int = Field(default=0, ge=0)
    swa_attn_implementation: Optional[Literal["sdpa", "flash_attention_2"]] = Field(
        default=None
    )

    # ── Real embeddings / vector manifold ────────────────────────────────
    swa_real_embeddings: bool = Field(default=True)
    swa_rebuild_manifold_on_boot: bool = Field(default=True)
    swa_runtime_real_embeddings: bool = Field(default=False)

    # ── Hardware state / NVMe ────────────────────────────────────────────
    hardware_state_device: str = Field(
        default="/dev/sde",
        description="Block device path for hardware-backed state storage",
    )
    hardware_state_size_gb: int = Field(default=24, ge=1, le=65536)
    hardware_state_fallback: Literal["ram", "file", "error"] = Field(
        default="ram",
        description="Fallback strategy when the block device is unavailable",
    )
    hardware_state_fallback_path: Optional[Path] = Field(
        default=None,
        description="File path used when fallback is 'file'",
    )

    # ── PostgreSQL / pgvector ────────────────────────────────────────────
    pg_dsn: str = Field(default="dbname=swa_matrix user=alifer")
    pg_pool_min: int = Field(default=2, ge=1)
    pg_pool_max: int = Field(default=10, ge=1, le=100)

    # ── OODA orchestrator ────────────────────────────────────────────────
    ooda_max_iterations: int = Field(default=100, ge=1, le=1_000_000)
    ooda_token_budget_min: int = Field(default=64, ge=1)
    ooda_token_budget_max: int = Field(default=2048, ge=1)
    ooda_temperature_min: float = Field(default=0.3, ge=0.0, le=2.0)
    ooda_temperature_max: float = Field(default=1.5, ge=0.0, le=2.0)

    # ── REST server ──────────────────────────────────────────────────────
    rest_host: str = Field(default="127.0.0.1")
    rest_port: int = Field(default=20000, ge=1024, le=65535)

    # ── Path security (blake_tools whitelist) ────────────────────────────
    allowed_write_prefixes: tuple[Path, ...] = Field(
        default=(Path("/tmp"), Path.home() / ".swa"),
        description="Directories where blake_tools is allowed to write files",
    )
    allowed_read_prefixes: tuple[Path, ...] = Field(
        default=(Path("/tmp"), Path.home(), Path("/services")),
        description="Directories where blake_tools is allowed to read files",
    )
    allowed_commands: tuple[str, ...] = Field(
        default=("ls", "cat", "pwd", "df", "du", "uptime", "whoami", "echo"),
        description="Shell commands whitelisted for execute_command",
    )

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def workspace_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def resolved_model_dir(self) -> Path:
        return self._resolve(self.model_dir)

    @property
    def resolved_vectorizer_model_dir(self) -> Path:
        return self._resolve(self.vectorizer_model_dir)

    def _resolve(self, p: Path) -> Path:
        if not p.is_absolute():
            return self.workspace_root / p
        return p

    @field_validator("hf_token", mode="before")
    @classmethod
    def _coerce_empty_token(cls, v: object) -> str:
        if v is None:
            return ""
        return str(v).strip()

    @model_validator(mode="after")
    def _validate_hardware_fallback_path(self) -> SWAConfig:
        if self.hardware_state_fallback == "file" and self.hardware_state_fallback_path is None:
            self.hardware_state_fallback_path = Path("/tmp/swa_hardware_fallback.bin")
        return self

    @model_validator(mode="after")
    def _warn_on_default_token(self) -> SWAConfig:
        token = self.hf_token
        if token and ("your_token" in token or "hf_" not in token):
            import warnings

            warnings.warn("HF_TOKEN appears to be a placeholder — model downloads may fail.")
        return self


_default_config: SWAConfig | None = None


def get_config() -> SWAConfig:
    global _default_config
    if _default_config is None:
        _default_config = SWAConfig()
    return _default_config


def set_config(cfg: SWAConfig) -> None:
    global _default_config
    _default_config = cfg
