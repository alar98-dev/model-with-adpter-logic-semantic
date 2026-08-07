from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


class TokenizerCompatibilityError(Exception):
    """Tokenizer exists but is incompatible with the expected runtime/architecture."""


class TokenizerCorruptedError(Exception):
    """Tokenizer files exist but are unreadable/corrupted."""


class TokenizerMissingError(Exception):
    """Required tokenizer files are missing."""


class TokenizerVersionMismatchError(Exception):
    """Tokenizer versions (tokenizers/transformers) mismatch expectations."""


class SentencePieceMissingError(Exception):
    """SLOW tokenizer path needs sentencepiece backend but it is not installed."""



@dataclass
class TokenizerHealth:
    state: str = "loading"
    fast_tokenizer_available: bool = False
    slow_tokenizer_active: bool = False
    model_type: Optional[str] = None
    transformers_version: Optional[str] = None
    tokenizers_version: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def verify_tokenizer_dir(model_dir: str) -> Dict[str, Any]:
    """Requisito 9: valida existência/integridade/tamanho mínimo/parse JSON/compatibilidade declarada.

    Não faz load do tokenizer.
    """
    path = os.path.abspath(model_dir)

    out: Dict[str, Any] = {
        "path": path,
        "exists": os.path.isdir(path),
        "issues": [],
        "files": {},
        "json": {},
        "declared": {},
        "min_size_bytes_ok": True,
    }

    if not out["exists"]:
        out["issues"].append("model_dir_missing")
        return out

    required = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ]

    min_bytes = 64
    for fname in required:
        fpath = os.path.join(path, fname)
        exists = os.path.isfile(fpath)
        size = os.path.getsize(fpath) if exists else None
        out["files"][fname] = {"exists": exists, "size": size}
        if not exists:
            out["issues"].append(f"missing_{fname}")
            continue
        if size is not None and size < min_bytes:
            out["issues"].append(f"too_small_{fname}")
            out["min_size_bytes_ok"] = False

    # Parse JSONs se existirem
    def _try_load_json(fp: str) -> Tuple[bool, Any, Optional[str]]:
        try:
            with open(fp, encoding="utf-8") as f:
                return True, json.load(f), None
        except json.JSONDecodeError as e:
            return False, None, f"json_decode_error:{e}"
        except OSError as e:
            return False, None, f"os_error:{e}"

    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "config.json",
    ]:
        info = out["files"].get(fname) or {}
        if not info.get("exists"):
            continue
        ok, data, err = _try_load_json(os.path.join(path, fname))
        if not ok:
            out["issues"].append(f"corrupt_{fname}:{err}")
        else:
            out["json"][fname] = data

    # Declared fields
    cfg = out["json"].get("config.json") or {}
    out["declared"]["model_type"] = cfg.get("model_type")
    out["declared"]["architecture"] = cfg.get("architectures", None)

    tok_cfg = out["json"].get("tokenizer_config.json") or {}
    out["declared"]["tokenizer_class"] = tok_cfg.get("tokenizer_class")

    return out


class TokenizerManager:
    """Tokenizer Compatibility Layer.

    Requisitos atendidos:
    - validação isolada antes de load
    - fallback fast->slow
    - estado explícito
    - falhas tipadas
    - NÃO derruba API (propaga apenas se o chamador decidir; aqui mantemos encapsulado em health())

    Hardenings (fase Runtime Compatibility Hardening):
    - diagnóstico de `tokenizer.json` (schema/graph presence)
    - classificação heurística de falhas FAST (Rust ModelWrapper / enum variant mismatch)
    """


    def __init__(self) -> None:
        self._health = TokenizerHealth(state="loading")
        self._tokenizer: Any = None
        self._model_dir: Optional[str] = None
        self._validated: Optional[Dict[str, Any]] = None

    def validate(self, model_dir: str) -> Dict[str, Any]:
        self._model_dir = model_dir
        report = verify_tokenizer_dir(model_dir)
        self._validated = report

        # Extract declared versions
        try:
            import transformers  # noqa

            self._health.transformers_version = getattr(transformers, "__version__", None)
        except Exception:
            self._health.transformers_version = None

        try:
            import tokenizers  # noqa

            self._health.tokenizers_version = getattr(tokenizers, "__version__", None)
        except Exception:
            self._health.tokenizers_version = None

        self._health.model_type = report.get("declared", {}).get("model_type")

        # Decide issues
        issues = report.get("issues") or []
        if issues:
            self._health.state = "degraded"
            self._health.error = "; ".join(issues)
            self._health.error_type = "validation_issues"
        else:
            self._health.state = "ready"

        return report

    def _set_failed(
        self,
        state: str,
        exc: Exception,
        *,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._health.state = state
        self._health.error_type = exc.__class__.__name__
        self._health.error = str(exc)
        self._health.details = details

    def _diagnose_tokenizer_json(self, model_dir: str) -> Dict[str, Any]:
        """Diagnóstico leve (sem instanciar tokenizer) para inspecionar schema/graph."""
        fp = os.path.join(os.path.abspath(model_dir), "tokenizer.json")
        out: Dict[str, Any] = {
            "tokenizer_json_exists": os.path.isfile(fp),
            "tokenizer_json_size_bytes": os.path.getsize(fp) if os.path.isfile(fp) else None,
            "key_presence": {},
            "declared_model_type": None,
            "parse_error": None,
        }

        if not os.path.isfile(fp):
            return out

        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
            # presence of major graphs
            for k in (
                "model",
                "normalizer",
                "pre_tokenizer",
                "post_processor",
                "decoder",
            ):
                out["key_presence"][k] = k in data

            model = data.get("model") or {}
            if isinstance(model, dict):
                out["declared_model_type"] = model.get("type")
        except Exception as e:
            out["parse_error"] = str(e)

        return out

    def _classify_fast_exception(self, exc: Exception, model_dir: str) -> Dict[str, Any]:
        msg = str(exc) or ""
        lower = msg.lower()

        # Heurísticas para incompatibilidade Rust tokenizer schema.
        is_rust_variant_mismatch = (
            "data did not match any variant of untagged enum" in msg
            or "modelwrapper" in msg
            or "untagged enum" in msg
        )

        if is_rust_variant_mismatch:
            return {
                "error_type": "fast_tokenizer_rust_schema_incompatible",
                "fast_error": msg,
                "details": {
                    "diagnostics": self._diagnose_tokenizer_json(model_dir),
                },
            }

        # Outros agrupamentos úteis.
        if "tokenizer.json" in msg and ("not found" in lower or "missing" in lower):
            return {
                "error_type": "fast_tokenizer_missing_tokenizer_json",
                "fast_error": msg,
                "details": {"diagnostics": self._diagnose_tokenizer_json(model_dir)},
            }

        return {
            "error_type": "fast_tokenizer_unknown_error",
            "fast_error": msg,
            "details": {"diagnostics": self._diagnose_tokenizer_json(model_dir)},
        }


    def load(self, model_dir: Optional[str] = None) -> Any:
        """Load com fallback fast->slow.

        Não lança exceção genérica; se falhar, retorna None e marca health.
        """
        if model_dir is not None:
            self._model_dir = model_dir
        if not self._model_dir:
            raise ValueError("model_dir is required")

        # Ensure validated (best-effort)
        if self._validated is None:
            self.validate(self._model_dir)

        # If validation already found fatal missing files, we still attempt load (HF may still work)
        try:
            from transformers import AutoTokenizer
        except Exception as e:
            self._set_failed("failed", TokenizerCompatibilityError(f"transformers import failed: {e}"))
            self._tokenizer = None
            return None

        # FAST attempt
        self._health.state = "loading"
        fast_exc: Optional[Exception] = None
        fast_ok = False

        try:
            self._health.fast_tokenizer_available = True
            tok = AutoTokenizer.from_pretrained(self._model_dir, use_fast=True)
            # Validate minimal behavior
            _ = getattr(tok, "tokenize", None)
            # SWA extensions: add special tokens for control, set padding left
            tok.add_special_tokens({'additional_special_tokens': ['<SWA_BLOCK>']})
            tok.padding_side = 'left'
            fast_ok = True
            self._health.slow_tokenizer_active = False
            self._health.state = "ready"
            self._tokenizer = tok
            self._health.error = None
            self._health.error_type = None
            self._health.details = None
            return tok
        except Exception as e:
            fast_exc = e
            # Structured incompatibility
            self._health.fast_tokenizer_available = False
            self._health.slow_tokenizer_active = False


        if not fast_ok:
            msg = f"FAST tokenizer failed → slow tokenizer fallback activated | fast_error={fast_exc}"
            # Structured log
            print(msg)

            # Fallback slow
            try:
                # Dependency diagnostics before attempting the SLOW path.
                # If slow tokenizer needs SentencePiece and it is missing, transformers
                # will raise an ImportError.
                try:
                    import sentencepiece  # type: ignore  # noqa: F401
                except ImportError as sp_e:
                    self._set_failed(
                        "missing_dependency",
                        SentencePieceMissingError("SentencePiece library not found"),
                        details={
                            "missing_dependency": "sentencepiece",
                            "install_hint": "pip install sentencepiece",
                            "fast_error": str(fast_exc) if fast_exc else None,
                            "fast_error_type": fast_exc.__class__.__name__ if fast_exc else None,
                            "slow_import_error": str(sp_e),
                            "fallback": {"use_fast": False, "legacy": True},
                        },
                    )
                    self._tokenizer = None
                    return None

                from transformers import AutoTokenizer

                # Mandatory fallback strategy:
                # 1) use_fast=False
                # 2) legacy=True (where supported)
                tok = AutoTokenizer.from_pretrained(
                    self._model_dir,
                    use_fast=False,
                    legacy=True,
                )

                # SWA extensions: add special tokens for control, set padding left
                tok.add_special_tokens({'additional_special_tokens': ['<SWA_BLOCK>']})
                tok.padding_side = 'left'

                self._tokenizer = tok
                self._health.fast_tokenizer_available = False
                self._health.slow_tokenizer_active = True
                self._health.state = "fallback_slow"
                self._health.error = "FAST tokenizer incompatible"

                classified = (
                    self._classify_fast_exception(fast_exc, self._model_dir)
                    if fast_exc is not None
                    else {"error_type": "fast_tokenizer_unknown_error"}
                )

                self._health.error_type = classified.get("error_type") or "TokenizerCompatibilityError"
                self._health.details = {
                    "fast_error": str(fast_exc) if fast_exc else None,
                    "fast_error_type": fast_exc.__class__.__name__ if fast_exc else None,
                    "classified": classified,
                    "fallback": {"use_fast": False, "legacy": True},
                }

                return tok
            except Exception as e:
                # Final failure: keep degraded state but do not crash.
                # We mark failed and leave tokenizer None.
                err_str = str(e)
                missing_sp = "SentencePiece" in err_str or "sentencepiece" in err_str

                if missing_sp:
                    self._set_failed(
                        "missing_dependency",
                        SentencePieceMissingError("SentencePiece library not found"),
                        details={
                            "missing_dependency": "sentencepiece",
                            "install_hint": "pip install sentencepiece",
                            "fast_error": str(fast_exc) if fast_exc else None,
                            "fast_error_type": fast_exc.__class__.__name__ if fast_exc else None,
                            "slow_error": err_str,
                            "fallback": {"use_fast": False, "legacy": True},
                        },
                    )
                else:
                    self._set_failed(
                        "failed",
                        TokenizerCompatibilityError(
                            f"tokenizer load failed (fast and slow): {e}"
                        ),
                        details={
                            "fast_error": str(fast_exc) if fast_exc else None,
                            "fast_error_type": fast_exc.__class__.__name__ if fast_exc else None,
                        },
                    )

                self._tokenizer = None
                return None



    def warmup(self) -> None:
        """Warmup leve para detectar tokenizer runtime issues (não derruba API)."""
        if self._tokenizer is None:
            return
        try:
            # Tiny encode/decode roundtrip
            if hasattr(self._tokenizer, "encode"):
                ids = self._tokenizer.encode("hello", add_special_tokens=False)
            elif hasattr(self._tokenizer, "__call__"):
                out = self._tokenizer("hello")
                ids = out.get("input_ids") if isinstance(out, dict) else None
            else:
                ids = None
            if not ids:
                self._health.state = "degraded"
                self._health.error = "tokenizer_warmup_failed"
                self._health.error_type = "TokenizerCorruptedError"
        except Exception as e:
            self._health.state = "degraded"
            self._health.error = str(e)
            self._health.error_type = "TokenizerCorruptedError"

    def metadata(self) -> Dict[str, Any]:
        tok = self._tokenizer
        if tok is None:
            return {"available": False}
        meta: Dict[str, Any] = {
            "available": True,
            "backend": tok.__class__.__name__,
        }
        for attr in ("model_max_length", "init_kwargs"):
            if hasattr(tok, attr):
                meta[attr] = getattr(tok, attr)
        return meta

    def health(self) -> Dict[str, Any]:
        # Keep health as plain dict
        d = asdict(self._health)
        # Normalize state mapping for required schema
        state = d.get("state")
        if state == "ready" and d.get("slow_tokenizer_active"):
            d["state"] = "fallback_slow"
        return d

    def get_tokenizer(self) -> Any:
        """Backward-compatible runtime API.

        Some runtime code expects TokenizerManager.get_tokenizer().
        """
        return self._tokenizer

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer


