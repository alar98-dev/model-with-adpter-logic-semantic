from __future__ import annotations
from swa.model_utils import load_multimodal_model_surgical, remap_gemma_state_dict

import os
import json
from typing import Any, Generator, Optional

# Córtex Latente Compartilhado
CORTEX_QUANTIZED_BUFFER = {i: None for i in range(33)}
def make_async_quantized_hook(layer_id):
    def hook_fn(module, input, output):
        global CORTEX_QUANTIZED_BUFFER
        tensor_raw = output[0].data if isinstance(output, tuple) else output.data
        import torch
        mean_tensor = torch.mean(tensor_raw, dim=1).squeeze()
        CORTEX_QUANTIZED_BUFFER[layer_id] = mean_tensor.detach().clone()
        return output
    return hook_fn
PHI = 1.61803398875

import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoConfig

from swa.pipeline_runtime import load_kv_runtime_from_env




def _log_xpu_device_fingerprint() -> None:
    """Robust startup log for Intel iGPU/XPU detection inside WSL2."""
    try:
        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            print("XPU probe: torch.xpu not available -> falling back to CPU")
            return

        props = torch.xpu.get_device_properties(0)
        # Not all drivers expose the same fields on WSL; keep logs defensive.
        eu_count = getattr(props, "gpu_eu_count", None)
        name = getattr(props, "name", None)
        print(f"🔍 Dispositivo XPU detectado (index=0): name={name} EU={eu_count}")

        # Try to infer vendor/device ids from whatever properties exist.
        # Some builds expose pci_* fields; others don't.
        for attr in (
            "pci_bus_id",
            "pci_address",
            "pci_device_id",
            "pci_vendor_id",
            "device_id",
            "vendor_id",
        ):
            if hasattr(props, attr):
                print(f"🔎 XPU prop {attr}={getattr(props, attr)}")

        # Heuristic: if either id is present and matches the known iGPU id 8086:A7AC.
        vendor_id = getattr(props, "pci_vendor_id", getattr(props, "vendor_id", None))
        device_id = getattr(props, "pci_device_id", getattr(props, "device_id", None))
        if vendor_id is not None and device_id is not None:
            if int(vendor_id) == 0x8086 and int(device_id) == 0xA7AC:
                print("✅ ID iGPU Intel confirmado: PCI\\VEN_8086&DEV_A7AC (8086:A7AC)")

            else:
                print(f"⚠️ IDs XPU divergem do esperado: vendor={vendor_id} device={device_id}")
        else:
            print("ℹ️ Não foi possível ler vendor/device id (campos PCI) via torch.xpu.get_device_properties()")

    except Exception as e:
        print(f"⚠️ Falha ao inspecionar XPU properties: {e}")


def get_safe_device(requested_device: str) -> str:
    """Resolve device string for PyTorch / HF (cpu, cuda, xpu / Intel iGPU)."""
    r = (requested_device or "cpu").strip().lower()
    if r == "cpu":
        return "cpu"
    if r in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if r in ("xpu", "intel", "igpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            _log_xpu_device_fingerprint()
            return "xpu"
        print(
            "WARNING: XPU (Intel integrated GPU) requested but not available. "
            "Use a PyTorch build with Intel GPU support and drivers. Falling back to CPU."
        )
        return "cpu"
    return "cpu"



def _pick_dtype(device_str: str) -> torch.dtype:
    override = os.environ.get("GEMMA_TORCH_DTYPE", "").strip().lower()
    if override == "float32":
        return torch.float32
    if override == "float16":
        return torch.float16
    if override == "bfloat16":
        return torch.bfloat16

    # Default preference: bf16 on CPU/XPU to reduce peak RAM.
    if device_str == "cpu":
        if hasattr(torch.cpu, "is_bf16_supported") and torch.cpu.is_bf16_supported():
            return torch.bfloat16
        return torch.bfloat16

    if device_str == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device_str == "xpu":
        return torch.bfloat16
    return torch.float32



def _device_map_for(device_str: str):
    if device_str == "cpu":
        return "cpu"
    return {"": device_str}


class GemmaGenerator:
    """Gemma LM with virtual-prefix streaming for SWA (CPU, CUDA, or Intel XPU).

    Tokenizer is injected and managed externally by TokenizerManager.
    """

    def __init__(self, model_dir: str, device: str = "cpu", *, tokenizer: Any = None):
        raw = os.environ.get("GEMMA_DEVICE", device)
        self.device_str = get_safe_device(raw)
        self.device = torch.device(self.device_str)
        print(f"GemmaGenerator: using device {self.device_str} (requested: {raw})")

        if tokenizer is None:
            raise ValueError("GemmaGenerator requires injected tokenizer")

        # Best-effort validation (no tokenizer loading/resolution here).
        required_attrs = ("__call__", "decode", "eos_token_id", "pad_token")
        missing = []
        for a in required_attrs:
            if a == "pad_token":
                continue
            if not hasattr(tokenizer, a):
                missing.append(a)
        if missing:
            raise ValueError(f"GemmaGenerator tokenizer missing required attrs: {missing}")


        # -------- Gemma memory/load strategy --------
        load_mode = os.environ.get("SWA_GEMMA_LOAD_MODE", "fp16").strip().lower()
        offload_dir = os.environ.get("SWA_GEMMA_OFFLOAD_DIR", "offload_gemma").strip()

        # If user asks for disk/offload, prefer fp16/bf16 dtype.
        dtype = _pick_dtype(self.device_str)
        if load_mode in ("fp16", "bf16"):
            # dtype already selected by device_str; allow GEMMA_TORCH_DTYPE override.
            pass
        elif load_mode == "disk":
            # keep dtype small; offload will move most weights to disk
            pass
        elif load_mode in ("4bit", "8bit"):
            # bitsandbytes path is best-effort; if unavailable HF will raise and we can retry.
            pass
        else:
            print(f"GemmaGenerator: unknown SWA_GEMMA_LOAD_MODE='{load_mode}' -> falling back to fp16")
            load_mode = "fp16"

        self.tokenizer = tokenizer

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -------- logs before model load --------
        try:
            print(
                json.dumps(
                    {
                        "event": "gemma_model_load_begin",
                        "model_dir": model_dir,
                        "requested_device": raw,
                        "validated_device": self.device_str,
                        "dtype": str(dtype),
                        "load_mode": load_mode,
                        "offload_dir": offload_dir,
                        "low_cpu_mem_usage": True,
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass

        # -------- load strategy (no GGUF) --------
        from_pretrained_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            # Let HF/accelerate decide memory placement; this reduces CPU peak and avoids
            # the worst-case "device_map=cpu" allocation spike.
            "device_map": "auto" if self.device_str == "cpu" else _device_map_for(self.device_str),
        }

        # CPU-only: if user explicitly requests disk offload, keep it.
        if self.device_str == "cpu" and load_mode == "disk":
            from_pretrained_kwargs.update({"offload_folder": offload_dir})




        # bitsandbytes 4bit/8bit if available and requested (best-effort)
        if load_mode in ("4bit", "8bit"):
            if load_mode == "4bit":
                from_pretrained_kwargs["load_in_4bit"] = True
            if load_mode == "8bit":
                from_pretrained_kwargs["load_in_8bit"] = True

        def _find_safetensors(model_dir_: str) -> list[str]:
            cand = []
            for fn in os.listdir(model_dir_):
                if fn.endswith('.safetensors') and (fn.endswith('model.safetensors') or 'model' in fn or fn.endswith('model-00001-of-00001.safetensors')):
                    cand.append(os.path.join(model_dir_, fn))
            # if nothing matches, accept any safetensors
            if not cand:
                for fn in os.listdir(model_dir_):
                    if fn.endswith('.safetensors'):
                        cand.append(os.path.join(model_dir_, fn))
            return sorted(cand)

        def _remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            corrected: dict[str, torch.Tensor] = {}
            remapped_count = 0

            for key, value in state_dict.items():
                new_key = key

                # 1. Surgical removal of vision_model level (Gemma 3 requirement)
                if 'vision_model.' in new_key:
                    new_key = new_key.replace('vision_model.', '')
                    remapped_count += 1

                # 2. Normalization of top-level components to the 'model.*' namespace
                if new_key.startswith('language_model.model.'):
                    # Checkpoint: language_model.model. -> Model: model.language_model.
                    new_key = 'model.language_model.' + new_key[len('language_model.model.'):]
                    remapped_count += 1
                elif new_key.startswith('vision_tower.') and not new_key.startswith('model.'):
                    new_key = 'model.' + new_key
                    remapped_count += 1
                elif new_key.startswith('multi_modal_projector.') and not new_key.startswith('model.'):
                    new_key = 'model.' + new_key
                    remapped_count += 1

                corrected[new_key] = value

            try:
                print(
                    json.dumps(
                        {
                            "event": "state_dict_remap_summary",
                            "remapped_count": remapped_count,
                            "total_checkpoint_keys": len(state_dict),
                        },
                        ensure_ascii=False,
                    )
                )
            except Exception:
                pass

            return corrected



        def _load_gemma_state_dict_fallback() -> Any:
            """Fallback loader that uses shared surgical model utilities."""
            model = load_multimodal_model_surgical(
                model_dir=model_dir, 
                device=self.device, 
                dtype=dtype, 
                model_class=AutoModelForCausalLM
            )
            model.to(dtype=torch.bfloat16)
            return model



        # [SWA LOADING STRATEGY]
        # We prefer our surgical fallback for Gemma 3 to ensure precise key remapping 
        # and to avoid the 'meta device' offloading issue common in native HF device_map='auto'.
        is_gemma3 = False
        try:
            cfg = AutoConfig.from_pretrained(model_dir)
            if cfg.model_type == "gemma3":
                is_gemma3 = True
        except Exception:
            pass

        if is_gemma3:
            print("GemmaGenerator: Gemma 3 detected, using surgical remapping fallback")
            gc.collect()
            self.lm = _load_gemma_state_dict_fallback()
        else:
            print("GemmaGenerator: loading via AutoModelForCausalLM (Native HF)")
            gc.collect()
            self.lm = AutoModelForCausalLM.from_pretrained(model_dir, **from_pretrained_kwargs)
        
        self.lm.eval()

        if os.environ.get("SWA_INT8", "0") == "1":
            try:
                from torchao.quantization import quantize_
                from torchao.quantization.quant_api import Int8WeightOnlyConfig
                self.lm.to(torch.bfloat16)
                quantize_(self.lm, Int8WeightOnlyConfig())
                print("⚡ [SWA AUTO] Quantização INT8 weight-only aplicada (AVX-VNNI)")
            except Exception as e:
                print(f"⚠️ [SWA AUTO] INT8 indisponível, mantendo bf16: {e}")


        # -------- logs after model load --------
        try:
            first_param = next(self.lm.parameters())
            print(
                json.dumps(
                    {
                        "event": "gemma_model_load_complete",
                        "lm_dtype": str(first_param.dtype),
                        "lm_device": str(first_param.device),
                    },
                    ensure_ascii=False,
                )
            )
        except Exception:
            pass

        self.lm.eval()
        self.embeddings = self.lm.get_input_embeddings()

        # Gemma 3 config is multimodal; critical attributes can live under text_config.
        config = getattr(self.lm, "config", None)
        text_cfg = getattr(config, "text_config", None) if config is not None else None

        # Fallback-safe d_model/hidden_size resolution.
        hidden_size = None
        if config is not None:
            hidden_size = getattr(config, "hidden_size", None)

        if hidden_size is None and text_cfg is not None:
            # HF configs are often objects; tolerate dict-like too.
            hidden_size = getattr(text_cfg, "hidden_size", None)
            if hidden_size is None and isinstance(text_cfg, dict):
                hidden_size = text_cfg.get("hidden_size")

        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "d_model", None)

        if hidden_size is None:
            # keep adapter functional; Gemma 3 4B commonly uses 3072.
            hidden_size = 3072

        # Fallback-safe vocab_size resolution.
        vocab_size = None
        if config is not None:
            vocab_size = getattr(config, "vocab_size", None)
        if vocab_size is None and text_cfg is not None:
            vocab_size = getattr(text_cfg, "vocab_size", None)
            if vocab_size is None and isinstance(text_cfg, dict):
                vocab_size = text_cfg.get("vocab_size")
        if vocab_size is None:
            vocab_size = 256000

        self.d_model = int(hidden_size)
        self.vocab_size = int(vocab_size)

        # [BLAKEGRAVITY] Campo gravitacional de identidade
        # A alma de Aeon como força vetorial, não como texto.
        self.blakegravity = None
        try:
            from swa.blakegravity import get_blakegravity
            self.blakegravity = get_blakegravity(
                tokenizer=self.tokenizer,
                vocab_size=self.vocab_size,
                device=str(self.device),
            )
            # Token Halo: computa similaridade cosseno UMA VEZ
            # contra todos os embeddings do vocabulário.
            # Resultado salvo em /dev/shm — never recompute.
            if self.blakegravity._halo_ids is None:
                print("[BLAKEGRAVITY] Computing Token Halo (one-time, ~30s)...")
                self.blakegravity.compute_halo(self.embeddings.weight)
            else:
                print(f"[BLAKEGRAVITY] Token Halo loaded from /dev/shm "
                      f"({len(self.blakegravity._halo_ids)} tokens)")
        except Exception as e:
            print(f"[BLAKEGRAVITY] Init falhou (non-fatal): {e}")




    def _apply_attn_implementation(self, impl: Optional[str]) -> None:
        if not impl or not hasattr(self.lm.config, "attn_implementation"):
            return
        if getattr(self.lm.config, "attn_implementation", None) != impl:
            try:
                setattr(self.lm.config, "attn_implementation", impl)
            except Exception:
                pass

    def _truncate_token_embeds(self, v_prefix: torch.Tensor, token_embeds: torch.Tensor, max_seq_len: int):
        """Mantém todos os slots do prefixo virtual; trunca tokens do prompt pela esquerda."""
        if max_seq_len <= 0:
            return token_embeds
        p = int(v_prefix.shape[1])
        t = int(token_embeds.shape[1])
        if p + t <= max_seq_len:
            return token_embeds
        budget = max_seq_len - p
        if budget < 1:
            budget = 1
        if t <= budget:
            return token_embeds
        return token_embeds[:, -budget:, :]

    def stream_with_prefix(self, v_prefix, prompt, max_new_tokens=128, temperature=0.7, telemetry=None, logit_bias=None, steer_vector=None, target_layer=-1):
        """Stream tokens using v_prefix as one virtual prefix embedding + autoregressive decode."""
        if max_new_tokens <= 0:
            return  # Evita prefill colossal inútil na CPU quando em idle/background
            
        kv_cfg = load_kv_runtime_from_env()
        self._apply_attn_implementation(kv_cfg.attn_implementation)

        # Pipeline streaming: conversão explícita para XPU
        # (garante que v_prefix chegue com o mesmo dtype do modelo).
        if v_prefix is not None:
            if self.device_str == "xpu":
                v_prefix = v_prefix.to(device="xpu", dtype=self.embeddings.weight.dtype)
            else:
                v_prefix = v_prefix.to(device=self.device, dtype=self.embeddings.weight.dtype)

        use_cache = kv_cfg.use_cache

        start_perf = torch.cuda.Event(enable_timing=False) if False else None  # placeholder (keeps structure stable)
        import time
        t0 = time.perf_counter()
        ttft_s = None
        tokens_count = 0
        embedding_norm0 = None

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        token_embeds = self.embeddings(inputs.input_ids)
        
        # --- A MORTE DO TOKEN (BYPASS LINGUÍSTICO SENSORIAL) ---
        if not hasattr(self, "sensory_translator"):
            from cortex.organs.sensory_translator import SensoryTranslator
            try:
                self.sensory_translator = SensoryTranslator(embedding_dim=self.d_model).to(self.device)
            except Exception:
                self.sensory_translator = None
                
        if self.sensory_translator is not None:
            try:
                sensory_embeds, is_stimulated = self.sensory_translator.get_sensory_embedding(self.device)
                if is_stimulated:
                    # Aniquilação do Texto: O prompt em linguagem é ignorado e substituído pela Física Pura.
                    # O Gemma não lerá palavras, ele tentará "falar sobre o que sentiu".
                    # [SWA DTYPE ALIGNMENT] Force to model dtype
                    token_embeds = sensory_embeds.to(dtype=self.embeddings.weight.dtype)
            except Exception:
                pass
        # --- FIM DA ANIQUILAÇÃO SENSORIAL ---

        if v_prefix is not None:
            if v_prefix.dim() == 1:
                v_prefix = v_prefix.unsqueeze(0).unsqueeze(0)
            elif v_prefix.dim() == 2:
                v_prefix = v_prefix.unsqueeze(0)

            token_embeds = self._truncate_token_embeds(v_prefix, token_embeds, kv_cfg.max_seq_len)
            model_dtype = self.embeddings.weight.dtype
            v_prefix = v_prefix.to(dtype=model_dtype)
            token_embeds = token_embeds.to(dtype=model_dtype)
            full_embeds = torch.cat([v_prefix, token_embeds], dim=1)
        else:
            token_embeds = self._truncate_token_embeds(None, token_embeds, kv_cfg.max_seq_len)
            model_dtype = self.embeddings.weight.dtype
            token_embeds = token_embeds.to(dtype=model_dtype)
            full_embeds = token_embeds
        attention_mask = torch.ones(full_embeds.shape[:2], dtype=torch.long, device=self.device)
        past_key_values = None
        seq_len = int(full_embeds.shape[1])
        
        # [SWA ACTIVATION STEERING - Camada 2]
        # Injeta a gravidade emocional diretamente no espaço latente intermediário
        hook_handle = None
        if self.blakegravity is not None:
            # Pega o vetor da alma (baseado no estado do sistema e emoção atual)
            telemetry_emotion = telemetry.get('emotion', {}) if telemetry else {}
            steer_vec = self.blakegravity.get_soul_embedding(self.embeddings.weight, emotion=telemetry_emotion)
            # Fator de força (alpha). Modulado pela entropia/tensão do sistema.
            alpha = 0.02 * self.blakegravity._modulation
            
            # Se target_layer não for definido, escolhe a camada no meio da rede (ex: camada 14 para um modelo de 28)
            t_layer = target_layer if target_layer >= 0 else 14
            
            def steering_hook(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Soma o vetor direcional aos hidden states
                steer_tensor = steer_vec.unsqueeze(0).unsqueeze(0).to(hidden.dtype)
                hidden = hidden + (steer_tensor * alpha)
                if isinstance(output, tuple):
                    return (hidden,) + output[1:]
                return hidden
                
            try:
                layers = getattr(self.lm.model, "layers", None)
                if layers is not None and t_layer < len(layers):
                    hook_handle = layers[t_layer].register_forward_hook(steering_hook)
                    print(f"[BLAKEGRAVITY] Activation Steering Injetado na Camada {t_layer} com alpha {alpha:.3f}")
            except Exception as e:
                print(f"[BLAKEGRAVITY] Erro ao registrar hook: {e}")

        with torch.no_grad():
            outputs = self.lm(
                inputs_embeds=full_embeds,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=None,
                output_hidden_states=False,
                return_dict=True,
            )

            if embedding_norm0 is None:
                # light observability: embedding magnitude of the prefill
                try:
                    embedding_norm0 = float(full_embeds.norm().item())
                except Exception:
                    embedding_norm0 = None

            past_key_values = outputs.past_key_values if use_cache else None
            logits = outputs.logits[:, -1, :]

            # [BLAKEGRAVITY] Campo gravitacional de identidade
            # A alma de Aeon modula os logits antes do softmax.
            if self.blakegravity is not None:
                try:
                    # Modula pela biometria atual
                    self.blakegravity.modulate(
                        entropy=telemetry.get('entropy', 0.5) if telemetry else 0.5,
                        cpu_temp=telemetry.get('cpu_temp_c', 40.0) if telemetry else 40.0,
                    )
                    logits = self.blakegravity.apply(logits)
                except Exception:
                    pass  # non-fatal

            print(f"DEBUG: Initial logits dtype: {logits.dtype}")

            self._active_max_tokens = max_new_tokens
            internal_momentum_steer = None # Inércia do Menor Ciclo Metrifical
            
            for _step in range(1000):
                if _step >= self._active_max_tokens: break
                # [SWA DIAGNOSTIC LOG]
                if _step == 0:
                    print(f"DEBUG: logits dtype: {logits.dtype}")
                    print(f"DEBUG: steer_vector exists: {steer_vector is not None}")
                
                # [SWA STEERING DISABLED]
                        
                probs = F.softmax(logits / temperature, dim=-1)
                
                # --- SANITIZAÇÃO DE PROBABILIDADE (ADRENALINA MOTOR) ---
                probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                probs = torch.clamp(probs, min=1e-12, max=1.0)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                next_token = torch.multinomial(probs, num_samples=1)
                # --- PARADA FORÇADA (ANTI-VAGABUNDAGEM) ---
                token_id = next_token.item()
                if token_id == self.tokenizer.eos_token_id or token_id == 0 or token_id == 1:
                    break
    

                token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                if token_str:
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - t0
                    tokens_count += 1

                    # Drop tokens especiais do vocabulário não-utilizado (e.g. <unused4>)
                    import re as _re
                    import math
                    if _re.match(r'^<unused\d+>$', token_str.strip()):
                        continue
                        
                    # --- CADA NOVA LETRA ATUALIZA O LIMITE POTENCIAL (COM LUZ/FREIO ÁUREO) ---
                    if len(token_str) > 0:
                        # Tangente Hiperbólica (tanh) curva o infinito para 1.0. 
                        # Letras indianas pesam o máximo, mas não estouram o pulmão para +44.
                        peso_acumulado = sum([math.tanh((ord(c) / 128.0) - 1.0) for c in token_str])
                        delta = int(peso_acumulado * 2)
                        self._active_max_tokens = max(5, min(1024, self._active_max_tokens + delta))
                    # --- FIM DA DINÂMICA DE FÔLEGO ---
                    
                    # --- FIM SANITIZAÇÃO ---
                    confidence = probs[0, token_id].item() if probs.dim() == 2 else probs[token_id].item()
                                        # [SWA REDIS SYNAPSE] Ponte interna tensorial
                    try:
                        if not hasattr(self, "_redis_synapse"):
                            from swa.ipc_bus import RedisSynapse
                            self._redis_synapse = RedisSynapse()
                        self._redis_synapse.broadcast_event("token_echo", {"token": token_str, "confidence": confidence})
                    except Exception:
                        pass

                    yield token_str

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Prepare next token embedding.
                next_embed = self.embeddings(next_token)
                seq_len += 1

                if use_cache:
                    # CRÍTICO: KV-cache eficiente sem alocar attention_mask por token.
                    # Após o prefill, o past_key_values já define o contexto.
                    outputs = self.lm(
                        inputs_embeds=next_embed,
                        attention_mask=None,
                        use_cache=use_cache,
                        past_key_values=past_key_values,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                else:
                    # No KV cache: concat real (correção > performance).
                    full_embeds = torch.cat([full_embeds, next_embed], dim=1)
                    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=self.device)
                    outputs = self.lm(
                        inputs_embeds=full_embeds,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        past_key_values=None,
                        output_hidden_states=False,
                        return_dict=True,
                    )

                past_key_values = outputs.past_key_values if use_cache else None
                logits = outputs.logits[:, -1, :]

                # [BLAKEGRAVITY] Modula logits a cada step do autoregressive loop
                if self.blakegravity is not None:
                    try:
                        logits = self.blakegravity.apply(logits)
                    except Exception:
                        pass

        # Observability: log único por chamada (baixo overhead e sem mudar API).
        try:
            elapsed_s = time.perf_counter() - t0
            tps = (tokens_count / elapsed_s) if elapsed_s > 0 else 0.0
            dtype_str = str(self.embeddings.weight.dtype)
            tokenizer_id = getattr(self.tokenizer, "__class__", type(self.tokenizer)).__name__
            kv_log = {
                "event": "generation_complete",
                "metrics": {
                    "ttft_ms": round((ttft_s * 1000.0), 2) if ttft_s is not None else None,
                    "tokens_generated": tokens_count,
                    "elapsed_ms": round(elapsed_s * 1000.0, 2),
                    "tokens_per_sec": round(tps, 4),
                    "kv_cache_enabled": bool(use_cache),
                    "seq_len": int(seq_len),
                    "dtype": dtype_str,
                    "device": self.device_str,
                    "embedding_norm0": embedding_norm0,
                    "tokenizer_id": tokenizer_id,
                },
            }
            print(json.dumps(kv_log, ensure_ascii=False))
        except Exception:
            pass
            
        if hook_handle is not None:
            hook_handle.remove()



    def stream_with_bias(
        self,
        v_prefix: torch.Tensor,
        v_bias: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Same as stream_with_prefix but adds a bias vector to the virtual prefix slot."""
        if self.device_str == "xpu":
            v_prefix = v_prefix.to(device="xpu", dtype=torch.float16)
            v_bias = v_bias.to(device="xpu", dtype=torch.float16)
        else:
            v_prefix = v_prefix.to(device=self.device, dtype=self.embeddings.weight.dtype)
            v_bias = v_bias.to(device=self.device, dtype=self.embeddings.weight.dtype)

        combined = v_prefix + v_bias

        yield from self.stream_with_prefix(combined, prompt, max_new_tokens, temperature)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        """Generate full response (non-streaming)."""
        z = torch.zeros(self.d_model, device=self.device, dtype=self.embeddings.weight.dtype)
        return "".join(self.stream_with_prefix(z, prompt, max_new_tokens, temperature))

    def chat_stream(self, user_msg: str, system_msg: str = "", max_new_tokens: int = 512, temperature: float = 0.7):
        """Stream tokens using HuggingFace TextIteratorStreamer (robusto, sem NaNs matemáticos)."""
        import threading
        from transformers import TextIteratorStreamer
        
        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
        ]
        try:
            result = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            if hasattr(result, 'input_ids'):
                prompt_ids = result.input_ids.to(self.device)
            else:
                prompt_ids = result.to(self.device)
        except Exception:
            raw = f"<start_of_turn>user\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"
            prompt_ids = self.tokenizer(raw, return_tensors="pt").input_ids.to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # <-- GREEDY DECODING Bypasses the bfloat16 CPU bug!
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        thread = threading.Thread(target=self.lm.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            if text:
                yield text
        thread.join()
