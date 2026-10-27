import asyncio
import json
import torch
import torch.nn.functional as F
from typing import AsyncGenerator, Dict, Any, Optional
from aeon.core.entropy import AdaptiveEntropyRegulator
from aeon.core.manifold import IsometricProjector
from aeon.ipc.shm_tensor import SharedMetamatterIPC

class BicameralSSEEngine:
    """
    Motor de Inferência Bicameral com Streaming Contínuo SSE.
    Alterna dinamicamente entre Câmara Rápida (270M) e Reflexiva (1B)
    sem interrupção do socket HTTP, auditado pelo Regulador de Entropia.
    """

    def __init__(
        self,
        model_fast,            # Gemma 270M
        tokenizer_fast,
        model_reflexive,       # Gemma 1B
        tokenizer_reflexive,
        projector_640_to_2560: IsometricProjector,
        shm_ipc: SharedMetamatterIPC,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.fast_model = model_fast
        self.fast_tok = tokenizer_fast
        self.reflex_model = model_reflexive
        self.reflex_tok = tokenizer_reflexive
        self.projector = projector_640_to_2560
        self.shm_ipc = shm_ipc
        self.device = device

        self.entropy_regulator = AdaptiveEntropyRegulator(
            window_size=50,
            z_threshold=2.0,
            sigma_min=0.05
        )

    def _compute_entropy(self, logits: torch.Tensor) -> float:
        """Calcula a entropia de Shannon pontual dos logits em FP32."""
        probs = F.softmax(logits.to(dtype=torch.float32), dim=-1)
        log_probs = F.log_softmax(logits.to(dtype=torch.float32), dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        # Normalização log_e para log_2 (bits de Shannon)
        return float(entropy / 0.693147)

    async def stream_bicameral(
        self,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> AsyncGenerator[str, None]:
        """
        Gerador SSE assíncrono que emite eventos em formato:
        data: {"token": "...", "chamber": "270M"|"1B", "entropy": float, "z_score": float}\n\n
        """
        self.entropy_regulator.reset()
        active_chamber = "270M"
        generated_tokens = []

        # 1. Ingestão do vetor de metamatéria e projeção isométrica
        raw_meta, _, _ = self.shm_ipc.read_latest_vector()
        meta_prefix = self.projector(raw_meta.to(self.device))  # [2560]

        # 2. Tokenização inicial
        current_input_ids = self.fast_tok.encode(prompt, return_tensors="pt").to(self.device)

        for step in range(max_new_tokens):
            if active_chamber == "270M":
                # Execução na Câmara Rápida
                with torch.no_grad():
                    outputs = self.fast_model(current_input_ids)
                    next_token_logits = outputs.logits[:, -1, :]

                    entropy_val = self._compute_entropy(next_token_logits)
                    should_fission, z_score, arm = self.entropy_regulator.update(entropy_val)

                    if should_fission:
                        # DISPARO DE FISSÃO: Transição de Fase para 1B
                        active_chamber = "1B"
                        # Re-tokenização do contexto consolidado para o vocabulário da Câmara Reflexiva
                        current_text = self.fast_tok.decode(current_input_ids[0], skip_special_tokens=True)
                        current_input_ids = self.reflex_tok.encode(current_text, return_tensors="pt").to(self.device)

                        # Emite frame de evento sinalizando a fissão semântica
                        fission_payload = {
                            "event": "fission",
                            "from": "270M",
                            "to": "1B",
                            "z_score": round(z_score, 3),
                            "entropy_bits": round(entropy_val, 3),
                        }
                        yield f"event: fission\ndata: {json.dumps(fission_payload)}\n\n"
                        await asyncio.sleep(0)
                        continue

                    # Seleção de token auto-regressiva
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    token_str = self.fast_tok.decode(next_token[0], skip_special_tokens=True)

            else:
                # Execução na Câmara Reflexiva (Gemma 1B)
                with torch.no_grad():
                    outputs = self.reflex_model(current_input_ids)
                    next_token_logits = outputs.logits[:, -1, :]

                    entropy_val = self._compute_entropy(next_token_logits)
                    _, z_score, _ = self.entropy_regulator.update(entropy_val)

                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    token_str = self.reflex_tok.decode(next_token[0], skip_special_tokens=True)

            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            generated_tokens.append(token_str)

            # Estrutura padrão de chunk SSE compatível com clientes OpenAI / Web
            event_payload = {
                "token": token_str,
                "chamber": active_chamber,
                "entropy": round(entropy_val, 3),
                "z_score": round(z_score, 3),
            }
            yield f"data: {json.dumps(event_payload)}\n\n"

            # Libera o event loop do asyncio para garantir flush imediato do streaming
            await asyncio.sleep(0)

            # Condição de parada (EOS)
            eos_id = self.fast_tok.eos_token_id if active_chamber == "270M" else self.reflex_tok.eos_token_id
            if next_token.item() == eos_id:
                break

        # Frame de finalização do protocolo SSE
        yield "data: [DONE]\n\n"
