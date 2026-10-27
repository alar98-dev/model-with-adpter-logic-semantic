"""
Hybrid Voice Shim — steering real via `inputs_embeds` no CausalLM do AeonCore.

O shim expõe as duas interfaces esperadas pelo plugin `attach_hybrid_steering`:

  stream_with_prefix(v_prefix, prompt, max_new_tokens, temperature)
      Injeta o steering vector (1,1,1152) como um token sintético concatenado
      aos embeddings do prompt (inputs_embeds no model.generate()).

  chat_stream(user_msg, system_msg, max_new_tokens, temperature)
      Delega ao pipeline nativo (fissão/gravidade/imune) do AeonCore.

Degradação graciosa: qualquer falha no caminho inputs_embeds cai para
`chat_stream` (pipeline original), garantindo que o benchmark nunca quebre.
"""

from __future__ import annotations

from typing import Iterator

import torch

# Prefixo sintético de 1 token embutido antes do prompt
PREFIX_TOKENS = 1


class HybridVoiceShim:
    def __init__(self, core_organism):
        self._core = core_organism
        self.model = core_organism.model
        self.tokenizer = core_organism.tokenizer
        self.device = core_organism.device
        self.embeddings = self.model.get_input_embeddings()

    # ---------------------------------------------------------------- build
    def _build_messages(self, prompt: str, system_msg: str = "") -> list[dict]:
        """Restaura a personalidade do core (profile/organism/narrative) no prompt."""
        msgs = []
        if system_msg:
            msgs.append({"role": "system", "content": system_msg})
        try:
            p = getattr(self._core, "profile", None)
            if p is not None and getattr(p, "instruction", None):
                msgs.append({"role": "system", "content": p.instruction})
            if getattr(self._core, "org_enabled", False):
                ctx = self._core.organism_context(prompt)
                if ctx:
                    msgs.append({"role": "system", "content": ctx})
            if getattr(self._core, "narrative_state", None):
                try:
                    nc = self._core.narrative_state.context()
                    if nc:
                        msgs.append({"role": "system", "content": nc})
                except Exception:
                    pass
        except Exception:
            pass
        msgs.append({"role": "user", "content": prompt})
        try:
            return self._core._merge_system_messages(msgs)
        except Exception:
            return [{"role": "user", "content": prompt}]

    # ------------------------------------------------------------- steering
    def stream_with_prefix(
        self,
        v_prefix: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """Steering: concatena v_prefix(1,1,1152) aos inputs_embeds do prompt."""
        try:
            msgs = self._build_messages(prompt)
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            else:
                formatted = prompt

            inputs = self.tokenizer(formatted, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            input_ids = torch.as_tensor(inputs["input_ids"], device=self.device)
            if input_ids.dim() > 2:
                input_ids = input_ids.squeeze(0)
            if input_ids.dim() != 2:
                input_ids = input_ids.unsqueeze(0)
            input_seq_len = input_ids.shape[1]

            prompt_embeds = self.embeddings(input_ids)  # (1, seq, d_model)

            v_pref = torch.as_tensor(
                v_prefix, device=self.device, dtype=prompt_embeds.dtype
            )
            if v_pref.dim() == 2:
                v_pref = v_pref.unsqueeze(0)
            if v_pref.dim() == 1:
                v_pref = v_pref.view(1, 1, -1)

            full_embeds = torch.cat([v_pref, prompt_embeds], dim=1)
            attn_mask = torch.ones(
                (full_embeds.shape[0], full_embeds.shape[1]),
                dtype=torch.long,
                device=self.device,
            )

            do_sample = float(temperature or 0.0) > 0.0
            gen_kwargs = {
                "inputs_embeds": full_embeds,
                "attention_mask": attn_mask,
                "max_new_tokens": max(1, int(max_new_tokens or 128)),
                "do_sample": do_sample,
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None)
                or getattr(self.tokenizer, "eos_token_id", None),
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                "return_dict_in_generate": True,
            }
            if do_sample:
                gen_kwargs["temperature"] = max(float(temperature or 0.0), 0.05)
                gen_kwargs["top_p"] = min(float(kwargs.get("top_p", 0.95)), 1.0)

            with torch.no_grad():
                outputs = self.model.generate(**gen_kwargs)

            # Descarta o prefixo sintético (1) + tokens do prompt: só tokens novos.
            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs[0]
            generated_ids = sequences[0, PREFIX_TOKENS + input_seq_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if text:
                yield text
        except Exception as err:
            print(
                f"⚠️ [HYBRID SHIM] Falha no steering inputs_embeds: {err}. "
                "Fallback stream_messages.",
                flush=True,
            )
            yield from self.chat_stream(
                user_msg=prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )

    # -------------------------------------------------------------- fallback
    def chat_stream(
        self,
        user_msg: str,
        system_msg: str = "",
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """Pipeline nativo do AeonCore (fissão/gravidade/imune)."""
        msgs = []
        if system_msg:
            msgs.append({"role": "system", "content": system_msg})
        msgs.append({"role": "user", "content": user_msg})
        return self._core.stream_messages(
            msgs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            profile=getattr(self._core, "profile", None),
        )