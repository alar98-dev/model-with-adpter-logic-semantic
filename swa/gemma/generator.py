from __future__ import annotations

from typing import Dict, Generator, Optional

import torch
import torch.nn.functional as F

from .gemma_wrapper import GemmaWrapper


class _CortexQuantizedBuffer:
    """Thread-safe dict-like buffer for per-layer intermediate tensors."""

    def __init__(self):
        self._data: Dict[int, torch.Tensor] = {}
        import threading
        self._lock = threading.Lock()

    def put(self, layer_id: int, tensor: torch.Tensor):
        with self._lock:
            self._data[layer_id] = tensor

    def get(self, layer_id: int, default=None):
        with self._lock:
            return self._data.get(layer_id, default)

    def clear(self):
        with self._lock:
            self._data.clear()


CORTEX_QUANTIZED_BUFFER = _CortexQuantizedBuffer()


def make_async_quantized_hook(model, layer_ids=None):
    """Register forward hooks on specified layers to populate CORTEX_QUANTIZED_BUFFER."""
    if layer_ids is None:
        layer_ids = [12]
    hooks = []
    for lid in layer_ids:
        if lid < len(model.model.layers):
            def _hook_fn(module, input, output, _lid=lid):
                if isinstance(output, tuple) and len(output) > 0:
                    CORTEX_QUANTIZED_BUFFER.put(_lid, output[0].detach().clone())
                elif hasattr(output, 'last_hidden_state'):
                    CORTEX_QUANTIZED_BUFFER.put(_lid, output.last_hidden_state.detach().clone())
            hooks.append(model.model.layers[lid].register_forward_hook(_hook_fn))
    return hooks


class GemmaGenerator:
    """Thin wrapper around GemmaWrapper for token-stream generation with
    virtual-prefix steering and logit-bias injection.

    The generator owns a GemmaWrapper (the real HF model) and a tokenizer.
    It exposes streaming helpers used by the OODA loop, the SWA adapter
    pipeline, and the REST bridge.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        tokenizer=None,
    ):
        self.wrapper = GemmaWrapper(model_dir, device=device)
        self.model = self.wrapper.model
        self.tokenizer = tokenizer if tokenizer is not None else self.wrapper.tokenizer
        self.device_str = self.wrapper.device_str
        self.device = self.wrapper.device
        self.dtype = getattr(self.model, "dtype", torch.float32)
        self.use_cache = True
        self.embed_tokens = getattr(self.model, "get_input_embeddings", lambda: self.model.model.embed_tokens)()
        self.lm_head = getattr(self.model, "lm_head", None)
        self.lm = self.model  # Legacy alias — código legado acessa .lm para o modelo bruto
        self._vocab_size = getattr(self.tokenizer, "vocab_size", 50000)
        self._d_model = getattr(self.embed_tokens, "weight", torch.zeros(1)).shape[-1] if hasattr(self.embed_tokens, "weight") else 2560

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def d_model(self) -> int:
        return self._d_model

    def embeddings(self, input_ids):
        return self.embed_tokens(input_ids)

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
        )
        return self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

    def chat_stream(self, prompt: str, system: str = "", max_new_tokens: int = 128, temperature: float = 0.7) -> Generator[str, None, None]:
        full_prompt = prompt
        if system:
            full_prompt = f"<start_of_turn>user\n{system}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
        past_key_values = None
        for step in range(max_new_tokens):
            with torch.inference_mode():
                if past_key_values is None:
                    outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=None, return_dict=True)
                else:
                    last_token = input_ids[:, -1:]
                    outputs = self.model(input_ids=last_token, use_cache=True, past_key_values=past_key_values, return_dict=True)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values
                scaled = logits / max(temperature, 1e-6)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_str = self.tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
                yield token_str
                input_ids = next_token.unsqueeze(0)

    def stream_with_prefix(
        self,
        v_prefix: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        temperature: float = 0.7,
        telemetry: Optional[Dict[str, float]] = None,
        logit_bias: Optional[torch.Tensor] = None,
        steer_vector: Optional[torch.Tensor] = None,
        target_layer: int = -1,
    ) -> Generator[str, None, None]:
        """Generate token-by-token with a virtual prefix embedding.

        *v_prefix* is prepended to the token embeddings to steer generation
        towards a desired semantic direction.
        """
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(self.device)

        prefix_embed = v_prefix.to(dtype=self.dtype, device=self.device) if v_prefix is not None else None
        token_embeds = self.embeddings(input_ids)
        if prefix_embed is not None:
            pfx = prefix_embed
            while pfx.dim() > 1:
                pfx = pfx.squeeze(0)
            pfx = pfx.unsqueeze(0).unsqueeze(0)
            batch_prefix = pfx.expand(token_embeds.shape[0], 1, -1)
            combined = torch.cat([batch_prefix, token_embeds], dim=1)
        else:
            combined = token_embeds

        from transformers import TextIteratorStreamer, LogitsProcessorList
        from threading import Thread

        processors = None
        if steer_vector is not None:
            steer = steer_vector.to(dtype=self.dtype, device=self.device)
            if steer.dim() == 1:
                steer = steer.unsqueeze(0)
            projected = self.lm_head(steer).squeeze(0).detach()
            state = {"first": True}

            def _steer_processor(input_ids, scores):
                if state["first"]:
                    state["first"] = False
                    return scores + projected
                return scores

            processors = LogitsProcessorList([_steer_processor])

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,
        )
        gen_kwargs = dict(
            inputs_embeds=combined,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            streamer=streamer,
            logits_processor=processors,
        )
        t_gen = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        t_gen.start()
        for token in streamer:
            yield token

    def stream_with_multimodal_bias(
        self,
        v_prefix: torch.Tensor,
        logit_bias: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """Stream with both prefix embedding and logit bias."""
        yield from self.stream_with_prefix(
            v_prefix, prompt, max_new_tokens, temperature,
            logit_bias=logit_bias,
        )
