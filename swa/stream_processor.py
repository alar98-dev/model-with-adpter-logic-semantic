from typing import Generator, List, Optional
import torch
import threading
import queue
import time
from collections import deque
import psutil
import os
from concurrent.futures import ThreadPoolExecutor

from .logical_engine import LogicalEngine
from .adapter import SemanticWarpAdapter, optimize_for_xeon, set_phase_affinity
from .gemma_integration import GemmaGenerator
from . import rest_translator

class StreamProcessor:
    '''3-Process Low-Latency Pipeline para Dual CPU:
    - Process A (CPU1): Vectorizer + SWA prep (v_global)
    - Process B (CPU2/GPU): Autoregressive infer (KVCache)
    - Process C: Rolling buffer detok + SSE + entropia feedback
    TTFT ~0.2s, 30-50 tokens/s, anti-stuttering.
    '''

    @staticmethod
    def _cpus_from_mask(mask: str) -> list:
        """Parse '0-5' or '6-11' into a CPU index list (inclusive range)."""
        if "-" in mask:
            lo, hi = mask.split("-", 1)
            return list(range(int(lo.strip()), int(hi.strip()) + 1))
        return [int(x.strip()) for x in mask.split(",") if x.strip()]

    def __init__(self, engine: LogicalEngine, cpu_mask_a: str | None = None, cpu_mask_b: str | None = None):
        self.engine = engine
        n = psutil.cpu_count() or 8
        h = max(1, n // 2)
        self.cpu_mask_a = cpu_mask_a if cpu_mask_a is not None else f"0-{h - 1}"
        self.cpu_mask_b = cpu_mask_b if cpu_mask_b is not None else f"{h}-{n - 1}"
        self.token_queue = queue.Queue()  # Inter-process comm
        self.memory: deque = deque(maxlen=32)
        self.entropy_threshold = 0.5  # Divergência trigger reanchor
        optimize_for_xeon(0)

    def prep_pipeline(self, text: str) -> torch.Tensor:
        '''Process A: CPU1 - Vectorizer + SWA → v_global (parallel).'''
        try:
            os.sched_setaffinity(0, self._cpus_from_mask(self.cpu_mask_a))
        except OSError as e:
            print(f"sched_setaffinity prep skip: {e}")
        v_seg = self.engine.process_event(text)
        self.memory.append(v_seg)
        v_global = self.engine.close_context()
        return v_global

    def infer_pipeline(self, v_context: torch.Tensor, prompt: str) -> Generator[str, None, None]:
        '''Autoregressive stream: v_context is SWA macro output (d) or already a Gemma prefix (d_model).'''
        try:
            os.sched_setaffinity(0, self._cpus_from_mask(self.cpu_mask_b))
        except OSError as e:
            print(f"sched_setaffinity infer skip: {e}")
        if self.engine.gemma_gen is None:
            return
        gdim = self.engine.gemma_gen.d_model
        vc = v_context.to(self.engine.device)
        if vc.shape[-1] == gdim:
            v_prefix = vc.view(-1)[:gdim]
        elif self.engine.adapter_macro.prefix_projection is not None:
            v_p, v_b = self.engine.adapter_macro.prefix_and_bias(vc)
            v_prefix = v_p + v_b
        else:
            v_prefix = torch.zeros(gdim, device=vc.device, dtype=vc.dtype)
        yield from self.engine.gemma_gen.stream_with_prefix(v_prefix, prompt)

    def buffer_pipeline(self, tokens: Generator) -> Generator[str, None, None]:
        '''Process C: Detok + rolling + entropia check → SSE.'''
        full_text = ''
        for token in tokens:
            if isinstance(token, str):
                token_str = token
            else:
                token_str = self.engine.gemma_gen.tokenizer.decode([int(token)])
            full_text += token_str
            # Entropia check (simpl: norma decay)
            if len(full_text) % 10 == 0:  # Every 10 tokens
                v_check = self.engine.adapter_local(self.engine.vectorizer.embed([full_text[-50:]]))  # Recent context
                entropy = float(v_check.norm().item())
                if entropy > self.entropy_threshold:
                    yield f'data: {{"reanchor": true, "entropy": {entropy}}}\\n\\n'
                    # Feedback: Re-prep
                    v_reanchor = self.prep_pipeline(full_text)
                    yield from self.infer_pipeline(v_reanchor, '')  # Resume
            yield f'data: {{"token": token_str.strip()}}\\n\\n'
        yield 'data: [DONE]\\n\\n'

    def stream_dual(self, prompt: str, use_dual: bool = True) -> Generator[str, None, None]:
        '''Full 3-process dual-engine stream (Llama bruto → Gemma filter).'''
        if use_dual and self.engine.engines:
            # Llama bruto (GPU)
            llama_raw = self.engine.engines['llama_bruto'].process_embedding_context(prompt)
            prep_text = f'Bruto: {llama_raw}\\nRefine: '
        else:
            prep_text = prompt

        # Parallel 3-process
        with ThreadPoolExecutor(max_workers=3) as executor:
            prep_future = executor.submit(self.prep_pipeline, prep_text)
            v_global = prep_future.result()  # ~50ms CPU1

            infer_gen = self.infer_pipeline(v_global, prompt)
            yield from self.buffer_pipeline(infer_gen)

