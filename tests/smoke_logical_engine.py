"""Smoke test for LogicalEngine without loading HF models.

This test monkeypatches `Vectorizer` in `swa.logical_engine` with a
`FakeVectorizer` that returns random embeddings so the pipeline can be
exercised on CPU quickly.

Run with:

    python -m tests.smoke_logical_engine

or

    python tests/smoke_logical_engine.py

"""
from swa import logical_engine
from swa.adapter import SemanticWarpAdapter
import torch


class FakeVectorizer:
    def __init__(self, model_dir=None, device='cpu'):
        self.device = torch.device(device)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        batch = len(texts)
        # return deterministic random for reproducibility
        return torch.randn(batch, 128)


def run_smoke():
    # patch Vectorizer to avoid HF model loading
    logical_engine.Vectorizer = FakeVectorizer

    # create small adapters matching d=128
    a_local = SemanticWarpAdapter(d=128, k=32)
    a_macro = SemanticWarpAdapter(d=128, k=32)

    eng = logical_engine.LogicalEngine(model_dir='dummy', adapter_local=a_local, adapter_macro=a_macro, device='cpu')

    # process a few events
    outs = eng.process_stream(["primeiro evento", "segundo evento", "terceiro evento"])

    # print shapes and norms
    for i, v in enumerate(outs):
        print(f"out[{i}] shape={v.shape} norm={v.norm().item():.4f}")

    # test close_context
    closed = eng.close_context()
    if closed is not None:
        print(f"closed shape={closed.shape} norm={closed.norm().item():.4f}")
    else:
        print("closed is None")


if __name__ == "__main__":
    run_smoke()
