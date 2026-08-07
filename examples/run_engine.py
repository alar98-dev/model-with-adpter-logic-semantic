"""Small demo: run LogicalEngine on a short stream (CPU-only).

Ensure `models/gemma1b` exists (local HF files) or change `MODEL_DIR`.
"""
from swa.adapter import SemanticWarpAdapter
from swa.logical_engine import LogicalEngine


MODEL_DIR = 'models/gemma3-4b'


def main():
    # create adapters (random init or load checkpoints)
    # dimension inferred at runtime when loading model in vectorizer
    # here we pick common k values configured earlier
    import torch
    # lazy load a model to read hidden size via vectorizer
    from swa.vectorizer import Vectorizer
    v = Vectorizer(MODEL_DIR, device='cpu')
    d = v.model.config.hidden_size

    adapter_local = SemanticWarpAdapter(d=d, k=64)
    adapter_macro = SemanticWarpAdapter(d=d, k=128)

    engine = LogicalEngine(model_dir=MODEL_DIR, adapter_local=adapter_local, adapter_macro=adapter_macro, device='cpu')

    stream = [
        "O sol esquenta a terra.",
        "A temperatura sobe devido à radiação solar.",
        "Isso causa expansão térmica." 
    ]

    outs = engine.process_stream(stream)
    for i, out in enumerate(outs):
        print(f"Event {i} -> anchored vector norm: {out.norm().item():.6f}")

    final = engine.close_context()
    print("Final global norm:", final.norm().item())


if __name__ == '__main__':
    main()
