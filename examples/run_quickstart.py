"""Quickstart example showing chunking + SWA local+macro flow.

Place a downloaded Gemma 270M HF-compatible checkpoint under `models/gemma270m/`.
"""
import torch
from swa.adapter import SemanticWarpAdapter, triplet_loss
from transformers import AutoTokenizer, AutoModel


def sliding_windows(toks, max_len, stride):
    # naive token chunker for dict with 'input_ids' and 'attention_mask'
    input_ids = toks['input_ids'][0]
    total = input_ids.size(0)
    segments = []
    i = 0
    while i < total:
        j = min(i + max_len, total)
        seg = {
            'input_ids': input_ids[i:j].unsqueeze(0),
            'attention_mask': torch.ones((1, j - i), dtype=torch.long)
        }
        segments.append(seg)
        if j == total:
            break
        i += stride
    return segments


def chunk_and_anchor(model, tokenizer, text, adapter_local, adapter_macro, max_len=256, stride=128, device='cpu'):
    toks = tokenizer(text, return_tensors='pt')
    segments = sliding_windows(toks, max_len, stride)
    v_segs = []
    for seg in segments:
        seg = {k: v.to(device) for k, v in seg.items()}
        out = model(**seg, output_hidden_states=True)
        # simple pooling
        v = out.last_hidden_state.mean(dim=1)
        v_seg = adapter_local(v)
        v_segs.append(v_seg)
    v_agg = torch.stack(v_segs, dim=1).mean(dim=1)
    v_global = adapter_macro(v_agg)
    return v_global


def main():
    model_dir = "models/gemma1b"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
    except Exception as e:
        print("Erro ao carregar o modelo. Certifique-se de baixar o checkpoint HF para:", model_dir)
        raise

    device = 'cpu'
    model.to(device)

    # dimension inference (simple): use model hidden_size if available
    d = model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else 768

    adapter_local = SemanticWarpAdapter(d=d, k=64)
    adapter_macro = SemanticWarpAdapter(d=d, k=128)

    text = "O sol esquenta a terra. Isso aumenta a temperatura." * 20
    v_global = chunk_and_anchor(model, tokenizer, text, adapter_local, adapter_macro, max_len=512, stride=256, device=device)
    print("v_global shape:", v_global.shape)


if __name__ == '__main__':
    main()
