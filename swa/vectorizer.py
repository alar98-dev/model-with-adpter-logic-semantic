from swa.model_utils import load_multimodal_model_surgical
from transformers import AutoConfig
import torch
from transformers import AutoTokenizer, AutoModel


def get_safe_device(requested_device: str) -> str:
    """Determine safe device with GPU fallback to CPU."""
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if requested_device in ("xpu", "intel", "igpu"):
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        print("WARNING: XPU requested but not available. Falling back to CPU.")
        return "cpu"
    return "cpu"


class Vectorizer:
    """Converts text to dense vectors using a HF model (CPU-friendly).

    Usage:
      v = Vectorizer(model_dir='models/gemma1b', device='cpu')
      emb = v.embed(["texto aqui"])  # tensor (batch, d)

    Architectural requirement:
      Prefer an injected tokenizer instance (TokenizerManager) to keep
      tokenization consistent with GemmaGenerator.
    """

    def __init__(self, model_dir: str, device: str = 'cpu', *, tokenizer: any = None, dtype: torch.dtype = torch.float32):
        self.device_str = get_safe_device(device)
        self.device = torch.device(self.device_str)
        self.dtype = dtype
        print(f"Vectorizer: using device {self.device_str} (requested: {device})")

        # If tokenizer is injected, do NOT load internally.
        # Backward compatible: if tokenizer is None, load default HF tokenizer.
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
                _ = getattr(self.tokenizer, "tokenize", None)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, legacy=True)

                # [SWA SURGICAL LOAD]
        is_gemma3 = False
        try:
            cfg = AutoConfig.from_pretrained(model_dir)
            if cfg.model_type == "gemma3":
                is_gemma3 = True
        except Exception:
            pass
            
        if is_gemma3:
            self.model = load_multimodal_model_surgical(model_dir, self.device, self.dtype, model_class=AutoModel)
        else:
            self.model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype).to(self.device)
        self.model.eval()


    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        toks = {k: v.to(self.device) for k, v in toks.items()}
        with torch.no_grad():
            out = self.model(**toks, output_hidden_states=True)
            emb = out.last_hidden_state.mean(dim=1)
        return emb
