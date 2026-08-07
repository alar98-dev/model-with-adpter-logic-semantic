"""
LLM Quantizado para PyTest Interpreter.

Gemma-3-270m quantizado INT4 (128MB) para explicar falhas de teste.
Roda em XPU (Intel Arc) ou CPU com mínimo consumo.
"""
import os
import sys
import json
import time
import torch
from pathlib import Path

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
MODEL_DIR = ROOT / "models" / "gemma-3-270m-it"


class QuantizedTestInterpreter:
    """Interpretador de testes com LLM quantizado INT4."""

    def __init__(self, device: str = "auto"):
        self.model = None
        self.tokenizer = None
        self.device = self._resolve_device(device)
        self._loaded = False

    def _resolve_device(self, device: str) -> str:
        if device == "auto":
            if torch.xpu.is_available():
                return "xpu"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def load(self):
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import BitsAndBytesConfig

        print(f"🔄 Carregando Gemma-270M quantizado INT4 em {self.device}...")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_DIR),
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_DIR),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        self.model.eval()
        self._loaded = True

        # Mostrar uso de memória
        if self.device == "xpu":
            mem = torch.xpu.memory_allocated() / 1024**2
            print(f"✅ Modelo carregado: {mem:.1f}MB em XPU")
        elif self.device == "cuda":
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"✅ Modelo carregado: {mem:.1f}MB em CUDA")
        else:
            print(f"✅ Modelo carregado em CPU")

    def explain_failure(self, test_name: str, error: str, traceback: str = "") -> str:
        """Explica uma falha de teste em linguagem natural."""
        if not self._loaded:
            self.load()

        prompt = f"""<bos><start_of_turn>user
You are a test failure analyst. Explain this test failure simply:

Test: {test_name}
Error: {error}
{f"Traceback: {traceback[:500]}" if traceback else ""}

Provide:
1. What failed
2. Why it failed
3. How to fix it<end_of_turn>
<start_of_turn>model"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extrair apenas a resposta do modelo
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]

        return response.strip()

    def suggest_fix(self, test_name: str, code_snippet: str) -> str:
        """Sugere uma correção para o código."""
        if not self._loaded:
            self.load()

        prompt = f"""<bos><start_of_turn>user
You are a Python test fixer. Suggest a fix for:

Test: {test_name}
Code:
```python
{code_snippet[:1000]}
```

Provide the corrected code:<end_of_turn>
<start_of_turn>model"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]

        return response.strip()

    def summarize_results(self, results: dict) -> str:
        """Resume resultados de teste em linguagem natural."""
        if not self._loaded:
            self.load()

        passed = results.get("passed", 0)
        failed = results.get("failed", 0)
        errors = results.get("errors", 0)
        duration = results.get("duration", 0)

        prompt = f"""<bos><start_of_turn>user
Summarize these pytest results concisely:

Passed: {passed}
Failed: {failed}
Errors: {errors}
Duration: {duration:.1f}s

Provide a brief status summary:<end_of_turn>
<start_of_turn>model"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<start_of_turn>model" in response:
            response = response.split("<start_of_turn>model")[-1]
        if "<end_of_turn>" in response:
            response = response.split("<end_of_turn>")[0]

        return response.strip()
