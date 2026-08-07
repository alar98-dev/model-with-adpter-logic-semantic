"""
DirectML Optimizer — Otimização do LLM para GPU via DML.

Converte e executa modelos quantizados usando DirectML (DxCore) no WSL2.
Compara performance entre PyTorch XPU e DirectML.
"""
import os
import sys
import time
import torch
from pathlib import Path

ROOT = Path(os.environ.get("SWA_PROJECT_ROOT", "/services/model-with-adpter-logic-semantic"))
sys.path.insert(0, str(ROOT))


class DirectMLOptimizer:
    """Otimizador para rodar LLM via DirectML."""

    def __init__(self):
        self.dml_available = False
        self.ort_session = None
        self.tokenizer = None
        self._check_dml()

    def _check_dml(self):
        """Verifica se DirectML está disponível."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self.dml_available = "DmlExecutionProvider" in providers
            print(f"DirectML providers: {providers}")
        except ImportError:
            print("onnxruntime não instalado. Instalando...")
            self._install_onnxruntime()

    def _install_onnxruntime(self):
        """Instala onnxruntime com DirectML."""
        import subprocess
        cmd = [
            sys.executable, "-m", "pip", "install",
            "onnxruntime-directml", "-q"
        ]
        subprocess.run(cmd, check=True)
        import onnxruntime as ort
        providers = ort.get_available_providers()
        self.dml_available = "DmlExecutionProvider" in providers
        print(f"DirectML providers após instalação: {providers}")

    def convert_to_onnx(
        self,
        model_path: str,
        output_path: str,
        opset: int = 14,
    ):
        """Converte modelo HuggingFace para ONNX."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"🔄 Convertendo modelo para ONNX...")
        print(f"   Modelo: {model_path}")
        print(f"   Output: {output_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )

        # Dummy input para trace
        dummy_input = tokenizer(
            "Hello, world!",
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )

        # Exportar para ONNX
        torch.onnx.export(
            model,
            (dummy_input["input_ids"], dummy_input["attention_mask"]),
            output_path,
            opset_version=opset,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size", 1: "sequence"},
            },
        )

        print(f"✅ Modelo exportado: {output_path}")
        return output_path

    def load_onnx_directml(self, onnx_path: str):
        """Carrega modelo ONNX com DirectML."""
        import onnxruntime as ort

        if not self.dml_available:
            raise RuntimeError("DirectML não disponível")

        print(f"🔄 Carregando modelo ONNX com DirectML...")

        providers = [
            ("DmlExecutionProvider", {
                "device_id": 0,
                "enable_dynamic_graph_fusion": True,
            }),
            "CPUExecutionProvider",
        ]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        self.ort_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

        print(f"✅ Modelo carregado com DirectML")
        print(f"   Providers: {self.ort_session.get_providers()}")

    def benchmark_xpu_vs_dml(
        self,
        model_path: str,
        num_tokens: int = 50,
    ):
        """Compara performance entre XPU e DirectML."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n" + "=" * 60)
        print("📊 BENCHMARK: XPU vs DirectML")
        print("=" * 60)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Benchmark XPU
        print("\n🔄 Testando PyTorch XPU...")
        model_xpu = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="xpu",
        )

        input_text = "Explain quantum computing in simple terms:"
        inputs = tokenizer(input_text, return_tensors="pt").to("xpu")

        # Warmup
        with torch.no_grad():
            _ = model_xpu.generate(**inputs, max_new_tokens=10)

        # Benchmark
        start = time.time()
        with torch.no_grad():
            outputs = model_xpu.generate(
                **inputs,
                max_new_tokens=num_tokens,
                do_sample=False,
            )
        xpu_time = time.time() - start
        xpu_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
        xpu_tps = xpu_tokens / xpu_time

        print(f"   XPU: {xpu_time:.2f}s para {xpu_tokens} tokens ({xpu_tps:.1f} tok/s)")

        # Benchmark DirectML (se disponível)
        if self.dml_available:
            print("\n🔄 Testando DirectML...")
            # Converter para ONNX primeiro
            onnx_path = str(ROOT / "models" / "gemma-3-270m-it.onnx")
            if not os.path.exists(onnx_path):
                self.convert_to_onnx(model_path, onnx_path)

            self.load_onnx_directml(onnx_path)

            # Benchmark DirectML
            start = time.time()
            # TODO: Implementar geração com DirectML
            dml_time = time.time() - start
            print(f"   DirectML: {dml_time:.2f}s")

        # Resumo
        print("\n" + "=" * 60)
        print("📊 RESUMO DO BENCHMARK")
        print("=" * 60)
        print(f"   PyTorch XPU: {xpu_tps:.1f} tok/s")
        if self.dml_available:
            print(f"   DirectML: {dml_time:.2f}s")
        print("=" * 60)

        # Liberar memória
        del model_xpu
        torch.xpu.empty_cache()

        return {
            "xpu_time": xpu_time,
            "xpu_tokens": xpu_tokens,
            "xpu_tps": xpu_tps,
        }


def main():
    """Entry point para o otimizador DirectML."""
    import argparse

    parser = argparse.ArgumentParser(description="DirectML Optimizer para LLM")
    parser.add_argument("--convert", action="store_true", help="Converter modelo para ONNX")
    parser.add_argument("--benchmark", action="store_true", help="Rodar benchmark")
    parser.add_argument("--model", default=str(ROOT / "models" / "gemma-3-270m-it"))
    args = parser.parse_args()

    optimizer = DirectMLOptimizer()

    if args.convert:
        onnx_path = str(ROOT / "models" / "gemma-3-270m-it.onnx")
        optimizer.convert_to_onnx(args.model, onnx_path)

    if args.benchmark:
        optimizer.benchmark_xpu_vs_dml(args.model)

    if not args.convert and not args.benchmark:
        print("Uso:")
        print("  python directml_optimizer.py --convert   # Converter para ONNX")
        print("  python directml_optimizer.py --benchmark # Comparar XPU vs DML")


if __name__ == "__main__":
    main()
