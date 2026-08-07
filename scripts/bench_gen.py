"""Benchmark de geração para gemma-3-1b no CPU-only.

Mede prefill (ttft) e decode (tps) com configurações comparáveis ao aeonc.
Uso:
  python3 scripts/bench_gen.py --dtype bf16 --threads 5
  python3 scripts/bench_gen.py --dtype int8 --threads 10 --compile
"""
import argparse
import json
import sys
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma-3-1b-it")
    ap.add_argument("--dtype", choices=["bf16", "int8"], default="bf16")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--prompt", default="qual o formato de persistência binária dos estados do aeon?")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    if args.dtype == "int8":
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
        quantize_(model, Int8WeightOnlyConfig())
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if args.compile:
        model = torch.compile(model)
    load_s = time.time() - t0

    ids = tok(args.prompt, return_tensors="pt")
    gen_kwargs = dict(
        do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=args.max_new_tokens
    )

    def run():
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**ids, **gen_kwargs)
        dt = time.time() - t0
        n = out.shape[1] - ids["input_ids"].shape[1]
        return n, dt

    run()  # warmup (desconta compile/uma-página-fria)
    n, dt = run()
    print(json.dumps({
        "model": args.model,
        "dtype": args.dtype,
        "threads": args.threads,
        "compile": args.compile,
        "load_s": round(load_s, 1),
        "tokens": n,
        "elapsed_s": round(dt, 2),
        "tps": round(n / dt, 2),
    }, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
