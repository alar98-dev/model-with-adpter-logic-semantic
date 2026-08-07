"""AutoLoRA — Geração automática de LoRA adapters a partir do token dataset.

Pipeline:
  parts.jsonl + opencode_sessions.jsonl →
    clustering por trace_classification + blake_signature →
    geração de dataset de treino por cluster →
    treino LoRA para cada cluster

Cada adapter captura um padrão de comportamento do agente
identificado pela assinatura Blake.
"""

import os
import sys
import json
import time
import shutil
import hashlib
import tempfile
from typing import Optional
from collections import Counter, defaultdict

TOKEN_DATASET_DIR = os.environ.get(
    "TOKEN_DATASET_DIR",
    "/services/model-with-adpter-logic-semantic/models/token_dataset"
)
ADAPTER_BASE = os.environ.get(
    "AUTOLORA_ADAPTER_BASE",
    "/services/model-with-adpter-logic-semantic/models/autolora-adapters"
)
MODEL_PATH = os.environ.get(
    "AUTOLORA_MODEL_PATH",
    "/services/model-with-adpter-logic-semantic/models/gemma-3-270m"
)
MIN_SAMPLES_PER_CLUSTER = 3
MAX_CLUSTERS = 8


def _blake3(text: str) -> str:
    try:
        import blake3
        return blake3.blake3(text.encode()).hexdigest()
    except ImportError:
        return hashlib.sha256(text.encode()).hexdigest()


def cluster_key(part: dict) -> str:
    trace = part.get("trace_classification") or part.get("label", "unknown")
    sig = part.get("blake_signature", [0.0] * 6)
    disc = tuple(round(v * 4) / 4.0 for v in sig)
    return f"{trace}/{disc[0]:.2f}_{disc[1]:.2f}"


def load_all_parts() -> list[dict]:
    path = os.path.join(TOKEN_DATASET_DIR, "parts.jsonl")
    if not os.path.exists(path):
        return []
    parts = []
    with open(path) as f:
        for line in f:
            try:
                parts.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return parts


def load_all_sessions() -> list[dict]:
    path = os.path.join(TOKEN_DATASET_DIR, "opencode_sessions.jsonl")
    if not os.path.exists(path):
        return []
    sessions = []
    with open(path) as f:
        for line in f:
            try:
                sessions.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return sessions


def build_training_pairs_from_parts(parts: list[dict]) -> list[dict]:
    """Converte parts em pares de treino (prompt → resposta)."""
    pairs = []
    for p in parts:
        text = p.get("text", "").strip()
        if not text:
            continue
        pt = p.get("part_type", "?")
        trace = p.get("trace_classification", "?")
        if pt == "reasoning":
            prompt = "Raciocine sobre o problema."
            response = text
        elif pt == "tool_call":
            tool = p.get("tool_name", "?")
            inp = p.get("tool_input", "")[:100]
            prompt = f"Execute {tool}: {inp}"
            response = text or f"<tool>{tool}</tool>"
        elif pt == "assistant_text":
            prompt = "Responda ao usuário."
            response = text
        elif pt == "user_text":
            prompt = text
            response = "<continue>"
        else:
            continue
        pairs.append({
            "text": f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>",
            "label": trace,
            "trace_classification": trace,
            "blake_signature": p.get("blake_signature", [0.0] * 6),
            "part_type": pt,
        })
    return pairs


def build_training_pairs_from_sessions(sessions: list[dict]) -> list[dict]:
    """Converte sessões completas em pares prompt→resposta multistep."""
    pairs = []
    for s in sessions:
        conv = s.get("conversation", [])
        for i, part in enumerate(conv):
            text = part.get("text", "").strip()
            if not text:
                continue
            pt = part.get("part_type", "?")
            trace = part.get("trace_classification", "?")
            if pt == "user_text":
                pairs.append({
                    "text": f"<start_of_turn>user\n{text}<end_of_turn>",
                    "label": "user",
                    "trace_classification": "user",
                    "blake_signature": part.get("blake_signature", [0.0] * 6),
                    "part_type": pt,
                    "session_id": s.get("session_id", ""),
                })
            elif pt == "assistant_text":
                pairs.append({
                    "text": f"<start_of_turn>model\n{text}<end_of_turn>",
                    "label": trace,
                    "trace_classification": trace,
                    "blake_signature": part.get("blake_signature", [0.0] * 6),
                    "part_type": pt,
                    "session_id": s.get("session_id", ""),
                })
            elif pt == "reasoning":
                pairs.append({
                    "text": f"<start_of_turn>user\n<thinking>{text}<end_of_turn>\n<start_of_turn>model\n<understood><end_of_turn>",
                    "label": trace,
                    "trace_classification": trace,
                    "blake_signature": part.get("blake_signature", [0.0] * 6),
                    "part_type": pt,
                    "session_id": s.get("session_id", ""),
                })
    return pairs


def cluster_pairs(pairs: list[dict], max_clusters: int = MAX_CLUSTERS) -> dict[str, list[dict]]:
    """Agrupa pares por chave de cluster."""
    clusters = defaultdict(list)
    for p in pairs:
        key = cluster_key(p)
        clusters[key].append(p)
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    result = {}
    for key, items in sorted_clusters[:max_clusters]:
        if len(items) >= MIN_SAMPLES_PER_CLUSTER:
            result[key] = items
    return result


def cluster_summary(clusters: dict[str, list[dict]]) -> list[dict]:
    return [
        {"cluster": k, "count": len(v),
         "labels": dict(Counter(p["label"] for p in v).most_common(3)),
         "types": dict(Counter(p["part_type"] for p in v).most_common(3))}
        for k, v in clusters.items()
    ]


def train_adapter(
    cluster_key: str,
    records: list[dict],
    output_dir: str,
) -> bool:
    """Treina um adapter LoRA para um cluster específico."""
    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer
    except ImportError:
        return False

    if len(records) < MIN_SAMPLES_PER_CLUSTER:
        return False

    safe_key = _blake3(cluster_key)[:12]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        return False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    dataset = Dataset.from_list(records)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=min(30, max(len(records), 5)),
        save_strategy="no",
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            args=training_args,
            processing_class=tokenizer,
        )
        trainer.train()
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
        return True
    except Exception:
        return False


def run_autolora(
    dataset_dir: str = TOKEN_DATASET_DIR,
    adapter_base: str = ADAPTER_BASE,
    max_clusters: int = MAX_CLUSTERS,
    dry_run: bool = False,
) -> dict:
    """Pipeline AutoLoRA completo."""
    os.makedirs(adapter_base, exist_ok=True)

    parts = load_all_parts()
    sessions = load_all_sessions()

    pairs_parts = build_training_pairs_from_parts(parts)
    pairs_sessions = build_training_pairs_from_sessions(sessions)
    all_pairs = pairs_parts + pairs_sessions

    if not all_pairs:
        return {"error": "no training pairs", "parts": len(parts), "sessions": len(sessions)}

    clusters = cluster_pairs(all_pairs, max_clusters=max_clusters)
    summary = cluster_summary(clusters)

    results = []
    if not dry_run:
        for key, records in clusters.items():
            safe_key = _blake3(key)[:12]
            output_dir = os.path.join(adapter_base, safe_key)
            ok = train_adapter(key, records, output_dir)
            results.append({
                "cluster": key,
                "count": len(records),
                "adapter_path": output_dir if ok else None,
                "trained": ok,
            })

    return {
        "total_parts": len(parts),
        "total_sessions": len(sessions),
        "total_pairs": len(all_pairs),
        "clusters_found": len(clusters),
        "cluster_summary": summary,
        "adapters_trained": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AutoLoRA — treino automático de adapters")
    parser.add_argument("--dry-run", action="store_true", help="Apenas analisa, não treina")
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--adapter-base", default=ADAPTER_BASE)
    args = parser.parse_args()

    result = run_autolora(
        adapter_base=args.adapter_base,
        max_clusters=args.max_clusters,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
