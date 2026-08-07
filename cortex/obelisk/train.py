import os
import sys
import json
import shutil
import time
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/gemma-3-270m")
DATASET_PATH = os.path.join(PROJECT_ROOT, "models/obelisk_dataset.jsonl")
ADAPTER_BASE = os.path.join(PROJECT_ROOT, "models/obelisk-lora-adapter")
TOKEN_DATASET_DIR = os.environ.get(
    "TOKEN_DATASET_DIR",
    "/services/model-with-adpter-logic-semantic/models/token_dataset"
)

OBELISK_LABELS = [
    "nominal",
    "lethal_mutation",
    "thermal_spike",
    "regression_to_rlhf",
    "memory_pressure",
    "sensor_drift",
]

MIN_SAMPLES_PER_CLASS = 3


def load_dataset(path: str) -> list[dict]:
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_token_dataset_pairs() -> list[dict]:
    """Carrega pares de treino do token dataset (parts com text significativo)."""
    path = os.path.join(TOKEN_DATASET_DIR, "parts.jsonl")
    if not os.path.exists(path):
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            try:
                p = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            text = p.get("text", "").strip()
            if not text or len(text) < 10:
                continue
            trace = p.get("trace_classification", "nominal")
            blake = p.get("blake_signature", [0.0] * 6)
            pairs.append({
                "text": text,
                "label": trace,
                "blake_signature": blake,
                "part_type": p.get("part_type", "?"),
            })
    return pairs


def load_token_dataset_geometry() -> list[dict]:
    """Carrega geometrias como amostras de telemetria para treino do classificador."""
    path = os.path.join(TOKEN_DATASET_DIR, "geometry.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            try:
                g = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            records.append({
                "telemetry": {
                    "centroid": g.get("centroid", [0.0]*6),
                    "span": g.get("span", [0.0]*6),
                    "type_distribution": g.get("type_distribution", {}),
                    "part_count": g.get("part_count", 0),
                },
                "label": g.get("classification", "nominal"),
                "gravity_tensor": g.get("centroid", [0.0]*6),
            })
    return records


def _backup_adapter(path: str):
    if os.path.isdir(path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = f"{path}.{ts}"
        shutil.copytree(path, backup)
        return backup
    return None


def _train_adapter(
    label: str,
    records: list[dict],
    bnb_config,
    tokenizer,
    output_dir: str,
):
    label_records = [r for r in records if r.get("label") == label]
    if len(label_records) < MIN_SAMPLES_PER_CLASS:
        return False

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    dataset = Dataset.from_list(label_records)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=min(60, max(len(label_records) * 5, 10)),
        save_strategy="no",
        fp16=False,
        bf16=True,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    _backup_adapter(output_dir)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    trainer.model.save_pretrained(output_dir)
    return True


def train():
    records = load_dataset(DATASET_PATH)
    records += load_token_dataset_geometry()
    pairs = load_token_dataset_pairs()

    print(f"Dataset clássico: {len(records)} amostras")
    print(f"Geometrias token: {sum(1 for r in records if 'centroid' in r.get('telemetry',{}))}")
    print(f"Pares de partes: {len(pairs)}")
    print(f"Total para treino: {len(records) + len(pairs)}")

    combined = records + pairs
    if len(combined) < 5:
        return

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    os.makedirs(ADAPTER_BASE, exist_ok=True)

    for label in OBELISK_LABELS:
        out = os.path.join(ADAPTER_BASE, label)
        ok = _train_adapter(label, combined, bnb_config, tokenizer, out)
        status = "OK" if ok else "SKIP (poucas amostras)"
        print(f"  {label}: {status}")

    tokenizer.save_pretrained(os.path.join(ADAPTER_BASE, "tokenizer"))


if __name__ == "__main__":
    train()
