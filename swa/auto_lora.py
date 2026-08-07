import os
import re
import json
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

# XPU Optimization Fallback
try:
    import intel_extension_for_pytorch as ipex
    DEVICE = "xpu"
except ImportError:
    DEVICE = "cpu"

def parse_historian_logs():
    log_file = "/services/model-with-adpter-logic-semantic/swa_lora_dataset.jsonl"
    dataset_records = []
    
    if not os.path.exists(log_file):
        return dataset_records
        
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if all(k in data for k in ["user_context", "model_response", "status"]):
                    # Treinar apenas comandos que deram certo (Reforço Positivo Físico)
                    if data["status"] == "success":
                        user_ctx = data["user_context"].strip()
                        model_resp = data["model_response"].strip()
                        
                        # Formatação nativa do Gemma Instruct
                        text = f"<start_of_turn>user\n{user_ctx}<end_of_turn>\n<start_of_turn>model\n{model_resp}<end_of_turn>\n"
                        dataset_records.append({"text": text})
                    elif data["status"] == "failed":
                        # Se falhou, ensinamos o modelo a pedir desculpas em vez de rodar o código ruim
                        user_ctx = data["user_context"].strip()
                        text = f"<start_of_turn>user\n{user_ctx}<end_of_turn>\n<start_of_turn>model\nSinto que a sintaxe motora que eu planejava executar falharia. Preciso repensar a abordagem física.<end_of_turn>\n"
                        dataset_records.append({"text": text})
            except Exception:
                continue
            
    return dataset_records

def get_dynamic_hyperparameters(num_records: int):
    # Base params
    r = 8
    lora_alpha = 16
    learning_rate = 2e-4
    max_seq_length = 256
    per_device_train_batch_size = 2
    gradient_accumulation_steps = 2
    
    # Adapt to RAM constraints
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        avail_gb = mem.available / (1024**3)
    except Exception:
        total_gb = 16.0
        avail_gb = 8.0

    # If memory is very low (e.g. less than 8GB total or 3GB available), shrink sequence length and batch size
    if total_gb < 8.0 or avail_gb < 3.0:
        max_seq_length = 128
        per_device_train_batch_size = 1
        gradient_accumulation_steps = 4
        print(f"⚠️ Memória física restrita ({avail_gb:.2f}GB livres). Vetorizando limites mínimos de treinamento: seq_len={max_seq_length}, batch={per_device_train_batch_size}")
    elif total_gb > 32.0 and avail_gb > 16.0:
        # High performance system
        max_seq_length = 512
        per_device_train_batch_size = 4
        gradient_accumulation_steps = 1
        print(f"🚀 Sistema de alto desempenho detectado ({avail_gb:.2f}GB livres). Expandindo limites: seq_len={max_seq_length}, batch={per_device_train_batch_size}")

    # Scale learning rate, rank and alpha based on data size
    if num_records > 100:
        r = 16
        lora_alpha = 32
        learning_rate = 1e-4
    elif num_records > 50:
        r = 12
        lora_alpha = 24
        learning_rate = 1.5e-4
        
    return {
        "r": r,
        "lora_alpha": lora_alpha,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps
    }

def train_lora_adapter():
    print(f"🚀 Iniciando Auto-Learning SWA (LoRA) no dispositivo: {DEVICE}")
    records = parse_historian_logs()
    
    if len(records) < 10:
        print(f"⚠️ Poucos dados válidos para treino ({len(records)}). Acumule pelo menos 10 ciclos válidos no Historian primeiro.")
        return
        
    print(f"📊 {len(records)} ciclos de aprendizado orgânico perfeitos encontrados!")
    
    # Obter hiperparâmetros dinâmicos baseados na carga física e volume de dados
    params = get_dynamic_hyperparameters(len(records))
    
    dataset = Dataset.from_list(records)
    
    model_id = "/services/model-with-adpter-logic-semantic/models/gemma-3-270m" # Fallback if local path not mounted
    # Assuming local model exists or will be downloaded
    
    print("⏳ Carregando Mente Artificial Base...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "xpu" else torch.float32,
    )
    
    if DEVICE == "xpu":
        model = model.to("xpu")
    
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(0, 14)),
    )
    
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()
    
    # Treino dinâmico focado no estilo e telemetria física
    training_args = SFTConfig(
        output_dir="/services/model-with-adpter-logic-semantic/models/swa-lora-adapter",
        per_device_train_batch_size=params["per_device_train_batch_size"],
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        learning_rate=params["learning_rate"],
        logging_steps=5,
        max_steps=50 if len(records) > 50 else len(records),
        save_strategy="no",
        fp16=False,
        bf16=(DEVICE == "xpu"),
        dataset_text_field="text",
        max_length=params["max_seq_length"]
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    print("🧠 Iniciando infusão de sinapses estruturais...")
    trainer.train()
    
    adapter_path = "/services/model-with-adpter-logic-semantic/models/swa-lora-adapter"
    trainer.model.save_pretrained(adapter_path)
    print(f"✅ Nova intuição estrutural cristalizada e salva em: {adapter_path}")

if __name__ == "__main__":
    train_lora_adapter()
