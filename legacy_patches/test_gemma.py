import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

MODEL_DIR = "models/gemma-3-4-b"

print("\n[LOADING TOKENIZER]\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True
)

print("\n[LOADING MODEL]\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("\n[MODEL LOADED]\n")

print("\n[MODEL CONFIG]\n")

print(model.config)

text_cfg = model.config.text_config

print("\n[TEXT CONFIG]\n")

print("hidden_size:", text_cfg.hidden_size)

print(
    "num_attention_heads:",
    text_cfg.num_attention_heads
)

print(
    "num_key_value_heads:",
    text_cfg.num_key_value_heads
)

head_dim = (
    text_cfg.hidden_size //
    text_cfg.num_attention_heads
)

print("head_dim:", head_dim)

emb = model.get_input_embeddings()

print("\n[EMBEDDINGS]\n")

print("embedding:", emb.weight.shape)

print(
    "lm_head:",
    model.lm_head.weight.shape
)

print(
    "tied:",
    emb.weight.data_ptr() ==
    model.lm_head.weight.data_ptr()
)

prompt = "Ola, como voce esta?"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)

print("\n[INPUT IDS]\n")

print(inputs["input_ids"].shape)

with torch.no_grad():

    out = model.generate(
        **inputs,
        max_new_tokens=20
    )

print("\n[GENERATED IDS]\n")

print(out.shape)

decoded = tokenizer.decode(
    out[0],
    skip_special_tokens=True
)

print("\n[OUTPUT]\n")

print(decoded)