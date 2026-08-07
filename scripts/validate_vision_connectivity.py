#!/usr/bin/env python3
"""Validate vision → latent space connectivity for SWA.

Checks shape consistency and embedding norms between vision tower and SWA latent space.
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

def _remap_state_dict_keys(state_dict):
    corrected = {}
    for key, value in state_dict.items():
        new_key = key
        if 'vision_tower' in key:
            key = key.replace('model.vision_tower.vision_model.', '')
            key = key.replace('vision_tower.vision_model.', '')
            key = key.replace('vision_model.', '')
            if 'encoder.layers.' in key:
                new_key = f'vision_tower.{key}'
            elif 'encoder.' in key and 'layers' not in key:
                key = key.replace('encoder.', 'encoder.layers.')
                new_key = f'vision_tower.{key}'
            else:
                new_key = f'vision_tower.{key}'
        elif 'multi_modal_projector' in key:
            parts = key.split('multi_modal_projector.')
            new_key = 'model.multi_modal_projector.' + parts[-1]
        elif 'language_model.model.' in key:
            new_key = 'model.language_model.' + key.split('language_model.model.')[-1]
        corrected[new_key] = value
    return corrected

def load_model(model_dir, device):
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir)
    safetensors_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.safetensors')]
    merged = {}
    for pth in safetensors_files:
        sd = load_file(pth)
        merged.update(sd)
    corrected = _remap_state_dict_keys(merged)
    corrected = {k: v.to(dtype=torch.bfloat16) for k, v in corrected.items()}

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    missing, unexpected = model.load_state_dict(corrected, strict=False, assign=True)
    if str(device) != 'cpu':
        model = model.to(device)
    model.eval()
    return model

def main():
    model_dir = "models/gemma-3-4-b"
    device = "cpu"

    print("Loading Gemma 3 model with vision remap...")
    from transformers import AutoModelForCausalLM
    model = load_model(model_dir, device)

    # Create dummy vision embeddings (batch, num_patches, vision_hidden)
    batch_size = 1
    num_patches = 4096  # (896/14)^2
    vision_hidden = 1152

    dummy_vision_embeds = torch.randn(batch_size, num_patches, vision_hidden, dtype=torch.bfloat16, device=device)
    print(f"Dummy vision embeds shape: {dummy_vision_embeds.shape}")

    with torch.no_grad():
        # Check if multi_modal_projector exists
        if hasattr(model, 'multi_modal_projector'):
            projector = model.multi_modal_projector
        elif hasattr(model, 'model') and hasattr(model.model, 'multi_modal_projector'):
            projector = model.model.multi_modal_projector
        else:
            print("multi_modal_projector not found")
            # List all attributes
            attrs = [attr for attr in dir(model) if 'projector' in attr.lower()]
            print(f"Projector-like attrs: {attrs}")
            # Check model.model
            if hasattr(model, 'model'):
                attrs2 = [attr for attr in dir(model.model) if 'projector' in attr.lower()]
                print(f"model.model projector attrs: {attrs2}")
            return 1

        # Project to text space
        projected_embeds = projector(dummy_vision_embeds)
        print(f"Projected embeds shape: {projected_embeds.shape}")

        # Get text embeddings
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        text_inputs = tokenizer("test", return_tensors="pt").to(device)
        text_embeds = model.get_input_embeddings()(text_inputs['input_ids'])
        print(f"Text embeds shape: {text_embeds.shape}")

        # Shapes
        vision_hidden_size = dummy_vision_embeds.shape[-1]
        projected_hidden_size = projected_embeds.shape[-1]
        text_hidden_size = text_embeds.shape[-1]

        print(f"Vision hidden size: {vision_hidden_size}")
        print(f"Projected hidden size: {projected_hidden_size}")
        print(f"Text hidden size: {text_hidden_size}")

        # Norms
        vision_norm = torch.norm(dummy_vision_embeds.float()).item()
        projected_norm = torch.norm(projected_embeds.float()).item()
        text_norm = torch.norm(text_embeds.float()).item()

        print(f"Vision embeds norm: {vision_norm:.4f}")
        print(f"Projected embeds norm: {projected_norm:.4f}")
        print(f"Text embeds norm: {text_norm:.4f}")

        connectivity_ok = projected_hidden_size == text_hidden_size == 2560
        norms_stable = all(5 <= n <= 80 for n in [vision_norm, projected_norm, text_norm])

        report = {
            "test": "vision_latent_connectivity",
            "shapes": {
                "vision_embeds": list(dummy_vision_embeds.shape),
                "projected_embeds": list(projected_embeds.shape),
                "text_embeds": list(text_embeds.shape),
            },
            "hidden_sizes": {
                "vision": vision_hidden_size,
                "projected": projected_hidden_size,
                "text": text_hidden_size,
            },
            "norms": {
                "vision": vision_norm,
                "projected": projected_norm,
                "text": text_norm,
            },
            "connectivity_ok": connectivity_ok,
            "norms_stable": norms_stable,
            "overall_success": connectivity_ok and norms_stable,
        }

    print(json.dumps(report, indent=2))
    return 0 if report["overall_success"] else 1

if __name__ == "__main__":
    exit(main())