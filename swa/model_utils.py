import torch
import json
import gc
from transformers import AutoConfig, AutoModelForCausalLM, AutoModel
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

def remap_gemma_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Surgically remap state dict keys for Gemma 3 and other multimodal variants."""
    corrected: dict[str, torch.Tensor] = {}
    remapped_count = 0

    for key, value in state_dict.items():
        new_key = key

        # 1. Surgical removal of vision_model level (Gemma 3 requirement)
        if 'vision_model.' in new_key:
            new_key = new_key.replace('vision_model.', '')
            remapped_count += 1

        # 2. Normalization of top-level components to the 'model.*' namespace
        if new_key.startswith('language_model.model.'):
            new_key = 'model.language_model.' + new_key[len('language_model.model.'):]
            remapped_count += 1
        elif new_key.startswith('vision_tower.') and not new_key.startswith('model.'):
            new_key = 'model.' + new_key
            remapped_count += 1
        elif new_key.startswith('multi_modal_projector.') and not new_key.startswith('model.'):
            new_key = 'model.' + new_key
            remapped_count += 1

        corrected[new_key] = value

    return corrected, remapped_count

def load_multimodal_model_surgical(model_dir: str, device: torch.device, dtype: torch.dtype, model_class=AutoModelForCausalLM):
    """Load a multimodal model (like Gemma 3) with surgical remapping and materialization."""
    print(f"SWA Loading: Surgical materialization for {model_dir} on {device}")
    
    config = AutoConfig.from_pretrained(model_dir)
    
    # Identify safetensors files
    import os
    files = sorted([f for f in os.listdir(model_dir) if f.endswith('.safetensors')])
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    # Initialize empty model
    with init_empty_weights():
        model = model_class.from_config(config, torch_dtype=dtype)

    all_params = dict(model.named_parameters())
    all_buffers = dict(model.named_buffers())
    
    # Load and remap keys from all checkpoint shards
    from safetensors.torch import load_file
    
    for f in files:
        shard_path = os.path.join(model_dir, f)
        sd = load_file(shard_path)
        corrected, _ = remap_gemma_state_dict(sd)
        
        # Assign to model
        for name in corrected:
            if name in all_params:
                set_module_tensor_to_device(model, name, device=device, value=corrected[name], dtype=dtype)
            elif name in all_buffers:
                set_module_tensor_to_device(model, name, device=device, value=corrected[name], dtype=dtype)
        
        del sd
        del corrected
        gc.collect()

    # Materialize missing weights with zeros
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            zero_data = torch.zeros(param.shape, dtype=dtype, device=device)
            set_module_tensor_to_device(model, name, device=device, value=zero_data, dtype=dtype)

    for name, buf in model.named_buffers():
        if buf.device.type == 'meta':
            zero_data = torch.zeros(buf.shape, dtype=dtype, device=device)
            set_module_tensor_to_device(model, name, device=device, value=zero_data, dtype=dtype)

    model.eval()
    return model
