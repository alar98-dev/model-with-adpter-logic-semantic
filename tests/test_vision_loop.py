# Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
"""PyTest suite to validate real-time vision processing in the SWA system."""

import os
import gc
import pytest
import torch
import numpy as np
from PIL import Image

from swa.gemma_integration import GemmaGenerator
from swa.depth_analyzer import (
    SpatialSemanticAnalyzer,
    ImageBufferMonitor,
    create_spatial_semantic_analyzer_from_gemma_config
)
from swa.world_state_engine import SpaceBiasProcessor
from swa.hardware_state import HardwareStateManager

MODEL_DIR = "/services/model-with-adpter-logic-semantic/models/gemma-3-4-b"

def test_hardware_mapping():
    """Verify that reading state from the /dev/sde loop device operates successfully."""
    # Check that /dev/sde exist
    assert os.path.exists("/dev/sde"), "Device /dev/sde does not exist. Ensure loop device setup is run."
    
    try:
        # Open in read binary mode
        with open("/dev/sde", "rb") as f:
            # Read first few bytes (can be empty due to loop device virtual size on DrvFs)
            header = f.read(10)
            assert isinstance(header, bytes)
            
        print("✅ Raw /dev/sde read validation completed successfully.")
    except PermissionError:
        pytest.fail("Permission denied accessing /dev/sde. Check device permissions (chmod 666 /dev/loop0).")
    except Exception as e:
        pytest.fail(f"Failed to access /dev/sde: {e}")


def test_hardware_state_manager():
    """Verify that HardwareStateManager initializes and handles state reading/writing."""
    manager = HardwareStateManager(device_path="/dev/sde")
    assert manager is not None
    
    # Test diagnostics
    diags = manager.get_organic_diagnostics()
    assert isinstance(diags, dict)
    assert "hardware_mode" in diags
    
    # Since we successfully attached SWA_ROOT.vhdx, hardware_mode should be True
    assert diags["hardware_mode"] is True


def test_logits_penalty_constraint():
    """Verify that SpaceBiasProcessor applies logits penalties correctly."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    processor = SpaceBiasProcessor(tokenizer=tokenizer, dtype=torch.bfloat16)
    assert processor is not None
    
    # Let's compute spatial bias for navigation 'north' and obstacle 'north' (blocked front)
    bias = processor.apply_spatial_bias(navigation_vector='north', obstacles=['north'])
    
    # Check that navigation direction is boosted
    north_tokens = processor.token_id_mappings['north']
    for token_id in north_tokens:
        if token_id < len(bias):
            # For north, we boosted (+2.0) and penalized (-8.0) so net change should be negative (penalized)
            assert bias[token_id] < 0, f"Token {token_id} for north should be penalized but got {bias[token_id]}"
            
    # Check that recovery directions (backward) are boosted
    backward_tokens = processor.token_id_mappings['backward']
    for token_id in backward_tokens:
        if token_id < len(bias):
            assert bias[token_id] > 0, f"Token {token_id} for backward should be boosted but got {bias[token_id]}"


def test_spatial_semantic_analyzer():
    """Test SpatialSemanticAnalyzer forward pass, shape verification, and obstacle detection."""
    # Gemma 3 vision hidden size is 1152
    analyzer = SpatialSemanticAnalyzer(vision_hidden_size=1152, spatial_resolution=(16, 16), device="cpu")
    assert analyzer is not None
    
    # Simulate vision embeddings of shape (batch_size=1, seq_len=256, hidden_size=1152)
    dummy_embeddings = torch.randn(1, 256, 1152, dtype=torch.float32)
    
    # Forward pass
    obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability = analyzer(dummy_embeddings)
    
    assert obstacle_likelihood_front.shape == (1, 1)
    assert vertical_semantic_planes.shape == (1, 256)
    assert spatial_navigability.shape == (1, 16, 16)
    
    # Force a frontal obstacle by setting likelihood to 0.9
    obstacle_likelihood_front[0, 0] = 0.95
    
    # Check obstacle detection
    obstacles = analyzer.detect_obstacles(obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability)
    assert len(obstacles) > 0
    assert any(o.direction == 'front' for o in obstacles)
    
    # Verify fused bias
    manual_bias = {'front': 0.8, 'left': 0.0, 'right': 0.0, 'backward': 0.0}
    fused = analyzer.fuse_multimodal_bias(manual_bias, obstacles)
    
    # Front should be penalized, backward/left/right boosted
    assert fused['front'] < manual_bias['front']
    assert fused['backward'] > 0.0


def test_real_vision_and_tower_loop():
    """Run model loading and verify the real processor and vision tower operations in a closed loop."""
    device = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
    print(f"Loading GemmaGenerator on device: {device}")
    
    # Force bfloat16 to avoid OOM
    generator = GemmaGenerator(model_dir=MODEL_DIR, device=device, dtype=torch.bfloat16)
    
    try:
        assert generator.lm is not None
        assert generator.vision_tower is not None
        assert generator.processor is not None
        
        # 1. Create a dummy PIL image of size (896, 896) matching Gemma 3 preprocessor_config
        img = Image.new("RGB", (896, 896), color="blue")
        
        # 2. Use image_processor directly (bypass Gemma3Processor boi_token validation)
        #    Gemma3Processor requires boi_token='<start_of_image>' in text when images given.
        #    For pure vision-tower extraction, image_processor alone suffices.
        pixel_values = generator.processor.image_processor(images=img, return_tensors="pt")["pixel_values"]
        # Match device and dtype
        pixel_values = pixel_values.to(device=generator.device, dtype=generator.dtype)
        
        # 3. Feed the pixel values to the vision tower to extract real embeddings
        with torch.no_grad():
            vision_outputs = generator.vision_tower(pixel_values)
            
            if hasattr(vision_outputs, "last_hidden_state"):
                embeddings = vision_outputs.last_hidden_state
            else:
                embeddings = vision_outputs[0]
                
        # Raw SiglipVisionModel output: (1, 4096, 1152)
        # 896×896px image @ 14×14 patch_size → 64×64 = 4096 spatial patches
        # The multi_modal_projector later compresses 4096 → 256 language tokens.
        assert embeddings.shape[0] == 1, "Batch dimension should be 1"
        assert embeddings.shape[2] == 1152, f"Vision hidden size should be 1152, got {embeddings.shape[2]}"
        num_patches = embeddings.shape[1]  # 4096 for 896×896 @ patch 14
        print(f"✅ Extracted real vision embeddings. Shape: {embeddings.shape}  (patches={num_patches})")
        
        # Pool 4096 patches → 256 tokens to feed SpatialSemanticAnalyzer
        import torch.nn.functional as F
        # Reshape to (1, 256, 16, hidden) then mean-pool over last dim group
        stride = num_patches // 256  # 16
        emb_cpu = embeddings.cpu().float()  # (1, 4096, 1152)
        # Use adaptive average pooling along sequence dimension
        emb_pooled = emb_cpu.permute(0, 2, 1)  # (1, 1152, 4096)
        emb_pooled = F.adaptive_avg_pool1d(emb_pooled, 256)  # (1, 1152, 256)
        emb_pooled = emb_pooled.permute(0, 2, 1)  # (1, 256, 1152)
        print(f"✅ Pooled vision embeddings to SpatialSemanticAnalyzer input shape: {emb_pooled.shape}")
        
        # 4. Initialize SpatialSemanticAnalyzer with Gemma 3's vision hidden size (1152)
        analyzer = create_spatial_semantic_analyzer_from_gemma_config(generator.lm.config, device="cpu")
        assert analyzer.vision_hidden_size == 1152, f"Expected hidden size 1152, got {analyzer.vision_hidden_size}"
        
        # 5. Feed the pooled embeddings to analyzer to verify navigability maps
        obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability = analyzer(emb_pooled)
        
        assert obstacle_likelihood_front.shape == (1, 1), f"obstacle_likelihood_front: {obstacle_likelihood_front.shape}"
        # vertical_semantic_planes: (batch, seq_len) — one semantic score per vision token
        assert vertical_semantic_planes.shape[0] == 1, "Batch dim should be 1"
        assert vertical_semantic_planes.shape[-1] == 256, f"Expected 256 tokens, got {vertical_semantic_planes.shape}"
        nav_h, nav_w = analyzer.spatial_resolution
        assert spatial_navigability.shape == (1, nav_h, nav_w), \
            f"spatial_navigability: {spatial_navigability.shape}, expected (1, {nav_h}, {nav_w})"
        print("✅ Spatial navigability maps constructed successfully from real vision embeddings.")
        
    finally:
        # Explicit memory cleanup to prevent any OOM or leftover XPU memory
        del generator
        gc.collect()
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.empty_cache()
