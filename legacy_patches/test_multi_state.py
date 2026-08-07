#!/usr/bin/env python3
"""Test multi-state simultaneous access and navigation map."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swa.hardware_state import HardwareStateManager
from swa.hw_eventual_state import HWEventualState
import torch

def test_multi_state_system():
    """Test multi-state simultaneous access with navigation map."""
    print("🗺️ Testing SWA Multi-State Simultaneous Access...")

    # Initialize hardware manager
    hw_manager = HardwareStateManager()
    print("✅ Hardware manager initialized")

    # Create test states
    reality_state = torch.randn(2560, dtype=torch.float32)
    prediction_state = reality_state + 0.1 * torch.randn(2560, dtype=torch.float32)
    navigation_map = torch.ones(2560, dtype=torch.float32) * 0.5  # Neutral map

    # Write states
    hw_manager.write_state("reality", reality_state)
    hw_manager.write_state("prediction", prediction_state)
    hw_manager.write_navigation_map(navigation_map)
    print("✅ States written to hardware")

    # Test parallel reading
    states = hw_manager.get_parallel_states(["reality", "prediction", "navigation_map"])
    print(f"✅ Parallel states read: {len(states)} states")

    # Test multi-state interference
    interference = hw_manager.get_multi_state_interference()
    print(f"✅ Multi-state interference computed: {interference.shape}")

    # Test HW Eventual State integration
    hw_state = HWEventualState(vocab_size=10000, hidden_size=2560)
    hw_state.update_navigation_map(navigation_map)

    retrieved_map = hw_state.get_navigation_map()
    if retrieved_map is not None:
        diff = torch.abs(retrieved_map - navigation_map).mean().item()
        print(".6f")

    # Test organic bias with multi-state
    bias = hw_state.get_organic_attention_bias()
    print(f"✅ Multi-state organic bias: {bias.shape}")

    print("🎉 Multi-state system test successful!")

if __name__ == "__main__":
    test_multi_state_system()