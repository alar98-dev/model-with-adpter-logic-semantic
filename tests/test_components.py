#!/usr/bin/env python3
"""Test SWA components without full Gemma model loading."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from swa.hw_eventual_state import HWEventualState
from swa.phase_shifter import PhaseShifter
from swa.hardware_state import HardwareStateManager

def test_swa_components():
    """Test SWA components individually."""
    print("🔧 Testing SWA components individually...")

    try:
        # Test Hardware State Manager
        hw_manager = HardwareStateManager()
        print("✅ Hardware State Manager initialized")

        # Test HW Eventual State
        hw_state = HWEventualState(vocab_size=10000, hidden_size=2560)
        print("✅ HW Eventual State initialized")

        # Test Phase Shifter
        phase_shifter = PhaseShifter(hw_manager)
        print("✅ Phase Shifter initialized")

        # Test organic properties
        entropy = hw_state.organic_entropy
        inertia = hw_state.organic_inertia
        bias = hw_state.get_organic_attention_bias()

        print(".4f")
        print(".4f")
        print(f"✅ Organic bias shape: {bias.shape}")

        # Test multi-state interference
        interference = hw_manager.get_multi_state_interference()
        print(f"✅ Multi-state interference shape: {interference.shape}")

        # Test phase injection logic
        should_inject = hw_state.should_inject_phase()
        print(f"✅ Phase injection check: {should_inject}")

        print("🎉 All SWA components test passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_swa_components()