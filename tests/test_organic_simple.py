#!/usr/bin/env python3
"""Test organic system without full model loading."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swa.hw_eventual_state import HWEventualState

def test_organic_system():
    """Test organic system components."""
    print("🧬 Testing SWA Organic System Components...")

    # Create HW eventual state
    hw_state = HWEventualState(vocab_size=10000, hidden_size=2560)
    print("✅ HW Eventual State created")

    # Test organic properties
    print(".4f")
    print(".4f")

    # Test bias generation
    bias = hw_state.get_organic_attention_bias()
    print(f"✅ Organic bias shape: {bias.shape}")

    # Test logit steering
    logit_bias = hw_state.get_logit_steering()
    if logit_bias is not None:
        print(f"✅ Logit steering shape: {logit_bias.shape}")
    else:
        print("⚠️ Logit steering returned None")

    # Test decision computation
    decision = hw_state.compute_swa_decision()
    print(f"✅ SWA decision shape: {decision.shape}")

    # Test phase injection check
    should_inject = hw_state.should_inject_phase()
    print(f"✅ Phase injection check: {should_inject}")

    print("🎉 Organic system components test successful!")

if __name__ == "__main__":
    test_organic_system()