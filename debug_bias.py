#!/usr/bin/env python3
"""Debug the 8 vs 256 error by checking vocab size and bias shapes."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer
import torch

def debug_vocab_and_bias():
    """Debug vocabulary size and bias tensor shapes."""
    print("🔍 Debugging vocab size and bias shapes...")

    # Load tokenizer
    model_dir = "models/gemma-3-4-b"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    vocab_size = len(tokenizer)
    print(f"📚 Vocabulary size: {vocab_size}")

    # Test HW Eventual State bias generation
    from swa.hw_eventual_state import HWEventualState

    hw_state = HWEventualState(vocab_size=vocab_size, hidden_size=2560)
    print("✅ HW Eventual State created")

    # Check logit steering shape
    logit_bias = hw_state.get_logit_steering()
    if logit_bias is not None:
        print(f"🎯 Logit bias shape: {logit_bias.shape}")
        print(f"   Expected vocab size: {vocab_size}")
        print(f"   Match: {logit_bias.shape[0] == vocab_size}")
    else:
        print("❌ Logit bias is None")

    # Check attention bias shape
    attn_bias = hw_state.get_organic_attention_bias()
    print(f"🎯 Attention bias shape: {attn_bias.shape}")

    print("🎉 Debug complete!")

if __name__ == "__main__":
    debug_vocab_and_bias()