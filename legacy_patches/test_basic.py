#!/usr/bin/env python3
"""Test basic model loading and inference without steering."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swa.gemma_integration import GemmaGenerator
from swa.tokenizer_manager import TokenizerManager

def test_basic_inference():
    """Test basic model loading and inference."""
    print("🧪 Testing basic model inference...")

    try:
        # Load tokenizer
        tokenizer_manager = TokenizerManager()
        tokenizer = tokenizer_manager.load("models/gemma-3-4-b")
        print("✅ Tokenizer loaded")

        # Load model
        generator = GemmaGenerator("models/gemma-3-4-b", device="cpu", tokenizer=tokenizer)
        print("✅ Model loaded")

        # Test basic generation
        prompt = "Hello world"
        response = generator.generate(prompt, max_new_tokens=5)
        print(f"✅ Basic generation successful: {response}")

        print("🎉 Basic inference test passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_inference()