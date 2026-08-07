#!/usr/bin/env python3
"""Simple test to check if AutoModelForCausalLM can load Gemma 3."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔧 Testing basic AutoModelForCausalLM import...")
    from transformers import AutoModelForCausalLM
    print("✅ AutoModelForCausalLM imported")

    print("🔧 Testing model loading...")
    model = AutoModelForCausalLM.from_pretrained(
        "models/gemma-3-4-b",
        torch_dtype="auto",
        low_cpu_mem_usage=True
    )
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
    print(f"📊 Config: {model.config}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()