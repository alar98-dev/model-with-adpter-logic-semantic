#!/usr/bin/env python3
"""Benchmark SWA multicore performance."""

import sys
import os
import time
import torch
from swa.gemma_integration import GemmaGenerator
from swa.hw_eventual_state import HWEventualState

def waiting_for_organic_inflation(generator, timeout=300):
    """
    Implementa a Espera Orgânica: Monitora a inflação dos pesos
    no vácuo antes de liberar o benchmark.
    """
    start_time = time.time()
    print("⏳ Iniciando inflação dos pulmões (Pesos do Modelo)...")

    while not getattr(generator, '_model_loaded', False):
        elapsed = time.time() - start_time

        # Log de status para acompanhar a inflação
        if int(elapsed) % 10 == 0:
            print(f"   [T+{int(elapsed)}s] Sincronizando vetores no /dev/sde...")

        if elapsed > timeout:
            raise RuntimeError(f"⚠️ Colapso de Massa: O modelo demorou mais de {timeout}s para inflar.")

        # Pequena pausa para não sobrecarregar o núcleo de controle
        time.sleep(1.0)

    print(f"✅ Sistema Inflado! Massa vetorial pronta em {int(time.time() - start_time)}s.")
    return True

def benchmark_generation(generator, prompt="Hello world", num_runs=3):
    """Benchmark generation performance."""
    print(f"🏃 Benchmarking generation: '{prompt}'")

    times = []
    for i in range(num_runs):
        start_time = time.time()
        response = generator.generate(prompt, max_new_tokens=50)
        end_time = time.time()

        elapsed = end_time - start_time
        tokens_generated = len(response.split())
        tps = tokens_generated / elapsed

        print(".2f")
        times.append(tps)

    avg_tps = sum(times) / len(times)
    print(".2f")
    return avg_tps

def benchmark_organic_threading(hw_state, generator):
    """Benchmark organic threading performance."""
    print("🧬 Benchmarking organic threading...")

    # Start organic threading
    hw_state.start_organic_threading(generator)
    time.sleep(1)  # Let it stabilize

    # Measure bias update frequency
    start_time = time.time()
    updates = 0
    for _ in range(100):  # 100 rapid checks
        bias = generator.get_organic_bias_realtime()
        if bias is not None:
            updates += 1
        time.sleep(0.001)  # 1ms intervals

    end_time = time.time()
    update_rate = updates / (end_time - start_time)
    print(".1f")

    return update_rate

def main():
    """Run comprehensive SWA performance benchmark."""
    print("🚀 SWA Multicore Performance Benchmark")
    print("=" * 50)

    # CPU information
    import psutil
    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)
    print(f"🧠 CPU: {physical_cores} physical, {logical_cores} logical cores")

    # PyTorch threading info
    intra_threads = torch.get_num_threads()
    inter_threads = torch.get_num_interop_threads()
    print(f"⚡ PyTorch: intra_op={intra_threads}, inter_op={inter_threads}")

    print("\n📊 Benchmarking SWA Components...")

    # Test HW Eventual State
    hw_state = HWEventualState(vocab_size=10000, hidden_size=2560)
    print("✅ HW Eventual State initialized")

    # Test Gemma Generator with multicore
    print("\n🔧 Initializing Gemma Generator...")
    try:
        print("🔧 About to create GemmaGenerator...")
        generator = GemmaGenerator("models/gemma-3-4-b", device="cpu")
        print("✅ Gemma Generator initialized with multicore optimization")

        # Implement Organic Waiting - monitor model inflation
        model_loaded = waiting_for_organic_inflation(generator, timeout=120)  # 2 minutes for 4B params

        # Benchmark generation
        gen_tps = benchmark_generation(generator)
        print(".2f")

        # Benchmark organic threading
        thread_rate = benchmark_organic_threading(hw_state, generator)
        print(".1f")

        # Overall assessment
        print("\n🎯 Performance Assessment:")
        if gen_tps > 10:
            print("✅ Excellent: High-throughput generation achieved")
        elif gen_tps > 5:
            print("✅ Good: Solid multicore performance")
        else:
            print("⚠️  Needs optimization: Consider further threading adjustments")

        if thread_rate > 50:
            print("✅ Real-time: Organic threading responding in real-time")
        else:
            print("⚠️  Threading: May need optimization for real-time response")

        print("\n🎉 SWA Multicore Benchmark Complete!")
        print("💡 Tip: Run with 'sudo' for hardware acceleration and true organic performance")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()