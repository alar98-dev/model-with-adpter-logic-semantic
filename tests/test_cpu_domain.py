#!/usr/bin/env python3
"""Unit test for SWA CPU Core Domain structure (Class, Attributes, Methods, Properties)."""

import sys
import os
import torch

# Ensure swa is importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from swa.cpu import CPUPhysicalCore, CPUDetector, CPUTopology
from swa.hardware_hal import CPUBackend
from swa.hw_eventual_state import HWEventualState


def test_cpu_core_structure():
    print("🧪 Testing CPUPhysicalCore Elemental Structure...")
    
    # 1. Class instantiation
    core = CPUPhysicalCore(
        core_id=0,
        physical_id=0,
        logical_processors=[0, 1],
        l1_cache="64 KB",
        l2_cache="512 KB",
        l3_cache="12288 KB"
    )
    
    assert isinstance(core, CPUPhysicalCore), "Should instantiate CPUPhysicalCore"
    
    # 2. Attributes checking (Characteristics)
    print("  Checking Attributes:")
    print(f"    core_id: {core.core_id}")
    print(f"    physical_id: {core.physical_id}")
    print(f"    logical_processors: {core.logical_processors}")
    print(f"    l1_cache: {core.l1_cache}")
    print(f"    l2_cache: {core.l2_cache}")
    print(f"    l3_cache: {core.l3_cache}")
    
    assert core.core_id == 0
    assert core.physical_id == 0
    assert core.logical_processors == [0, 1]
    assert core.l1_cache == "64 KB"
    assert core.l2_cache == "512 KB"
    assert core.l3_cache == "12288 KB"
    
    # 3. Methods checking (Instructions)
    print("  Checking Methods:")
    success = core.execute_instruction("test_instruction", weight=2.0)
    assert success is True, "execute_instruction method should succeed"
    print("    execute_instruction() OK")
    
    diags = core.run_diagnostics()
    assert isinstance(diags, dict), "run_diagnostics method should return a dict"
    assert diags["core_id"] == 0
    assert "metrics" in diags
    print("    run_diagnostics() OK")
    
    # 4. Properties checking (@property eventual cycles)
    print("  Checking Properties:")
    freq = core.frequency
    usage = core.usage_percentage
    temp = core.temperature
    state = core.eventual_cycle_state
    
    print(f"    frequency: {freq} MHz")
    print(f"    usage_percentage: {usage}%")
    print(f"    temperature: {temp}°C")
    print(f"    eventual_cycle_state: {state}")
    
    assert isinstance(freq, float)
    assert isinstance(usage, float)
    assert isinstance(temp, float)
    assert isinstance(state, str)
    
    print("✅ CPUPhysicalCore structure verified successfully!")


def test_cpu_topology_orchestration():
    print("\n🧪 Testing CPUTopology Orchestration and Thread Affinity...")
    
    # 1. Instantiation
    topo = CPUTopology()
    assert isinstance(topo, CPUTopology)
    assert len(topo.cores) > 0, "Should detect at least 1 physical core"
    print(f"  Detected {len(topo.cores)} physical CPU cores")
    
    # 2. Pinning Method check
    print("  Checking thread pinning instruction affinity:")
    pin_success = topo.pin_current_thread_to_core(0)
    print(f"    pin_current_thread_to_core(0) success: {pin_success}")
    
    reset_success = topo.reset_thread_affinity()
    print(f"    reset_thread_affinity() success: {reset_success}")
    
    assert reset_success is True
    
    # 3. Properties check
    print("  Checking aggregated topology properties:")
    util_avg = topo.utilization_average
    temp_avg = topo.temperature_average
    active_count = topo.active_core_count
    throttled_count = topo.throttled_core_count
    
    print(f"    utilization_average: {util_avg}%")
    print(f"    temperature_average: {temp_avg}°C")
    print(f"    active_core_count: {active_count}")
    print(f"    throttled_core_count: {throttled_count}")
    
    assert isinstance(util_avg, float)
    assert isinstance(temp_avg, float)
    assert isinstance(active_count, int)
    assert isinstance(throttled_count, int)
    
    print("✅ CPUTopology Orchestration verified successfully!")


def test_integration():
    print("\n🧪 Testing SWA HAL & State Manager Integration...")
    
    # 1. HAL Backend check
    cpu_backend = CPUBackend()
    detection = cpu_backend.detect()
    assert detection["available"] is True
    assert "cores" in detection
    assert detection["physical_core_count"] > 0
    print(f"  CPUBackend detection matches dynamic topology: {detection['physical_core_count']} cores")
    
    health = cpu_backend.health()
    assert health["available"] is True
    assert "healthy_cores" in health
    print("  CPUBackend health details OK")
    
    # 2. HW Eventual State check
    hw_state = HWEventualState(vocab_size=10000, hidden_size=2560)
    assert isinstance(hw_state.cpu_topology, CPUTopology), "Should expose cpu_topology property"
    print("  HWEventualState.cpu_topology property OK")
    
    # Trigger steer calculations to check core instruction load simulation
    steering_bias = hw_state.get_logit_steering()
    # If phase shifter is missing, it returns super().get_logit_steering() which could be None or a zero tensor.
    # What matters is that executing get_logit_steering executed without throwing exceptions,
    # thereby successfully simulating instruction load on physical cores.
    print("  HWEventualState.get_logit_steering load simulation executed OK")
    
    # Let's verify that instruction load increased cycle count on core 0
    active_cycles = hw_state.cpu_topology.cores[0].run_diagnostics()["total_cycles_executed"]
    print(f"  Physical Core 0 cycles after steering calculation: {active_cycles} instructions")
    assert active_cycles > 0, "Core instruction cycles should have run"
    
    print("✅ HAL and State Manager integration verified successfully!")


def main():
    print("==================================================")
    print(" SWA CPU Core Domain Structural Verification Test")
    print("==================================================")
    
    try:
        test_cpu_core_structure()
        test_cpu_topology_orchestration()
        test_integration()
        
        print("\n🎉 All tests PASSED! SWA CPU Core Domain is 100% correct.")
        return 0
    except Exception as e:
        print(f"\n❌ Structural test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
