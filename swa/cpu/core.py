from __future__ import annotations

import time
import random
from typing import Dict, Any, List
from threading import Lock
import psutil


class CPUPhysicalCore:
    """Models a real physical CPU core with strict structural and behavioral boundaries.

    Elemental structure:
    - Class: CPUPhysicalCore
    - Attributes: characteristics (core_id, socket, logical processors, caches, execution counts)
    - Methods: instructions and actions (execute_instruction, run_diagnostics)
    - @property: eventual process cycles (frequency, usage_percentage, temperature, eventual_cycle_state)
    """

    def __init__(
        self,
        core_id: int,
        physical_id: int,
        logical_processors: List[int],
        l1_cache: str = "64 KB",
        l2_cache: str = "512 KB",
        l3_cache: str = "12288 KB",
    ):
        # 1. ATTRIBUTES (Core Characteristics & backing fields)
        self.core_id: int = core_id
        self.physical_id: int = physical_id
        self.logical_processors: List[int] = logical_processors
        self.l1_cache: str = l1_cache
        self.l2_cache: str = l2_cache
        self.l3_cache: str = l3_cache

        # Telemetry backing states
        self._last_telemetry_scan: float = time.time()
        self._total_cycles_executed: int = 0
        self._simulated_temp_delta: float = 0.0
        self._lock: Lock = Lock()

    # 2. METHODS (Core Instructions)
    def execute_instruction(self, instruction_type: str, weight: float = 1.0) -> bool:
        """Instruction Execution Method (Operational Control).

        Simulates or tracks execution load directed to this physical core.
        Influences the thermal/load cycle properties eventually.
        """
        with self._lock:
            # Simulate instruction cycles
            cycles = int(1000 * weight)
            self._total_cycles_executed += cycles

            # Execute load changes thermal status eventually
            self._simulated_temp_delta = min(
                35.0, self._simulated_temp_delta + (0.05 * weight)
            )

            # Minimal back-off simulating cycle execution overhead
            time.sleep(0.0001 * min(weight, 10))
            return True

    def run_diagnostics(self) -> Dict[str, Any]:
        """Core Self-Diagnostics Method.

        Executes physical core health verification and returns status mapping.
        """
        with self._lock:
            usage = self.usage_percentage
            freq = self.frequency
            temp = self.temperature
            state = self.eventual_cycle_state

            health_status = "healthy"
            if temp > 80.0:
                health_status = "overheated"
            elif usage > 90.0:
                health_status = "overloaded"

            return {
                "core_id": self.core_id,
                "physical_id": self.physical_id,
                "status": health_status,
                "total_cycles_executed": self._total_cycles_executed,
                "metrics": {
                    "usage_percent": usage,
                    "frequency_mhz": freq,
                    "temperature_c": temp,
                    "eventual_state": state,
                },
            }

    def _decay_thermal_load(self) -> None:
        """Internal thermal dissipation simulation (eventual decay)."""
        elapsed = time.time() - self._last_telemetry_scan
        if elapsed > 0:
            # Heat dissipates naturally over time
            decay = elapsed * 2.0  # 2 degrees per second decay
            self._simulated_temp_delta = max(0.0, self._simulated_temp_delta - decay)

    # 3. PROPERTIES (@property - Eventual Process Cycles)
    @property
    def frequency(self) -> float:
        """Eventual process cycle property: Dynamic CPU Frequency.

        Reads system telemetry or returns active frequency.
        """
        try:
            freq_info = psutil.cpu_freq()
            if freq_info:
                return float(freq_info.current)
        except Exception:
            pass
        return 2400.0  # Baseline fallback

    @property
    def usage_percentage(self) -> float:
        """Eventual process cycle property: Core-Specific Workload.

        Calculated by scanning and averaging the load on its logical processors.
        """
        try:
            # Fetch load for all logical cpus in interval
            per_cpu_usage = psutil.cpu_percent(interval=None, percpu=True)
            if per_cpu_usage:
                # Average the usage across the logical cpus assigned to this physical core
                core_usages = [
                    per_cpu_usage[cpu_idx]
                    for cpu_idx in self.logical_processors
                    if cpu_idx < len(per_cpu_usage)
                ]
                if core_usages:
                    return sum(core_usages) / len(core_usages)
        except Exception:
            pass
        return 0.0

    @property
    def temperature(self) -> float:
        """Eventual process cycle property: Core Temperature telemetry.

        Reads system temperature where available, adding simulated instruction load shifts.
        """
        base_temp = 40.0
        try:
            # Try to get real CPU temp from psutil
            temps = psutil.sensors_temperatures()
            if "coretemp" in temps and len(temps["coretemp"]) > self.core_id:
                base_temp = temps["coretemp"][self.core_id].current
            elif "cpu_thermal" in temps and temps["cpu_thermal"]:
                base_temp = temps["cpu_thermal"][0].current
        except Exception:
            pass

        self._decay_thermal_load()
        self._last_telemetry_scan = time.time()

        return base_temp + self._simulated_temp_delta

    @property
    def eventual_cycle_state(self) -> str:
        """Eventual process cycle property: Nominal Control State.

        Classifies core execution status based on dynamic telemetry signals.
        """
        usage = self.usage_percentage
        temp = self.temperature

        if temp > 82.0:
            return "throttled"
        elif usage > 85.0:
            return "saturated"
        elif usage > 30.0:
            return "processing"
        else:
            return "latent_cycle"
