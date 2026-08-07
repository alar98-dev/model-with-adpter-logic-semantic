from __future__ import annotations

import psutil
from typing import List, Dict, Any, Optional
from threading import Lock
from swa.cpu.detector import CPUDetector
from swa.cpu.core import CPUPhysicalCore


class CPUTopology:
    """Orchestrates and coordinates the physical CPU cores topology of the host.

    Provides a centralized point for:
    - Thread pinning (CPU affinity isolation)
    - Consolidated dynamic thermal and load telemetry
    - Real-time diagnostic reports
    """

    def __init__(self, detector: Optional[CPUDetector] = None):
        self._detector: CPUDetector = detector or CPUDetector()
        self._cores: List[CPUPhysicalCore] = []
        self._lock: Lock = Lock()

        self._initialize_topology()

    def _initialize_topology(self) -> None:
        """Discover and initialize physical cores list."""
        with self._lock:
            self._cores = self._detector.detect_cores()

    # 1. METHODS
    def get_core(self, core_id: int) -> Optional[CPUPhysicalCore]:
        """Fetch a specific physical core by its ID."""
        with self._lock:
            if 0 <= core_id < len(self._cores):
                return self._cores[core_id]
        return None

    def pin_current_thread_to_core(self, core_id: int) -> bool:
        """Physical instruction pinning method.

        Pins the current OS process affinity exclusively to the logical processors
        of the specified physical CPU core. Enables strict execution isolation.
        """
        core = self.get_core(core_id)
        if not core:
            return False

        try:
            # Set CPU affinity of current process to the core's logical processors
            proc = psutil.Process()
            proc.cpu_affinity(core.logical_processors)
            return True
        except Exception as e:
            print(f"Warning: Failed to set CPU affinity to physical core {core_id} ({e})")
            return False

    def reset_thread_affinity(self) -> bool:
        """Restores thread execution affinity to all available logical processors."""
        try:
            proc = psutil.Process()
            # Reset to allow all processors
            proc.cpu_affinity([])
            return True
        except Exception as e:
            print(f"Warning: Failed to reset CPU affinity ({e})")
            return False

    def run_diagnostics(self) -> List[Dict[str, Any]]:
        """Consolidated Diagnostics Method.

        Runs self-diagnostics across all physical cores.
        """
        with self._lock:
            return [core.run_diagnostics() for core in self._cores]

    # 2. PROPERTIES (@property - Eventual Multi-Core Telemetry Cycles)
    @property
    def cores(self) -> List[CPUPhysicalCore]:
        """Active physical core representation mapping."""
        with self._lock:
            return list(self._cores)

    @property
    def utilization_average(self) -> float:
        """Eventual cycle property: Average load across all physical cores."""
        with self._lock:
            if not self._cores:
                return 0.0
            return sum(core.usage_percentage for core in self._cores) / len(self._cores)

    @property
    def temperature_average(self) -> float:
        """Eventual cycle property: Average CPU temperature across all cores."""
        with self._lock:
            if not self._cores:
                return 0.0
            return sum(core.temperature for core in self._cores) / len(self._cores)

    @property
    def active_core_count(self) -> int:
        """Count of physical cores currently processing loads."""
        with self._lock:
            return sum(1 for core in self._cores if core.eventual_cycle_state != "latent_cycle")

    @property
    def throttled_core_count(self) -> int:
        """Count of physical cores currently experiencing thermal throttle status."""
        with self._lock:
            return sum(1 for core in self._cores if core.eventual_cycle_state == "throttled")
