from __future__ import annotations

import os
import re
from typing import Dict, List, Any, Tuple
import psutil
from swa.cpu.core import CPUPhysicalCore


class CPUDetector:
    """Discovers and maps the real physical core topology on the CPU.

    Divides discovery into two layers:
    - Main: Parse `/proc/cpuinfo` for detailed physical sockets, core IDs, and cache specs.
    - Fallback: Use `psutil` to construct logical core mapping when CPU info is inaccessible.
    """

    def __init__(self, cpuinfo_path: str = "/proc/cpuinfo"):
        self.cpuinfo_path = cpuinfo_path

    def detect_cores(self) -> List[CPUPhysicalCore]:
        """Detect and return list of instantiated CPUPhysicalCore objects."""
        if os.path.exists(self.cpuinfo_path):
            try:
                cores = self._parse_proc_cpuinfo()
                if cores:
                    return cores
            except Exception as e:
                print(f"Warning: Failed parsing {self.cpuinfo_path} ({e}). Using psutil fallback.")

        return self._detect_via_psutil()

    def _parse_proc_cpuinfo(self) -> List[CPUPhysicalCore]:
        """Parse Linux /proc/cpuinfo to build authentic hardware map."""
        processors: List[Dict[str, Any]] = []
        current_proc: Dict[str, Any] = {}

        # 1. Parse lines into structured processor list
        with open(self.cpuinfo_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_proc:
                        processors.append(current_proc)
                        current_proc = {}
                    continue

                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().lower()
                    val = val.strip()
                    current_proc[key] = val

            if current_proc:
                processors.append(current_proc)

        # 2. Group logical processors under unique (physical id, core id)
        # Socket mapping: (physical_id, core_id) -> list of logical processor IDs
        topology: Dict[Tuple[int, int], List[int]] = {}
        cache_specs: Dict[Tuple[int, int], str] = {}

        for proc in processors:
            proc_id_str = proc.get("processor")
            phys_id_str = proc.get("physical id")
            core_id_str = proc.get("core id")
            cache_str = proc.get("cache size", "12288 KB")

            if proc_id_str is not None:
                proc_id = int(proc_id_str)
                # Fallbacks if keys are not present
                phys_id = int(phys_id_str) if phys_id_str is not None else 0
                core_id = int(core_id_str) if core_id_str is not None else proc_id

                key = (phys_id, core_id)
                if key not in topology:
                    topology[key] = []
                topology[key].append(proc_id)
                cache_specs[key] = cache_str

        # If topology is empty, return empty list to trigger fallback
        if not topology:
            return []

        # 3. Instantiate physical core models
        physical_cores: List[CPUPhysicalCore] = []
        for idx, (key, logical_procs) in enumerate(sorted(topology.items())):
            phys_id, core_id = key
            l3_cache = cache_specs.get(key, "12288 KB")
            
            core_obj = CPUPhysicalCore(
                core_id=idx,  # Logical sequential core ID
                physical_id=phys_id,
                logical_processors=logical_procs,
                l1_cache="64 KB",
                l2_cache="512 KB",
                l3_cache=l3_cache
            )
            physical_cores.append(core_obj)

        return physical_cores

    def _detect_via_psutil(self) -> List[CPUPhysicalCore]:
        """Construct fallback topology using psutil metadata."""
        physical_count = psutil.cpu_count(logical=False) or 1
        logical_count = psutil.cpu_count(logical=True) or 1

        # Hyperthreading threads per core
        threads_per_core = max(1, logical_count // physical_count)

        physical_cores: List[CPUPhysicalCore] = []
        for i in range(physical_count):
            # Approximate which logical processors map to this physical core
            logical_procs = [
                i * threads_per_core + t
                for t in range(threads_per_core)
                if (i * threads_per_core + t) < logical_count
            ]

            core_obj = CPUPhysicalCore(
                core_id=i,
                physical_id=0,
                logical_processors=logical_procs,
                l1_cache="64 KB",
                l2_cache="512 KB",
                l3_cache="12288 KB"
            )
            physical_cores.append(core_obj)

        return physical_cores
