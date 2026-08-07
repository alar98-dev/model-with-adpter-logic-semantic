"""OODA Observe phase — captures system state and entropy."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from swa.ooda.interfaces import AbstractState


class Observer:
    """Captures the current cognitive and physical state for the OODA cycle."""

    def __init__(self, state: AbstractState) -> None:
        self._state = state

    def observe(
        self,
        debian_physics_path: str = "/tmp/debian_physics.json",
    ) -> ObservedState:
        """Collect all available sensor data into an ``ObservedState``."""
        vectors = self._state.vectors if hasattr(self._state, "vectors") else {}
        obstacles = self._state.obstacles if hasattr(self._state, "obstacles") else []

        # Entropy
        entropy = self._resolve_entropy()
        heat, mass, resonance = self._read_debian_physics(debian_physics_path)
        entropy += (heat / 50.0) + (mass / 100.0) + (resonance / 200.0)

        # Temporal inertia / viscosity
        temporal_inertia: Dict[str, float] = getattr(
            self._state, "temporal_inertia", {}
        )
        fatigue_str = ", ".join(
            f"{k}:{v:.2f}" for k, v in temporal_inertia.items() if v > 0.1
        ) or "Nenhuma"

        obs_str = ", ".join(
            f"{k}:{v:.2f}" for k, v in vectors.items()
        ) or "none:0.0"

        return ObservedState(
            vectors=vectors,
            obstacles=obstacles,
            entropy=entropy,
            fatigue=fatigue_str,
            obs_str=obs_str,
            heat=heat,
            mass=mass,
            resonance=resonance,
            temporal_inertia=temporal_inertia,
        )

    def _resolve_entropy(self) -> float:
        if hasattr(self._state, "organic_entropy"):
            return self._state.organic_entropy
        metrics = getattr(self._state, "runtime_metrics", None)
        if metrics is not None and isinstance(metrics, dict):
            return metrics.get("entropy", 0.0)
        return 0.0

    def _read_debian_physics(
        self, path: str
    ) -> Tuple[float, float, float]:
        heat = mass = resonance = 0.0
        if not os.path.exists(path):
            return heat, mass, resonance
        try:
            with open(path) as f:
                physics = json.load(f)
            heat = physics.get("kinetic_energy", {}).get("global_heat", 0)
            mass = physics.get("gravitational_density", {}).get("mass_pressure", 0)
            inward = physics.get("string_resonance", {}).get("inward_pull", 0)
            outward = physics.get("string_resonance", {}).get("outward_pull", 0)
            resonance = inward + outward
        except Exception:
            pass
        return float(heat), float(mass), float(resonance)


from dataclasses import dataclass, field


@dataclass
class ObservedState:
    vectors: Dict[str, float] = field(default_factory=dict)
    obstacles: List[Dict[str, Any]] = field(default_factory=list)
    entropy: float = 0.0
    fatigue: str = "Nenhuma"
    obs_str: str = "none:0.0"
    heat: float = 0.0
    mass: float = 0.0
    resonance: float = 0.0
    temporal_inertia: Dict[str, float] = field(default_factory=dict)
    visual_energy: float = 0.0
    biological_entropy: float = 0.0
