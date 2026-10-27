"""
ToroidalConfinement — Flux Confinement as Context Isolation

In real toroidal inductors, the magnetic flux Φ is confined INSIDE the
toroid, with zero field outside. This is the ideal isolation mechanism.

In computational terms, a toroidal confinement is a CONTEXT BOUNDARY:
  - Inside the toroid: computation happens freely (high coherence)
  - At the boundary: information is reflected (prevents leakage)
  - Outside the toroid: zero field (complete isolation)

The SWA-Aeon system has multiple contexts that must be isolated:
  - OODA loop internal state
  - External sensor inputs (subcortex, auditory)
  - User interaction (IPC bus)
  - Autonomous loops (orchestrator)

ToroidalConfinement creates MAGNETICALLY ISOLATED CONTEXTS where
interference between unrelated computations is suppressed.

The mathematical model:
  Φ = ∮ B · dA = μ₀ · N · I   (Ampère's law for toroid)
  B(r) = μ₀ · N · I / (2πr)    (field inside toroid)
  B(r) = 0                      (field outside toroid)
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ToroidalRegion:
    """A toroidally confined computational region."""
    name: str
    inner_radius: float
    outer_radius: float
    n_windings: int       # Number of "turns" — determines field strength
    current: float        # "Current" — activation level of this context
    permeability: float   # μ_r — how easily this context accepts flux
    stored_flux: float    # Φ — accumulated information
    coherence: float      # Phase coherence inside the toroid
    birth_time: float = 0.0
    last_update: float = 0.0


class ToroidalConfinement:
    """
    Manages multiple toroidally confined computational contexts.

    Each context is an isolated "bubble" where computation proceeds
    without interference from other contexts. The confinement is
    maintained by the toroidal geometry — flux cannot leak outside.

    When two contexts NEED to communicate, they do so through
    MUTUAL INDUCTANCE — a shared boundary where flux can couple
    in a controlled manner (like two adjacent toroids sharing a wall).
    """

    def __init__(self, max_regions: int = 16):
        self._regions: dict[str, ToroidalRegion] = {}
        self._max_regions = max_regions
        self._coupling_map: dict[tuple[str, str], float] = {}
        self._global_flux = 0.0
        self._timestamp = time.time()

    def create_region(
        self,
        name: str,
        inner_radius: float = 1.0,
        outer_radius: float = 5.0,
        n_windings: int = 100,
        permeability: float = 1.0,
    ) -> ToroidalRegion:
        """Create a new toroidally confined context."""
        if len(self._regions) >= self._max_regions:
            # Evict least active region
            self._evict_weakest()

        region = ToroidalRegion(
            name=name,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            n_windings=n_windings,
            current=0.0,
            permeability=permeability,
            stored_flux=0.0,
            coherence=1.0,
            birth_time=self._timestamp,
            last_update=self._timestamp,
        )
        self._regions[name] = region
        return region

    def activate_region(self, name: str, current: float):
        """Activate a region by injecting 'current' (activation level)."""
        if name in self._regions:
            region = self._regions[name]
            region.current = current
            region.last_update = self._timestamp

            # Compute confined flux: Φ = μ₀ · N · I
            mu_0 = 4 * math.pi * 1e-7  # vacuum permeability (normalized to 1.0)
            region.stored_flux = mu_0 * region.n_windings * current * region.permeability

            # Update global flux
            self._global_flux = sum(r.stored_flux for r in self._regions.values())

    def deactivate_region(self, name: str):
        """Deactivate a region — reduce current to zero."""
        if name in self._regions:
            self._regions[name].current = 0.0
            self._regions[name].last_update = self._timestamp

    def destroy_region(self, name: str):
        """Destroy a confined context completely."""
        self._regions.pop(name, None)
        # Remove coupling connections
        self._coupling_map = {
            k: v for k, v in self._coupling_map.items()
            if name not in k
        }

    def set_mutual_inductance(self, name_a: str, name_b: str, coupling: float):
        """
        Set mutual inductance between two regions — how much flux
        can leak from one to the other. coupling ∈ [0, 1]:
          0 = complete isolation
          1 = full coupling (equivalent to merging)
        """
        key = tuple(sorted([name_a, name_b]))
        self._coupling_map[key] = np.clip(coupling, 0, 1)

    def compute_field_at_point(
        self,
        name: str,
        r: float,
    ) -> float:
        """
        Compute the magnetic field B at distance r from center of a toroid.
        B(r) = μ₀·N·I / (2πr)  for inner_r < r < outer_r
        B(r) = 0                 otherwise
        """
        region = self._regions.get(name)
        if region is None:
            return 0.0

        if region.inner_radius <= r <= region.outer_radius:
            mu_0 = 1.0  # normalized
            circumference = 2 * math.pi * r
            if circumference > 1e-8:
                return mu_0 * region.n_windings * region.current / circumference
        return 0.0

    def field_profile(self, name: str, n_points: int = 100) -> list[tuple[float, float]]:
        """Get the radial field profile of a toroid."""
        region = self._regions.get(name)
        if region is None:
            return []

        r_min = region.inner_radius * 0.5
        r_max = region.outer_radius * 1.5
        points = np.linspace(r_min, r_max, n_points)

        return [(float(r), self.compute_field_at_point(name, float(r))) for r in points]

    def confinement_ratio(self, name: str) -> float:
        """
        How well is the flux confined?
        Ratio of flux inside the toroid to total generated flux.
        Ideal toroid: 1.0 (zero leakage)
        """
        region = self._regions.get(name)
        if region is None:
            return 0.0

        # Flux at inner boundary
        B_inner = self.compute_field_at_point(name, region.inner_radius)
        # Flux at outer boundary (should be zero for ideal toroid)
        B_outer = self.compute_field_at_point(name, region.outer_radius * 1.1)

        if B_inner < 1e-10:
            return 1.0

        leakage = abs(B_outer) / abs(B_inner)
        return float(np.clip(1.0 - leakage, 0, 1))

    def filter_interference(
        self,
        signal: np.ndarray,
        source_region: str,
        target_region: str,
    ) -> np.ndarray:
        """
        Filter a signal passing between two regions based on mutual inductance.
        High coupling → signal passes through.
        Low coupling → signal is attenuated (confined).
        """
        key = tuple(sorted([source_region, target_region]))
        coupling = self._coupling_map.get(key, 0.0)

        # Attenuation factor
        isolation = 1.0 - coupling
        return signal * (1.0 - isolation * 0.9)  # Never fully block, always allow some leakage

    def get_all_regions(self) -> dict[str, dict]:
        """Get summary of all confined regions."""
        result = {}
        for name, region in self._regions.items():
            result[name] = {
                "current": region.current,
                "stored_flux": region.stored_flux,
                "coherence": region.coherence,
                "confinement": self.confinement_ratio(name),
                "age": self._timestamp - region.birth_time,
                "activity": self._timestamp - region.last_update,
            }
        return result

    def _evict_weakest(self):
        """Evict the least active region to make room."""
        if not self._regions:
            return
        weakest = min(self._regions.values(), key=lambda r: r.current)
        self.destroy_region(weakest.name)

    def step(self, dt: float = 0.001):
        """Advance all regions by one time step."""
        self._timestamp += dt

        for name, region in self._regions.items():
            # Decay current over time (L/R time constant)
            tau = region.n_windings * region.permeability / max(region.current, 1e-6)
            decay = math.exp(-dt / max(tau, 1e-6))
            region.current *= decay

            # Update flux
            mu_0 = 1.0
            region.stored_flux = mu_0 * region.n_windings * region.current * region.permeability

        # Mutual inductance coupling
        for (name_a, name_b), coupling in self._coupling_map.items():
            if name_a in self._regions and name_b in self._regions:
                ra = self._regions[name_a]
                rb = self._regions[name_b]
                # Flux transfer proportional to coupling
                transfer = coupling * ra.stored_flux * 0.01
                ra.stored_flux -= transfer
                rb.stored_flux += transfer
