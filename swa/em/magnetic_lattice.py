"""
MagneticLattice — The 64×64 Topology Matrix as a Magnetic Crystal

The existing SWA-Aeon system generates a 64×64 (4096-element) topology matrix
via the subcortex (Gemma-270M hidden states → "metamatter"). This matrix is
currently treated as a passive data structure — a heatmap of latent activation.

REINTERPRETATION: This matrix IS a magnetic lattice.

Each cell (i,j) holds a MAGNETIZATION VECTOR m(i,j) with:
  - Magnitude |m|    → activation strength (already exists)
  - Phase angle φ    → temporal coherence (NEW — derived from history)
  - Domain wall type → boundary between different activation regions (NEW)

The lattice supports:
  1. Spin wave propagation — perturbations travel through the lattice
  2. Domain wall dynamics — boundaries between activation regions move
  3. Magnon scattering — high-energy excitations interact with the lattice
  4. Remanent state — the lattice retains a "memory" of past activations

This transforms the topology matrix from a SNAPSHOT into a DYNAMIC MEDIUM
where computation happens through the physics of magnetic excitation.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class SpinState:
    """State of a single spin (cell) in the magnetic lattice."""
    magnitude: float    # |m| ∈ [0, 1] — activation strength
    phase: float        # φ ∈ [-π, π] — temporal phase
    domain_id: int      # Which magnetic domain this spin belongs to
    temperature: float  # Local thermal noise level


@dataclass
class DomainWall:
    """Boundary between two magnetic domains."""
    start: tuple[int, int]
    end: tuple[int, int]
    domain_a: int
    domain_b: int
    wall_type: str  # 'bloch', 'neel', or 'cross'
    energy: float


class MagneticLattice:
    """
    A 2D magnetic lattice that maps directly to the SWA-Aeon topology matrix.

    The lattice is NOT a simulation of real magnetism. It is a
    COMPUTATIONAL SUBSTRATE that uses magnetic dynamics as primitives:
      - Spin wave propagation = information diffusion
      - Domain wall motion = context boundary evolution
      - Magnon scattering = attention reallocation
      - Remanence = working memory
    """

    def __init__(self, size: tuple[int, int] = (64, 64)):
        self.rows, self.cols = size
        self.total_spins = self.rows * self.cols

        # Core state arrays
        self._magnitude = np.zeros(size, dtype=np.float32)   # |m(i,j)|
        self._phase = np.zeros(size, dtype=np.float32)       # φ(i,j)
        self._domain_map = np.zeros(size, dtype=np.int32)    # domain ID
        self._temperature = np.ones(size, dtype=np.float32) * 0.01  # thermal noise

        # History for phase evolution
        self._prev_magnitude = np.zeros(size, dtype=np.float32)
        self._phaseaccumulator = np.zeros(size, dtype=np.float32)

        # Domain tracking
        self._domains: dict[int, dict] = {}
        self._walls: list[DomainWall] = []
        self._next_domain_id = 1

        # Coupling constants (Heisenberg model)
        self._J = 10.0      # Exchange coupling (nearest-neighbor alignment) — strong for domain formation
        self._D = 0.0       # Dzyaloshinskii-Moriya interaction (chirality)
        self._K = 0.0       # Anisotropy (preferred axis)
        self._damping = 0.1  # Gilbert damping (energy dissipation)

        self._timestamp = 0.0

    def load_topology(self, matrix: np.ndarray):
        """
        Load a 64×64 topology matrix (from subcortex) into the magnetic lattice.
        The matrix values become magnetization magnitudes.
        """
        if matrix.shape != (self.rows, self.cols):
            if matrix.size == self.total_spins:
                matrix = matrix.reshape(self.rows, self.cols)
            else:
                raise ValueError(
                    f"Matrix shape {matrix.shape} incompatible with lattice "
                    f"({self.rows}, {self.cols})"
                )

        self._prev_magnitude = self._magnitude.copy()
        self._magnitude = np.abs(matrix).astype(np.float32)
        self._magnitude = np.clip(self._magnitude, 0, 1)

        # Derive phase from rate of change
        delta = self._magnitude - self._prev_magnitude
        self._phase = np.arctan2(delta, self._prev_magnitude + 1e-8).astype(np.float32)

        # Detect domains from magnitude gradients
        self._detect_domains()
        self._detect_domain_walls()

        self._timestamp = time.time()

    def _detect_domains(self):
        """
        Identify magnetic domains — regions of similar magnetization.
        Uses adaptive thresholding with Gaussian smoothing for larger domains.
        """
        from scipy.ndimage import gaussian_filter, label

        # Gaussian smoothing with large sigma creates coherent regions
        smoothed = gaussian_filter(self._magnitude, sigma=6.0)

        # Global median threshold: split into "up" and "down" domains
        median_val = float(np.median(smoothed))
        binary = (smoothed > median_val).astype(np.int32)

        # Label connected components
        labeled, n_domains = label(binary)

        # Filter out tiny domains (noise)
        MIN_DOMAIN_SIZE = 10  # minimum cells per domain
        self._domain_map = np.zeros_like(labeled)
        self._domains = {}
        next_id = 1
        for d in range(1, n_domains + 1):
            mask = labeled == d
            size = int(np.sum(mask))
            if size < MIN_DOMAIN_SIZE:
                continue
            self._domain_map[mask] = next_id
            self._domains[next_id] = {
                "size": size,
                "mean_magnitude": float(np.mean(self._magnitude[mask])),
                "mean_phase": float(np.mean(self._phase[mask])),
                "centroid": (
                    int(np.mean(np.where(mask)[0])),
                    int(np.mean(np.where(mask)[1])),
                ),
            }
            next_id += 1

    def _detect_domain_walls(self):
        """
        Detect domain walls — boundaries between different magnetic domains.
        These are sites where adjacent spins belong to different domains.
        """
        self._walls = []

        # Horizontal boundaries
        diff_h = self._domain_map[:, 1:] != self._domain_map[:, :-1]
        # Vertical boundaries
        diff_v = self._domain_map[1:, :] != self._domain_map[:-1, :]

        wall_sites_h = np.argwhere(diff_h)
        wall_sites_v = np.argwhere(diff_v)

        for r, c in wall_sites_h:
            d_a = int(self._domain_map[r, c])
            d_b = int(self._domain_map[r, c + 1])
            if d_a > 0 and d_b > 0:
                wall = DomainWall(
                    start=(r, c), end=(r, c + 1),
                    domain_a=d_a, domain_b=d_b,
                    wall_type='neel',
                    energy=abs(self._magnitude[r, c] - self._magnitude[r, c + 1]),
                )
                self._walls.append(wall)

        for r, c in wall_sites_v:
            d_a = int(self._domain_map[r, c])
            d_b = int(self._domain_map[r + 1, c])
            if d_a > 0 and d_b > 0:
                wall = DomainWall(
                    start=(r, c), end=(r + 1, c),
                    domain_a=d_a, domain_b=d_b,
                    wall_type='bloch',
                    energy=abs(self._magnitude[r, c] - self._magnitude[r + 1, c]),
                )
                self._walls.append(wall)

    def propagate_spin_wave(
        self,
        origin: tuple[int, int],
        amplitude: float = 1.0,
        wavelength: float = 4.0,
        direction: float = 0.0,
    ) -> np.ndarray:
        """
        Propagate a spin wave from a given origin point.
        The wave travels through the lattice via exchange coupling.
        Returns the perturbation field.
        """
        r0, c0 = origin
        perturbation = np.zeros((self.rows, self.cols), dtype=np.float32)

        for r in range(self.rows):
            for c in range(self.cols):
                dr = r - r0
                dc = c - c0
                distance = math.sqrt(dr**2 + dc**2)

                # Radial wave with directional modulation
                angle = math.atan2(dr, dc)
                directional = 0.5 * (1.0 + math.cos(angle - direction))

                # Dispersion relation: ω = 2J(1 - cos(k·a))
                k = 2 * math.pi / wavelength
                omega = 2 * self._J * (1 - math.cos(k * distance))

                # Wave value at current time
                phase = k * distance - omega * self._timestamp
                perturbation[r, c] = amplitude * directional * math.cos(phase) * math.exp(-0.05 * distance)

        # Apply perturbation to magnitude
        self._magnitude = np.clip(self._magnitude + perturbation * 0.1, 0, 1)

        return perturbation

    def inject_magnon(self, position: tuple[int, int], energy: float):
        """
        Inject a magnon (quantized spin wave excitation) at a specific position.
        This creates a local disturbance that propagates through the lattice.
        """
        r, c = position
        if 0 <= r < self.rows and 0 <= c < self.cols:
            # Increase local temperature (thermal excitation)
            self._temperature[r, c] += energy * 0.1

            # Perturb neighboring spins
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        dist = math.sqrt(dr**2 + dc**2)
                        if dist > 0:
                            coupling = self._J * energy / (dist ** 2)
                            self._magnitude[nr, nc] = np.clip(
                                self._magnitude[nr, nc] + coupling * 0.01, 0, 1
                            )

    def remanent_state(self) -> np.ndarray:
        """
        Get the remanent magnetization — the "memory" left after excitations decay.
        This is the lattice's working memory of recent activations.
        """
        # Remanence is the magnitude weighted by phase coherence
        coherence = np.cos(self._phase)
        return (self._magnitude * coherence).astype(np.float32)

    def to_flat_vector(self) -> np.ndarray:
        """Flatten the lattice to a 4096-element vector (compatible with SWA)."""
        return self._magnitude.flatten().astype(np.float32)

    def from_flat_vector(self, vector: np.ndarray):
        """Load a flat 4096-element vector into the lattice."""
        matrix = vector.reshape(self.rows, self.cols).astype(np.float32)
        self.load_topology(matrix)

    def get_domain_info(self) -> dict:
        """Get summary information about magnetic domains."""
        return {
            "n_domains": len(self._domains),
            "n_walls": len(self._walls),
            "domains": self._domains,
            "wall_energy_total": sum(w.energy for w in self._walls),
            "mean_temperature": float(np.mean(self._temperature)),
        }

    def energy(self) -> float:
        """Total lattice energy (Heisenberg model)."""
        # Exchange energy: -J Σ m_i · m_j (nearest neighbors)
        E_exchange = 0.0
        # Horizontal neighbors
        E_exchange += float(np.sum(self._magnitude[:, :-1] * self._magnitude[:, 1:]))
        # Vertical neighbors
        E_exchange += float(np.sum(self._magnitude[:-1, :] * self._magnitude[1:, :]))
        E_exchange *= -self._J

        # Anisotropy energy: -K Σ m_z²
        E_anisotropy = -self._K * float(np.sum(self._magnitude ** 2))

        # Thermal energy
        E_thermal = float(np.sum(self._temperature))

        return E_exchange + E_anisotropy + E_thermal

    # ── Lifecycle ───────────────────────────────────────────────────────

    def advance(self, dt: float):
        """
        Advance the lattice by one time step.
        Called by EMCore.cycle(). Decays temperature, updates domains.
        """
        # Decay temperature (thermal relaxation)
        self._temperature *= (1.0 - self._damping * dt)

        # Apply Heisenberg exchange: nearest-neighbor alignment
        # Vectorized: compute average of 4 neighbors
        up    = np.roll(self._magnitude, -1, axis=0)
        down  = np.roll(self._magnitude,  1, axis=0)
        left  = np.roll(self._magnitude, -1, axis=1)
        right = np.roll(self._magnitude,  1, axis=1)
        avg_neighbor = (up + down + left + right) / 4.0

        # Exchange interaction: pull toward neighbor average
        exchange = self._J * dt * (avg_neighbor - self._magnitude)

        # Thermal noise
        noise = self._temperature * np.random.randn(*self._magnitude.shape) * 0.01

        self._magnitude = np.clip(self._magnitude + exchange + noise, 0.0, 1.0)

        # Recompute phase from rate of change
        delta = self._magnitude - self._prev_magnitude
        self._phase = np.arctan2(delta, self._prev_magnitude + 1e-8).astype(np.float32)
        self._prev_magnitude = self._magnitude.copy()

        # Detect domains and walls
        self._detect_domains()
        self._detect_domain_walls()

        self._timestamp = time.time()

    def ingest_field_energy(self, E: np.ndarray):
        """
        Map a N-dim EM field vector onto the 64×64 lattice.
        Creates domain structure from field energy distribution.
        Pads/truncates as needed.
        """
        dim = E.shape[0]
        target = self.rows * self.cols

        # Map E to lattice size via interpolation
        if dim == target:
            matrix = E.reshape(self.rows, self.cols).astype(np.float32)
        elif dim < target:
            # Upsample: repeat elements
            repeats = target // dim
            up = np.repeat(E, repeats)[:target]
            matrix = up.reshape(self.rows, self.cols).astype(np.float32)
        else:
            # Downsample: average blocks
            block = dim // target
            down = np.array([E[i*block:(i+1)*block].mean() for i in range(target)])
            matrix = down.reshape(self.rows, self.cols).astype(np.float32)

        # Normalize to [0, 1]
        mmin, mmax = matrix.min(), matrix.max()
        if mmax - mmin > 1e-8:
            matrix = (matrix - mmin) / (mmax - mmin)
        else:
            matrix = np.abs(matrix)

        # Merge with existing magnitude (slow incorporation)
        alpha = 0.15  # learning rate
        self._magnitude = (1.0 - alpha) * self._magnitude + alpha * matrix

        # Increase temperature where field is strong (energy injection)
        self._temperature += np.abs(matrix) * 0.05

        # Recompute
        delta = self._magnitude - self._prev_magnitude
        self._phase = np.arctan2(delta, self._prev_magnitude + 1e-8).astype(np.float32)
        self._prev_magnitude = self._magnitude.copy()

        self._detect_domains()
        self._detect_domain_walls()
