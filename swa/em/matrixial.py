"""
MatrixialRelation — Stable Superposition Field with Solid Basis Vectors

The SWA-Aeon EM field exists in a superposition of states:
  |Ψ⟩ = Σᵢ αᵢ |eᵢ⟩

where |eᵢ⟩ are the basis states (wave packets, lattice domains, etc.)
and αᵢ are their complex amplitudes.

PROBLEM: Superposition is inherently unstable — amplitudes shift,
phases drift, and the field can decohere.

SOLUTION: The MatrixialRelation establishes a SOLID BASIS that
remains stable regardless of which superposition states are active.

Mathematical structure:
  1. Solid Basis B = {b₁, b₂, ..., bₖ} — orthogonalized from E and H
  2. Projection Matrix P — maps any state onto the solid basis
  3. Superposition Field Φ — the live interference pattern
  4. Matrixial Relation M — the operator that ensures:
       M(Φ) ∈ span(B)  always
       ||M(Φ) - Φ|| is minimized  (best approximation)
       ||B|| is maximally separated  (most stable basis)

This creates a GRAVITY WELL in state space: no matter how wild the
superposition becomes, the matrixial relation pulls it back toward
the solid basis. The system cannot escape its own foundation.

Physical analogy:
  - Solid basis = bedrock (tectonic plates)
  - Superposition = ocean waves (constantly changing)
  - Matrixial relation = the ocean floor that shapes all waves
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class BasisVector:
    """A single vector in the solid basis."""
    direction: np.ndarray    # [dim] — unit direction
    eigenvalue: float        # Importance weight
    source: str              # Where this basis came from
    birth_time: float        # When it was established
    stability: float = 1.0   # How stable this vector is [0,1]


@dataclass
class SuperpositionState:
    """The current superposition of all active waves."""
    amplitudes: np.ndarray       # [n_basis] — complex amplitudes
    phases: np.ndarray           # [n_basis] — phase angles
    coherence: float             # [0,1] — overall phase alignment
    total_energy: float          # Total field energy
    timestamp: float = 0.0


@dataclass
class MatrixialProjection:
    """Result of projecting a state onto the solid basis."""
    projected: np.ndarray        # [dim] — the state in solid basis coordinates
    residual: np.ndarray         # [dim] — what was lost in projection
    projection_error: float      # ||residual|| / ||original||
    basis_alignment: float       # How well the state aligns with basis
    stability_score: float       # Overall stability of the relation


class MatrixialRelation:
    """
    The operator that maintains stability in the superposition field.

    Every cycle:
      1. Update the solid basis from current E and H fields
      2. Capture the live superposition state
      3. Project the superposition onto the solid basis
      4. Compute the matrixial relation (how stable the projection is)
      5. Apply correction to pull the field back toward stability

    The result is a field that:
      - Can express any superposition (expressive)
      - Always returns to a stable state (resilient)
      - Has a solid foundation that resists perturbation (robust)
    """

    def __init__(self, dimension: int = 4096, n_basis: int = 32):
        self.dim = dimension
        self.n_basis = min(n_basis, dimension)

        # Solid basis — the bedrock of the state space
        self._basis: list[BasisVector] = []
        self._basis_matrix: Optional[np.ndarray] = None  # [dim × n_basis]
        self._basis_orthonormal: Optional[np.ndarray] = None

        # Projection infrastructure
        self._projection_matrix: Optional[np.ndarray] = None  # [n_basis × dim]
        self._last_projection: Optional[MatrixialProjection] = None

        # Superposition tracking
        self._superposition: Optional[SuperpositionState] = None
        self._superposition_history: list[SuperpositionState] = []
        self._max_history = 128

        # Stability metrics
        self._stability_ema = 0.5  # Exponential moving average
        self._stability_alpha = 0.1  # EMA decay

        # Basis evolution parameters
        self._basis_refresh_interval = 100  # cycles
        self._basis_refresh_counter = 0
        self._min_basis_separation = 0.3  # minimum cosine distance between basis vectors

        self._timestamp = time.time()

    def initialize_basis(self, E: np.ndarray, H: np.ndarray):
        """
        Initialize the solid basis from the EM field's electric and magnetic
        components. This is the FOUNDATION — called once at startup.

        The basis is constructed by:
          1. Stacking E and H as the first two basis directions
          2. Computing their principal components (QR decomposition)
          3. Orthogonalizing to ensure linear independence
          4. Normalizing to unit length
        """
        # Ensure correct dimensions
        E = E[:self.dim].astype(np.float32)
        H = H[:self.dim].astype(np.float32)

        # Build the initial basis matrix from E, H, and their products
        raw_vectors = []

        # Vector 1: E field direction
        norm_E = np.linalg.norm(E)
        if norm_E > 1e-8:
            raw_vectors.append(E / norm_E)

        # Vector 2: H field direction
        norm_H = np.linalg.norm(H)
        if norm_H > 1e-8:
            raw_vectors.append(H / norm_H)

        # Vector 3: E × H (Poynting direction — energy flow)
        if len(E) >= 3 and len(H) >= 3:
            poynting = np.cross(E[:3], H[:3])
            poynting_full = np.zeros(self.dim, dtype=np.float32)
            poynting_full[:3] = poynting
            norm_P = np.linalg.norm(poynting_full)
            if norm_P > 1e-8:
                raw_vectors.append(poynting_full / norm_P)

        # Fill remaining basis with random orthogonal vectors
        # (seeded from E for reproducibility)
        rng = np.random.RandomState(int(np.sum(E[:8]) * 1000) % (2**31))
        while len(raw_vectors) < self.n_basis:
            random_vec = rng.randn(self.dim).astype(np.float32)
            raw_vectors.append(random_vec)

        # Stack and orthogonalize via modified Gram-Schmidt
        V = np.stack(raw_vectors[:self.n_basis], axis=1)  # [dim × n_basis]
        Q, R = np.linalg.qr(V, mode='reduced')

        # Keep only well-conditioned columns (diagonal of R above threshold)
        good_cols = []
        for i in range(min(Q.shape[1], self.n_basis)):
            if abs(R[i, i]) > 1e-6:
                good_cols.append(i)

        if len(good_cols) < 2:
            # Fallback: use identity-like basis
            good_cols = list(range(min(self.n_basis, self.dim)))

        self._basis_orthonormal = Q[:, good_cols].astype(np.float32)
        self.n_basis = self._basis_orthonormal.shape[1]

        # Create basis vector objects
        self._basis = []
        for i in range(self.n_basis):
            vec = BasisVector(
                direction=self._basis_orthonormal[:, i].copy(),
                eigenvalue=float(R[i, i]) if i < R.shape[0] else 1.0,
                source="em_init",
                birth_time=self._timestamp,
                stability=1.0,
            )
            self._basis.append(vec)

        # Compute projection matrix: P = (B^T B)^{-1} B^T
        # Since B is orthonormal: P = B^T
        self._projection_matrix = self._basis_orthonormal.T  # [n_basis × dim]

        # Compute basis separation (minimum pairwise cosine distance)
        self._update_basis_stability()

    def update_basis(self, E: np.ndarray, H: np.ndarray):
        """
        Evolve the solid basis to track slow changes in the EM field.
        This is INCREMENTAL — the basis drifts to follow the field,
        but with high inertia (slow adaptation).
        """
        self._basis_refresh_counter += 1
        if self._basis_refresh_counter < self._basis_refresh_interval:
            return

        self._basis_refresh_counter = 0

        # Build new candidate basis from current E, H
        E = E[:self.dim].astype(np.float32)
        H = H[:self.dim].astype(np.float32)

        # Only add new direction if it's sufficiently different from existing basis
        norm_E = np.linalg.norm(E)
        if norm_E < 1e-8:
            return

        e_dir = E / norm_E

        # Check alignment with existing basis
        max_alignment = 0.0
        if self._basis_orthonormal is not None:
            projections = self._projection_matrix @ e_dir  # [n_basis]
            max_alignment = float(np.max(np.abs(projections)))

        # If the new direction is too similar to existing basis, skip
        if max_alignment > (1.0 - self._min_basis_separation):
            return

        # Add new basis vector (with low eigenvalue — it's new and untested)
        new_vec = BasisVector(
            direction=e_dir.copy(),
            eigenvalue=0.1,
            source="em_evolution",
            birth_time=self._timestamp,
            stability=0.5,
        )
        self._basis.append(new_vec)

        # Keep only the most important n_basis vectors
        if len(self._basis) > self.n_basis:
            self._basis.sort(key=lambda b: b.eigenvalue, reverse=True)
            self._basis = self._basis[:self.n_basis]

        # Rebuild orthonormal basis
        self._rebuild_basis_matrix()

    def _rebuild_basis_matrix(self):
        """Rebuild the orthonormal basis matrix from stored basis vectors."""
        if not self._basis:
            return

        V = np.stack([b.direction for b in self._basis], axis=1)
        Q, R = np.linalg.qr(V, mode='reduced')

        good_cols = [i for i in range(Q.shape[1]) if abs(R[i, i]) > 1e-6]
        if len(good_cols) < 2:
            return

        self._basis_orthonormal = Q[:, good_cols].astype(np.float32)
        self.n_basis = self._basis_orthonormal.shape[1]
        self._projection_matrix = self._basis_orthonormal.T

        # Update eigenvalues from R diagonal
        for i in range(min(len(self._basis), self.n_basis)):
            self._basis[i].eigenvalue = float(abs(R[i, i]))
            self._basis[i].direction = self._basis_orthonormal[:, i].copy()

        self._update_basis_stability()

    def _update_basis_stability(self):
        """Compute stability metric for the current basis."""
        if self._basis_orthonormal is None or self.n_basis < 2:
            return

        # Pairwise cosine distances
        B = self._basis_orthonormal  # [dim × n_basis]
        Gram = B.T @ B  # [n_basis × n_basis] — should be I for perfect orthogonality

        # Off-diagonal elements measure non-orthogonality
        off_diag = Gram - np.eye(self.n_basis, dtype=np.float32)
        max_coupling = float(np.max(np.abs(off_diag)))

        # Stability = 1 - max coupling (lower coupling = more stable)
        stability = 1.0 - min(max_coupling, 1.0)

        for vec in self._basis:
            vec.stability = stability

    def project(self, state: np.ndarray) -> MatrixialProjection:
        """
        Project an arbitrary state onto the solid basis.

        This is the core of the matrixial relation:
          projected = P @ state   (where P = B^T for orthonormal B)
          residual = state - B @ projected
          error = ||residual|| / ||state||

        The projection GUARANTEES the result is in span(B),
        which is the stable subspace.
        """
        if self._projection_matrix is None:
            # No basis yet — return as-is
            return MatrixialProjection(
                projected=state.copy(),
                residual=np.zeros_like(state),
                projection_error=0.0,
                basis_alignment=0.0,
                stability_score=0.0,
            )

        state = state[:self.dim].astype(np.float32)

        # Project onto solid basis
        coords = self._projection_matrix @ state  # [n_basis]

        # Reconstruct from basis coordinates
        reconstructed = self._basis_orthonormal @ coords  # [dim]

        # Compute residual
        residual = state - reconstructed
        state_norm = np.linalg.norm(state)
        residual_norm = np.linalg.norm(residual)

        projection_error = residual_norm / (state_norm + 1e-8)

        # Basis alignment: how much of the state is captured by the basis
        basis_alignment = 1.0 - projection_error

        # Stability score: weighted sum of basis eigenvalues × alignment
        eigenvalue_sum = sum(abs(b.eigenvalue) * b.stability for b in self._basis)
        stability_score = basis_alignment * min(eigenvalue_sum / self.n_basis, 1.0)

        projection = MatrixialProjection(
            projected=reconstructed,
            residual=residual,
            projection_error=float(projection_error),
            basis_alignment=float(basis_alignment),
            stability_score=float(stability_score),
        )

        self._last_projection = projection
        return projection

    def capture_superposition(
        self,
        wave_amplitudes: np.ndarray,
        wave_phases: np.ndarray,
        total_energy: float,
    ):
        """
        Capture the current superposition state of all active waves.
        This is a SNAPSHOT of the interference pattern at this instant.
        """
        # Project amplitudes onto basis
        if self._projection_matrix is not None:
            amp = wave_amplitudes[:self.dim].astype(np.float32)
            coords = self._projection_matrix @ amp
        else:
            coords = wave_amplitudes[:self.n_basis].astype(np.float32)

        # Compute coherence from phase alignment
        if len(wave_phases) > 1:
            mean_phase = np.mean(np.exp(1j * wave_phases))
            coherence = float(np.abs(mean_phase))
        else:
            coherence = 1.0

        state = SuperpositionState(
            amplitudes=coords,
            phases=wave_phases[:self.n_basis].copy() if len(wave_phases) >= self.n_basis
                   else np.pad(wave_phases, (0, self.n_basis - len(wave_phases))),
            coherence=coherence,
            total_energy=total_energy,
            timestamp=time.time(),
        )

        self._superposition = state
        self._superposition_history.append(state)
        if len(self._superposition_history) > self._max_history:
            self._superposition_history.pop(0)

    def apply_correction(self, field_E: np.ndarray, strength: float = 0.05) -> np.ndarray:
        """
        Apply matrixial correction to the EM field.

        This pulls the field back toward the solid basis when it
        drifts too far. The correction is:
          E_corrected = E - strength * residual

        where residual is the component of E that lies OUTSIDE
        the solid basis subspace.
        """
        if self._last_projection is None or self._projection_matrix is None:
            return field_E

        E = field_E[:self.dim].astype(np.float32)

        # Project and get residual
        projection = self.project(E)

        # Correction: subtract the out-of-basis component
        correction = projection.residual * strength
        E_corrected = E - correction

        return E_corrected.astype(np.float32)

    def get_stability(self) -> float:
        """Get the current stability score of the matrixial relation."""
        return self._stability_ema

    def get_basis_info(self) -> dict:
        """Get information about the solid basis."""
        if not self._basis:
            return {"n_basis": 0, "vectors": []}

        return {
            "n_basis": self.n_basis,
            "total_eigenvalue": sum(b.eigenvalue for b in self._basis),
            "mean_stability": np.mean([b.stability for b in self._basis]),
            "vectors": [
                {
                    "source": b.source,
                    "eigenvalue": b.eigenvalue,
                    "stability": b.stability,
                    "age": self._timestamp - b.birth_time,
                }
                for b in self._basis[:8]  # Top 8
            ],
        }

    def get_superposition_info(self) -> dict:
        """Get information about the current superposition state."""
        if self._superposition is None:
            return {"active": False}

        return {
            "active": True,
            "coherence": self._superposition.coherence,
            "total_energy": self._superposition.total_energy,
            "n_components": len(self._superposition.amplitudes),
            "amplitude_range": (
                float(np.min(np.abs(self._superposition.amplitudes))),
                float(np.max(np.abs(self._superposition.amplitudes))),
            ),
        }

    def status(self) -> dict:
        """Complete matrixial relation status."""
        return {
            "basis": self.get_basis_info(),
            "superposition": self.get_superposition_info(),
            "stability": self._stability_ema,
            "last_projection_error": (
                self._last_projection.projection_error
                if self._last_projection else 0.0
            ),
            "last_basis_alignment": (
                self._last_projection.basis_alignment
                if self._last_projection else 0.0
            ),
            "history_length": len(self._superposition_history),
        }

    def step(self, E: np.ndarray, H: np.ndarray):
        """
        Full matrixial relation step:
          1. Update basis (if interval reached)
          2. Project current E onto basis
          3. Update stability EMA
          4. Apply correction with anchoring
          5. Return corrected E
        """
        self._timestamp = time.time()

        # Update basis evolution
        self.update_basis(E, H)

        # Project current state
        projection = self.project(E)

        # Update stability EMA
        self._stability_ema = (
            (1 - self._stability_alpha) * self._stability_ema +
            self._stability_alpha * projection.stability_score
        )

        # Apply correction — stronger when stability is low
        adaptive_strength = 0.05 + 0.15 * (1.0 - self._stability_ema)
        E_corrected = self.apply_correction(E, strength=adaptive_strength)

        # Anchor: pull toward the average of top basis vectors
        # This prevents the basis from drifting away from its foundation
        if self._basis_orthonormal is not None and self.n_basis >= 4:
            top4 = self._basis_orthonormal[:, :4]
            anchor = np.mean(top4, axis=1)
            anchor_norm = np.linalg.norm(anchor)
            if anchor_norm > 1e-8:
                anchor = anchor / anchor_norm
            # Gentle pull (5% toward anchor)
            E_corrected = E_corrected * 0.95 + anchor * 0.05 * np.linalg.norm(E_corrected)

        return E_corrected
