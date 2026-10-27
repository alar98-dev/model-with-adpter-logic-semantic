"""
EMField — The Electromagnetic Abstraction Layer

Treats every computational signal as an electromagnetic wave propagating
through a virtual medium. Each node in the system has:
  - An electric field component E(x,t)   → represents instantaneous state
  - A magnetic field component H(x,t)    → represents rate of change
  - An impedance Z(x)                    → represents resistance to change
  - A coupling matrix Γ                  → represents cross-node interference

The fundamental equation:
  V_noise(t) = Z_coupling · I_sources(t) + Γ_field · [E(t), H(t)]^T

This is NOT a simulation of real EM physics. It is a COMPUTATIONAL PRIMITIVE
that uses EM mathematics as a universal language for describing interference,
coupling, and resonance in any system — silicon, software, or hybrid.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EMState:
    """Complete electromagnetic state of a single node or the global field."""
    E: np.ndarray          # Electric field vector [N] — instantaneous state
    H: np.ndarray          # Magnetic field vector [N] — rate of change
    Z: np.ndarray          # Impedance vector [N] — resistance to perturbation
    timestamp: float = 0.0
    coherence: float = 1.0  # Phase coherence [0,1] — how synchronized the field is

    def energy(self) -> float:
        """Total electromagnetic energy: E = 0.5 * (ε·E² + μ·H²)"""
        eps = 1.0  # permittivity (normalized)
        mu = 1.0   # permeability (normalized)
        return 0.5 * float(np.sum(eps * self.E**2 + mu * self.H**2))

    def poynting_flux(self) -> np.ndarray:
        """Poynting vector S = E × H — energy flow direction."""
        if self.E.shape == self.H.shape and len(self.E) >= 3:
            return np.cross(self.E[:3], self.H[:3])
        return np.zeros(3)

    def spectral_density(self) -> np.ndarray:
        """Frequency spectrum via FFT of the electric field."""
        return np.abs(np.fft.rfft(self.E))


class EMField:
    """
    The global electromagnetic field that wraps all computational state.

    Every subsystem (orchestrator, subcortex, phase_shifter, etc.) reads
    and writes to this field. Interference between subsystems happens
    naturally through superposition of their contributions.
    """

    def __init__(self, dimension: int = 4096, dt: float = 0.001):
        self.dim = dimension
        self.dt = dt
        self.state = EMState(
            E=np.zeros(dimension, dtype=np.float32),
            H=np.zeros(dimension, dtype=np.float32),
            Z=np.ones(dimension, dtype=np.float32) * 50.0,  # 50Ω default impedance
        )
        self._sources: dict[str, np.ndarray] = {}
        self._coupling_matrix: Optional[np.ndarray] = None
        self._history: list[EMState] = []
        self._max_history = 64

    def register_source(self, name: str, contribution: np.ndarray):
        """Register a subsystem's contribution to the EM field."""
        assert contribution.shape[0] == self.dim, (
            f"Source '{name}' dimension {contribution.shape[0]} != field dimension {self.dim}"
        )
        self._sources[name] = contribution.astype(np.float32)

    def unregister_source(self, name: str):
        """Remove a subsystem's contribution."""
        self._sources.pop(name, None)

    def compute_coupling(self) -> np.ndarray:
        """
        Compute the coupling matrix Γ from all registered sources.
        Γ_ij = cosine_similarity(source_i, source_j) — how much source i
        interferes with source j.
        """
        names = sorted(self._sources.keys())
        n = len(names)
        if n == 0:
            return np.eye(self.dim, dtype=np.float32)

        # Build source matrix [n_sources × dim]
        S = np.stack([self._sources[name] for name in names], axis=0)

        # Normalize each source vector
        norms = np.linalg.norm(S, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        S_norm = S / norms

        # Coupling matrix [n_sources × n_sources]
        Gamma = S_norm @ S_norm.T

        # Expand to full dimension via Kronecker product with identity
        # Each source couples to the field proportionally
        self._coupling_matrix = Gamma
        return Gamma

    def step(self):
        """
        Advance the EM field by one time step using FDTD-inspired update:

        E(n+1) = E(n) + dt/ε · ∇×H(n) - dt/σ · E(n) + Σ source_i + hysteresis
        H(n+1) = H(n) - dt/μ · ∇×E(n+1)

        MEMORY: Hysteresis term retains residual from previous E state.
        CONSTRUCTIVE: Phase coherence between sources amplifies energy.
        COHERENCE PUMP: Amplifies dominant singular modes, suppresses weak ones.
        """
        E = self.state.E
        H = self.state.H
        Z_inv = 1.0 / np.maximum(self.state.Z, 1e-6)

        # Damping — very low to allow energy accumulation (memory)
        damping = 0.001

        # Sum all registered sources
        source_sum = np.zeros(self.dim, dtype=np.float32)
        for contrib in self._sources.values():
            source_sum += contrib

        # Coupling feedback (interference between sources)
        coupling_feedback = np.zeros(self.dim, dtype=np.float32)
        names = sorted(self._sources.keys())
        n = len(names)
        if n > 1 and self._coupling_matrix is not None:
            S = np.stack([self._sources[nm] for nm in names], axis=0)
            Gamma = self.compute_coupling()
            Gamma_off = Gamma - np.eye(Gamma.shape[0], dtype=np.float32)
            interference = Gamma_off @ S
            for i in range(n):
                coupling_feedback += interference[i]

        # ── CONSTRUCTIVE INTERFERENCE: detect in-phase sources ──
        if n > 1:
            S = np.stack([self._sources[nm] for nm in names], axis=0)
            norms = np.linalg.norm(S, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            S_norm = S / norms
            coherence_matrix = S_norm @ S_norm.T
            mean_coherence = float(np.mean(coherence_matrix[np.triu_indices(n, k=1)]))
            if mean_coherence > 0.5:
                gain = 1.0 + (mean_coherence - 0.5) * 2.0
                source_sum *= gain

        # ── HYSTERESIS: retain residual from previous E ──
        hysteresis_retention = 0.02
        hysteresis = hysteresis_retention * E

        # FDTD update with hysteresis
        E_new = E + self.dt * H * Z_inv - damping * E + source_sum * 0.35 + hysteresis
        H_new = H - self.dt * E_new + coupling_feedback * 0.015

        # ═══════════════════════════════════════════════════════════════════
        # COHERENCE PUMP — Non-linear SVD modulation (laser analogy)
        #
        # Mathematical formulation:
        #   M = E.reshape(64, 64) = U Σ Vᵀ
        #   σ̃ᵢ = σᵢ · (σᵢ / (σ₀ + ε))^γ   where γ ≥ 1.0
        #   M_pumped = U diag(σ̃) Vᵀ
        #   E_out = (1-η)E + η·vec(M_pumped)
        #
        # γ controls filtering sharpness:
        #   γ=1.0 → gentle (modes衰减 slowly)
        #   γ=2.0 → moderate (current choice)
        #   γ=3.0 → aggressive (only top modes survive)
        # ═══════════════════════════════════════════════════════════════════
        E_norm = np.linalg.norm(E_new)
        coherence = 0.0
        if E_norm > 1e-6:
            dim = E_new.shape[0]
            side = int(np.ceil(np.sqrt(dim)))
            if side * side == dim:
                mat = E_new.reshape(side, side).astype(np.float64)
            else:
                mat = E_new.reshape(1, dim).astype(np.float64)

            U, s, Vt = np.linalg.svd(mat, full_matrices=False)
            total = np.sum(s ** 2)

            if total > 1e-12 and s[0] > 1e-8:
                # Non-linear gain: σ̃ᵢ = σᵢ · (σᵢ / σ₀)^γ
                gain_gamma = 2.0
                epsilon = 1e-8
                norm_s = s / (s[0] + epsilon)
                s_pumped = s * (norm_s ** gain_gamma)

                # Energy conservation: preserve total energy
                energy_orig = np.sum(s ** 2)
                energy_new = np.sum(s_pumped ** 2)
                if energy_new > 1e-8:
                    s_pumped *= np.sqrt(energy_orig / energy_new)

                # Temporal relaxation: E_out = (1-η)E + η·vec(M_pumped)
                pump_rate = 0.15
                M_pumped = (U * s_pumped[:, np.newaxis]) @ Vt
                E_pumped = M_pumped.flatten().astype(np.float32)
                E_new = (1.0 - pump_rate) * E_new + pump_rate * E_pumped

                # Recompute coherence after pump
                top_k = min(5, len(s_pumped))
                total_after = np.sum(s_pumped ** 2)
                if total_after > 1e-12:
                    top_energy = np.sum(s_pumped[:top_k] ** 2)
                    coherence = float(np.clip(top_energy / total_after, 0.0, 1.0))
            else:
                coherence = 1.0  # zero field = coherent vacuum
        else:
            coherence = 1.0

        # Save snapshot
        new_state = EMState(
            E=E_new,
            H=H_new,
            Z=self.state.Z.copy(),
            timestamp=time.time(),
            coherence=coherence,
        )

        self._history.append(new_state)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        self.state = new_state

    def inject_perturbation(self, target: str, amplitude: float, frequency: float):
        """
        Inject a sinusoidal perturbation into a specific source.
        This simulates an external EM event (cosmic ray, motor noise, etc.)
        """
        if target in self._sources:
            t = self.state.timestamp
            wave = amplitude * np.sin(2 * np.pi * frequency * t + np.arange(self.dim) * 0.1)
            self._sources[target] += wave.astype(np.float32)

    def resonance_map(self) -> dict[str, float]:
        """
        Compute resonance frequencies — frequencies where the field
        has constructive interference (peaks in spectral density).
        """
        spectrum = self.state.spectral_density()
        if len(spectrum) < 2:
            return {}

        # Find peaks (local maxima above mean)
        mean_val = np.mean(spectrum)
        peaks = []
        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > spectrum[i-1] and spectrum[i] > spectrum[i+1]:
                if spectrum[i] > mean_val:
                    freq = i / (self.dim * self.dt)
                    peaks.append((freq, float(spectrum[i])))

        return {f"resonance_{i:.2f}Hz": amp for i, amp in sorted(peaks, key=lambda x: -x[1])[:5]}

    def get_impedance_profile(self) -> np.ndarray:
        """Return the current impedance profile — low Z = easy to perturb."""
        return self.state.Z.copy()

    def set_impedance(self, regions: dict[str, tuple[int, int]], value: float):
        """Set impedance for specific regions of the field."""
        for name, (start, end) in regions.items():
            self.state.Z[start:end] = value
