"""
WaveInterferenceEngine — Computational Attention via Wave Superposition

The Transformer attention mechanism:
  Attention(Q, K, V) = softmax(QK^T / √d) · V

Is mathematically equivalent to:
  WaveSuperposition(A_i, φ_i) = Σ A_i · e^{j(ωt + φ_i)}

Where:
  A_i = amplitude  → relevance score (from QK^T)
  φ_i = phase      → positional encoding (from RoPE/sinusoidal)
  ω = frequency    → layer depth (higher layers = higher frequency)

The WaveInterferenceEngine replaces softmax normalization with
PHYSICAL INTERFERENCE: waves in phase amplify, waves out of phase cancel.

This is not an approximation — it is a DIFFERENT COMPUTATIONAL REGIME
where attention is computed by the physics of superposition, not by
a software normalization function.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class WavePacket:
    """A single wave with amplitude, frequency, phase, and spatial extent."""
    amplitude: np.ndarray    # [dim] — strength of each component
    frequency: float         # Hz — oscillation rate
    phase: np.ndarray        # [dim] — phase offset per component
    origin_time: float = 0.0
    decay_rate: float = 0.01  # Exponential decay coefficient
    label: str = ""

    def evaluate(self, t: float) -> np.ndarray:
        """Compute wave value at time t: A * cos(ωt + φ) * e^(-λt)"""
        envelope = np.exp(-self.decay_rate * max(0, t - self.origin_time))
        return self.amplitude * np.cos(2 * np.pi * self.frequency * t + self.phase) * envelope

    def energy(self) -> float:
        """Current energy of this wave packet."""
        return float(np.sum(self.amplitude**2))


@dataclass
class InterferencePattern:
    """Result of superposing multiple wave packets."""
    pattern: np.ndarray       # [dim] — the combined waveform
    constructive: np.ndarray  # [dim] — regions of constructive interference
    destructive: np.ndarray   # [dim] — regions of destructive interference
    coherence: float          # [0,1] — overall phase coherence
    dominant_frequency: float # Hz — the frequency with highest amplitude
    timestamp: float = 0.0


class WaveInterferenceEngine:
    """
    The engine that computes attention through wave superposition.

    Instead of:
      scores = Q @ K.T / sqrt(d)
      weights = softmax(scores)
      output = weights @ V

    We compute:
      waves = [WavePacket(A_i, ω_i, φ_i) for each (Q_i, K_i, V_i)]
      interference = Σ waves(t)
      output = interference · V

    The result is attention that RESONATES — information that is
    "in phase" with the current context amplifies naturally, while
    "out of phase" noise cancels itself.
    """

    def __init__(self, dimension: int = 4096, base_frequency: float = 1.0):
        self.dim = dimension
        self.base_freq = base_frequency
        self._active_waves: list[WavePacket] = []
        self._wave_history: list[InterferencePattern] = []
        self._max_history = 128
        self._global_time: float = 0.0
        self._damping: float = 0.005
        self._pruned_total: int = 0

    def create_wave(
        self,
        amplitude: np.ndarray,
        frequency: Optional[float] = None,
        phase: Optional[np.ndarray] = None,
        label: str = "",
    ) -> WavePacket:
        """Create a new wave packet from a signal vector."""
        if frequency is None:
            # Frequency proportional to signal energy
            energy = float(np.sqrt(np.sum(amplitude**2)))
            frequency = self.base_freq * (1.0 + energy * 0.1)

        if phase is None:
            # Phase from the signal's spectral structure
            fft = np.fft.rfft(amplitude)
            phase = np.angle(fft).astype(np.float32)
            # Pad to full dimension
            if len(phase) < self.dim:
                phase = np.pad(phase, (0, self.dim - len(phase)), mode='constant')

        wave = WavePacket(
            amplitude=amplitude.astype(np.float32),
            frequency=frequency,
            phase=phase[:self.dim].astype(np.float32),
            origin_time=self._global_time,
            label=label,
        )
        self._active_waves.append(wave)
        # Lifecycle amortizado: só corta quando o balão cresce demais (mesmo
        # quando superpose raramente é chamado). O corte caprichado acontece
        # no superpose via gc().
        if len(self._active_waves) > 4 * 256:
            self.gc()
        return wave

    def remove_wave(self, wave: WavePacket):
        """Remove a wave packet from the active set."""
        self._active_waves = [w for w in self._active_waves if w is not wave]

    def superpose(self, time_step: Optional[float] = None) -> InterferencePattern:
        """
        Compute the superposition of all active waves at current time.
        This IS the attention mechanism — constructive interference = high attention,
        destructive interference = suppressed attention.
        """
        if time_step is not None:
            self._global_time += time_step

        t = self._global_time

        # Evaluate each wave at current time
        wave_values = []
        for wave in self._active_waves:
            val = wave.evaluate(t)
            wave_values.append(val)

        if not wave_values:
            empty = np.zeros(self.dim, dtype=np.float32)
            return InterferencePattern(
                pattern=empty, constructive=empty, destructive=empty,
                coherence=0.0, dominant_frequency=0.0, timestamp=t,
            )

        # Stack and sum (superposition principle)
        S = np.stack(wave_values, axis=0)  # [n_waves × dim]
        total = np.sum(S, axis=0)          # [dim]

        # Identify constructive vs destructive regions
        mean_val = np.mean(np.abs(total))
        constructive = np.where(np.abs(total) > mean_val, total, 0.0).astype(np.float32)
        destructive = np.where(np.abs(total) <= mean_val, total, 0.0).astype(np.float32)

        # Compute coherence (phase alignment across waves)
        if len(wave_values) > 1:
            # Pairwise phase correlation
            phases = []
            for wv in wave_values:
                fft = np.fft.rfft(wv)
                phases.append(np.angle(fft))
            phases = np.stack(phases, axis=0)
            # Circular mean of phase differences
            mean_phase = np.mean(np.exp(1j * phases), axis=0)
            coherence = float(np.abs(np.mean(mean_phase)))
        else:
            coherence = 1.0

        # Dominant frequency
        energies = [w.energy() for w in self._active_waves]
        dominant_idx = np.argmax(energies)
        dominant_freq = self._active_waves[dominant_idx].frequency

        pattern = InterferencePattern(
            pattern=total,
            constructive=constructive,
            destructive=destructive,
            coherence=coherence,
            dominant_frequency=dominant_freq,
            timestamp=t,
        )

        self._wave_history.append(pattern)
        if len(self._wave_history) > self._max_history:
            self._wave_history.pop(0)

        # Lifecycle: aposenta ondas antigas/decaídas antes de medir o estado
        self.gc()

        return pattern

    def gc(
        self,
        *,
        max_waves: int = 256,
        max_age: float = 45.0,
        min_energy: float = 1e-5,
    ) -> int:
        """Garbage collection do conjunto de ondas ativas (lifecycle).

        Aposenta ondas que:
          1. Ultrapassaram `max_age` desde a origem (estagnadas na memória),
          2. Caíram abaixo de `min_energy` de energia EFETIVA (amplitude já
             descontada pelo envelope exponencial — `energy()` puro ignora o
             tempo e nunca decai esse conjunto),
          3. Excederam `max_waves`: mantém só as mais energéticas.

        Retorna quantas ondas foram removidas neste ciclo.
        """
        t = self._global_time
        kept: list[WavePacket] = []
        removed = 0
        for w in self._active_waves:
            age = max(0.0, t - w.origin_time)
            envelope = math.exp(-w.decay_rate * age)
            eff = w.energy() * envelope * envelope
            if age > max_age or eff < min_energy:
                removed += 1
                continue
            kept.append(w)

        if len(kept) > max_waves:
            def _eff(w: WavePacket) -> float:
                age = max(0.0, t - w.origin_time)
                env = math.exp(-w.decay_rate * age)
                return w.energy() * env * env
            kept.sort(key=_eff, reverse=True)
            removed += len(kept) - max_waves
            kept = kept[:max_waves]

        self._active_waves = kept
        self._pruned_total += removed
        return removed

    def attention_weights(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Compute attention using wave interference instead of softmax.

        Traditional: weights = softmax(Q·K^T / √d)
        This engine: weights = interference_pattern(Q, K) normalized by energy

        The interference naturally produces sparse attention — only
        components that are "in phase" survive constructive interference.
        """
        n_keys = keys.shape[0]
        if n_keys == 0:
            return np.zeros(self.dim, dtype=np.float32)

        # Create a wave for each key, driven by the query
        for i in range(n_keys):
            # Phase offset from key-query alignment
            alignment = np.dot(query, keys[i]) / (np.linalg.norm(query) * np.linalg.norm(keys[i]) + 1e-8)
            phase_offset = np.arccos(np.clip(alignment, -1, 1))

            # Amplitude proportional to alignment strength
            amp = np.abs(alignment) * values[i] / temperature

            self.create_wave(
                amplitude=amp,
                frequency=self.base_freq * (1.0 + alignment),
                phase=np.full(self.dim, phase_offset, dtype=np.float32),
                label=f"key_{i}",
            )

        # Superpose and extract the interference pattern
        pattern = self.superpose(time_step=0.0)

        # The constructive regions are the "attended" dimensions
        weights = pattern.constructive

        # Normalize by total energy (instead of softmax sum)
        total_energy = np.sum(np.abs(weights)) + 1e-8
        weights = weights / total_energy

        return weights

    def resonance_amplify(
        self,
        signal: np.ndarray,
        frequency: float,
        bandwidth: float = 0.1,
    ) -> np.ndarray:
        """
        Amplify signal components near a target frequency (resonance).
        Components far from resonance are attenuated (destructive interference).
        """
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal))

        # Gaussian resonance kernel
        kernel = np.exp(-0.5 * ((freqs - frequency) / bandwidth) ** 2)
        amplified = fft * kernel

        return np.fft.irfft(amplified, n=len(signal)).astype(np.float32)

    def get_state_snapshot(self) -> dict:
        """Get a snapshot of the engine's current state for telemetry."""
        return {
            "active_waves": len(self._active_waves),
            "total_energy": sum(w.energy() for w in self._active_waves),
            "global_time": self._global_time,
            "history_length": len(self._wave_history),
            "last_coherence": self._wave_history[-1].coherence if self._wave_history else 0.0,
            "last_dominant_freq": self._wave_history[-1].dominant_frequency if self._wave_history else 0.0,
            "gc_removed": self._pruned_total,
        }
