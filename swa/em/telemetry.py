"""
EMTelemetry — Bit-Flip Awareness as Environmental Sensing

The existing entropy_daemon collects CPU temp, RAM, swap, etc.
EMTelemetry EXPANDS this to treat the hardware as an electromagnetic
environment where perturbations are meaningful signals, not just noise.

Core concept: A bit-flip is not an ERROR — it is a TELEMETRY POINT
that reveals the electromagnetic state of the environment.

  [Ambiente Externo] → [Acoplamento Físico] → [Estado de Silício]
       → [Dado Corrompido] ← PONTO DE OBSERVAÇÃO
       → [Computação Resiliente]

The EMTelemetry module:
  1. Monitors hardware signals (extending entropy_daemon)
  2. Detects patterns in perturbation timing (periodic = motor noise, random = cosmic)
  3. Maps perturbation locations to lattice regions (where on the die?)
  4. Provides adaptive response (throttling, shielding, migration)
"""

import math
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class EMPerturbation:
    """A detected electromagnetic perturbation event."""
    timestamp: float
    magnitude: float           # Strength of the perturbation
    frequency: float           # Estimated frequency (if periodic)
    spatial_region: tuple[int, int]  # Where on the lattice (i,j)
    source_class: str          # 'cosmic', 'thermal', 'switching', 'external', 'unknown'
    confidence: float          # Classification confidence [0,1]
    affected_bits: int         # Number of bits potentially affected


@dataclass
class EMSignature:
    """Electromagnetic signature of the current environment."""
    noise_floor: float         # Baseline noise level
    peak_frequency: float      # Dominant interference frequency
    spectral_centroid: float   # Center of mass of the frequency spectrum
    temporal_regularity: float # How periodic the noise is [0=random, 1=perfectly periodic]
    threat_level: float        # Overall threat assessment [0=safe, 1=critical]
    recommendations: list[str] = field(default_factory=list)


class EMTelemetry:
    """
    Extended hardware telemetry that treats the system as an
    electromagnetic environment.

    Combines data from:
      - entropy_daemon (CPU temp, RAM, swap, I/O, context switches)
      - Direct hardware registers (when available)
      - Software-level anomaly detection (bit patterns, hash consistency)
      - Temporal pattern analysis (periodicity detection)
    """

    def __init__(self, buffer_size: int = 256):
        self._history: list[dict] = []
        self._buffer_size = buffer_size
        self._perturbations: list[EMPerturbation] = []
        self._baseline: Optional[dict] = None
        self._baseline_samples = 0
        self._baseline_target = 50

        # Spectral analysis buffer
        self._spectral_buffer: list[float] = []
        self._spectral_window = 128

        # Pattern detection
        self._recent_hashes: list[str] = []
        self._hash_window = 32

    def ingest_hardware_state(self, state: dict):
        """
        Ingest hardware state from entropy_daemon or similar source.
        Expected keys: cpu_temp, ram_used, swap_used, context_switches,
                       interrupts, disk_read, disk_write
        """
        self._history.append({**state, "timestamp": time.time()})
        if len(self._history) > self._buffer_size:
            self._history.pop(0)

        # Update baseline
        if self._baseline_samples < self._baseline_target:
            self._update_baseline(state)
            self._baseline_samples += 1

        # Detect anomalies against baseline
        if self._baseline is not None:
            anomalies = self._detect_anomalies(state)
            for anomaly in anomalies:
                self._perturbations.append(anomaly)
                if len(self._perturbations) > self._buffer_size:
                    self._perturbations.pop(0)

        # Update spectral analysis
        self._update_spectral(state)

    def _update_baseline(self, state: dict):
        """Build baseline statistics from initial observations."""
        if self._baseline is None:
            self._baseline = {k: {"sum": 0.0, "sum_sq": 0.0, "count": 0}
                             for k in state.keys() if isinstance(state[k], (int, float))}

        for k, v in state.items():
            if k in self._baseline and isinstance(v, (int, float)):
                self._baseline[k]["sum"] += v
                self._baseline[k]["sum_sq"] += v ** 2
                self._baseline[k]["count"] += 1

    def _detect_anomalies(self, state: dict) -> list[EMPerturbation]:
        """Compare current state against baseline to detect perturbations."""
        anomalies = []
        now = time.time()

        for k, v in state.items():
            if k in self._baseline and isinstance(v, (int, float)):
                b = self._baseline[k]
                if b["count"] < 10:
                    continue

                mean = b["sum"] / b["count"]
                variance = b["sum_sq"] / b["count"] - mean ** 2
                std = math.sqrt(max(variance, 1e-10))

                # Z-score based anomaly detection
                z_score = abs(v - mean) / std if std > 1e-10 else 0

                if z_score > 3.0:  # 3-sigma threshold
                    anomaly = EMPerturbation(
                        timestamp=now,
                        magnitude=z_score,
                        frequency=self._estimate_frequency(k),
                        spatial_region=self._map_to_lattice(k),
                        source_class=self._classify_source(k, z_score),
                        confidence=min(z_score / 10.0, 1.0),
                        affected_bits=1,
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _estimate_frequency(self, metric: str) -> float:
        """Estimate the frequency of perturbations for a given metric."""
        recent = [
            p for p in self._perturbations
            if p.spatial_region == self._map_to_lattice(metric)
        ]
        if len(recent) < 3:
            return 0.0

        # Time intervals between recent perturbations
        intervals = [
            recent[i+1].timestamp - recent[i].timestamp
            for i in range(len(recent) - 1)
        ]
        mean_interval = sum(intervals) / len(intervals)
        return 1.0 / mean_interval if mean_interval > 0 else 0.0

    def _map_to_lattice(self, metric: str) -> tuple[int, int]:
        """Map a hardware metric to a position on the 64×64 lattice."""
        # Deterministic hash-based mapping
        h = int(hashlib.md5(metric.encode()).hexdigest()[:8], 16)
        row = h % 64
        col = (h >> 6) % 64
        return (row, col)

    def _classify_source(self, metric: str, magnitude: float) -> str:
        """Classify the source of the perturbation."""
        if magnitude > 10:
            return "external"   # Strong external influence
        elif magnitude > 5:
            return "switching"  # Digital switching noise
        elif magnitude > 3:
            return "thermal"    # Thermal fluctuation
        else:
            return "cosmic"     # Low-level cosmic/background

    def _update_spectral(self, state: dict):
        """Maintain spectral analysis buffer for frequency detection."""
        # Use a composite metric as the "signal"
        signal_value = sum(
            v for v in state.values() if isinstance(v, (int, float))
        )
        self._spectral_buffer.append(signal_value)
        if len(self._spectral_buffer) > self._spectral_window:
            self._spectral_buffer.pop(0)

    def compute_signature(self) -> EMSignature:
        """Compute the current electromagnetic signature of the environment."""
        if not self._history:
            return EMSignature(
                noise_floor=0, peak_frequency=0, spectral_centroid=0,
                temporal_regularity=0, threat_level=0,
            )

        # Noise floor from recent history
        recent = self._history[-32:]
        values = []
        for h in recent:
            values.extend(v for v in h.values() if isinstance(v, (int, float)))

        noise_floor = float(np.std(values)) if values else 0.0

        # Spectral analysis
        peak_freq = 0.0
        centroid = 0.0
        if len(self._spectral_buffer) >= 16:
            spectrum = np.abs(np.fft.rfft(self._spectral_buffer))
            freqs = np.fft.rfftfreq(len(self._spectral_buffer))
            total_power = np.sum(spectrum) + 1e-10

            peak_idx = np.argmax(spectrum)
            peak_freq = float(freqs[peak_idx])
            centroid = float(np.sum(freqs * spectrum) / total_power)

        # Temporal regularity (autocorrelation of perturbation timestamps)
        regularity = 0.0
        if len(self._perturbations) >= 5:
            timestamps = [p.timestamp for p in self._perturbations[-20:]]
            intervals = np.diff(timestamps)
            if len(intervals) > 1:
                # Coefficient of variation — low CV = high regularity
                cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                regularity = float(1.0 / (1.0 + cv))

        # Threat level
        recent_perturbations = [
            p for p in self._perturbations
            if time.time() - p.timestamp < 60
        ]
        threat = min(len(recent_perturbations) / 10.0, 1.0)
        threat = min(max(threat, noise_floor / 100.0), 1.0)

        # Recommendations
        recs = []
        if threat > 0.8:
            recs.append("CRITICAL: Consider throttling clock frequency")
            recs.append("Activate toroidal shielding on critical regions")
        elif threat > 0.5:
            recs.append("WARNING: Elevated EM activity — increase ECC checking")
        elif threat > 0.3:
            recs.append("NOTICE: Moderate interference — monitor lattice coherence")
        if regularity > 0.8:
            recs.append("Periodic interference detected — likely switching noise")

        return EMSignature(
            noise_floor=noise_floor,
            peak_frequency=peak_freq,
            spectral_centroid=centroid,
            temporal_regularity=regularity,
            threat_level=threat,
            recommendations=recs,
        )

    def detect_bitflip_pattern(self, data: bytes) -> Optional[dict]:
        """
        Detect if data has been corrupted by comparing against
        expected patterns (checksums, known structures).
        """
        current_hash = hashlib.sha256(data).hexdigest()
        self._recent_hashes.append(current_hash)
        if len(self._recent_hashes) > self._hash_window:
            self._recent_hashes.pop(0)

        # Check for duplicate hashes (data stagnation)
        if len(self._recent_hashes) > 1:
            unique = len(set(self._recent_hashes))
            total = len(self._recent_hashes)
            if unique < total * 0.8:
                return {
                    "type": "stagnation",
                    "unique_ratio": unique / total,
                    "severity": "low",
                }

        return None

    def get_perturbation_map(self) -> np.ndarray:
        """
        Generate a 64×64 spatial map of recent perturbations.
        This IS the EM telemetry overlay on the magnetic lattice.
        """
        perturbation_map = np.zeros((64, 64), dtype=np.float32)

        for p in self._perturbations:
            if time.time() - p.timestamp < 300:  # Last 5 minutes
                r, c = p.spatial_region
                perturbation_map[r, c] += p.magnitude

        # Normalize
        max_val = np.max(perturbation_map)
        if max_val > 0:
            perturbation_map /= max_val

        return perturbation_map
