"""
AutoImmuneShield — Self-Monitoring Electromagnetic Immunity

The biological immune system works by:
  1. Detecting "non-self" patterns (antigens)
  2. Activating targeted responses (antibodies)
  3. Remembering past threats (immune memory)
  4. Maintaining tolerance to "self" (preventing autoimmunity)

The AutoImmuneShield applies this to electromagnetic health:
  1. Detects anomalous EM patterns (perturbations, bit-flips, resonance)
  2. Activates targeted responses (shielding, throttling, migration)
  3. Remembers past threat signatures (pattern library)
  4. Distinguishes normal operation from genuine threats

This extends the existing integrity_shield.py (14-line decorator) into a
full immune system that is AWARE of the electromagnetic environment.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class ThreatSignature:
    """A known threat pattern for the immune memory."""
    pattern: np.ndarray       # The detected anomaly pattern
    severity: float           # How dangerous [0,1]
    response: str             # What action was taken
    timestamp: float
    false_positive: bool = False


@dataclass
class ImmuneResponse:
    """An action taken by the immune system."""
    action: str               # 'shield', 'throttle', 'migrate', 'isolate', 'alert'
    target: str               # What component is affected
    intensity: float          # How strongly to apply [0,1]
    reason: str
    timestamp: float = 0.0


class AutoImmuneShield:
    """
    The electromagnetic immune system of the AEON architecture.

    Monitors the EM field, magnetic lattice, and telemetry for threats.
    Responds autonomously to protect computational integrity.
    """

    def __init__(self):
        # Threat pattern library (immune memory)
        self._memory: list[ThreatSignature] = []
        self._max_memory = 256

        # Response history
        self._responses: list[ImmuneResponse] = []

        # Active shields
        self._active_shields: dict[str, float] = {}  # region → intensity
        self._throttle_level: float = 0.0
        self._isolated_regions: set[str] = set()

        # Tolerance threshold (prevents autoimmunity)
        self._tolerance: float = 0.3

        # Callbacks for system integration
        self._on_shield: Optional[Callable] = None
        self._on_throttle: Optional[Callable] = None
        self._on_isolate: Optional[Callable] = None
        self._on_alert: Optional[Callable] = None

    def register_callbacks(
        self,
        on_shield: Optional[Callable] = None,
        on_throttle: Optional[Callable] = None,
        on_isolate: Optional[Callable] = None,
        on_alert: Optional[Callable] = None,
    ):
        """Register callbacks for immune responses."""
        self._on_shield = on_shield
        self._on_throttle = on_throttle
        self._on_isolate = on_isolate
        self._on_alert = on_alert

    def scan(
        self,
        em_field_energy: float,
        lattice_coherence: float,
        perturbation_count: int,
        threat_level: float,
        resonance_map: Optional[dict] = None,
        energy_delta: float = 0.0,
        wave_count: int = 0,
    ) -> list[ImmuneResponse]:
        """
        Perform a full immune scan and generate responses.
        This is called periodically by the orchestrator.
        """
        responses = []
        self._prev_energy = getattr(self, '_prev_energy', em_field_energy)

        # Check 1: Excessive perturbation rate
        if perturbation_count > 5:
            severity = min(perturbation_count / 15.0, 1.0)
            if severity > self._tolerance:
                resp = ImmuneResponse(
                    action="shield",
                    target="global",
                    intensity=severity,
                    reason=f"High perturbation rate: {perturbation_count} events",
                    timestamp=time.time(),
                )
                responses.append(resp)

        # Check 2: Low lattice coherence (field is dephasing)
        if lattice_coherence < 0.5:
            resp = ImmuneResponse(
                action="throttle",
                target="lattice",
                intensity=1.0 - lattice_coherence,
                reason=f"Low coherence: {lattice_coherence:.3f}",
                timestamp=time.time(),
            )
            responses.append(resp)

        # Check 3: Rapid energy increase (runaway oscillation)
        if energy_delta > 0.01:
            severity = min(energy_delta / 0.05, 1.0)
            resp = ImmuneResponse(
                action="throttle",
                target="em_field",
                intensity=severity,
                reason=f"Rapid energy increase: Δ{energy_delta:.4f}",
                timestamp=time.time(),
            )
            responses.append(resp)

        # Check 4: Dangerous resonance frequencies
        if resonance_map:
            for freq_key, amplitude in resonance_map.items():
                if amplitude > 2.0:
                    resp = ImmuneResponse(
                        action="isolate",
                        target=freq_key,
                        intensity=min(amplitude / 5.0, 1.0),
                        reason=f"Dangerous resonance at {freq_key}: amplitude {amplitude:.2f}",
                        timestamp=time.time(),
                    )
                    responses.append(resp)

        # Check 5: Overall threat level
        if threat_level > 0.3:
            resp = ImmuneResponse(
                action="alert",
                target="system",
                intensity=threat_level,
                reason=f"Critical threat level: {threat_level:.2f}",
                timestamp=time.time(),
            )
            responses.append(resp)

        # Check 6: Wave count too high (information overload)
        if wave_count > 20:
            resp = ImmuneResponse(
                action="throttle",
                target="wave_engine",
                intensity=min((wave_count - 20) / 20.0, 1.0),
                reason=f"Wave overload: {wave_count} active waves",
                timestamp=time.time(),
            )
            responses.append(resp)

        # Check 7: Energy spike (sudden burst)
        energy_spike = em_field_energy - self._prev_energy
        self._prev_energy = em_field_energy
        if energy_spike > 0.005:
            resp = ImmuneResponse(
                action="shield",
                target="energy_spike",
                intensity=min(energy_spike / 0.02, 1.0),
                reason=f"Energy spike: {energy_spike:.4f}",
                timestamp=time.time(),
            )
            responses.append(resp)

        # Remember threat patterns
        for resp in responses:
            self._remember(ThreatSignature(
                pattern=np.array([em_field_energy, lattice_coherence,
                                 perturbation_count, threat_level]),
                severity=resp.intensity,
                response=resp.action,
                timestamp=resp.timestamp,
            ))

        # Execute responses
        for resp in responses:
            self._execute_response(resp)

        self._responses.extend(responses)
        if len(self._responses) > self._max_memory:
            self._responses = self._responses[-self._max_memory:]

        return responses

    def _execute_response(self, response: ImmuneResponse):
        """Execute an immune response action."""
        if response.action == "shield" and self._on_shield:
            self._active_shields[response.target] = response.intensity
            self._on_shield(response.target, response.intensity)

        elif response.action == "throttle" and self._on_throttle:
            self._throttle_level = max(self._throttle_level, response.intensity)
            self._on_throttle(response.intensity)

        elif response.action == "isolate" and self._on_isolate:
            self._isolated_regions.add(response.target)
            self._on_isolate(response.target)

        elif response.action == "alert" and self._on_alert:
            self._on_alert(response.severity, response.reason)

    def _remember(self, signature: ThreatSignature):
        """Store a threat signature in immune memory."""
        # Don't remember false positives
        if signature.false_positive:
            return

        # Deduplication: don't store similar patterns
        for existing in self._memory[-10:]:
            if self._patterns_similar(existing.pattern, signature.pattern):
                return

        self._memory.append(signature)
        if len(self._memory) > self._max_memory:
            self._memory.pop(0)

    def _patterns_similar(self, a: np.ndarray, b: np.ndarray, threshold: float = 0.9) -> bool:
        """Check if two threat patterns are similar (cosine similarity)."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return False
        similarity = np.dot(a, b) / (norm_a * norm_b)
        return similarity > threshold

    def recognize_threat(self, current_pattern: np.ndarray) -> Optional[ThreatSignature]:
        """
        Check if a current pattern matches any known threat in memory.
        This is the immune system's RECOGNITION capability.
        """
        for signature in reversed(self._memory):
            if self._patterns_similar(current_pattern, signature.pattern):
                return signature
        return None

    def reset_throttle(self):
        """Reset throttle level (called when threat subsides)."""
        self._throttle_level *= 0.9  # Gradual release

    def deactivate_shield(self, target: str):
        """Deactivate a specific shield."""
        self._active_shields.pop(target, None)

    def get_status(self) -> dict:
        """Get the current immune system status."""
        return {
            "active_shields": dict(self._active_shields),
            "throttle_level": self._throttle_level,
            "isolated_regions": list(self._isolated_regions),
            "memory_size": len(self._memory),
            "recent_responses": len([
                r for r in self._responses
                if time.time() - r.timestamp < 60
            ]),
            "total_responses": len(self._responses),
        }
