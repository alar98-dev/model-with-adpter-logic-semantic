"""
EM Bridge — Bidirectional LLM ↔ Electromagnetic Core Translator

This module closes the loop between the LLM's internal representation
and the EM field, creating a true feedback cycle:

  LLM hidden states  →  EM field sources  →  magnetic lattice
       ↑                                            │
       └──── matrixial projection ←── field state ──┘

The bridge operates in three directions:

  1. WRITE: LLM hidden states become EM field sources
     The last decoder layer's output (4096-dim for Gemma-1B)
     is registered as a new source in the EM field. This means
     the LLM's "thought" becomes an electromagnetic wave that
     propagates through the field, interferes with other sources,
     and excites the magnetic lattice.

  2. READ: EM field state becomes embedding injection
     The matrixial projection (stabilized E vector) is projected
     back into the LLM's embedding space and injected as a soft
     prompt via the embed_hook. This means the EM field's
     "opinion" modulates what the LLM generates next.

  3. MODULATE: EM resonance shapes logit probabilities
     The resonance map from the EM field is used to create
     a logit bias that amplifies tokens associated with
     constructive interference regions and suppresses tokens
     from destructive interference regions.

Physical analogy:
  - LLM is a radio transmitter (encodes thoughts as waves)
  - EM field is the atmosphere (waves propagate, interfere)
  - Magnetic lattice is the antenna array (receives and stores)
  - Matrixial relation is the signal processor (extracts stable signal)
  - The bridge is the coaxial cable (bidirectional feedline)
"""

import os
import time
import logging
import numpy as np
import torch
from typing import Optional
from swa.dimensions import MODEL_DIM
from swa.projections import load_or_create_projection


class SpectralSignature:
    """Extracts statistical and spectral features from a hidden state vector."""

    @staticmethod
    def extract(h: np.ndarray) -> dict:
        """
        Compute spectral signature of a hidden state vector.
        Returns dict with statistical moments and frequency band energies.
        """
        n = len(h)
        if n == 0:
            return {"variance": 0.0, "kurtosis": 0.0, "skewness": 0.0,
                    "low_band": 0.0, "mid_band": 0.0, "high_band": 0.0,
                    "spectral_entropy": 0.0, "peak_freq": 0.0}

        # Statistical moments (normalized)
        mean = np.mean(h)
        centered = h - mean
        variance = float(np.var(centered))
        std = np.sqrt(max(variance, 1e-12))

        skewness = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3.0)  # excess

        # Frequency band energies via FFT
        spectrum = np.abs(np.fft.rfft(centered))
        total_energy = float(np.sum(spectrum ** 2))
        if total_energy < 1e-12:
            return {"variance": variance, "kurtosis": kurtosis, "skewness": skewness,
                    "low_band": 0.0, "mid_band": 0.0, "high_band": 0.0,
                    "spectral_entropy": 0.0, "peak_freq": 0.0}

        # Band energies: low (0-25%), mid (25-75%), high (75-100%)
        n_bins = len(spectrum)
        low_end = max(1, n_bins // 4)
        mid_end = max(low_end + 1, 3 * n_bins // 4)

        low_energy = float(np.sum(spectrum[:low_end] ** 2))
        mid_energy = float(np.sum(spectrum[low_end:mid_end] ** 2))
        high_energy = float(np.sum(spectrum[mid_end:] ** 2))

        low_band = low_energy / total_energy
        mid_band = mid_energy / total_energy
        high_band = high_energy / total_energy

        # Spectral entropy (higher = more complex signal)
        probs = spectrum / np.sum(spectrum)
        probs = probs[probs > 1e-12]
        spectral_entropy = float(-np.sum(probs * np.log2(probs)))

        # Peak frequency index
        peak_freq = float(np.argmax(spectrum)) / max(n_bins, 1)

        return {
            "variance": variance,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "low_band": low_band,
            "mid_band": mid_band,
            "high_band": high_band,
            "spectral_entropy": spectral_entropy,
            "peak_freq": peak_freq,
        }

    @staticmethod
    def classify(sig: dict) -> str:
        """
        Classify input based on spectral signature.

        Categories:
          existential  — high complexity, balanced bands, high variance
          mathematical — low entropy, concentrated low/mid band
          adversarial  — high kurtosis, high-band dominance, skewed
          factual      — low complexity, low entropy, narrow spectrum
        """
        high_complexity = (
            sig["spectral_entropy"] > 3.0 or
            sig["kurtosis"] > 1.5 or
            (sig["high_band"] > 0.3 and sig["variance"] > 0.01)
        )

        adversarial = (
            sig["skewness"] > 2.0 or
            sig["kurtosis"] > 3.0 or
            (sig["high_band"] > 0.5 and sig["low_band"] < 0.1)
        )

        factual = (
            sig["spectral_entropy"] < 2.0 and
            sig["variance"] < 0.005 and
            abs(sig["skewness"]) < 0.5
        )

        if adversarial:
            return "adversarial"
        if factual:
            return "factual"
        if high_complexity:
            return "existential"
        return "mathematical"


class EMBridge:
    """
    Bidirectional translator between LLM and EMCore.

    Usage in aeonc.py:
        bridge = EMBridge(em_core, model_dim=2048)

        # In _embed_hook: read EM state, return injection vector
        em_injection = bridge.read_for_embedding(current_embeddings)

        # In _capture_hidden: write LLM state to EM field
        bridge.write_from_hidden(hidden_state)

        # In logits processor: modulate logits with EM resonance
        bridge.modulate_logits(logits)
    """

    def __init__(self, em_core, model_dim: int = MODEL_DIM):
        """
        Args:
            em_core: EMCore instance (from swa.em.integration)
            model_dim: LLM hidden dimension (2048 for Gemma-1B)
        """
        self.em = em_core
        self._logger = logging.getLogger("swa.em.bridge")
        self.model_dim = model_dim
        self.em_dim = em_core.dim  # 4096

        # Projection layers: LLM dim ↔ EM dim
        self._llm_to_em = self._make_projection(model_dim, self.em_dim)
        self._em_to_llm = self._make_projection(self.em_dim, model_dim)

        # State tracking
        self._last_em_injection: Optional[np.ndarray] = None
        self._last_write_time: float = 0.0
        self._write_interval: float = 0.05  # Minimum seconds between writes
        self._cycle_count: int = 0

        # Modulation strength (how much EM state influences generation)
        self.modulation_strength: float = 0.05

        # Logit bias map (computed from resonance frequencies)
        self._logit_bias_cache: Optional[torch.Tensor] = None
        self._logit_bias_ttl: float = 0.0
        self._logit_bias_max_age: float = 2.0  # seconds

        # ── REACTIVITY: spectral signature tracking ──
        self._input_norm_history: list[float] = []
        self._input_direction_ema: Optional[np.ndarray] = None
        self._direction_ema_alpha: float = 0.1
        self._last_signature: Optional[dict] = None
        self._last_category: str = "unknown"
        self._category_history: list[str] = []

        # Category-specific injection parameters
        self._category_params = {
            "existential":  {"harmonics": 5, "amplitude_mult": 1.5, "spread": 0.8},
            "mathematical": {"harmonics": 2, "amplitude_mult": 1.0, "spread": 0.3},
            "factual":      {"harmonics": 1, "amplitude_mult": 0.6, "spread": 0.1},
            "adversarial":  {"harmonics": 3, "amplitude_mult": 2.0, "spread": 1.0},
        }

        # ── MEMORY: accumulate energy from past writes ──
        self._accumulated_energy: float = 0.0
        self._source_stack: list[np.ndarray] = []  # recent sources
        self._max_stack: int = 16
        self._hysteresis_bias: np.ndarray = np.zeros(em_core.dim, dtype=np.float32)
        self._hysteresis_alpha: float = 0.05  # retention of previous E state

        # ── ADAPTATION: momentum phase vector ──
        self._momentum: Optional[np.ndarray] = None
        self._momentum_alpha: float = 0.3
        self._phase_history: list[float] = []  # phase angles of successive inputs
        self._amplitude_history: list[float] = []

    def _make_projection(self, in_dim: int, out_dim: int) -> np.ndarray:
        """Load a stable, versioned projection (returned as ``in × out``)."""
        root = os.environ.get("SWA_EM_PROJECTION_PATH", "models/em_projections")
        suffix = f"{in_dim}_to_{out_dim}.pt"
        path = os.path.join(root, suffix)
        layer = load_or_create_projection(in_dim, out_dim, path, device="cpu")
        return layer.weight.detach().cpu().numpy().T.astype(np.float32)

    # ── WRITE: LLM → EM Field ──────────────────────────────────────────

    def write_from_hidden(self, hidden: torch.Tensor):
        """
        Write the LLM's last hidden state into the EM field as a source.

        REACTIVITY: Extracts spectral signature, classifies input, and
        uses category-specific injection parameters (harmonics, amplitude,
        spread).

        MEMORY: Applies hysteresis — blends current projection with
        residual from previous E field. Constructive interference for
        in-phase successive inputs.

        ADAPTATION: Momentum vector accumulates trajectory. Amplitude
        modulation: A(t) = A_0 * (1 + tanh(momentum)).
        """
        now = time.time()
        if now - self._last_write_time < self._write_interval:
            return

        if hidden is None:
            return

        self._last_write_time = now
        self._cycle_count += 1

        try:
            if isinstance(hidden, torch.Tensor):
                h = hidden.detach().cpu().float().numpy()
            else:
                h = np.array(hidden, dtype=np.float32)

            if h.ndim > 1:
                h = h.flatten()

            if len(h) > self.model_dim:
                h = h[:self.model_dim]
            elif len(h) < self.model_dim:
                padded = np.zeros(self.model_dim, dtype=np.float32)
                padded[:len(h)] = h
                h = padded

            # ═══════════════════════════════════════════════════════════
            # REACTIVITY: Spectral signature extraction
            # ═══════════════════════════════════════════════════════════
            sig = SpectralSignature.extract(h)
            self._last_signature = sig
            category = SpectralSignature.classify(sig)
            self._last_category = category
            self._category_history.append(category)
            if len(self._category_history) > 64:
                self._category_history = self._category_history[-64:]

            cat_params = self._category_params[category]
            harmonics = cat_params["harmonics"]
            amp_mult = cat_params["amplitude_mult"]
            spread = cat_params["spread"]

            # Raw magnitude for energy scaling
            raw_norm = float(np.linalg.norm(h))
            self._input_norm_history.append(raw_norm)
            if len(self._input_norm_history) > 64:
                self._input_norm_history = self._input_norm_history[-64:]

            # ═══════════════════════════════════════════════════════════
            # ADAPTATION: Momentum and phase tracking
            # ═══════════════════════════════════════════════════════════
            h_unit = h / max(raw_norm, 1e-8)
            if self._input_direction_ema is None:
                self._input_direction_ema = h_unit.copy()
            else:
                a = self._direction_ema_alpha
                self._input_direction_ema = (1 - a) * self._input_direction_ema + a * h_unit

            # Phase angle (arctan of first two principal components)
            cos_sim = float(np.dot(h_unit, self._input_direction_ema))
            phase = np.arccos(np.clip(cos_sim, -1.0, 1.0))
            self._phase_history.append(phase)
            if len(self._phase_history) > 32:
                self._phase_history = self._phase_history[-32:]

            # Momentum: running average of phase differences
            if len(self._phase_history) >= 2:
                phase_diff = self._phase_history[-1] - self._phase_history[-2]
                if self._momentum is None:
                    self._momentum = np.array([phase_diff], dtype=np.float32)
                else:
                    self._momentum = (1 - self._momentum_alpha) * self._momentum + \
                                     self._momentum_alpha * np.array([phase_diff])
            momentum_mag = float(np.linalg.norm(self._momentum)) if self._momentum is not None else 0.0

            # Amplitude modulation: A(t) = A_0 * (1 + tanh(momentum))
            amplitude_mod = 1.0 + np.tanh(momentum_mag)

            # Novelty (how different from running average)
            novelty = 1.0 - max(-1.0, min(1.0, cos_sim))

            # ═══════════════════════════════════════════════════════════
            # Project to EM dimension with category-specific parameters
            # ═══════════════════════════════════════════════════════════
            em_vector = h @ self._llm_to_em
            em_norm = float(np.linalg.norm(em_vector))
            if em_norm > 1e-8:
                # Base scale: magnitude × category amplitude × novelty × momentum
                energy_scale = min(3.0, 0.3 + self._accumulated_energy * 0.005)
                scale = (raw_norm * 0.15) * amp_mult * (0.3 + novelty * 0.7) * \
                        energy_scale * amplitude_mod
                em_vector = em_vector / em_norm * scale

            # ═══════════════════════════════════════════════════════════
            # REACTIVITY: Category-specific harmonic injection
            # ═══════════════════════════════════════════════════════════
            if harmonics > 1:
                harmonic_vector = em_vector.copy()
                for k in range(2, harmonics + 1):
                    shifted = np.roll(em_vector, k * 37) * (1.0 / k)
                    harmonic_vector += shifted
                em_vector = harmonic_vector

            if spread > 0.5:
                noise = np.random.randn(self.em_dim).astype(np.float32)
                noise = noise / max(np.linalg.norm(noise), 1e-8)
                em_vector[:self.em_dim] += noise * spread * 0.3

            # ═══════════════════════════════════════════════════════════
            # RESILIENCE: Adversarial input → suppress energy injection
            # ═══════════════════════════════════════════════════════════
            if category == "adversarial":
                # Trigger shield scan with elevated threat
                self.em.shield.scan(
                    em_field_energy=self.em.field.state.energy(),
                    lattice_coherence=self.em.field.state.coherence,
                    perturbation_count=10,  # simulate high perturbation
                    threat_level=0.8,       # adversarial = high threat
                    energy_delta=float(np.sum(em_vector ** 2)),
                    wave_count=len(self._source_stack),
                )
                # Suppress amplitude: adversarial energy should be absorbed
                em_vector *= 0.15  # strong suppression

            # Track accumulated energy
            self._accumulated_energy += float(np.sum(em_vector ** 2))

            # ═══════════════════════════════════════════════════════════
            # MEMORY: Constructive interference for in-phase inputs
            # ═══════════════════════════════════════════════════════════
            if len(self._source_stack) > 0:
                prev_source = self._source_stack[-1]
                # Phase coherence: dot product of current and previous
                coherence = float(np.dot(
                    em_vector[:len(prev_source)] / max(np.linalg.norm(em_vector[:len(prev_source)]), 1e-8),
                    prev_source / max(np.linalg.norm(prev_source), 1e-8)
                ))
                # In-phase (coherence > 0.5): constructive gain
                if coherence > 0.5:
                    gain = 1.0 + coherence * 0.5
                    em_vector[:len(prev_source)] *= gain

            # ═══════════════════════════════════════════════════════════
            # MEMORY: Push source onto stack
            # ═══════════════════════════════════════════════════════════
            self._source_stack.append(em_vector[:self.em_dim].copy())
            if len(self._source_stack) > self._max_stack:
                self._source_stack = self._source_stack[-self._max_stack:]

            # Combined source with exponential decay for older memories
            combined = np.zeros(self.em_dim, dtype=np.float32)
            weight = 1.0
            for src in reversed(self._source_stack):
                n = len(src)
                combined[:n] += src[:n] * weight
                weight *= 0.75

            # Normalize combined source to unit norm (prevents energy explosion)
            # Category differentiation happens via wave amplitude, not source norm
            combined_norm = np.linalg.norm(combined)
            if combined_norm > 1e-8:
                combined = combined / combined_norm

            # ═══════════════════════════════════════════════════════════
            # MEMORY: Hysteresis — blend with previous E state
            # ═══════════════════════════════════════════════════════════
            E_prev = self.em.field.state.E
            combined_with_hysteresis = (
                (1 - self._hysteresis_alpha) * combined +
                self._hysteresis_alpha * E_prev
            )
            # Update hysteresis bias for next cycle
            self._hysteresis_bias = combined_with_hysteresis[:self.em_dim].copy()

            # ═══════════════════════════════════════════════════════════
            # REACTIVITY: Category-specific source amplitude
            # ═══════════════════════════════════════════════════════════
            # The combined source is normalized, but we scale the FIELD
            # SOURCE contribution based on category. This makes the field
            # respond with different energy levels per category.
            cat_source_scale = {
                "factual": 0.3,
                "mathematical": 0.6,
                "existential": 1.2,
                "philosophical": 0.9,
                "emergent": 1.0,
                "adversarial": 0.05,
            }
            source_scale = cat_source_scale.get(category, 0.3)
            self.em.field.register_source("llm_thought", combined * source_scale)

            # Inject into lattice
            center = (32, 32)
            self.em.lattice.inject_magnon(center, self._accumulated_energy * 0.002)

            # ═══════════════════════════════════════════════════════════
            # REACTIVITY: Category-specific wave amplitude
            # ═══════════════════════════════════════════════════════════
            cat_wave_amp = {
                "factual": 0.2,
                "mathematical": 0.5,
                "existential": 1.2,
                "philosophical": 0.8,
                "emergent": 1.0,
                "adversarial": 0.05,
            }
            wave_amp = cat_wave_amp.get(category, 0.3)
            self.em.wave_engine.create_wave(
                amplitude=combined * wave_amp,
                label=f"thought_{self._cycle_count}_{category}",
            )

        except Exception as exc:
            self._logger.warning("hidden-state write failed: %s", exc)

    # ── READ: EM Field → LLM Embedding ─────────────────────────────────

    def read_for_embedding(self, current_embeddings: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Read the EM field state and create an injection vector for the LLM.

        ADAPTATION: Injects accumulated memory + momentum-based amplitude
        modulation into the projection. Emotion state scales injection.
        """
        try:
            E = self.em.field.state.E.copy()

            # Apply matrixial correction
            E_corrected = self.em.matrixial.apply_correction(E, strength=0.02)

            # ── MEMORY: blend with accumulated source history ──
            if self._source_stack:
                recent_avg = np.mean(self._source_stack[-4:], axis=0)
                n = len(recent_avg)
                memory_weight = min(0.4, self._accumulated_energy * 0.0008)
                E_corrected[:n] = (
                    (1 - memory_weight) * E_corrected[:n] +
                    memory_weight * recent_avg
                )

            # ── ADAPTATION: momentum-based amplitude modulation ──
            momentum_mag = float(np.linalg.norm(self._momentum)) if self._momentum is not None else 0.0
            amplitude_mod = 1.0 + np.tanh(momentum_mag)

            # ── Emotion-based scaling ──
            emotion_scale = 1.0
            if hasattr(self.em, '_emotion_state') and self.em._emotion_state is not None:
                emo = self.em._emotion_state
                arousal = emo.get("arousal", 0.0)
                # High arousal → stronger injection (more urgent)
                # Low arousal → weaker injection (calm processing)
                emotion_scale = 0.5 + arousal * 1.0

            # Project from EM dimension to LLM dimension
            llm_vector = E_corrected @ self._em_to_llm

            # Scale: stability × modulation × novelty × momentum × emotion
            stability = self.em.matrixial.get_stability()
            novelty_factor = 0.5 + min(0.5, self._accumulated_energy * 0.0005)
            llm_vector *= stability * self.modulation_strength * novelty_factor * \
                          amplitude_mod * emotion_scale

            if isinstance(current_embeddings, torch.Tensor):
                device = current_embeddings.device
                dtype = current_embeddings.dtype
                injection = torch.tensor(
                    llm_vector, dtype=dtype, device=device,
                ).unsqueeze(0).unsqueeze(0)
            else:
                injection = torch.tensor(llm_vector).unsqueeze(0).unsqueeze(0)

            self._last_em_injection = llm_vector
            return injection

        except Exception as exc:
            self._logger.warning("EM embedding read failed: %s", exc)
            return None

    # ── MODULATE: EM Resonance → Logit Bias ────────────────────────────

    def modulate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Modulate logit probabilities using the EM field's resonance map.

        ADAPTATION: When field is overwhelmed (high energy, low coherence),
        applies entropy penalty to force concise/defensive responses.
        Category-specific logit biases based on input type.
        """
        now = time.time()

        # Refresh bias cache if stale
        if (self._logit_bias_cache is None or
                now - self._logit_bias_ttl > self._logit_bias_max_age):
            self._update_logit_bias_cache()

        # ── Entropy penalty for overwhelmed state ──
        energy = self.em.field.state.energy()
        coherence = self.em.field.state.coherence
        entropy_penalty = 0.0
        if energy > 0.8 and coherence < 0.3:
            # Field is overwhelmed: flatten logits (increase entropy)
            entropy_penalty = -0.1 * (energy - 0.8) * (0.3 - coherence)

        if self._logit_bias_cache is None:
            # Still apply entropy penalty even without resonance cache
            if entropy_penalty != 0.0:
                logits = logits * (1.0 + entropy_penalty)
            return logits

        try:
            vocab_size = logits.shape[-1]
            bias = self._logit_bias_cache

            if bias.shape[0] != vocab_size:
                if bias.shape[0] > vocab_size:
                    bias = bias[:vocab_size]
                else:
                    repeats = (vocab_size // bias.shape[0]) + 1
                    bias = bias.repeat(repeats)[:vocab_size]

            # Apply bias with modulation strength
            strength = self.modulation_strength
            if entropy_penalty != 0.0:
                strength *= (1.0 + entropy_penalty)

            if logits.dim() == 1:
                logits = logits + bias.to(logits.device) * strength
            else:
                logits = logits + bias.to(logits.device).unsqueeze(0) * strength

        except Exception:
            pass

        return logits

    def _update_logit_bias_cache(self):
        """Recompute the logit bias from current EM resonance state."""
        try:
            resonance_map = self.em.field.resonance_map()

            if not resonance_map:
                self._logit_bias_cache = None
                return

            spectrum = np.abs(np.fft.rfft(self.em.field.state.E))
            vocab_bias = np.zeros(256000, dtype=np.float32)

            for i, (freq_key, amplitude) in enumerate(resonance_map.items()):
                freq_val = float(freq_key.split("_")[1].replace("Hz", ""))
                token_center = int(abs(freq_val * 1000)) % 256000
                spread = max(10, int(amplitude * 5))

                for j in range(max(0, token_center - spread),
                               min(256000, token_center + spread)):
                    dist = abs(j - token_center)
                    vocab_bias[j] += amplitude * np.exp(-0.5 * (dist / (spread * 0.3)) ** 2)

            max_bias = np.max(np.abs(vocab_bias))
            if max_bias > 0:
                vocab_bias = vocab_bias / max_bias

            self._logit_bias_cache = torch.tensor(vocab_bias, dtype=torch.float32)
            self._logit_bias_ttl = time.time()

        except Exception:
            self._logit_bias_cache = None

    # ── Status ─────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "cycle_count": self._cycle_count,
            "last_write_time": self._last_write_time,
            "modulation_strength": self.modulation_strength,
            "em_stability": self.em.matrixial.get_stability(),
            "has_injection": self._last_em_injection is not None,
            "has_logit_bias": self._logit_bias_cache is not None,
            "last_category": self._last_category,
            "last_signature": self._last_signature,
            "accumulated_energy": self._accumulated_energy,
            "momentum_magnitude": float(np.linalg.norm(self._momentum)) if self._momentum is not None else 0.0,
            "stack_size": len(self._source_stack),
            "hysteresis_energy": float(np.linalg.norm(self._hysteresis_bias)),
        }
