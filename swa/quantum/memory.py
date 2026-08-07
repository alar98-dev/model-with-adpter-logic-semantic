from __future__ import annotations

import gc
import math
from typing import Any

import torch


class SemanticMemory:
    """In-memory semantic cache, decay, rotation, and manifold operations.

    Holds the cached anchor embeddings (``_cached_L``), weights
    (``_cached_w``), obsolescence counters (``_anchor_irrelevance``),
    and the recursive thought trajectory.
    """

    def __init__(
        self,
        conn: Any,
        cursor: Any,
        gold_dimension: int = 2560,
        phi_ratio: float = 1.61803398875,
        apoptosis_threshold: int = 400,
        recursion_decay: float = 0.85,
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.gold_dimension = gold_dimension
        self.phi_ratio = phi_ratio
        self.apoptosis_threshold = apoptosis_threshold
        self._recursion_decay = recursion_decay
        self._phase_theta = 0.0
        self.current_jitter_mag = 0.0

        self._cached_L: torch.Tensor | None = None
        self._cached_w: torch.Tensor | None = None
        self._anchor_irrelevance: torch.Tensor | None = None
        self._recursive_thought_accumulator: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Cache loading
    # ------------------------------------------------------------------

    def load_cache(
        self,
        d_model: int | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Load all semantic anchor embeddings into a PyTorch cache."""
        if d_model is not None:
            self.gold_dimension = d_model

        if self.conn is None:
            self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
            self._cached_w = torch.ones((1,), dtype=dtype, device=device)
            return

        try:
            self.cursor.execute(
                "SELECT embedding_2560, COALESCE(weight, 1.0) "
                "FROM blake_semantic_anchors WHERE embedding_2560 IS NOT NULL;"
            )
            rows = self.cursor.fetchall()
            if rows:
                embeddings = [r[0] for r in rows]
                weights = [r[1] for r in rows]

                parsed_embeddings = []
                for emb in embeddings:
                    if isinstance(emb, str):
                        emb = [float(v) for v in emb.strip("[]").split(",")]
                    parsed_embeddings.append(emb)

                raw_tensor = torch.tensor(parsed_embeddings, dtype=dtype, device=device)
                self._anchor_irrelevance = torch.zeros(
                    raw_tensor.size(0), dtype=torch.long, device=device
                )

                if self.gold_dimension != 2560:
                    g = torch.Generator(device=device).manual_seed(42)
                    P = torch.randn(
                        (2560, self.gold_dimension), generator=g, dtype=dtype, device=device
                    )
                    P = P / P.norm(dim=0, keepdim=True).clamp(min=1e-9)
                    self._cached_L = torch.matmul(raw_tensor, P)
                else:
                    self._cached_L = raw_tensor

                self._cached_w = torch.tensor(weights, dtype=dtype, device=device)
            else:
                self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
                self._cached_w = torch.ones((1,), dtype=dtype, device=device)
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
            self._cached_w = torch.ones((1,), dtype=dtype, device=device)

    def invalidate_cache(self) -> None:
        """Mark the cached embedding matrix as stale (forces reload)."""
        self._cached_L = None

    # ------------------------------------------------------------------
    # Semantic mass computation (RBF + apoptosis)
    # ------------------------------------------------------------------

    def compute_mass(
        self,
        x: torch.Tensor,
        sigma: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Compute the continuous Semantic Mass Tensor M(x) for a given activation vector x."""
        x_tensor = x.to(device=device, dtype=dtype).view(-1)
        d_model = x_tensor.size(0)

        if self._cached_L is None or self._cached_L.size(-1) != d_model:
            self.load_cache(d_model=d_model, device=device, dtype=dtype)

        if x_tensor.size(0) < self.gold_dimension:
            import torch.nn.functional as F

            x_tensor = F.pad(
                x_tensor,
                (0, self.gold_dimension - x_tensor.size(0)),
                "constant",
                0.0,
            )
        elif x_tensor.size(0) > self.gold_dimension:
            x_tensor = x_tensor[: self.gold_dimension]

        dists = torch.sum((x_tensor - self._cached_L) ** 2, dim=-1)
        rbf = torch.exp(-dists / (2.0 * (sigma**2)))

        # Semantic apoptosis — active forgetting
        if self._anchor_irrelevance is not None:
            mask_unused = rbf < 1e-4
            self._anchor_irrelevance[mask_unused] += 1
            self._anchor_irrelevance[~mask_unused] = 0

            if torch.any(self._anchor_irrelevance > self.apoptosis_threshold):
                keep_mask = self._anchor_irrelevance <= self.apoptosis_threshold
                old_count = self._cached_L.size(0)
                self._cached_L = self._cached_L[keep_mask]
                self._cached_w = self._cached_w[keep_mask]
                self._anchor_irrelevance = self._anchor_irrelevance[keep_mask]
                new_count = self._cached_L.size(0)
                if old_count - new_count > 0:
                    gc.collect()

        weighted_pull = (self._cached_w * rbf).unsqueeze(1) * self._cached_L
        pull_vector = torch.sum(weighted_pull, dim=0)
        return pull_vector

    # ------------------------------------------------------------------
    # Decay vector
    # ------------------------------------------------------------------

    @property
    def decay_vector(self) -> torch.Tensor:
        """Dynamic selective memory decay based on thermal stress.

        Higher temperatures cause faster decay to avoid fixation loops.
        """
        stress = self._compute_stress()

        base_decay = 0.88 - (stress * 0.20)
        device = "cpu"
        dtype = torch.float32
        if self._cached_L is not None:
            device = self._cached_L.device
            dtype = self._cached_L.dtype

        g_dec = torch.Generator(device=device).manual_seed(137)
        noise = torch.randn(
            self.gold_dimension, generator=g_dec, dtype=dtype, device=device
        ) * 0.05
        return torch.clamp(base_decay + noise, min=0.60, max=0.98)

    def _compute_stress(self) -> float:
        """Estimate environmental stress from CPU temperature telemetry."""
        try:
            import psutil

            temp = 45.0
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = [
                    entry.current for entries in temps.values() for entry in entries
                ]
                if all_temps:
                    temp = sum(all_temps) / len(all_temps)
            return max(0.0, min(1.0, (temp - 30.0) / 60.0))
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Phase rotation (RoPE-style)
    # ------------------------------------------------------------------

    def apply_phase_rotation(
        self, x: torch.Tensor, theta: float
    ) -> torch.Tensor:
        """Apply 2D phase rotation (RoPE-style) to pairs of dimensions."""
        if x is None:
            return x
        device = x.device
        dtype = x.dtype
        d_model = x.size(-1)
        padded = False
        if d_model % 2 != 0:
            import torch.nn.functional as F

            x = F.pad(x, (0, 1), "constant", 0.0)
            d_model += 1
            padded = True

        x_reshaped = x.view(-1, d_model // 2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        y1 = x1 * cos_theta - x2 * sin_theta
        y2 = x1 * sin_theta + x2 * cos_theta

        y = torch.stack([y1, y2], dim=-1).view(*x.shape[:-1], d_model)

        if padded:
            y = y[..., :-1]
        return y

    # ------------------------------------------------------------------
    # Recursive thought
    # ------------------------------------------------------------------

    def feed_thought(self, current_thought_vector: torch.Tensor) -> None:
        """Accumulate and mutate the recursive thought trajectory."""
        try:
            device = current_thought_vector.device
            dtype = current_thought_vector.dtype
            d_model = current_thought_vector.size(-1)

            freq, jitter = self._compute_clock_parameters(device, dtype)

            self._phase_theta += freq + jitter
            if self._phase_theta > math.pi * 2:
                self._phase_theta -= math.pi * 2

            self.current_jitter_mag = abs(jitter)

            x_tensor = current_thought_vector.detach().clone().view(-1)
            if x_tensor.size(0) < self.gold_dimension:
                import torch.nn.functional as F

                x_tensor = F.pad(
                    x_tensor,
                    (0, self.gold_dimension - x_tensor.size(0)),
                    "constant",
                    0.0,
                )
            elif x_tensor.size(0) > self.gold_dimension:
                x_tensor = x_tensor[: self.gold_dimension]

            x_tensor = self.apply_phase_rotation(x_tensor, self._phase_theta)

            if self._recursive_thought_accumulator is None:
                self._recursive_thought_accumulator = x_tensor
            else:
                decay = self.decay_vector.to(device=device, dtype=dtype)
                self._recursive_thought_accumulator = (
                    self._recursive_thought_accumulator.to(device=device, dtype=dtype)
                    * decay
                ) + (x_tensor * (1.0 - decay))
                n = self._recursive_thought_accumulator.norm()
                if n > 0:
                    self._recursive_thought_accumulator = (
                        self._recursive_thought_accumulator / n
                    )
        except Exception as e:
            pass

    def _compute_clock_parameters(
        self, device: str, dtype: torch.dtype
    ) -> tuple[float, float]:
        """Compute phase clock frequency and jitter based on thermal telemetry and semantic alignment."""
        try:
            import psutil

            temp = 45.0
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = [
                    entry.current for entries in temps.values() for entry in entries
                ]
                if all_temps:
                    temp = sum(all_temps) / len(all_temps)
            freq = max(0.01, min(0.5, (temp - 30.0) / 100.0))
        except Exception:
            freq = 0.1

        jitter = 0.0
        if self._cached_L is not None and self._cached_L.size(0) > 0:
            x_comp = (
                torch.zeros(self.gold_dimension, device=device, dtype=dtype)
                if self._recursive_thought_accumulator is None
                else self._recursive_thought_accumulator.detach().clone().view(-1).to(
                    device=device, dtype=dtype
                )
            )
            if x_comp.size(0) < self.gold_dimension:
                import torch.nn.functional as F

                x_comp = F.pad(
                    x_comp,
                    (0, self.gold_dimension - x_comp.size(0)),
                    "constant",
                    0.0,
                )
            elif x_comp.size(0) > self.gold_dimension:
                x_comp = x_comp[: self.gold_dimension]

            x_comp = x_comp.to(self._cached_L.device, dtype=self._cached_L.dtype)
            norm_x = x_comp.norm().clamp(min=1e-9)
            norms_L = self._cached_L.norm(dim=1).clamp(min=1e-9)
            cos_sim = torch.matmul(self._cached_L, x_comp) / (norms_L * norm_x)
            alignment = float(torch.max(cos_sim).item())

            misalignment = max(0.0, 1.0 - alignment)
            variance = float(torch.var(x_comp).item())
            sum_active = float(torch.sum(x_comp).item())
            natural_wave = math.sin(sum_active)
            semantic_stagnation = 1.0 / (variance + 1e-4)

            try:
                import psutil

                cpu_times = psutil.cpu_times()
                io_wait = getattr(cpu_times, "iowait", 0.0)
                system_stagnation = io_wait / 10.0
            except Exception:
                system_stagnation = 0.0

            total_anxiety = semantic_stagnation + system_stagnation
            amplitude = misalignment * total_anxiety
            jitter = natural_wave * amplitude * 0.02

        return freq, jitter

    # ------------------------------------------------------------------
    # Manifold deformation
    # ------------------------------------------------------------------

    def deform_manifold(
        self, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> None:
        """Warp cached anchors based on the accumulated recursive thought vector."""
        if (
            self._recursive_thought_accumulator is None
            or self._cached_L is None
        ):
            return

        try:
            q_rec = self._recursive_thought_accumulator.to(
                device=device, dtype=dtype
            ).view(1, self.gold_dimension)

            distortion_strength = 0.05 / self.phi_ratio
            delta = q_rec - self._cached_L
            self._cached_L = self._cached_L + (distortion_strength * delta)

            norms = self._cached_L.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            self._cached_L = self._cached_L / norms
        except Exception:
            pass
