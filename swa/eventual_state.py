import torch
import time
from typing import Dict, List, Any, Optional
from threading import Lock
from swa.depth_analyzer import RuntimeEvent


class EventualState:
    """
    Reactive state management for SWA Broadcast Eventual Runtime.
    
    Uses @property for controlled access to state variables, triggering
    side effects like bias recalculation, event broadcasting, and telemetry updates.
    """

    def __init__(self, vocab_size: int, hidden_size: int = 2560, dtype: torch.dtype = torch.float32, tokenizer = None):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.tokenizer = tokenizer
        
        # Core state variables with private backing
        self._vectors: Dict[str, float] = {
            "forward": 0.0,
            "backward": 0.0,
            "left": 0.0,
            "right": 0.0
        }
        self._obstacles: List[Dict[str, Any]] = []
        self._last_update: float = time.time()
        
        # Cached bias tensor for performance
        self._cached_bias: Optional[torch.Tensor] = None
        self._bias_dirty: bool = True
        
        # Telemetry for drift monitoring
        self._latent_drift: float = 0.0
        self._update_count: int = 0
        
        # Thread safety for concurrent access
        self._state_lock = Lock()
        
        # Event broadcasting callback
        self._event_callback: Optional[callable] = None

    @property
    def vectors(self) -> Dict[str, float]:
        """Get current navigation vectors."""
        with self._state_lock:
            return self._vectors.copy()

    @vectors.setter
    def vectors(self, new_values: Dict[str, float]) -> None:
        """Set navigation vectors and trigger side effects."""
        with self._state_lock:
            old_vectors = self._vectors.copy()
            self._vectors.update(new_values)
            self._last_update = time.time()
            self._bias_dirty = True
            self._update_count += 1
            
            # Calculate drift from previous state
            self._calculate_vector_drift(old_vectors)
            
            # Broadcast event
            self._broadcast_event(RuntimeEvent(
                type="vectors_updated",
                payload={
                    "old_vectors": old_vectors,
                    "new_vectors": self._vectors.copy(),
                    "drift": self._latent_drift
                }
            ))

    def update_vectors(self, values: Dict[str, float]) -> None:
        """Atomically update navigation vectors (the getter is intentionally a copy)."""
        self.vectors = values

    def set_coordinate_vector(self, vector: Optional[torch.Tensor]) -> None:
        """Shared coordinate contract for steering implementations."""
        with self._state_lock:
            self._coordinate_vector = None if vector is None else vector.detach().clone()

    def get_coordinate_vector(self) -> Optional[torch.Tensor]:
        with self._state_lock:
            value = getattr(self, "_coordinate_vector", None)
            return None if value is None else value.clone()

    @property
    def obstacles(self) -> List[Dict[str, Any]]:
        """Get current obstacle list."""
        with self._state_lock:
            return self._obstacles.copy()

    @obstacles.setter
    def obstacles(self, new_obstacles: List[Dict[str, Any]]) -> None:
        """Set obstacles and trigger side effects."""
        with self._state_lock:
            old_obstacles = self._obstacles.copy()
            self._obstacles = new_obstacles.copy()
            self._last_update = time.time()
            self._bias_dirty = True
            self._update_count += 1
            
            # Update drift based on obstacle changes
            self._calculate_obstacle_drift(old_obstacles)
            
            # Broadcast event
            self._broadcast_event(RuntimeEvent(
                type="obstacles_updated",
                payload={
                    "old_obstacles": old_obstacles,
                    "new_obstacles": self._obstacles.copy(),
                    "drift": self._latent_drift
                }
            ))

    @property
    def is_obstructed(self) -> bool:
        """Computed property: check if there are critical obstacles."""
        with self._state_lock:
            return len(self._obstacles) > 0

    @property
    def latent_drift(self) -> float:
        """Get current latent drift metric."""
        with self._state_lock:
            return self._latent_drift

    @property
    def last_update(self) -> float:
        """Get timestamp of last state update."""
        with self._state_lock:
            return self._last_update

    @property
    def update_count(self) -> int:
        """Get total number of state updates."""
        with self._state_lock:
            return self._update_count

    @property
    def cached_bias(self) -> Optional[torch.Tensor]:
        """Get cached logit bias tensor, recomputing if dirty."""
        with self._state_lock:
            if self._bias_dirty or self._cached_bias is None:
                self._cached_bias = self._recompute_logit_bias()
                self._bias_dirty = False
            return self._cached_bias.clone() if self._cached_bias is not None else None

    def get_embedding_steering(self) -> Optional[torch.Tensor]:
        """Get embedding steering tensor based on navigation vectors."""
        # Simple implementation: create a bias vector in embedding space
        # In real implementation, this would map vectors to embedding modifications
        bias = torch.zeros(self.hidden_size, dtype=self.dtype)
        
        # Apply vector biases (positive for directions)
        for direction, strength in self._vectors.items():
            if strength > 0:
                # Placeholder: boost certain embedding dimensions based on direction
                if direction == "forward":
                    bias[0:100] += strength * 0.1  # Boost forward-related concepts
                elif direction == "backward":
                    bias[100:200] += strength * 0.1
                # Add more directions as needed
        
        return bias if torch.any(bias != 0) else None

    def get_logit_steering(self) -> Optional[torch.Tensor]:
        """Get logit steering tensor based on obstacles."""
        return self.cached_bias

    def set_event_callback(self, callback: callable) -> None:
        """Set callback for broadcasting state change events."""
        self._event_callback = callback

    def _calculate_vector_drift(self, old_vectors: Dict[str, float]) -> None:
        """Calculate semantic drift from vector changes."""
        total_drift = 0.0
        for direction, new_val in self._vectors.items():
            old_val = old_vectors.get(direction, 0.0)
            total_drift += abs(new_val - old_val)
        
        # Exponential decay for drift accumulation
        self._latent_drift = self._latent_drift * 0.9 + total_drift * 0.1

    def _calculate_obstacle_drift(self, old_obstacles: List[Dict[str, Any]]) -> None:
        """Calculate drift from obstacle changes."""
        old_count = len(old_obstacles)
        new_count = len(self._obstacles)
        
        # Drift increases with obstacle changes
        obstacle_drift = abs(new_count - old_count) * 0.5
        self._latent_drift = self._latent_drift * 0.95 + obstacle_drift * 0.05

    def _recompute_logit_bias(self) -> torch.Tensor:
        """Recompute spatial bias tensor dynamically mapping direction words via active tokenizer."""
        bias = torch.zeros(self.vocab_size, dtype=self.dtype)
        
        # Mapping definition for spatial routing directions
        direction_words = {
            "forward": ["frente", "forward", "avançar", "adiante"],
            "backward": ["trás", "backward", "recuar", "retroceder"],
            "left": ["esquerda", "left", "desviar"],
            "right": ["direita", "right"],
            "stop": ["parar", "stop", "frear"],
            "front": ["frente", "forward", "avançar"]
        }

        # Dynamically resolve token IDs
        resolved_mappings = {}
        for dir_key, words in direction_words.items():
            ids = []
            if self.tokenizer is not None:
                for w in words:
                    try:
                        token_ids = self.tokenizer.encode(w, add_special_tokens=False)
                        if token_ids:
                            ids.extend(token_ids)
                    except Exception:
                        pass
            # Fallback to deterministic pseudo-hash space if no tokenizer is loaded
            if not ids:
                import hashlib
                seed = int(hashlib.md5(dir_key.encode()).hexdigest(), 16)
                ids = [(seed + i) % self.vocab_size for i in range(100)]
            resolved_mappings[dir_key] = list(set(ids))

        # Apply vector biases (positive for directions)
        for direction, strength in self._vectors.items():
            if strength > 0 and direction in resolved_mappings:
                for tid in resolved_mappings[direction]:
                    if tid < self.vocab_size:
                        bias[tid] += strength * 2.0
        
        # Apply obstacle penalties (negative for blocked directions)
        for obstacle in self._obstacles:
            direction = obstacle.get("direction", "")
            confidence = obstacle.get("confidence", 0.0)
            if confidence > 0.5 and direction in resolved_mappings:
                penalty = -confidence * 8.0
                for tid in resolved_mappings[direction]:
                    if tid < self.vocab_size:
                        bias[tid] += penalty
        
        return bias

    def _broadcast_event(self, event: RuntimeEvent) -> None:
        """Broadcast state change event if callback is set."""
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                # Log error but don't crash
                print(f"Event broadcast failed: {e}")

    def get_telemetry(self) -> Dict[str, Any]:
        """Get current telemetry snapshot."""
        with self._state_lock:
            return {
                "vectors": self._vectors.copy(),
                "obstacles": self._obstacles.copy(),
                "latent_drift": self._latent_drift,
                "last_update": self._last_update,
                "update_count": self._update_count,
                "bias_dirty": self._bias_dirty
            }