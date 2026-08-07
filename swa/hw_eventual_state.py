from __future__ import annotations
from typing import Dict, Any, List, Optional
import torch
import time
from swa.eventual_state import EventualState
from swa.depth_analyzer import RuntimeEvent
from swa.hardware_state import get_swa_hardware_manager, HardwareStateManager
from swa.phase_shifter import PhaseShifter
from swa.cpu import CPUTopology


class HWEventualState(EventualState):
    """Hardware-persistent eventual state extending EventualState with NVMe storage.

    Implements multiple simultaneous states:
    - Reality: Current sensor/environment state
    - Prediction: AI-simulated next state
    - Shadow Bias: Attention steering vectors
    - Recursion: High-frequency interference space
    - Dreamwords: Persistent long-term context
    """

    def __init__(self, vocab_size: int, hidden_size: int = 2560, dtype: torch.dtype = torch.float32):
        # Hardware-backed state tensors
        self._hw_manager: Optional[HardwareStateManager] = None
        self._phase_shifter: Optional[PhaseShifter] = None
        self._reality_state: Optional[torch.Tensor] = None
        self._prediction_state: Optional[torch.Tensor] = None
        self._shadow_bias: Optional[torch.Tensor] = None
        self._recursion_space: Optional[torch.Tensor] = None
        self._dreamwords: Optional[torch.Tensor] = None

        # Initialize hardware manager first
        try:
            self._hw_manager = get_swa_hardware_manager()
            self._phase_shifter = PhaseShifter(self._hw_manager)
        except Exception as e:
            print(f"Hardware state initialization failed: {e}. Falling back to RAM-only mode.")
            self._hw_manager = None
            self._phase_shifter = None

        # Initialize parent class
        super().__init__(vocab_size, hidden_size, dtype)

        # Initialize CPU Core Topology (Characteristics & Eventual Process Cycles)
        self._cpu_topology: CPUTopology = CPUTopology()

        # Initialize hardware states after parent
        if self._hw_manager:
            self._initialize_hardware_states()

    def start_organic_threading(self, gemma_generator):
        """Start organic threading integration with Gemma generator and hot-plug the active tokenizer."""
        if hasattr(gemma_generator, 'start_organic_threading'):
            gemma_generator.start_organic_threading(self)
            print("🧬 HW Eventual State connected to organic threading")
        
        # Dynamic Tokenizer Hot-plug injection from active Gemma Generator
        if hasattr(gemma_generator, 'tokenizer') and gemma_generator.tokenizer is not None:
            self.tokenizer = gemma_generator.tokenizer
            print("🧬 HW Eventual State: Tokenizer dynamically hot-plugged from active model.")

        # Initialize hardware manager
        try:
            self._hw_manager = get_swa_hardware_manager()
            self._initialize_hardware_states()
        except Exception as e:
            print(f"Hardware state initialization failed: {e}. Falling back to RAM-only mode.")
            self._hw_manager = None

    @EventualState.vectors.setter
    def vectors(self, new_values: Dict[str, float]) -> None:
        """Set navigation vectors and trigger side effects, including hardware sync."""
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

            # Sync to hardware organically
            self.update_organic_state({"vectors": self._vectors, "obstacles": self._obstacles})

    def get_embedding_steering(self) -> Optional[torch.Tensor]:
        """Get organic embedding steering from phase shifter interference."""
        organic_bias = self.get_organic_attention_bias()
        if organic_bias is not None:
            # Convert to embedding space by averaging heads
            return organic_bias.mean(dim=[1, 2])  # [320] shape
        return super().get_embedding_steering()

    def get_logit_steering(self) -> Optional[torch.Tensor]:
        """Get organic logit steering dynamically mapping attention head weights to real tokenizer tokens."""
        # Trigger actual load simulation on physical cores corresponding to this cycle
        if hasattr(self, '_cpu_topology') and self._cpu_topology.cores:
            core_idx = self._update_count % len(self._cpu_topology.cores)
            self._cpu_topology.cores[core_idx].execute_instruction("steering_calculation", weight=1.2)

        if self._phase_shifter:
            # Create logit bias based on organic attention bias
            organic_bias = self._phase_shifter.organic_bias

            # Use entropy to modulate the strength of spatial biases
            entropy_factor = min(1.0, self._phase_shifter.entropy * 2.0)

            # Create base logit bias
            bias_logits = torch.zeros(self.vocab_size, dtype=self.dtype)

            # Apply organic modulation to directional tokens
            directions = ['forward', 'left', 'right', 'reverse', 'stop']
            head_bias = organic_bias.squeeze(0).squeeze(-1)  # [8]

            # Define semantic mapping lists for dynamic encoding
            direction_words = {
                "forward": ["frente", "forward", "avançar", "adiante"],
                "left": ["esquerda", "left", "desviar"],
                "right": ["direita", "right"],
                "reverse": ["trás", "backward", "recuar", "retroceder"],
                "stop": ["parar", "stop", "frear"]
            }

            for i, direction in enumerate(directions):
                if i < len(head_bias):
                    strength = head_bias[i].mean().item() * (1.0 - entropy_factor)
                    words = direction_words.get(direction, [direction])
                    ids = []
                    
                    # Dynamically encode tokens if tokenizer is injected
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                        for w in words:
                            try:
                                token_ids = self.tokenizer.encode(w, add_special_tokens=False)
                                if token_ids:
                                    ids.extend(token_ids)
                            except Exception:
                                pass
                    
                    # Fallback to deterministic pseudorandom range to prevent empty mapping
                    if not ids:
                        import hashlib
                        seed = int(hashlib.md5(direction.encode()).hexdigest(), 16)
                        ids = [(seed + j) % self.vocab_size for j in range(100)]
                    
                    for tid in ids:
                        if tid < self.vocab_size:
                            bias_logits[tid] += strength

            # Ensure the bias tensor has exactly the right size
            assert bias_logits.shape[0] == self.vocab_size, f"Bias size {bias_logits.shape[0]} != vocab size {self.vocab_size}"

            return bias_logits

        return super().get_logit_steering()

    def _initialize_hardware_states(self):
        """Initialize hardware-backed state tensors."""
        if self._hw_manager is None:
            return

        state_dim = 2560  # As per SWA spec
        bias_dim = 2560   # 8x320 attention bias

        # Initialize with zeros or try to load from hardware
        try:
            self._reality_state = self._hw_manager.read_state("reality")
            if self._reality_state is None:
                self._reality_state = torch.zeros(state_dim)
        except:
            self._reality_state = torch.zeros(state_dim)

        try:
            self._prediction_state = self._hw_manager.read_state("prediction")
            if self._prediction_state is None:
                self._prediction_state = torch.zeros(state_dim)
        except:
            self._prediction_state = torch.zeros(state_dim)

        try:
            self._shadow_bias = self._hw_manager.read_state("shadow_bias")
            if self._shadow_bias is None:
                self._shadow_bias = torch.zeros(bias_dim)
        except:
            self._shadow_bias = torch.zeros(bias_dim)

        try:
            self._recursion_space = self._hw_manager.read_state("recursion")
            if self._recursion_space is None:
                self._recursion_space = torch.zeros(state_dim)
        except:
            self._recursion_space = torch.zeros(state_dim)

        try:
            self._dreamwords = self._hw_manager.read_state("dreamwords")
            if self._dreamwords is None:
                self._dreamwords = torch.zeros(state_dim)
        except:
            self._dreamwords = torch.zeros(state_dim)

    def update_reality_state(self, sensor_data: Dict[str, Any]):
        """Update reality state from sensors."""
        # Convert sensor data to vector representation
        reality_vector = self._encode_sensor_data(sensor_data)
        self._reality_state = reality_vector

        # Persist to hardware
        if self._hw_manager:
            self._hw_manager.write_state("reality", reality_vector)

    def update_prediction_state(self, predicted_state: torch.Tensor):
        """Update AI prediction state."""
        self._prediction_state = predicted_state

        # Persist to hardware
        if self._hw_manager:
            self._hw_manager.write_state("prediction", predicted_state)

    def update_shadow_bias(self, attention_bias: torch.Tensor):
        """Update attention steering bias."""
        self._shadow_bias = attention_bias

        # Persist to hardware
        if self._hw_manager:
            self._hw_manager.write_state("shadow_bias", attention_bias)

    def update_recursion_space(self, interference_data: torch.Tensor):
        """Update high-frequency recursion space."""
        self._recursion_space = interference_data

        # Persist to hardware
        if self._hw_manager:
            self._hw_manager.write_state("recursion", interference_data)

    def update_dreamwords(self, persistent_context: torch.Tensor):
        """Update long-term persistent state."""
        self._dreamwords = persistent_context

        # Persist to hardware
        if self._hw_manager:
            self._hw_manager.write_state("dreamwords", persistent_context)

    def compute_swa_decision(self) -> torch.Tensor:
        """Compute SWA decision using phase shifter organic interference."""
        if self._phase_shifter:
            bias = self._phase_shifter.organic_bias
            return bias.squeeze(0).squeeze(-1).mean(dim=-1)
        else:
            # Fallback to simple vector sum
            return torch.tensor([0.5] * 8)

    def get_organic_attention_bias(self) -> torch.Tensor:
        """Get organic attention bias for Gemma heads [1, 8, 1, 320] via phase shifter."""
        if self._phase_shifter:
            return self._phase_shifter.organic_bias
        else:
            return torch.zeros(1, 8, 1, 320)

    def should_inject_phase(self) -> bool:
        """Check if phase injection should be triggered based on phase shifter entropy."""
        if self._phase_shifter:
            return self._phase_shifter.entropy > 0.7  # Threshold for phase injection
        return False

    def update_organic_state(self, sensor_data: Dict[str, Any], predicted_state: Optional[torch.Tensor] = None):
        """Update organic state with sensor data and optional prediction."""
        # Update reality from sensors
        self.update_reality_state(sensor_data)

        # Update prediction if provided
        if predicted_state is not None:
            self.update_prediction_state(predicted_state)

        # Update circular buffer with current reality
        if self._hw_manager and self._reality_state is not None:
            self._hw_manager.update_circular_buffer(self._reality_state)

        # Calculate and update inertia based on state changes
        if self._hw_manager and self._reality_state is not None:
            momentum = self._hw_manager.calculate_momentum_vector(self._reality_state)
            self._hw_manager.update_inertia_accumulator(momentum)

        # Sync to hardware
        self.sync_to_hardware()

    def update_navigation_map(self, map_data: torch.Tensor):
        """Update the persistent navigation map in hardware."""
        if self._hw_manager:
            self._hw_manager.write_navigation_map(map_data)

    def get_navigation_map(self) -> Optional[torch.Tensor]:
        """Retrieve the navigation map from hardware."""
        if self._hw_manager:
            return self._hw_manager.read_navigation_map()
        return None

    def get_parallel_states(self) -> Dict[str, torch.Tensor]:
        """Get all hardware states simultaneously."""
        if self._hw_manager:
            return self._hw_manager.get_parallel_states([
                "reality", "prediction", "shadow_bias", "recursion", "dreamwords"
            ])
        else:
            return {
                "reality": self._reality_state,
                "prediction": self._prediction_state,
                "shadow_bias": self._shadow_bias,
                "recursion": self._recursion_space,
                "dreamwords": self._dreamwords,
            }

    def _encode_sensor_data(self, sensor_data: Dict[str, Any]) -> torch.Tensor:
        """Encode sensor data into state vector."""
        # Simple encoding: map common sensor types to vector dimensions
        state_vector = torch.zeros(2560)

        # Movement vectors
        if "vectors" in sensor_data:
            vectors = sensor_data["vectors"]
            state_vector[0] = vectors.get("forward", 0.0)
            state_vector[1] = vectors.get("left", 0.0)
            state_vector[2] = vectors.get("right", 0.0)
            state_vector[3] = vectors.get("reverse", 0.0)
            state_vector[4] = vectors.get("stop", 0.0)

        # Obstacles
        if "obstacles" in sensor_data:
            obstacles = sensor_data["obstacles"]
            # Encode obstacles in dimensions 5-10
            obstacle_types = ["front", "left", "right", "back"]
            for obs in obstacles:
                direction = obs.get("direction", "")
                confidence = obs.get("confidence", 1.0)
                if direction in obstacle_types:
                    idx = obstacle_types.index(direction)
                    state_vector[5 + idx] = confidence

        # Telemetry
        if "telemetry" in sensor_data:
            telemetry = sensor_data["telemetry"]
            # Encode numeric telemetry values
            idx = 10
            for key, value in telemetry.items():
                if isinstance(value, (int, float)) and idx < 50:
                    state_vector[idx] = float(value)
                    idx += 1

        return state_vector

    def sync_from_hardware(self):
        """Sync in-memory state from hardware storage."""
        if self._hw_manager:
            states = self.get_parallel_states()
            self._reality_state = states.get("reality")
            self._prediction_state = states.get("prediction")
            self._shadow_bias = states.get("shadow_bias")
            self._recursion_space = states.get("recursion")
            self._dreamwords = states.get("dreamwords")

    def sync_to_hardware(self):
        """Sync in-memory state to hardware storage."""
        if self._hw_manager:
            if self._reality_state is not None:
                self._hw_manager.write_state("reality", self._reality_state)
            if self._prediction_state is not None:
                self._hw_manager.write_state("prediction", self._prediction_state)
            if self._shadow_bias is not None:
                self._hw_manager.write_state("shadow_bias", self._shadow_bias)
            if self._recursion_space is not None:
                self._hw_manager.write_state("recursion", self._recursion_space)
            if self._dreamwords is not None:
                self._hw_manager.write_state("dreamwords", self._dreamwords)

    @property
    def organic_entropy(self) -> float:
        """Get current organic entropy from phase shifter."""
        if self._phase_shifter:
            return self._phase_shifter.entropy
        return 0.0

    @property
    def organic_inertia(self) -> float:
        """Get current organic inertia weight from phase shifter."""
        if self._phase_shifter:
            return self._phase_shifter.inertia_weight
        return 0.95

    @property
    def cpu_topology(self) -> CPUTopology:
        """Get the physical CPU core topology (Characteristics & Process Cycles)."""
        return self._cpu_topology