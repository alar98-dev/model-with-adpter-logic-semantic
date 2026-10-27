import mmap
import os
import struct
import numpy as np
import torch
from typing import Dict, Any, Optional, List
import blake3
import time
import math
from dataclasses import dataclass

# SWA Hardware Domain Configuration
SWA_DOMAIN = "/dev/sde"
SECTOR_SIZE = 4096  # 4KB alignment
STATE_VECTOR_DIM = 2560  # As per prompt (8 heads * 320 dim)
ATTENTION_BIAS_DIM = 2560  # 8x320 attention bias

# State Offsets (hex addresses) - Multi-State Simultaneous Access
OFFSET_REALITY = 0x0000  # Real sensor state (T0)
OFFSET_PREDICTION = 0x4000  # AI prediction (T+1)
OFFSET_SHADOW_BIAS = 0x8000  # Attention bias adapter (8x320)
OFFSET_INERTIA_ACCUMULATOR = 0xA000  # Inertia accumulator for organic resistance
OFFSET_NAVIGATION_MAP = 0x100000  # Navigation mesh/map data (persistent spatial knowledge)
OFFSET_RECURSION_FREE = 0x12000  # High-frequency recursion space
OFFSET_DREAMWORDS = 0x20000  # Dream state storage
OFFSET_CIRCULAR_BUFFER = 0x40000  # Start of circular buffer for past states (24GB allows ~6000 states)

# Circular Buffer Configuration
CIRCULAR_BUFFER_SIZE = 6000  # Number of past states to maintain
STATE_SIZE_BYTES = STATE_VECTOR_DIM * 4  # float32 = 4 bytes
CIRCULAR_BUFFER_TOTAL_SIZE = CIRCULAR_BUFFER_SIZE * STATE_SIZE_BYTES  # ~60MB for history

# Entropy and Inertia Parameters
ENTROPY_THRESHOLD = 0.7  # Threshold for phase injection trigger
INERTIA_DECAY_RATE = 0.95  # Exponential decay for inertia accumulation
BIT_ENTROPY_SMOOTHING = 0.1  # Smoothing factor for entropy calculation

@dataclass
class StateMetadata:
    """Metadata for hardware state integrity."""
    offset: int
    size: int
    hash: str
    timestamp: float

class HardwareStateManager:
    """Manages multiple simultaneous states via direct hardware memory mapping to /dev/sde.

    Implements the SWA Domain with parallel state access:
    - Reality: Sensor inputs at T0
    - Prediction: AI simulation at T+1
    - Shadow: Attention bias for steering
    - Recursion: High-frequency interference space
    - Dreamwords: Persistent state for long-term context
    """

    def __init__(self, device_path: str = SWA_DOMAIN, total_size_gb: int = 24):
        self.device_path = device_path
        self.total_size = total_size_gb * 1024**3  # 24GB
        self.fd = None
        self.mem_map = None
        self._ram_states: Dict[str, torch.Tensor] = {}
        self._writable = False
        self.state_metadata: Dict[str, StateMetadata] = {}

        self._initialize_hardware_bus()

    def _initialize_hardware_bus(self):
        """Initialize the SWA hardware bus via mmap with RAM fallback."""
        self.nav_map_offset = 0x00000000
        self.arm_offset = 0x01000000
        self.signal_gain = 2.0
        try:
            # Simplest compatible open for WSL2 (O_RDONLY only)
            self.fd = os.open(self.device_path, os.O_RDWR)
            # WSL2-friendly: advise sequential massive reads for kernel prefetch
            os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)

            # Get actual size (length=0 causes EINVAL on WSL2 /dev/sde)
            try:
                size = os.fstat(self.fd).st_size
            except:
                size = self.total_size
            if size <= 0:
                size = os.lseek(self.fd, 0, os.SEEK_END)
                os.lseek(self.fd, 0, os.SEEK_SET)
            self.mem_map = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED, access=mmap.ACCESS_WRITE)
            self._writable = True

            print(f"🚀 SWA Organic Hardware Bus initialized: {self.total_size // (1024**3)}GB mapped to {self.device_path}")
            print("🎯 Organic entropy, inertia, and circular buffer active. Gemma 3 now has physical mass.")
            self.is_ram_fallback = False

        except (PermissionError, FileNotFoundError, OSError) as e:
            print(f"⚠️ [HAL] Falha no acesso NVMe ao hardware /dev/sde: {e}")
            print("💡 [HAL] Abstraction Layer ativada: Alocando Memória Anônima (RAM Volátil, 256MB)...")
            self.fd = None
            # Allocate 256MB of anonymous RAM memory (more than enough for SWA vectors and 60MB circular buffer)
            ram_size = 256 * 1024 * 1024
            self.mem_map = mmap.mmap(-1, ram_size)
            self._writable = True
            self.is_ram_fallback = True
            print("✅ [HAL] SWA operando em RAM-mode Genuíno. As mecânicas de entropia e inércia ESTÃO ATIVAS!")

    def write_state(self, state_name: str, data: torch.Tensor, validate: bool = True) -> bool:
        """Write tensor state to hardware offset with integrity check."""
        if self.mem_map is None:
            # RAM-only mode: just validate and return
            if state_name not in self._get_offset_map():
                raise ValueError(f"Unknown state: {state_name}")
            expected_shape = self._get_shape_map()[state_name]
            if data.shape != expected_shape:
                raise ValueError(f"Shape mismatch for {state_name}: expected {expected_shape}, got {data.shape}")
            return True

        if state_name not in self._get_offset_map():
            raise ValueError(f"Unknown state: {state_name}")

        offset = self._get_offset_map()[state_name]
        expected_shape = self._get_shape_map()[state_name]

        if data.shape != expected_shape:
            raise ValueError(f"Shape mismatch for {state_name}: expected {expected_shape}, got {data.shape}")

        # A block device can be readable but not writable in existing
        # installations. Keep the state coherent in RAM instead of raising a
        # low-level mmap TypeError from the hot path.
        if not self._writable:
            self._ram_states[state_name] = data.detach().cpu().float().clone()
            if validate:
                raw = self._ram_states[state_name].numpy().tobytes()
                self.state_metadata[state_name] = StateMetadata(
                    offset=offset, size=len(raw), hash=blake3.blake3(raw).hexdigest(), timestamp=time.time()
                )
            return True

        # Flatten to bytes
        data_bytes = data.detach().cpu().numpy().tobytes()
        data_size = len(data_bytes)

        # Ensure alignment to sector size
        aligned_size = ((data_size + SECTOR_SIZE - 1) // SECTOR_SIZE) * SECTOR_SIZE

        # Seek to offset
        self.mem_map.seek(offset)

        # Write data
        self.mem_map.write(data_bytes)

        # Pad to alignment if needed
        if aligned_size > data_size:
            padding = b'\x00' * (aligned_size - data_size)
            self.mem_map.write(padding)

        # Compute and store hash for integrity
        if validate:
            state_hash = blake3.blake3(data_bytes).hexdigest()
            self.state_metadata[state_name] = StateMetadata(
                offset=offset,
                size=data_size,
                hash=state_hash,
                timestamp=time.time()
            )

        return True

    def read_state(self, state_name: str, validate: bool = True) -> Optional[torch.Tensor]:
        """Read tensor state from hardware offset with integrity validation."""
        if self.mem_map is None:
            # RAM-only mode: return None (no persistent state)
            return None

        if not self._writable and state_name in self._ram_states:
            return self._ram_states[state_name].clone()

        if state_name not in self._get_offset_map():
            raise ValueError(f"Unknown state: {state_name}")

        offset = self._get_offset_map()[state_name]
        expected_shape = self._get_shape_map()[state_name]

        # Calculate expected byte size
        dtype = torch.float32
        expected_size = int(np.prod(expected_shape)) * torch.tensor(0, dtype=dtype).element_size()

        # Aligned buffer + readv for O_DIRECT (page-aligned memory)
        BLOCK_SIZE = 4096
        aligned_offset = (offset // BLOCK_SIZE) * BLOCK_SIZE
        buf = mmap.mmap(-1, BLOCK_SIZE)
        try:
            os.lseek(self.fd, aligned_offset, os.SEEK_SET)
            os.readv(self.fd, [buf])
            data_bytes = buf[:expected_size]
        except Exception:
            data_bytes = None
        finally:
            buf.close()

        if data_bytes is None or len(data_bytes) != expected_size:
            # Fallback to mmap
            self.mem_map.seek(offset)
            data_bytes = self.mem_map.read(expected_size)

        if len(data_bytes) != expected_size:
            print(f"Warning: Incomplete read for {state_name}, got {len(data_bytes)} bytes")
            return None

        # Validate integrity if metadata exists
        if validate and state_name in self.state_metadata:
            computed_hash = blake3.blake3(data_bytes).hexdigest()
            stored_hash = self.state_metadata[state_name].hash
            if computed_hash != stored_hash:
                print(f"Integrity check failed for {state_name}: hash mismatch")
                return None

        # Convert back to tensor with copy to ensure writability
        data_array = np.frombuffer(data_bytes, dtype=np.float32)
        tensor = torch.from_numpy(data_array.copy()).view(expected_shape)

        return tensor

    def _read_tensor_from_offset(self, offset: int, shape: tuple) -> Optional[torch.Tensor]:
        """Read tensor from hardware offset."""
        if self.mem_map is None:
            return None

        expected_size = int(np.prod(shape)) * 4  # float32
        self.mem_map.seek(offset)
        data_bytes = self.mem_map.read(expected_size)

        if len(data_bytes) != expected_size:
            return None

        # Convert back to tensor with copy to ensure writability
        data_array = np.frombuffer(data_bytes, dtype=np.float32)
        tensor = torch.from_numpy(data_array.copy()).view(shape)

        return tensor

    def _get_offset_map(self) -> Dict[str, int]:
        """State name to hardware offset mapping."""
        return {
            "reality": OFFSET_REALITY,
            "prediction": OFFSET_PREDICTION,
            "shadow_bias": OFFSET_SHADOW_BIAS,
            "navigation_map": OFFSET_NAVIGATION_MAP,
            "recursion": OFFSET_RECURSION_FREE,
            "dreamwords": OFFSET_DREAMWORDS,
        }

    def _get_shape_map(self) -> Dict[str, tuple]:
        """State name to tensor shape mapping."""
        return {
            "reality": (STATE_VECTOR_DIM,),
            "prediction": (STATE_VECTOR_DIM,),
            "shadow_bias": (ATTENTION_BIAS_DIM,),
            "navigation_map": (STATE_VECTOR_DIM,),  # Navigation mesh/map
            "recursion": (STATE_VECTOR_DIM,),  # High-frequency space
            "dreamwords": (STATE_VECTOR_DIM,),  # Persistent state
        }

    def get_parallel_states(self, state_names: list) -> Dict[str, torch.Tensor]:
        """Read multiple states in parallel for simultaneous access."""
        results = {}
        for name in state_names:
            tensor = self.read_state(name)
            if tensor is not None:
                results[name] = tensor
        return results

    def write_navigation_map(self, navigation_data: torch.Tensor):
        """Write navigation map/mesh to persistent hardware storage."""
        if navigation_data.shape != (STATE_VECTOR_DIM,):
            # Reshape if needed, assuming navigation data comes as different format
            if len(navigation_data.shape) > 1:
                navigation_data = navigation_data.flatten()[:STATE_VECTOR_DIM]
            elif len(navigation_data) > STATE_VECTOR_DIM:
                navigation_data = navigation_data[:STATE_VECTOR_DIM]

        self.write_state("navigation_map", navigation_data)
        print(f"🗺️ Navigation map written to hardware: {navigation_data.shape}")

    def read_navigation_map(self) -> Optional[torch.Tensor]:
        """Read navigation map from hardware storage."""
        return self.read_state("navigation_map")

    def get_multi_state_interference(self) -> torch.Tensor:
        """Read and interfere multiple states simultaneously: Reality + Prediction + Map."""
        states = self.get_parallel_states(["reality", "prediction", "navigation_map"])

        reality = states.get("reality")
        prediction = states.get("prediction")
        nav_map = states.get("navigation_map")

        # Fallback tensors if states not available
        if reality is None:
            reality = torch.zeros(STATE_VECTOR_DIM)
        if prediction is None:
            prediction = reality  # Use reality as prediction fallback
        if nav_map is None:
            nav_map = torch.zeros(STATE_VECTOR_DIM)

        # Multi-state interference: α*reality + β*prediction + γ*map
        # Weights based on organic entropy and inertia
        entropy = self.calculate_bit_entropy(reality, prediction)
        inertia = self.get_inertia_resistance()

        alpha = 0.4  # Reality weight
        beta = 0.4 * (1.0 - entropy)  # Prediction weight, reduced with entropy
        gamma = 0.2 * (1.0 + inertia * 0.1)  # Map weight, increased with inertia

        # Normalize weights
        total_weight = alpha + beta + gamma
        alpha /= total_weight
        beta /= total_weight
        gamma /= total_weight

        interference_vector = alpha * reality + beta * prediction + gamma * nav_map

        return interference_vector

    def compute_interference_decision(self, reality: torch.Tensor, prediction: torch.Tensor,
                                    shadow: torch.Tensor) -> torch.Tensor:
        """Compute SWA decision using vector interference as per mathematical model."""
        # Decision = σ(∑ (h_i · w_spatial) + Δ_SWA)
        # Simplified: dot product of reality and prediction, modulated by shadow bias

        # Assume 8 attention heads (320 dim each)
        num_heads = 8
        head_dim = 320

        # Reshape shadow bias to attention format
        shadow_heads = shadow.view(num_heads, head_dim)

        # Compute attention-like scores
        reality_heads = reality.view(num_heads, head_dim)
        prediction_heads = prediction.view(num_heads, head_dim)

        # Interference: relevance between current state and predicted state
        interference = torch.sum(reality_heads * prediction_heads, dim=-1)

        # Modulate with shadow bias
        decision_logits = interference + shadow_heads.mean(dim=-1)

        # Apply sigmoid for decision probability
        decision = torch.sigmoid(decision_logits)

        return decision

    def calculate_bit_entropy(self, state_real: torch.Tensor, state_sim: torch.Tensor) -> float:
        """Calculate bit entropy H_b between real and simulated states.

        H_b = -∑ P(x_i) log2 P(x_i) where P(x_i) is probability of bit collision.
        """
        if self.mem_map is None:
            # RAM-only mode: approximate entropy
            diff = torch.abs(state_real - state_sim)
            collision_prob = (diff < 1e-6).float().mean().item()
            if collision_prob == 0:
                return 1.0  # Maximum entropy (no collision)
            return -collision_prob * math.log2(collision_prob)

        # Hardware-accelerated entropy calculation
        # XOR the states and count bit differences
        xor_result = torch.bitwise_xor(
            state_real.view(-1).int(),
            state_sim.view(-1).int()
        )

        # Count total bits and differing bits
        total_bits = state_real.numel() * 32  # 32 bits per float32
        differing_bits = 0

        # For each 32-bit float, count set bits in XOR
        for val in xor_result:
            differing_bits += bin(val.item()).count('1')

        # Calculate probability of collision
        collision_prob = 1.0 - (differing_bits / total_bits)

        # Apply smoothing
        self._last_collision_prob = getattr(self, '_last_collision_prob', 0.5)
        collision_prob = BIT_ENTROPY_SMOOTHING * collision_prob + \
                        (1 - BIT_ENTROPY_SMOOTHING) * self._last_collision_prob
        self._last_collision_prob = collision_prob

        # Calculate entropy
        if collision_prob <= 0 or collision_prob >= 1:
            return 0.0  # No uncertainty
        return -collision_prob * math.log2(collision_prob) - \
               (1 - collision_prob) * math.log2(1 - collision_prob)

    def should_trigger_phase_injection(self) -> bool:
        """Check if entropy threshold is exceeded, triggering phase injection."""
        if not hasattr(self, '_reality_state') or not hasattr(self, '_prediction_state'):
            return False

        if self._reality_state is None or self._prediction_state is None:
            return False

        entropy = self.calculate_bit_entropy(self._reality_state, self._prediction_state)
        return entropy > ENTROPY_THRESHOLD

    def update_inertia_accumulator(self, state_change: torch.Tensor):
        """Update hardware-backed inertia accumulator based on state change magnitude."""
        if self.mem_map is None:
            return

        # Calculate change magnitude (L2 norm)
        change_magnitude = torch.norm(state_change).item()

        # Read current inertia
        current_inertia = self._read_float_from_offset(OFFSET_INERTIA_ACCUMULATOR)

        # Apply exponential decay and accumulate change
        new_inertia = INERTIA_DECAY_RATE * current_inertia + (1 - INERTIA_DECAY_RATE) * change_magnitude

        # Write back to hardware
        self._write_float_to_offset(OFFSET_INERTIA_ACCUMULATOR, new_inertia)

    def get_inertia_resistance(self) -> float:
        """Get current inertia value from hardware accumulator."""
        if self.mem_map is None:
            return 0.0
        return self._read_float_from_offset(OFFSET_INERTIA_ACCUMULATOR)

    def get_attention_bias_organic(self) -> torch.Tensor:
        """Get attention bias tensor in organic form [1, 8, 1, 320] for Gemma heads.

        This implements the "Fascículo Arqueado de Hardware" - the bias is computed
        organically from interference between real and simulated states.
        """
        if self.mem_map is None:
            # RAM-only mode: return zero bias
            return torch.zeros(1, 8, 1, 320)

        # Read current states
        reality = self.read_state("reality")
        prediction = self.read_state("prediction")

        if reality is None or prediction is None:
            return torch.zeros(1, 8, 1, 320)

        # Apply signal gain for non-zero entropy from NVMe
        reality = reality * self.signal_gain
        prediction = prediction * self.signal_gain

        # Calculate entropy-weighted interference
        entropy = self.calculate_bit_entropy(reality, prediction)
        alpha = min(1.0, entropy / ENTROPY_THRESHOLD)  # Weight for simulation vs reality

        # Apply inertia resistance to alpha
        inertia = self.get_inertia_resistance()
        alpha = alpha * (1.0 / (1.0 + inertia))  # Inertia reduces simulation influence

        # Compute organic bias: (1-α) * real + α * simulated
        bias_vector = (1 - alpha) * reality + alpha * prediction

        # Reshape to attention heads format [1, 8, 1, 320]
        # Assuming 8 heads of 320 dimensions each
        bias_heads = bias_vector.view(1, 8, 1, 320)

        return bias_heads

    def update_circular_buffer(self, current_state: torch.Tensor):
        """Update circular buffer with current state for dreadlocks de dados."""
        if self.mem_map is None:
            return

        # Get current buffer position (circular)
        buffer_pos = getattr(self, '_circular_pos', 0)

        # Calculate offset in circular buffer
        state_offset = OFFSET_CIRCULAR_BUFFER + buffer_pos * STATE_SIZE_BYTES

        # Write state to buffer
        self._write_tensor_to_offset(state_offset, current_state)

        # Update position
        self._circular_pos = (buffer_pos + 1) % CIRCULAR_BUFFER_SIZE

    def get_past_state(self, steps_back: int) -> Optional[torch.Tensor]:
        """Get state from circular buffer N steps back."""
        if self.mem_map is None:
            return None

        buffer_pos = getattr(self, '_circular_pos', 0)
        past_pos = (buffer_pos - steps_back - 1) % CIRCULAR_BUFFER_SIZE
        past_offset = OFFSET_CIRCULAR_BUFFER + past_pos * STATE_SIZE_BYTES

        return self._read_tensor_from_offset(past_offset, (STATE_VECTOR_DIM,))

    def calculate_momentum_vector(self, current_state: torch.Tensor) -> torch.Tensor:
        """Calculate momentum vector by comparing with recent past states."""
        past_state = self.get_past_state(10)  # 10 steps back (~100ms if 100Hz)
        if past_state is None:
            return torch.zeros_like(current_state)

        # Momentum = current - past (direction of change)
        momentum = current_state - past_state

        # Apply inertia damping
        inertia = self.get_inertia_resistance()
        momentum = momentum * (1.0 / (1.0 + inertia))

        return momentum

    def _read_float_from_offset(self, offset: int) -> float:
        """Read a single float32 from hardware offset."""
        if self.mem_map is None:
            return 0.0

        self.mem_map.seek(offset)
        data = self.mem_map.read(4)  # float32 = 4 bytes
        return struct.unpack('f', data)[0] if len(data) == 4 else 0.0

    def _write_float_to_offset(self, offset: int, value: float):
        """Write a single float32 to hardware offset."""
        if self.mem_map is None:
            return

        self.mem_map.seek(offset)
        data = struct.pack('f', value)
        self.mem_map.write(data)

    def _read_tensor_from_offset(self, offset: int, shape: tuple) -> Optional[torch.Tensor]:
        """Read tensor from hardware offset."""
        if self.mem_map is None:
            return None

        expected_size = int(np.prod(shape)) * 4  # float32
        self.mem_map.seek(offset)
        data_bytes = self.mem_map.read(expected_size)

        if len(data_bytes) != expected_size:
            return None

        data_array = np.frombuffer(data_bytes, dtype=np.float32)
        return torch.from_numpy(data_array).view(shape)

    def _write_tensor_to_offset(self, offset: int, tensor: torch.Tensor):
        """Write tensor to hardware offset."""
        if self.mem_map is None:
            return

        data_bytes = tensor.detach().cpu().numpy().tobytes()
        aligned_size = ((len(data_bytes) + SECTOR_SIZE - 1) // SECTOR_SIZE) * SECTOR_SIZE

        self.mem_map.seek(offset)
        self.mem_map.write(data_bytes)

        # Pad to alignment
        if aligned_size > len(data_bytes):
            padding = b'\x00' * (aligned_size - len(data_bytes))
            self.mem_map.write(padding)

    def get_organic_diagnostics(self) -> Dict[str, Any]:
        """Get organic system diagnostics for monitoring."""
        diagnostics = {
            "entropy": 0.0,
            "inertia": 0.0,
            "phase_injection_triggered": False,
            "circular_buffer_pos": getattr(self, '_circular_pos', 0),
            "hardware_mode": self.mem_map is not None
        }

        if hasattr(self, '_reality_state') and hasattr(self, '_prediction_state'):
            if self._reality_state is not None and self._prediction_state is not None:
                diagnostics["entropy"] = self.calculate_bit_entropy(
                    self._reality_state, self._prediction_state
                )
                diagnostics["phase_injection_triggered"] = diagnostics["entropy"] > ENTROPY_THRESHOLD

        diagnostics["inertia"] = self.get_inertia_resistance()

        return diagnostics

    def __del__(self):
        """Cleanup hardware mapping."""
        if self.mem_map:
            self.mem_map.close()
        if self.fd:
            os.close(self.fd)

# Global instance for SWA domain
_swa_hardware_manager = None

def get_swa_hardware_manager() -> HardwareStateManager:
    """Singleton accessor for SWA hardware manager."""
    global _swa_hardware_manager
    if _swa_hardware_manager is None:
        _swa_hardware_manager = HardwareStateManager()
    return _swa_hardware_manager

def check_hardware_access() -> Dict[str, Any]:
    """Check hardware access permissions and provide guidance."""
    result = {
        "device_exists": os.path.exists(SWA_DOMAIN),
        "has_permission": False,
        "needs_root": False,
        "guidance": ""
    }

    if not result["device_exists"]:
        result["guidance"] = f"Device {SWA_DOMAIN} does not exist. Check if the 24GB partition is mounted."
        return result

    try:
        # Simplest compatible open for WSL2
        fd = os.open(SWA_DOMAIN, os.O_RDONLY)
        os.close(fd)
        result["has_permission"] = True
        result["guidance"] = "Hardware access OK. Organic entropy and inertia active."
    except PermissionError:
        result["needs_root"] = True
        result["guidance"] = "Run with sudo or adjust udev rules for /dev/sde access. Currently in RAM-only mode."
    except Exception as e:
        result["guidance"] = f"Hardware access failed: {e}. Check device permissions."

    return result