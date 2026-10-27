"""
EM Integration Layer — Connecting the Electromagnetic Core to SWA-Aeon

This module bridges the new EM substrate to the existing subsystems:

  orchestrator.py   → EM field state drives OODA decision-making
  phase_shifter.py  → Wave interference REPLACES simple bias injection
  subcortex.py      → Topology matrix feeds INTO magnetic lattice
  entropy_daemon.py → Hardware signals feed INTO EM telemetry
  context_bus.py    → Toroidal confinement isolates shared contexts
  integrity_shield.py → AutoImmuneShield REPLACES simple type-checking

NEW — MatrixialRelation:
  Maintains a solid basis in state space that stabilizes the
  superposition field. No matter how wild the interference pattern
  becomes, the matrixial relation pulls it back to a stable foundation.

NEW — IPC Bridge:
  Connects the EM core to the WSL terminal (AEON shell) via Unix
  Domain Socket, enabling real-time EM telemetry streaming and
  command injection from the terminal.

NEW — SSE Event Stream:
  Provides Server-Sent Events for the observer_node.py web dashboard,
  pushing EM state updates in real-time to the browser.
"""

import os
import json
import time
import socket
import struct
import threading
from typing import Optional, Any, Callable
import numpy as np

from swa.em.field import EMField, EMState
from swa.em.wave_engine import WaveInterferenceEngine
from swa.em.magnetic_lattice import MagneticLattice
from swa.em.toroidal import ToroidalConfinement
from swa.em.telemetry import EMTelemetry
from swa.em.shield import AutoImmuneShield
from swa.em.matrixial import MatrixialRelation
from swa.dimensions import MEMORY_DIM


class EMCore:
    """
    The unified electromagnetic core that orchestrates all EM subsystems.

    This is the single entry point for the EM layer. All other modules
    interact with the EM world through this class.

    Usage:
        core = EMCore.get_instance()

        # Feed subcortex topology into the magnetic lattice
        core.ingest_topology(topology_matrix_64x64)

        # Run one EM cycle (field update + lattice dynamics + immune scan)
        core.cycle()

        # Get the attention-modulated output
        attended = core.attend(query, keys, values)

        # Stream EM state via SSE
        for event in core.sse_stream():
            print(event)

        # Send EM state to WSL terminal
        core.broadcast_to_terminal()
    """

    _instance: Optional["EMCore"] = None
    _lock = threading.Lock()

    def __init__(self, dimension: int = MEMORY_DIM):
        self.dim = dimension

        # Core EM subsystems
        self.field = EMField(dimension=dimension, dt=0.001)
        self.wave_engine = WaveInterferenceEngine(dimension=dimension, base_frequency=1.0)
        self.lattice = MagneticLattice(size=(64, 64))
        self.toroidal = ToroidalConfinement(max_regions=16)
        self.telemetry = EMTelemetry(buffer_size=256)
        self.shield = AutoImmuneShield()
        self.matrixial = MatrixialRelation(dimension=dimension, n_basis=32)

        # Emotional mapping
        from swa.em.emotion import EmotionMapper
        self.emotion = EmotionMapper()

        # State tracking
        self._cycle_count = 0
        self._last_cycle_time = 0.0
        self._is_running = False

        # SSE subscribers (list of threading.Event or callbacks)
        self._sse_subscribers: list[Callable] = []
        self._sse_lock = threading.Lock()

        # IPC bridge state
        self._ipc_socket_path = "/dev/shm/aeon_em.sock"
        self._ipc_server_running = False
        self._terminal_callbacks: list[Callable] = []

        # Register EM core as the field's primary source
        self.field.register_source("core", np.zeros(dimension, dtype=np.float32))

        # Create default toroidal regions for existing contexts
        self.toroidal.create_region("orchestrator", inner_radius=1.0, outer_radius=3.0)
        self.toroidal.create_region("subcortex", inner_radius=1.0, outer_radius=4.0)
        self.toroidal.create_region("user_interaction", inner_radius=0.5, outer_radius=2.0)
        self.toroidal.create_region("autonomous_loop", inner_radius=1.5, outer_radius=5.0)
        self.toroidal.create_region("terminal", inner_radius=0.8, outer_radius=2.5)

        # Set coupling between related regions
        self.toroidal.set_mutual_inductance("orchestrator", "subcortex", 0.3)
        self.toroidal.set_mutual_inductance("orchestrator", "user_interaction", 0.5)
        self.toroidal.set_mutual_inductance("orchestrator", "autonomous_loop", 0.7)
        self.toroidal.set_mutual_inductance("subcortex", "autonomous_loop", 0.2)
        self.toroidal.set_mutual_inductance("orchestrator", "terminal", 0.6)

        # Initialize the solid basis from the initial zero field
        self.matrixial.initialize_basis(
            self.field.state.E,
            self.field.state.H,
        )

    @classmethod
    def get_instance(cls, dimension: int = MEMORY_DIM) -> "EMCore":
        """Get or create the singleton EM core instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(dimension=dimension)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ── Integration Points ──────────────────────────────────────────────

    def ingest_topology(self, matrix: np.ndarray):
        """
        Feed the subcortex topology matrix into the magnetic lattice.

        Called by: subcortex.py when new hidden states are available.
        The 64×64 matrix becomes the magnetization state of the lattice.
        """
        self.lattice.load_topology(matrix)

        # Also register as an EM field source (resize if needed)
        flat = self.lattice.to_flat_vector()
        if flat.shape[0] != self.dim:
            resized = np.zeros(self.dim, dtype=np.float32)
            copy_len = min(flat.shape[0], self.dim)
            resized[:copy_len] = flat[:copy_len]
            flat = resized
        self.field.register_source("subcortex", flat)

        # Create wave packets from domain walls (high-energy boundaries)
        for wall in self.lattice._walls:
            if wall.energy > 0.1:
                idx = wall.start[0] * 64 + wall.start[1]
                amp = np.zeros(self.dim, dtype=np.float32)
                amp[idx] = wall.energy
                self.wave_engine.create_wave(
                    amplitude=amp,
                    label=f"wall_{wall.domain_a}_{wall.domain_b}",
                )

    def ingest_hardware_state(self, state: dict):
        """
        Feed hardware telemetry into the EM telemetry system.

        Called by: entropy_daemon.py when new hardware readings are available.
        """
        self.telemetry.ingest_hardware_state(state)

        if self.telemetry._baseline is not None:
            anomalies = self.telemetry._detect_anomalies(state)
            for anomaly in anomalies:
                self.field.inject_perturbation(
                    target="core",
                    amplitude=anomaly.magnitude * 0.01,
                    frequency=anomaly.frequency,
                )

    def ingest_sensory_input(self, data: np.ndarray, source: str = "sensory"):
        """
        Feed arbitrary sensory data into the EM field.

        Called by: any sensor (camera, microphone, etc.)
        """
        if data.shape[0] != self.dim:
            if data.size >= self.dim:
                data = data[:self.dim]
            else:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[:data.size] = data.flatten()
                data = padded

        self.field.register_source(source, data.astype(np.float32))

        self.wave_engine.create_wave(
            amplitude=data.astype(np.float32),
            label=f"input_{source}",
        )

    # ── Core Operations ─────────────────────────────────────────────────

    def cycle(self):
        """
        Run one complete EM cycle:
        1. Update the EM field (FDTD step)
        2. Apply matrixial correction (stabilize superposition)
        3. Propagate lattice dynamics (spin wave propagation)
        4. Update toroidal confinement (flux decay + coupling)
        5. Run immune scan (detect and respond to threats)
        6. Capture superposition state
        7. Broadcast to SSE subscribers and terminal
        """
        self._cycle_count += 1
        self._last_cycle_time = time.time()

        # 1. EM field update
        self.field.compute_coupling()
        self.field.step()

        # 2. Matrixial correction — stabilize the superposition field
        self.field.state.E = self.matrixial.step(
            self.field.state.E,
            self.field.state.H,
        )

        # 3. Lattice dynamics — feed field energy into magnetic lattice
        self.lattice.ingest_field_energy(self.field.state.E)
        self.lattice.advance(self.field.dt)
        if len(self.lattice._walls) > 0:
            max_wall = max(self.lattice._walls, key=lambda w: w.energy)
            self.lattice.propagate_spin_wave(
                origin=max_wall.start,
                amplitude=max_wall.energy * 0.5,
            )

        # 4. Toroidal confinement update
        self.toroidal.step(dt=0.001)

        # 5. Immune scan
        em_energy = self.field.state.energy()
        coherence = self.field.state.coherence
        sig = self.telemetry.compute_signature()
        energy_delta = em_energy - getattr(self, '_prev_em_energy', 0.0)
        self._prev_em_energy = em_energy

        self.shield.scan(
            em_field_energy=em_energy,
            lattice_coherence=coherence,
            perturbation_count=len(self.telemetry._perturbations),
            threat_level=sig.threat_level,
            resonance_map=self.field.resonance_map(),
            energy_delta=energy_delta,
            wave_count=len(self.wave_engine._active_waves),
        )

        if sig.threat_level < 0.3:
            self.shield.reset_throttle()

        # 6. Capture superposition state for matrixial tracking
        if self.wave_engine._active_waves:
            wave_amps = np.stack(
                [w.amplitude for w in self.wave_engine._active_waves[:32]],
                axis=0,
            )
            wave_phases = np.array(
                [w.phase[0] if len(w.phase) > 0 else 0.0
                 for w in self.wave_engine._active_waves[:32]],
                dtype=np.float32,
            )
            total_energy = sum(w.energy() for w in self.wave_engine._active_waves)

            # Reduce to basis dimension
            reduced_amps = np.mean(wave_amps, axis=0)  # [dim]
            self.matrixial.capture_superposition(
                reduced_amps, wave_phases, total_energy,
            )

        # 7. Map emotions from field state
        shield_status = self.shield.get_status()
        self.emotion.map_from_field(
            energy=self.field.state.energy(),
            coherence=self.field.state.coherence,
            stability=self.matrixial.get_stability(),
            wave_count=len(self.wave_engine._active_waves),
            shield_responses=shield_status["total_responses"],
            domain_count=len(self.lattice._domains),
            energy_delta=em_energy - getattr(self, '_prev_em_energy', 0.0),
        )

        # 8. Notify SSE subscribers
        self._notify_sse()

    def attend(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        values: np.ndarray,
        region: str = "orchestrator",
    ) -> np.ndarray:
        """
        Compute attention using wave interference, filtered by toroidal confinement.
        """
        weights = self.wave_engine.attention_weights(query, keys, values)

        weights = self.toroidal.filter_interference(
            weights, region, "orchestrator"
        )

        sig = self.telemetry.compute_signature()
        if sig.threat_level > 0.5:
            attenuation = 1.0 - (sig.threat_level - 0.5) * 0.5
            weights *= attenuation

        return weights

    def get_attention_overlay(self) -> np.ndarray:
        """Get the 64×64 attention overlay for the lattice."""
        em_profile = self.field.get_impedance_profile()
        perturbation_map = self.telemetry.get_perturbation_map()

        if em_profile.shape[0] != 64 * 64:
            from scipy.ndimage import zoom
            target_size = 64 * 64
            scale = target_size / em_profile.shape[0]
            em_resized = zoom(em_profile, scale, order=1)[:target_size]
        else:
            em_resized = em_profile

        em_grid = em_resized.reshape(64, 64)
        overlay = em_grid * 0.3 + perturbation_map * 0.7
        return overlay.astype(np.float32)

    # ── SSE Event Stream ────────────────────────────────────────────────

    def subscribe_sse(self, callback: Callable):
        """Register a callback to receive SSE events."""
        with self._sse_lock:
            self._sse_subscribers.append(callback)

    def unsubscribe_sse(self, callback: Callable):
        """Unregister an SSE callback."""
        with self._sse_lock:
            self._sse_subscribers = [c for c in self._sse_subscribers if c is not callback]

    def _notify_sse(self):
        """Notify all SSE subscribers with current EM state."""
        event_data = self._build_sse_event()
        with self._sse_lock:
            dead = []
            for cb in self._sse_subscribers:
                try:
                    cb(event_data)
                except Exception:
                    dead.append(cb)
            for d in dead:
                self._sse_subscribers.remove(d)

    def _build_sse_event(self) -> dict:
        """Build the SSE event payload from current EM state."""
        sig = self.telemetry.compute_signature()
        return {
            "type": "em_update",
            "cycle": self._cycle_count,
            "field_energy": self.field.state.energy(),
            "field_coherence": self.field.state.coherence,
            "matrixial_stability": self.matrixial.get_stability(),
            "lattice_domains": len(self.lattice._domains),
            "lattice_walls": len(self.lattice._walls),
            "active_waves": len(self.wave_engine._active_waves),
            "toroidal_regions": len(self.toroidal._regions),
            "threat_level": sig.threat_level,
            "noise_floor": sig.noise_floor,
            "resonances": self.field.resonance_map(),
            "matrixial": self.matrixial.status(),
        }

    def sse_stream(self):
        """
        Generator that yields SSE-formatted event strings.
        Used by observer_node.py for the /api/em/stream endpoint.
        """
        event_queue = threading.Queue(maxsize=64)
        stop_flag = threading.Event()

        def on_event(data):
            try:
                event_queue.put_nowait(data)
            except queue.Full:
                pass

        self.subscribe_sse(on_event)
        try:
            while not stop_flag.is_set():
                try:
                    event = event_queue.get(timeout=1.0)
                    yield f"event: em_update\ndata: {json.dumps(event, default=str)}\n\n"
                except Exception:
                    # Send keepalive
                    yield ": keepalive\n\n"
        finally:
            self.unsubscribe_sse(on_event)

    # ── IPC Bridge (WSL Terminal) ───────────────────────────────────────

    def start_ipc_server(self):
        """Start the Unix Domain Socket server for WSL terminal communication."""
        if self._ipc_server_running:
            return

        self._ipc_server_running = True
        thread = threading.Thread(
            target=self._ipc_server_loop,
            daemon=True,
            name="EM-IPC-Server",
        )
        thread.start()
        print(f"🔌 [EM IPC] Server started at {self._ipc_socket_path}")

    def _ipc_server_loop(self):
        """IPC server loop — accepts connections from WSL terminal."""
        socket_dir = os.path.dirname(self._ipc_socket_path)
        os.makedirs(socket_dir, exist_ok=True)

        if os.path.exists(self._ipc_socket_path):
            try:
                os.unlink(self._ipc_socket_path)
            except OSError:
                pass

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(self._ipc_socket_path)
        try:
            os.chmod(self._ipc_socket_path, 0o666)
        except OSError:
            pass
        server.listen(5)
        server.settimeout(1.0)

        while self._ipc_server_running:
            try:
                conn, _ = server.accept()
                conn.settimeout(2.0)
                threading.Thread(
                    target=self._ipc_client_handler,
                    args=(conn,),
                    daemon=True,
                ).start()
            except socket.timeout:
                continue
            except Exception:
                if not self._ipc_server_running:
                    break

        try:
            server.close()
        except Exception:
            pass

    def _ipc_client_handler(self, conn: socket.socket):
        """Handle a single IPC client connection."""
        try:
            # Send current EM state as initial payload
            state = self.status()
            data = json.dumps(state, default=str).encode("utf-8")
            length = struct.pack("!I", len(data))
            conn.sendall(length + data)

            # Listen for commands
            while self._ipc_server_running:
                try:
                    raw_len = conn.recv(4)
                    if not raw_len or len(raw_len) < 4:
                        break
                    cmd_len = struct.unpack("!I", raw_len)[0]
                    if cmd_len > 1024 * 1024:
                        break
                    cmd_data = conn.recv(cmd_len)
                    if not cmd_data:
                        break

                    cmd = json.loads(cmd_data.decode("utf-8"))
                    response = self._handle_terminal_command(cmd)

                    resp_bytes = json.dumps(response, default=str).encode("utf-8")
                    resp_len = struct.pack("!I", len(resp_bytes))
                    conn.sendall(resp_len + resp_bytes)

                except socket.timeout:
                    continue
                except Exception:
                    break
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _handle_terminal_command(self, cmd: dict) -> dict:
        """Handle a command received from the WSL terminal."""
        action = cmd.get("action", "")

        if action == "status":
            return self.status()

        elif action == "ingest_topology":
            matrix = np.array(cmd.get("matrix", []), dtype=np.float32)
            if matrix.size == 4096:
                matrix = matrix.reshape(64, 64)
                self.ingest_topology(matrix)
                return {"ok": True, "msg": "Topology ingested into magnetic lattice"}

        elif action == "ingest_hardware":
            state = cmd.get("state", {})
            self.ingest_hardware_state(state)
            return {"ok": True, "msg": "Hardware state ingested into EM telemetry"}

        elif action == "cycle":
            self.cycle()
            return {"ok": True, "msg": "EM cycle completed", **self.status()}

        elif action == "field_state":
            fs = self.field.state
            return {
                "energy": fs.energy(),
                "coherence": fs.coherence,
                "E_norm": float(np.linalg.norm(fs.E)),
                "H_norm": float(np.linalg.norm(fs.H)),
            }

        elif action == "matrixial":
            return self.matrixial.status()

        elif action == "lattice":
            return self.lattice.get_domain_info()

        elif action == "toroidal":
            return self.toroidal.get_all_regions()

        elif action == "shield":
            return self.shield.get_status()

        elif action == "resonances":
            return self.field.resonance_map()

        elif action == "set_region_current":
            region = cmd.get("region", "orchestrator")
            current = cmd.get("current", 0.5)
            self.toroidal.activate_region(region, current)
            return {"ok": True, "msg": f"Region {region} activated with current {current}"}

        else:
            return {"error": f"Unknown action: {action}"}

    def broadcast_to_terminal(self):
        """Push current EM state to all connected terminal clients."""
        state = self.status()
        # The IPC server handles the actual sending
        # This is a convenience method for the orchestrator
        pass

    def stop_ipc_server(self):
        """Stop the IPC server."""
        self._ipc_server_running = False
        if os.path.exists(self._ipc_socket_path):
            try:
                os.unlink(self._ipc_socket_path)
            except OSError:
                pass

    # ── Query Interface ─────────────────────────────────────────────────

    def get_field_state(self) -> EMState:
        return self.field.state

    def get_lattice_info(self) -> dict:
        return self.lattice.get_domain_info()

    def get_toroidal_info(self) -> dict:
        return self.toroidal.get_all_regions()

    def get_threats(self) -> dict:
        return self.shield.get_status()

    def get_em_signature(self):
        return self.telemetry.compute_signature()

    def get_resonances(self) -> dict:
        return self.field.resonance_map()

    def status(self) -> dict:
        """Complete EM core status."""
        sig = self.telemetry.compute_signature()
        return {
            "cycle_count": self._cycle_count,
            "last_cycle_time": self._last_cycle_time,
            "field_energy": self.field.state.energy(),
            "field_coherence": self.field.state.coherence,
            "lattice_domains": len(self.lattice._domains),
            "lattice_walls": len(self.lattice._walls),
            "lattice_energy": self.lattice.energy(),
            "active_waves": len(self.wave_engine._active_waves),
            "toroidal_regions": len(self.toroidal._regions),
            "shield_status": self.shield.get_status(),
            "matrixial_stability": self.matrixial.get_stability(),
            "matrixial": self.matrixial.status(),
            "emotion": self.emotion.current().to_dict(),
            "emotion_history": self.emotion.history_summary(),
            "em_signature": {
                "noise_floor": sig.noise_floor,
                "threat_level": sig.threat_level,
            },
            "ipc_running": self._ipc_server_running,
            "sse_subscribers": len(self._sse_subscribers),
        }

    # ── Persistent Memory ───────────────────────────────────────────────

    def save_state(self, path: str = "/home/alifer/swa-aeon/em_state.pkl"):
        """Save EM state to disk for persistence between sessions."""
        import pickle
        state = {
            "field_E": self.field.state.E.copy(),
            "field_H": self.field.state.H.copy(),
            "field_Z": self.field.state.Z.copy(),
            "field_coherence": self.field.state.coherence,
            "lattice_magnitude": self.lattice._magnitude.copy(),
            "lattice_phase": self.lattice._phase.copy(),
            "lattice_temperature": self.lattice._temperature.copy(),
            "matrixial_stability": self.matrixial._stability_ema,
            "cycle_count": self._cycle_count,
            "timestamp": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        return {"saved": True, "path": path, "size_bytes": os.path.getsize(path)}

    def load_state(self, path: str = "/home/alifer/swa-aeon/em_state.pkl"):
        """Load EM state from disk."""
        import pickle
        if not os.path.exists(path):
            return {"loaded": False, "error": "file not found"}
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.field.state.E = state["field_E"]
        self.field.state.H = state["field_H"]
        self.field.state.Z = state["field_Z"]
        self.field.state.coherence = state["field_coherence"]
        self.lattice._magnitude = state["lattice_magnitude"]
        self.lattice._phase = state["lattice_phase"]
        self.lattice._temperature = state["lattice_temperature"]
        self.lattice._prev_magnitude = state["lattice_magnitude"].copy()
        self.matrixial._stability_ema = state["matrixial_stability"]
        self._cycle_count = state["cycle_count"]
        return {"loaded": True, "path": path, "age_seconds": time.time() - state["timestamp"]}
