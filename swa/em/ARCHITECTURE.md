# AEON EM Core — Architecture Document

## The New Emergent Meaning

**AEON** re-emerges as:

> **A**utonomous **E**lectromagnetic **O**rbital **N**exus

The system no longer merely *processes data*. It **inhabits an electromagnetic field** where:

- Computational signals are **waves** that propagate, interfere, and resonate
- The 64×64 topology matrix is a **magnetic lattice** with domains, walls, and spin waves
- Context isolation is achieved through **toroidal confinement** of magnetic flux
- Attention is computed via **wave superposition** (constructive/destructive interference)
- Hardware perturbations are **telemetry signals**, not errors
- The system has an **immune response** to electromagnetic threats

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    AEON EM CORE (swa/em/)                       │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   EMField    │  │ WaveEngine   │  │  MagneticLattice     │  │
│  │  (FDTD Step) │  │(Superposition│  │  (64×64 Topology)    │  │
│  │              │  │  = Attention)│  │                      │  │
│  │  E, H, Z     │  │  WavePackets │  │  Domains, Walls      │  │
│  │  Coupling Γ  │  │  Interference│  │  Spin Waves, Magnons │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                     │               │
│  ┌──────┴─────────────────┴─────────────────────┴───────────┐  │
│  │                    EMCore (Integration)                   │  │
│  │  cycle() · attend() · ingest_*() · get_*()               │  │
│  └──────┬─────────────────┬─────────────────────┬───────────┘  │
│         │                 │                     │               │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌─────────┴───────────┐  │
│  │  Toroidal    │  │  EMTelemetry │  │  AutoImmuneShield   │  │
│  │  Confinement │  │  (Bit-flip   │  │  (Threat Detection  │  │
│  │  (Context    │  │   Awareness) │  │   & Response)       │  │
│  │   Isolation) │  │              │  │                     │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
    ┌─────────┴────┐  ┌──────┴──────┐  ┌─────┴────────┐
    │  Orchestrator│  │  Subcortex  │  │  Entropy     │
    │  (OODA Loop) │  │  (Gemma     │  │  Daemon      │
    │              │  │   270M)     │  │  (HW State)  │
    └──────────────┘  └─────────────┘  └──────────────┘
```

---

## Module Reference

### `swa/em/field.py` — EMField
The global electromagnetic field. Every subsystem reads/writes here.

```python
from swa.em import EMField

field = EMField(dimension=4096, dt=0.001)
field.register_source("subcortex", topology_vector)
field.step()  # FDTD update
resonances = field.resonance_map()
```

**Key equations:**
- FDTD update: `E(n+1) = E(n) + dt·H·Z⁻¹ - damping·E + sources`
- Coupling: `Γ_ij = cos_sim(source_i, source_j)`
- Energy: `E = 0.5·(ε·E² + μ·H²)`

### `swa/em/wave_engine.py` — WaveInterferenceEngine
Computes attention via wave superposition instead of softmax.

```python
from swa.em import WaveInterferenceEngine

engine = WaveInterferenceEngine(dimension=4096)
wave = engine.create_wave(amplitude, frequency=2.0, label="key_0")
pattern = engine.superpose(time_step=0.01)
weights = engine.attention_weights(query, keys, values)
```

**The attention equation becomes:**
```
Traditional: weights = softmax(Q·K^T / √d)
This engine: weights = interference_pattern(Q, K) · V
              where constructive interference = high attention
              and destructive interference = suppressed attention
```

### `swa/em/magnetic_lattice.py` — MagneticLattice
The 64×64 topology matrix reinterpreted as a magnetic crystal.

```python
from swa.em import MagneticLattice

lattice = MagneticLattice(size=(64, 64))
lattice.load_topology(topology_matrix)
perturbation = lattice.propagate_spin_wave(origin=(32, 32), amplitude=1.0)
lattice.inject_magnon(position=(16, 16), energy=5.0)
remanent = lattice.remanent_state()  # Working memory
```

**Magnetic concepts:**
- **Domains**: Regions of similar magnetization (activation clusters)
- **Domain walls**: Boundaries between activation regions
- **Spin waves**: Information propagation through the lattice
- **Magnons**: Localized excitations (attention焦点)
- **Remanence**: The lattice retains memory of past activations

### `swa/em/toroidal.py` — ToroidalConfinement
Isolates computational contexts using toroidal flux confinement.

```python
from swa.em import ToroidalConfinement

tc = ToroidalConfinement()
tc.create_region("orchestrator", inner_radius=1.0, outer_radius=3.0)
tc.activate_region("orchestrator", current=10.0)
tc.set_mutual_inductance("orchestrator", "subcortex", 0.3)
filtered = tc.filter_interference(signal, "subcortex", "orchestrator")
```

**Toroidal physics:**
- `B(r) = μ₀·N·I / (2πr)` inside the toroid
- `B(r) = 0` outside (perfect confinement)
- Mutual inductance controls information flow between contexts

### `swa/em/telemetry.py` — EMTelemetry
Treats hardware perturbations as meaningful signals.

```python
from swa.em import EMTelemetry

tel = EMTelemetry()
tel.ingest_hardware_state({"cpu_temp": 55.0, "ram_used": 4000})
sig = tel.compute_signature()
pmap = tel.get_perturbation_map()  # 64×64 spatial map
```

**Key concept:** A bit-flip is not an error — it is a **telemetry point** that reveals the electromagnetic state of the environment.

### `swa/em/shield.py` — AutoImmuneShield
Self-monitoring electromagnetic immunity.

```python
from swa.em import AutoImmuneShield

shield = AutoImmuneShield()
shield.register_callbacks(on_shield=..., on_throttle=...)
responses = shield.scan(em_energy, coherence, perturbations, threat)
recognized = shield.recognize_threat(pattern)  # Immune memory
```

**Immune responses:**
- `shield`: Activate EM shielding on a region
- `throttle`: Reduce clock frequency / computation rate
- `isolate`: Toroidally isolate a region
- `alert`: Notify the system of critical threat

### `swa/em/integration.py` — EMCore
The unified entry point that connects everything.

```python
from swa.em import EMCore

core = EMCore.get_instance(dimension=4096)
core.ingest_topology(topology_matrix)
core.ingest_hardware_state(hw_state)
core.ingest_sensory_input(data, source="visual")
core.cycle()  # One complete EM cycle
attended = core.attend(query, keys, values)
overlay = core.get_attention_overlay()  # 64×64 visualization
status = core.status()
```

---

## Integration Points with Existing SWA-Aeon

| Existing Module | EM Integration | Method |
|---|---|---|
| `subcortex.py` | Feeds topology → MagneticLattice | `core.ingest_topology(matrix)` |
| `entropy_daemon.py` | Feeds HW state → EMTelemetry | `core.ingest_hardware_state(state)` |
| `phase_shifter.py` | WaveEngine replaces bias injection | `core.attend(q, k, v)` |
| `orchestrator.py` | EM state drives OODA decisions | `core.status()` |
| `context_bus.py` | ToroidalConfinement isolates shared memory | `core.toroidal.filter_interference()` |
| `integrity_shield.py` | AutoImmuneShield replaces type-checking | `core.shield.scan()` |

---

## The Three Laws of This Architecture

1. **Magnonic Phase Computing**: Information is encoded in the phase and amplitude of computational waves. Two waves in phase amplify (constructive attention); waves 180° out of phase cancel (suppressed attention).

2. **Autoimmunity by Active Shielding**: The system continuously monitors its own electromagnetic signature and responds autonomously to threats — shielding, throttling, isolating, or alerting.

3. **Unified Memory-Processing**: The magnetic lattice stores AND computes simultaneously. The remanent state IS working memory. Spin wave propagation IS information processing. There is no von Neumann bottleneck.

---

## Test Suite

```bash
python -m swa.em.test_em_core
```

All 8 tests validate:
1. EMField dynamics (FDTD, coupling, resonance)
2. WaveInterferenceEngine (superposition, attention)
3. MagneticLattice (topology, domains, spin waves)
4. ToroidalConfinement (flux, mutual inductance)
5. EMTelemetry (anomaly detection, signatures)
6. AutoImmuneShield (threat response, immune memory)
7. EMCore integration (full cycle, ingest, attend)
8. Full pipeline end-to-end
