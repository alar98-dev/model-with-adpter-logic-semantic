"""
Tests for the AEON EM Core — Electromagnetic Orbital Nexus

Validates:
  1. EMField dynamics (FDTD step, coupling, resonance)
  2. WaveInterferenceEngine (superposition, attention weights)
  3. MagneticLattice (topology loading, domain detection, spin waves)
  4. ToroidalConfinement (flux confinement, mutual inductance)
  5. EMTelemetry (anomaly detection, signature computation)
  6. AutoImmuneShield (threat detection, response execution)
  7. EMCore integration (full cycle, ingest, attend)
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_emfield():
    """Test EMField dynamics."""
    from swa.em.field import EMField, EMState

    field = EMField(dimension=256, dt=0.001)

    # Test initial state
    assert field.state.E.shape == (256,)
    assert field.state.H.shape == (256,)
    assert field.state.Z.shape == (256,)
    assert field.state.energy() == 0.0

    # Test source registration
    source = np.random.randn(256).astype(np.float32)
    field.register_source("test", source)
    assert "test" in field._sources

    # Test FDTD step
    field.step()
    assert field.state.timestamp > 0
    assert len(field._history) == 1

    # Test coupling computation
    field.register_source("test2", np.random.randn(256).astype(np.float32))
    gamma = field.compute_coupling()
    assert gamma.shape[0] == 2

    # Test resonance map
    # Inject a strong sinusoidal signal
    t = np.arange(256, dtype=np.float32)
    signal = np.sin(2 * np.pi * 5 * t / 256).astype(np.float32)
    field.register_source("strong", signal * 10.0)
    for _ in range(10):
        field.step()
    resonances = field.resonance_map()
    assert isinstance(resonances, dict)

    print("  [PASS] EMField")


def test_wave_engine():
    """Test WaveInterferenceEngine."""
    from swa.em.wave_engine import WaveInterferenceEngine, WavePacket

    engine = WaveInterferenceEngine(dimension=256)

    # Create waves
    amp1 = np.ones(256, dtype=np.float32)
    amp2 = np.zeros(256, dtype=np.float32)
    amp2[:128] = 1.0

    w1 = engine.create_wave(amp1, frequency=1.0, label="wave1")
    w2 = engine.create_wave(amp2, frequency=2.0, label="wave2")

    assert len(engine._active_waves) == 2

    # Test superposition
    pattern = engine.superpose(time_step=0.01)
    assert pattern.pattern.shape == (256,)
    assert pattern.coherence >= 0.0
    assert pattern.coherence <= 1.0

    # Test attention weights
    query = np.random.randn(256).astype(np.float32)
    keys = np.random.randn(3, 256).astype(np.float32)
    values = np.random.randn(3, 256).astype(np.float32)

    weights = engine.attention_weights(query, keys, values)
    assert weights.shape == (256,)
    assert np.sum(np.abs(weights)) > 0

    # Test resonance amplification
    signal = np.random.randn(256).astype(np.float32)
    amplified = engine.resonance_amplify(signal, frequency=0.5)
    assert amplified.shape == (256,)

    print("  [PASS] WaveInterferenceEngine")


def test_magnetic_lattice():
    """Test MagneticLattice."""
    from swa.em.magnetic_lattice import MagneticLattice

    lattice = MagneticLattice(size=(64, 64))

    # Test topology loading
    matrix = np.random.rand(64, 64).astype(np.float32)
    lattice.load_topology(matrix)

    assert lattice._magnitude.shape == (64, 64)
    assert lattice._phase.shape == (64, 64)

    # Test domain detection
    assert len(lattice._domains) > 0
    assert len(lattice._walls) >= 0

    # Test spin wave propagation
    perturbation = lattice.propagate_spin_wave(
        origin=(32, 32),
        amplitude=1.0,
        wavelength=8.0,
    )
    assert perturbation.shape == (64, 64)

    # Test magnon injection
    lattice.inject_magnon((16, 16), energy=5.0)

    # Test remanent state
    remanent = lattice.remanent_state()
    assert remanent.shape == (64, 64)

    # Test flat vector conversion
    flat = lattice.to_flat_vector()
    assert flat.shape == (4096,)

    lattice2 = MagneticLattice(size=(64, 64))
    lattice2.from_flat_vector(flat)
    assert np.allclose(lattice2._magnitude, lattice._magnitude, atol=1e-6)

    # Test energy computation
    energy = lattice.energy()
    assert isinstance(energy, float)

    print("  [PASS] MagneticLattice")


def test_toroidal():
    """Test ToroidalConfinement."""
    from swa.em.toroidal import ToroidalConfinement

    tc = ToroidalConfinement(max_regions=4)

    # Create regions
    tc.create_region("ctx_a", inner_radius=1.0, outer_radius=3.0, n_windings=100)
    tc.create_region("ctx_b", inner_radius=1.0, outer_radius=4.0, n_windings=200)

    # Set coupling
    tc.set_mutual_inductance("ctx_a", "ctx_b", 0.5)

    # Activate regions
    tc.activate_region("ctx_a", current=10.0)
    tc.activate_region("ctx_b", current=5.0)

    # Check flux
    assert tc._regions["ctx_a"].stored_flux > 0
    assert tc._regions["ctx_b"].stored_flux > 0

    # Check field profile
    profile_a = tc.field_profile("ctx_a", n_points=50)
    assert len(profile_a) == 50

    # Check confinement ratio
    ratio_a = tc.confinement_ratio("ctx_a")
    assert 0.0 <= ratio_a <= 1.0

    # Check filtering
    signal = np.ones(256, dtype=np.float32)
    filtered = tc.filter_interference(signal, "ctx_a", "ctx_b")
    assert filtered.shape == (256,)

    # Step and check decay
    tc.step(dt=0.01)
    tc.step(dt=0.01)
    # Current should decay over time
    assert tc._regions["ctx_a"].current < 10.0

    # Test eviction
    tc.create_region("ctx_c")
    tc.create_region("ctx_d")
    tc.create_region("ctx_e")  # Should trigger eviction
    assert len(tc._regions) <= 4

    print("  [PASS] ToroidalConfinement")


def test_telemetry():
    """Test EMTelemetry."""
    from swa.em.telemetry import EMTelemetry

    tel = EMTelemetry(buffer_size=128)

    # Feed normal data to build baseline
    for i in range(60):
        tel.ingest_hardware_state({
            "cpu_temp": 55.0 + np.random.normal(0, 0.5),
            "ram_used": 4000 + np.random.normal(0, 50),
            "swap_used": 100 + np.random.normal(0, 5),
            "context_switches": 1000 + np.random.normal(0, 20),
        })

    # Feed anomalous data
    tel.ingest_hardware_state({
        "cpu_temp": 95.0,  # Huge spike
        "ram_used": 8000,
        "swap_used": 500,
        "context_switches": 5000,
    })

    # Check that anomalies were detected
    assert len(tel._perturbations) > 0

    # Compute signature
    sig = tel.compute_signature()
    assert sig.noise_floor >= 0
    assert 0.0 <= sig.threat_level <= 1.0
    assert isinstance(sig.recommendations, list)

    # Test perturbation map
    pmap = tel.get_perturbation_map()
    assert pmap.shape == (64, 64)

    print("  [PASS] EMTelemetry")


def test_shield():
    """Test AutoImmuneShield."""
    from swa.em.shield import AutoImmuneShield

    shield = AutoImmuneShield()

    # Track callbacks
    shield_actions = []
    shield.register_callbacks(
        on_shield=lambda t, i: shield_actions.append(("shield", t, i)),
        on_throttle=lambda i: shield_actions.append(("throttle", i)),
    )

    # Normal conditions — no response
    responses = shield.scan(
        em_field_energy=10.0,
        lattice_coherence=0.8,
        perturbation_count=2,
        threat_level=0.1,
    )
    assert len(responses) == 0

    # Threat conditions
    responses = shield.scan(
        em_field_energy=150.0,
        lattice_coherence=0.2,
        perturbation_count=15,
        threat_level=0.9,
    )
    assert len(responses) > 0

    # Check that callbacks were triggered
    assert len(shield_actions) > 0

    # Check immune memory
    assert len(shield._memory) > 0

    # Test threat recognition
    pattern = np.array([150.0, 0.2, 15.0, 0.9])
    recognized = shield.recognize_threat(pattern)
    # Should recognize a similar pattern
    assert recognized is not None or len(shield._memory) == 0

    # Check status
    status = shield.get_status()
    assert "active_shields" in status
    assert "throttle_level" in status

    print("  [PASS] AutoImmuneShield")


def test_em_core():
    """Test EMCore integration."""
    from swa.em.integration import EMCore

    EMCore.reset()
    core = EMCore.get_instance(dimension=256)

    # Test topology ingestion
    topology = np.random.rand(64, 64).astype(np.float32)
    core.ingest_topology(topology)

    lattice_info = core.get_lattice_info()
    assert lattice_info["n_domains"] > 0

    # Test hardware state ingestion
    for i in range(20):
        core.ingest_hardware_state({
            "cpu_temp": 55.0 + np.random.normal(0, 0.5),
            "ram_used": 4000 + np.random.normal(0, 50),
        })

    # Test sensory input
    sensory = np.random.randn(256).astype(np.float32)
    core.ingest_sensory_input(sensory, source="test_sensor")

    # Run a cycle
    core.cycle()
    assert core._cycle_count == 1

    # Test attention
    query = np.random.randn(256).astype(np.float32)
    keys = np.random.randn(3, 256).astype(np.float32)
    values = np.random.randn(3, 256).astype(np.float32)

    attended = core.attend(query, keys, values)
    assert attended.shape == (256,)

    # Test attention overlay
    overlay = core.get_attention_overlay()
    assert overlay.shape == (64, 64)

    # Test full status
    status = core.status()
    assert "cycle_count" in status
    assert "field_energy" in status
    assert "lattice_domains" in status

    # Test toroidal info
    toroidal_info = core.get_toroidal_info()
    assert "orchestrator" in toroidal_info
    assert "subcortex" in toroidal_info

    # Run multiple cycles
    for _ in range(10):
        core.cycle()
    assert core._cycle_count == 11

    # Test singleton
    core2 = EMCore.get_instance(dimension=256)
    assert core is core2
    # Different dimension should still return same instance
    core3 = EMCore.get_instance(dimension=1024)
    assert core is core3

    EMCore.reset()
    print("  [PASS] EMCore Integration")


def test_full_pipeline():
    """Test the complete EM pipeline end-to-end."""
    from swa.em.integration import EMCore

    EMCore.reset()
    core = EMCore.get_instance(dimension=512)

    # Simulate a complete cognitive cycle:
    # 1. Subcortex provides topology
    topology = np.random.rand(64, 64).astype(np.float32)
    # Create some clear domains
    topology[16:32, 16:32] += 2.0
    topology[40:56, 40:56] += 1.5
    core.ingest_topology(topology)

    # 2. Hardware reports normal state
    for i in range(30):
        core.ingest_hardware_state({
            "cpu_temp": 55.0 + np.random.normal(0, 0.5),
            "ram_used": 4000 + np.random.normal(0, 50),
            "swap_used": 100 + np.random.normal(0, 5),
        })

    # 3. Sensory input arrives
    visual = np.sin(np.linspace(0, 2*np.pi, 512)).astype(np.float32)
    core.ingest_sensory_input(visual, source="visual")

    # 4. Run 20 EM cycles
    for _ in range(20):
        core.cycle()

    # 5. Compute attention
    q = np.random.randn(512).astype(np.float32)
    k = np.random.randn(5, 512).astype(np.float32)
    v = np.random.randn(5, 512).astype(np.float32)
    result = core.attend(q, k, v)

    # 6. Check everything is coherent
    status = core.status()
    assert status["cycle_count"] == 20
    assert status["field_energy"] > 0
    assert status["lattice_domains"] > 0

    # 7. Check EM signature
    sig = core.get_em_signature()
    assert sig.noise_floor >= 0

    # 8. Check resonances
    res = core.get_resonances()
    assert isinstance(res, dict)

    EMCore.reset()
    print("  [PASS] Full Pipeline")


if __name__ == "__main__":
    print("\n═══════════════════════════════════════════════")
    print("  AEON EM Core — Validation Suite")
    print("═══════════════════════════════════════════════\n")

    tests = [
        test_emfield,
        test_wave_engine,
        test_magnetic_lattice,
        test_toroidal,
        test_telemetry,
        test_shield,
        test_em_core,
        test_full_pipeline,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n═══════════════════════════════════════════════")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"═══════════════════════════════════════════════\n")

    sys.exit(1 if failed > 0 else 0)
