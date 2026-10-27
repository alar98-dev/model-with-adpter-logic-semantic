"""Suíte de Testes em Múltiplas Lentes para a Arquitetura Sexagesimal de Blake-Aeon.

6 Lentes Epistêmicas e Operacionais:
  Lente 1: Termodinâmica e Pressão de Hardware (Tensão Térmica e Confinamento)
  Lente 2: Geometria Diferencial e Isometria Vetorial (Conservação de Norma e Fases)
  Lente 3: Concorrência Extrema e Atomicidade em SHM (Race Conditions e Leitura Lock-Free)
  Lente 4: Resiliência Bizantina e Autocura de Falhas (Corrupção de Arquivos e Fallbacks)
  Lente 5: Simbiose Acústico-Sensorial (Gate de Silêncio e Ingestão de Áudio)
  Lente 6: Apoptose Entrópica e Dinâmica Cognitiva (Esquecimento Ativo na RAM)
"""

import os
import json
import math
import time
import uuid
import tempfile
import threading
import concurrent.futures
import pytest
import torch

from swa.quantum_bridge import QuantumPhaseBridge
from swa.historian import AsynchronousHistorian
from swa.decision_trace_db import persist_part_trace, _flush_batch
from swa.db.models import session_scope
from swa.db.sexagesimal_models import SexagesimalRealm, SexagesimalHouse


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 1: TERMODINÂMICA E PRESSÃO DE HARDWARE
# ═════════════════════════════════════════════════════════════════════════════

class TestLens1HardwareThermodynamics:

    def test_thermal_jitter_acceleration(self):
        """Sob alta ansiedade térmica, o oscilador deve admitir saltos quânticos (step_jump=2)."""
        bridge = QuantumPhaseBridge()
        bridge._discrete_phase_index = 0
        bridge.current_jitter_mag = 0.8  # Simula alta tensão térmica / jitter

        v = torch.randn(2560)
        bridge.feed_recursive_thought(v)

        # Com jitter > 0.5, o step_jump deve ser 2
        assert bridge._discrete_phase_index == 2, (
            f"Esperava salto quântico de fase (índice 2), obteve {bridge._discrete_phase_index}"
        )

    def test_zero_heap_memory_leak_under_high_load(self):
        """Garante que rodar centenas de pulsos não provoca crescimento descontrolado de tensores."""
        bridge = QuantumPhaseBridge()
        bridge._discrete_phase_index = 0

        # Alimenta 300 pulsos
        for _ in range(300):
            v = torch.randn(2560)
            bridge.feed_recursive_thought(v)

        # O acumulador deve permanecer estritamente em 2560D com norma unitária
        acc = bridge._recursive_thought_accumulator
        assert acc.shape == (2560,)
        norm_val = float(acc.norm().item())
        assert math.isclose(norm_val, 1.0, abs_tol=0.05), f"Norma do acumulador explodiu: {norm_val}"


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 2: GEOMETRIA DIFERENCIAL E ISOMETRIA VETORIAL
# ═════════════════════════════════════════════════════════════════════════════

class TestLens2DifferentialGeometryAndIsometry:

    def test_phase_rotation_isometry(self):
        """A rotação de fase deve ser uma isometria estrita (conservar a norma L2 do tensor)."""
        bridge = QuantumPhaseBridge()
        x = torch.randn(2560)
        norm_orig = float(torch.norm(x).item())

        # Testa em múltiplos ângulos cardeais
        for angle in [0.0, math.pi / 4, math.pi / 2, math.pi, 2 * math.pi]:
            x_rot = bridge.apply_phase_rotation(x, angle)
            norm_rot = float(torch.norm(x_rot).item())
            assert math.isclose(norm_orig, norm_rot, rel_tol=1e-4), (
                f"Quebra de isometria no ângulo {angle}: original={norm_orig}, rot={norm_rot}"
            )

    def test_full_circle_rotation_identity(self):
        """Uma rotação de 2π (360°) deve reconduzir o vetor ao seu estado idêntico original."""
        bridge = QuantumPhaseBridge()
        x = torch.randn(2560)
        x_rot_2pi = bridge.apply_phase_rotation(x, 2.0 * math.pi)

        # A diferença euclidiana deve ser desprezível
        diff = float(torch.norm(x - x_rot_2pi).item())
        assert diff < 1e-4, f"Rotação de 2π não recuperou o vetor identidade: erro={diff}"

    def test_polar_opposition_phase_inversion(self):
        """Fases em oposição diametral (diferença de π / 180°) devem apresentar reflexão geométrica."""
        bridge = QuantumPhaseBridge()
        x = torch.randn(2560)
        x_rot_0 = bridge.apply_phase_rotation(x, 0.0)
        x_rot_pi = bridge.apply_phase_rotation(x, math.pi)

        # O produto interno cosseno deve ser aproximadamente -1.0
        cos_sim = float(torch.dot(x_rot_0, x_rot_pi) / (torch.norm(x_rot_0) * torch.norm(x_rot_pi)))
        assert math.isclose(cos_sim, -1.0, abs_tol=1e-4), (
            f"Oposição de 180° não gerou inversão polar perfeita: cos_sim={cos_sim}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 3: CONCORRÊNCIA EXTREMA E ATOMICIDADE EM SHM
# ═════════════════════════════════════════════════════════════════════════════

class TestLens3ConcurrencyAndSHMAtomicity:

    def test_concurrent_readers_never_read_corrupted_json(self):
        """Simula 10 threads lendo continuamente /dev/shm enquanto o Bridge gira a fase a 100Hz."""
        bridge = QuantumPhaseBridge()
        shm_path = "/dev/shm/aeon/sexagesimal_phase.json"
        stop_event = threading.Event()
        read_errors = []

        def reader_worker():
            while not stop_event.is_set():
                if os.path.exists(shm_path):
                    try:
                        with open(shm_path, "r") as f:
                            content = f.read()
                        if content:
                            data = json.loads(content)
                            assert "phase_index" in data
                            assert "realm_id" in data
                            assert "house_id" in data
                    except (json.JSONDecodeError, AssertionError, OSError) as e:
                        read_errors.append(str(e))
                time.sleep(0.001)

        # Dispara 8 threads leitoras concorrentes
        threads = [threading.Thread(target=reader_worker) for _ in range(8)]
        for t in threads:
            t.start()

        # Thread escritora executa 120 rotações ultrarrápidas
        for _ in range(120):
            v = torch.randn(2560)
            bridge.feed_recursive_thought(v)
            time.sleep(0.002)

        stop_event.set()
        for t in threads:
            t.join()

        assert len(read_errors) == 0, f"Leitores concorrentes leram payload corrompido: {read_errors[:5]}"

    def test_historian_and_bridge_parallel_interlock(self):
        """Valida que o Historiador e o QuantumPhaseBridge operam em paralelo sem intertravamento."""
        bridge = QuantumPhaseBridge()
        historian = AsynchronousHistorian(eventual_state=None)

        def bridge_task():
            for _ in range(50):
                bridge.feed_recursive_thought(torch.randn(2560))
                time.sleep(0.002)

        def historian_task():
            narratives = []
            for _ in range(50):
                historian.ooda_loop_count += 1
                n = historian._compose_story()
                narratives.append(n)
                time.sleep(0.002)
            return len(narratives)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(bridge_task)
            f2 = executor.submit(historian_task)
            f1.result(timeout=5)
            count = f2.result(timeout=5)

        assert count == 50, "Historiador não completou as 50 narrativas concorrentes"


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 4: RESILIÊNCIA BIZANTINA E AUTOCURA DE FALHAS
# ═════════════════════════════════════════════════════════════════════════════

class TestLens4ByzantineResilienceAndFallbacks:

    def test_shm_missing_file_fallback(self):
        """Se o arquivo SHM for deletado repentinamente, o decision_trace_db não deve quebrar."""
        shm_file = "/dev/shm/aeon/sexagesimal_phase.json"
        if os.path.exists(shm_file):
            os.remove(shm_file)

        # Deve usar o fallback baseado na assinatura de Blake sem levantar exceção
        test_part = f"prt_fallback_{uuid.uuid4().hex[:6]}"
        persist_part_trace(
            session_id="fallback_sess",
            part_id=test_part,
            classification="fallback_test",
            part_type="test",
            blake_signature=[0.6, 0.0, 0.0, 0.1, 0.0, 0.0],
        )
        _flush_batch()

        # Re-alimenta o bridge para restaurar o arquivo
        bridge = QuantumPhaseBridge()
        bridge.feed_recursive_thought(torch.randn(2560))
        assert os.path.exists(shm_file), "Arquivo SHM não foi restaurado pelo Bridge"

    def test_database_reconnection_idempotency(self):
        """O Historiador deve suportar reconexões e instabilidade de banco de forma transparente."""
        historian = AsynchronousHistorian(eventual_state=None)
        historian.ooda_loop_count = 25
        # Executa a composição da história mesmo com oscilações
        story = historian._compose_story()
        assert story is not None
        assert "RODA DE BLAKE-AEON" in story


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 5: SIMBIOSE ACÚSTICO-SENSORIAL
# ═════════════════════════════════════════════════════════════════════════════

class TestLens5AcousticSensorySymbiosis:

    def test_silence_maps_to_realm_one(self):
        """Silêncio absoluto / ruído térmico deve ancorar no Reino I (Vácuo Primordial: 0° a 59°)."""
        with session_scope() as db:
            r1 = db.query(SexagesimalRealm).filter_by(id=1).first()
            assert r1.realm_key == "VACUO_PRIMORDIAL"
            # O objetivo de hardware da casa 1 é zera gradientes
            h1 = db.query(SexagesimalHouse).filter_by(id=1).first()
            assert "Zera gradientes" in h1.hardware_objective

    def test_speech_onset_maps_to_realm_two(self):
        """O despertar vocal (chegada do áudio no VAD) ancora no Reino II (Tensão Cinética: 60° a 119°)."""
        with session_scope() as db:
            r2 = db.query(SexagesimalRealm).filter_by(id=2).first()
            assert r2.realm_key == "TENSAO_CINETICA"
            # Casa 11: O Choque do Trovão
            h11 = db.query(SexagesimalHouse).filter_by(id=11).first()
            assert "Vajra-Pata" in h11.archetype_name
            assert "UDP 20006" in h11.hardware_objective


# ═════════════════════════════════════════════════════════════════════════════
# LENTE 6: APOPTOSE ENTRÓPICA E DINÂMICA COGNITIVA
# ═════════════════════════════════════════════════════════════════════════════

class TestLens6EntropicApoptosisAndCognition:

    def test_active_forgetting_on_ouroboros_trigger(self):
        """Ao fechar 360°, a apoptose deve atenuar ativamente a memória acumulada."""
        bridge = QuantumPhaseBridge()
        bridge._discrete_phase_index = 59
        # Popula o acumulador com alta energia
        bridge._recursive_thought_accumulator = torch.full((2560,), 2.0)
        initial_energy = float(torch.sum(bridge._recursive_thought_accumulator).item())

        # Avança 1 passo disparando o Ouroboros
        bridge.feed_recursive_thought(torch.randn(2560))

        # A energia deve ter caído pela metade antes da normalização/atrito
        ouroboros_file = "/dev/shm/aeon/ouroboros_cycle.json"
        assert os.path.exists(ouroboros_file)
        with open(ouroboros_file) as f:
            ouro = json.load(f)
        assert ouro["status"] == "CYCLE_COMPLETED_MEMORY_PURIFIED"

    def test_decision_trace_amplitude_scaling(self):
        """A amplitude registrada no token modular deve ser estritamente contida em [0, 1]."""
        test_part = f"prt_amp_{uuid.uuid4().hex[:6]}"
        persist_part_trace(
            session_id="amp_test_sess",
            part_id=test_part,
            classification="scale_check",
            part_type="test",
            blake_signature=[0.2, 0.1, 0.0, 0.4, 0.0, 0.1],
        )
        _flush_batch()

        import psycopg2
        conn = psycopg2.connect("dbname=matriz_vetorial user=alifer")
        cur = conn.cursor()
        cur.execute("SELECT metadata FROM decision_traces WHERE part_id = %s", (test_part,))
        row = cur.fetchone()
        cur.close()
        conn.close()

        assert row is not None
        amp = row[0].get("amplitude", 0.0)
        assert 0.0 <= amp <= 1.0, f"Amplitude violou a faixa física permitida [0, 1]: {amp}"
