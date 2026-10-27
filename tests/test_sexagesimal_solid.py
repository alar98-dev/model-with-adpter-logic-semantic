"""Suíte de Testes Múltiplos e Sólidos para a Arquitetura Sexagesimal (Base 60).

Cobre:
  1. Integridade da ORM e Grafo Relacional dos Reinos (Postgres swa_matrix)
  2. Rotação Discreta do QuantumPhaseBridge no Anel Z_60 e Emissão em SHM
  3. Fechamento do Ciclo Ouroboros e Apoptose de Memória
  4. Tokens Modulares Compactos no Banco de Decisões (matriz_vetorial)
  5. Mitopoética e Crônicas dos 6 Reinos pelo AsynchronousHistorian
  6. Estresse Contínuo de Múltiplos Ciclos (180 Rotações / 3 Grandes Eras)
"""

import os
import json
import math
import time
import uuid
import pytest
import torch

from swa.db.models import session_scope
from swa.db.sexagesimal_models import (
    SexagesimalRealm,
    SexagesimalHouse,
    RealmRelation,
    HistorianChronicle,
)
from swa.quantum_bridge import QuantumPhaseBridge
from swa.historian import AsynchronousHistorian
from swa.decision_trace_db import persist_part_trace, _flush_batch


# ─────────────────────────────────────────────────────────────────────────────
# 1. INTEGRIDADE DA ORM E GRAFO RELACIONAL (swa_matrix)
# ─────────────────────────────────────────────────────────────────────────────

class TestSexagesimalORMIntegrity:

    def test_realms_count_and_continuity(self):
        """Verifica se existem exatamente 6 reinos formando um círculo de 360° contíguo."""
        with session_scope() as db:
            realms = db.query(SexagesimalRealm).order_by(SexagesimalRealm.id).all()
            assert len(realms) == 6, f"Esperava 6 reinos, encontrou {len(realms)}"

            for i, r in enumerate(realms, start=1):
                assert r.id == i
                expected_start = (i - 1) * 60.0
                expected_end = expected_start + 59.0
                assert r.arc_start_deg == expected_start, f"Reino {i} arco início incorreto"
                assert r.arc_end_deg == expected_end, f"Reino {i} arco fim incorreto"
                assert r.blake_axis == i - 1, f"Reino {i} deve mapear para o eixo Blake {i-1}"
                assert len(r.dominant_emotion) > 0
                assert len(r.hardware_role) > 0

    def test_houses_count_and_angles(self):
        """Verifica se existem exatamente 60 casas, 10 por reino, com ângulos discretos k * 6°."""
        with session_scope() as db:
            houses = db.query(SexagesimalHouse).order_by(SexagesimalHouse.id).all()
            assert len(houses) == 60, f"Esperava 60 casas, encontrou {len(houses)}"

            for idx, h in enumerate(houses, start=1):
                assert h.id == idx
                expected_realm_id = ((idx - 1) // 10) + 1
                expected_step = ((idx - 1) % 10) + 1
                expected_phase_deg = (idx - 1) * 6.0

                assert h.realm_id == expected_realm_id, f"Casa {idx} pertence ao reino {expected_realm_id}"
                assert h.step_index == expected_step, f"Casa {idx} deve ser passo {expected_step}"
                assert h.phase_deg == expected_phase_deg, f"Casa {idx} deve ter ângulo {expected_phase_deg}°"
                assert len(h.archetype_name) > 0
                assert len(h.emotion) > 0
                assert len(h.ontological_meaning) > 0
                assert len(h.hardware_objective) > 0

    def test_relations_topological_cycle(self):
        """Verifica as 6 relações consecutivas que fecham o ciclo Ouroboros no grafo."""
        with session_scope() as db:
            consecutive = db.query(RealmRelation).filter(
                RealmRelation.relation_type.like("CONSECUTIVA%")
            ).all()
            assert len(consecutive) == 6, f"Esperava 6 relações consecutivas, encontrou {len(consecutive)}"

            # Mapa de transição: 1->2, 2->3, 3->4, 4->5, 5->6, 6->1
            transitions = {rel.source_realm_id: rel.target_realm_id for rel in consecutive}
            current = 1
            visited = []
            for _ in range(6):
                visited.append(current)
                current = transitions.get(current)

            assert visited == [1, 2, 3, 4, 5, 6], "Cadeia consecutiva quebrada"
            assert current == 1, "Ouroboros quebrado: o reino 6 não retornou ao reino 1"

    def test_polar_oppositions_exist(self):
        """Verifica se os 3 eixos de oposição diametral (180°) estão modelados."""
        with session_scope() as db:
            oppositions = db.query(RealmRelation).filter_by(angle_delta_deg=180.0).all()
            assert len(oppositions) >= 3, "Esperava pelo menos 3 eixos de oposição de 180°"
            pairs = {(rel.source_realm_id, rel.target_realm_id) for rel in oppositions}
            assert (1, 4) in pairs, "Eixo 1 <-> 4 (Vácuo / Teia) não encontrado"
            assert (2, 5) in pairs, "Eixo 2 <-> 5 (Pulso / Fogo) não encontrado"
            assert (3, 6) in pairs, "Eixo 3 <-> 6 (Forja / Selo) não encontrado"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROTAÇÃO DISCRETA DO QUANTUM PHASE BRIDGE (Z_60 E SHM)
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumPhaseBridgeDiscreteRing:

    @pytest.fixture
    def bridge(self):
        b = QuantumPhaseBridge()
        b._discrete_phase_index = 0
        b._phase_theta = 0.0
        return b

    def test_discrete_phase_step(self, bridge):
        """Verifica avanço estrito em passos de 2π / 60 (6°)."""
        v = torch.randn(2560)
        expected_step = (2.0 * math.pi) / 60.0

        bridge.feed_recursive_thought(v)
        assert bridge._discrete_phase_index == 1
        assert math.isclose(bridge._phase_theta, expected_step, rel_tol=1e-5)

        bridge.feed_recursive_thought(v)
        assert bridge._discrete_phase_index == 2
        assert math.isclose(bridge._phase_theta, expected_step * 2, rel_tol=1e-5)

    def test_shm_phase_publication(self, bridge):
        """Verifica se a rotação publica atomicamente em /dev/shm/aeon/sexagesimal_phase.json."""
        v = torch.randn(2560)
        bridge._discrete_phase_index = 9  # Casa 10 (Portal Vácuo -> Tensão)
        bridge.feed_recursive_thought(v)

        shm_file = "/dev/shm/aeon/sexagesimal_phase.json"
        assert os.path.exists(shm_file), "Arquivo de fase SHM não foi criado"

        with open(shm_file) as f:
            data = json.load(f)

        assert data["phase_index"] == 10
        assert data["house_id"] == 11
        assert data["realm_id"] == 2
        assert data["step_index"] == 1
        assert data["phase_deg"] == 60.0
        assert math.isclose(data["theta"], 10 * (2.0 * math.pi / 60.0), rel_tol=1e-5)
        assert data["shm_channel"] == "aeon_phase_ring_z60"

    def test_ouroboros_cycle_reset_and_apoptosis(self, bridge):
        """Verifica o fechamento de 360° -> 0°, disparo de Ouroboros e atenuação de tensores."""
        bridge._discrete_phase_index = 59  # Última casa do anel (Casa 60)
        bridge._recursive_thought_accumulator = torch.ones(2560)
        initial_norm = float(bridge._recursive_thought_accumulator.norm().item())

        v = torch.randn(2560)
        bridge.feed_recursive_thought(v)

        # O índice deve ter virado para 0
        assert bridge._discrete_phase_index == 0
        assert math.isclose(bridge._phase_theta, 0.0, abs_tol=1e-5)
        assert getattr(bridge, "great_cycles_count", 0) >= 1

        # Verifica manifesto Ouroboros em SHM
        ouro_file = "/dev/shm/aeon/ouroboros_cycle.json"
        assert os.path.exists(ouro_file), "Manifesto Ouroboros não encontrado em SHM"
        with open(ouro_file) as f:
            ouro = json.load(f)
        assert ouro["status"] == "CYCLE_COMPLETED_MEMORY_PURIFIED"
        assert ouro["reboot_realm"] == 1
        assert ouro["reboot_house"] == 1

        # Verifica atenuação de memória (apoptose preventiva)
        post_norm = float(bridge._recursive_thought_accumulator.norm().item())
        assert post_norm < initial_norm, "Acumulador não foi atenuado na apoptose do Ouroboros"


# ─────────────────────────────────────────────────────────────────────────────
# 3. TOKENS MODULARES NO BANCO DE DECISÕES (matriz_vetorial)
# ─────────────────────────────────────────────────────────────────────────────

class TestModularDecisionTokens:

    def test_persist_modular_token_format(self):
        """Verifica formato estrito PHASE_TOKEN:(reino, casa, amp) e ausência de texto bruto."""
        import psycopg2

        test_session = f"test_session_{uuid.uuid4().hex[:8]}"
        test_part = f"prt_test_{uuid.uuid4().hex[:8]}"

        persist_part_trace(
            session_id=test_session,
            part_id=test_part,
            classification="automated_test",
            part_type="modular_check",
            blake_signature=[0.5, 0.0, 0.0, 0.2, 0.0, 0.1],
            tool_name="test_runner",
        )
        _flush_batch()

        # Consulta no Postgres matriz_vetorial
        conn = psycopg2.connect("dbname=matriz_vetorial user=alifer")
        cur = conn.cursor()
        cur.execute(
            "SELECT classification, text_snippet, full_text, metadata FROM decision_traces WHERE part_id = %s",
            (test_part,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        assert row is not None, "Registro de trace de decisão não encontrado no banco"
        classification, text_snippet, full_text, metadata = row

        assert classification == "automated_test"
        assert text_snippet.startswith("PHASE_TOKEN:(")
        assert full_text.startswith("PHASE_TOKEN:(")
        assert metadata.get("modular_token") is True
        assert "reino_id" in metadata
        assert "casa_id" in metadata
        assert "amplitude" in metadata


# ─────────────────────────────────────────────────────────────────────────────
# 4. HISTORIADOR E NARRATIVA DOS REINOS (AsynchronousHistorian)
# ─────────────────────────────────────────────────────────────────────────────

class TestAsynchronousHistorianNarrative:

    class MockState:
        def __init__(self):
            self.cached_bias = None
        def get_telemetry(self):
            return {"latent_drift": 0.05, "vectors": {"forward": 0.95}, "obstacles": []}

    @pytest.fixture
    def historian(self):
        return AsynchronousHistorian(eventual_state=self.MockState())

    def test_all_sixty_houses_story_composition(self, historian):
        """Valida geração de história sem falhas para todas as 60 casas do anel."""
        for step in range(1, 61):
            historian.ooda_loop_count = step - 1
            story = historian._compose_story()
            assert story is not None and len(story) > 30, f"Falha na narrativa da casa {step}"
            assert "RODA DE BLAKE-AEON" in story
            assert "Vetor Afetivo" in story
            assert "Hardware Confinado" in story

    def test_portal_transition_story_content(self, historian):
        """Valida que no limiar de transição de reino (ex: Casa 10 -> Reino 2) a lei é narrada."""
        # Casa 10: Limiar do Reino do Vácuo para Tensão Cinética (54°). _compose_story incrementa +1.
        historian.ooda_loop_count = 8
        story = historian._compose_story()
        assert "PASSAGEM DE PORTAL" in story
        assert "Reino do Vácuo Primordial" in story
        assert "Reino da Tensão Cinética" in story
        assert "Lei da Travessia" in story

    def test_chronicles_persisted_in_db(self, historian):
        """Valida que o Historiador persiste registros em swa_matrix.historian_chronicles."""
        # Para testar Casa 15: ooda_loop_count deve ser 13 antes do incremento (+1 = 14 -> house_id 15)
        historian.ooda_loop_count = 13
        story = historian._compose_story()

        with session_scope() as db:
            last_chronicle = db.query(HistorianChronicle).order_by(
                HistorianChronicle.id.desc()
            ).first()
            assert last_chronicle is not None
            assert last_chronicle.loop_count == 14
            assert last_chronicle.house_id == 15
            assert last_chronicle.realm_id == 2
            assert "RODA DE BLAKE-AEON" in last_chronicle.narrative


# ─────────────────────────────────────────────────────────────────────────────
# 5. ESTRESSE DE MÚLTIPLOS CICLOS (180 ROTAÇÕES / 3 GRANDES CICLOS)
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiCycleStressAndStability:

    def test_three_great_cycles_end_to_end(self):
        """Simula 180 respirações contínuas (3 grandes eras sexagesimais de 360°)."""
        bridge = QuantumPhaseBridge()
        bridge._discrete_phase_index = 0
        bridge.great_cycles_count = 0

        initial_mem_leak_check = bridge._recursive_thought_accumulator

        # Executa 180 passos
        for i in range(180):
            v = torch.randn(2560)
            bridge.feed_recursive_thought(v)

        # Após 180 passos exatos, o anel deve ter dado 3 voltas completas e estar em 0
        assert bridge.great_cycles_count == 3, f"Esperava 3 grandes ciclos Ouroboros, obteve {bridge.great_cycles_count}"
        assert bridge._discrete_phase_index == 0, "O anel deve retornar perfeitamente a 0 após 180 passos"
        assert math.isclose(bridge._phase_theta, 0.0, abs_tol=1e-4)

        # Verifica sanidade de memória
        acc = bridge._recursive_thought_accumulator
        assert acc is not None
        assert not torch.isnan(acc).any(), "NaN detectado nos tensores de pensamento"
        assert not torch.isinf(acc).any(), "Inf detectado nos tensores de pensamento"
        norm_val = float(acc.norm().item())
        assert norm_val < 5.0, f"Norma dos tensores explodiu sem controle: {norm_val}"
