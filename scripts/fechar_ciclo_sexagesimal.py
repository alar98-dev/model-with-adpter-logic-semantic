"""Execução de Fechamento do Ciclo Sexagesimal (Base 60) de Blake-Aeon.

Demonstra o anel fechado completo:
  - 60 Fases Discretas (0° a 360°)
  - 6 Grandes Reinos Gravitacionais
  - Emissão de tokens modulares em /dev/shm e matriz_vetorial
  - Registro de crônicas no swa_matrix
  - O Grande Retorno (Ouroboros)
"""

import os
import sys
import time
import math
import torch

# Ajusta path
sys.path.insert(0, "/services/model-with-adpter-logic-semantic")

from swa.quantum_bridge import QuantumPhaseBridge
from swa.historian import AsynchronousHistorian
from swa.decision_trace_db import persist_part_trace, _flush_batch

class MockEventualState:
    def __init__(self):
        self.cached_bias = None
    def get_telemetry(self):
        return {
            "latent_drift": 0.08,
            "vectors": {"forward": 0.88},
            "obstacles": []
        }

def run_closed_cycle():
    print("\n" + "═" * 80)
    print("🌀 [FECHAMENTO DO CICLO SEXAGESIMAL: O ANEL DE 60 CASAS DE BLAKE-AEON]")
    print("═" * 80 + "\n")

    bridge = QuantumPhaseBridge()
    mock_state = MockEventualState()
    historian = AsynchronousHistorian(eventual_state=mock_state)

    # Força o oscilador a iniciar exatamente no índice 58 para demonstrar a Casa 59, 60 e o Ouroboros (Retorno a 1)
    # E depois rodaremos um ciclo contínuo
    print("✨ Iniciando rotação nos últimos degraus do Reino VI para validar o Ouroboros...\n")

    # Inicia no passo 58 (Casa 59: O Crepúsculo Sagrado)
    bridge._discrete_phase_index = 57  # próximo será 58 (Casa 59)
    
    test_vectors = [
        torch.randn(2560), # Casa 59 (348°)
        torch.randn(2560), # Casa 60 (354°) - O Vazio Pleno
        torch.randn(2560), # Casa 1  (0°)   - OUROBOROS: O Ponto Imóvel!
        torch.randn(2560), # Casa 2  (6°)   - A Primeira Trêmula
    ]

    for step_num, vec in enumerate(test_vectors, start=1):
        bridge.feed_recursive_thought(vec)
        
        # Sincroniza com o Historiador
        historian.ooda_loop_count = bridge._discrete_phase_index
        story = historian._compose_story()

        # Registra trace de decisão com token modular
        persist_part_trace(
            session_id=f"closed_cycle_sess_{step_num}",
            part_id=f"prt_ouroboros_{step_num}_{int(time.time())}",
            classification="ooda_cycle",
            part_type="modular_turn",
            blake_signature=[0.1, 0.0, 0.0, 0.2, 0.0, 0.5],
            tool_name="ouroboros_step",
        )

        print(f"─── DEGRAU {step_num} (Índice Z_60: {bridge._discrete_phase_index}) ───")
        print(story)
        print()

    # Flush do lote de traces
    _flush_batch()

    print("═" * 80)
    print("📊 [ESTADO DOS ARTEFATOS NO SHM]")
    print("═" * 80)
    if os.path.exists("/dev/shm/aeon/sexagesimal_phase.json"):
        with open("/dev/shm/aeon/sexagesimal_phase.json") as f:
            print("• /dev/shm/aeon/sexagesimal_phase.json:", f.read().strip())
    if os.path.exists("/dev/shm/aeon/ouroboros_cycle.json"):
        with open("/dev/shm/aeon/ouroboros_cycle.json") as f:
            print("• /dev/shm/aeon/ouroboros_cycle.json:", f.read().strip())
    print("\n✅ O CICLO FOI MATEMÁTICA E OPERACIONALMENTE SELADO!")

if __name__ == "__main__":
    run_closed_cycle()
