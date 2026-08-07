import pytest
import time

def test_system_homeostasis(agent):
    """Verifica se o sistema está em paz algorítmica após a reinicialização."""
    state = agent.probe_state()
    assert "status" in state or "fase" in state
    print("✅ [TESTE] Homeostase verificada.")

def test_hardware_bridge_connectivity(agent):
    """Verifica se o micro-agente consegue disparar um pulso de hardware."""
    response = agent.send_pulse(intensity=0.9)
    assert response.status_code == 200
    print("✅ [TESTE] Ponte de hardware responsiva.")
