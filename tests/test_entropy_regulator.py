import pytest
from aeon.core.entropy import AdaptiveEntropyRegulator

def test_constant_entropy_stability():
    regulator = AdaptiveEntropyRegulator(window_size=50, z_threshold=2.0, sigma_min=0.05)
    
    # 20 iterações com entropia idêntica (ex: blocos de código/chaves repetidas)
    for _ in range(20):
        should_fission, z_score, arm = regulator.update_and_evaluate(5.0)
        assert not should_fission
        assert arm in ["FOLLOW", "CONTINUAR"]
        assert abs(z_score) < 1.0, f"Z-score instável sob entropia constante: {z_score}"

def test_anomaly_fission_trigger():
    regulator = AdaptiveEntropyRegulator(window_size=50, z_threshold=2.0, sigma_min=0.05)
    
    # Enche a janela com nível de entropia baseline ~ 4.0
    for _ in range(30):
        regulator.update_and_evaluate(4.0)
        
    # Injeta pico de entropia (anomalia semântica)
    should_fission, z_score, arm = regulator.update_and_evaluate(12.0)
    assert should_fission, "Deveria ter ativado fissão para pico de entropia!"
    assert arm == "DIVERGIR"
    assert z_score > 2.0, f"Z-score esperado > 2.0, obtido: {z_score}"
