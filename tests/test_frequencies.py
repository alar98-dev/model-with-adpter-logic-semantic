import pytest
import torch
import math

# ==============================================================================
# FÍSICA DO ESPAÇO LATENTE: Tabela de Frequências (Channels)
# ==============================================================================
# CANAL ALFA (Tédio / Inércia): Norma < 5.0 (Respostas curtas: ~60 tokens)
# CANAL BETA (Foco Operacional): Norma 5.0 a 20.0 (Respostas normais: 100~200 tokens)
# CANAL GAMA (Epifania / Complexidade): Norma 20.0 a 100.0 (Respostas longas: 300~800 tokens)
# CANAL ÔMEGA (Colapso / Superposição): Norma > 100.0 ou Variância Absurda -> Exige Warp Drive.
# ==============================================================================

def calcular_duracao_onda(semantic_energy: float) -> int:
    """Função simulando a equação injetada no aeonc.py"""
    dynamic_tokens = int(50 + (semantic_energy * 2.5))
    return min(800, max(50, dynamic_tokens))

def identificar_canal_cognitivo(tensor_3d: torch.Tensor) -> str:
    """Dado o tensor, classifica em qual canal o pensamento está surfando."""
    norm = torch.norm(tensor_3d).item()
    std = torch.std(tensor_3d).item()
    
    if std > 20.0 or norm > 100.0:
        return "OMEGA_COLLAPSE"
    elif norm < 5.0:
        return "ALFA_INERTIA"
    elif 5.0 <= norm < 20.0:
        return "BETA_FOCUS"
    else:
        return "GAMA_EPIPHANY"

# ------------------------------------------------------------------------------
# TESTES DO MOTOR TERMÓDINAMICO E ONDAS DE PENSAMENTO
# ------------------------------------------------------------------------------

def test_canal_alfa_tedio():
    """Testa uma onda de pensamento curta (Tédio/Respostas Simples)."""
    # Tensor de baixíssima energia
    t = torch.randn(1, 1, 1152) * 0.01 
    energia = torch.norm(t).item()
    tokens = calcular_duracao_onda(energia)
    
    assert identificar_canal_cognitivo(t) == "ALFA_INERTIA", "Deveria estar no Canal ALFA."
    assert tokens == 50, "A onda deveria bater no limite mínimo (50 tokens) para poupar CPU."

def test_canal_beta_foco():
    """Testa uma onda de pensamento operacional padrão."""
    # Tensor de energia média
    t = (torch.randn(1, 1, 1152) * 0.1) + 0.3
    energia = torch.norm(t).item()
    tokens = calcular_duracao_onda(energia)
    
    assert identificar_canal_cognitivo(t) == "BETA_FOCUS", "Deveria estar no Canal BETA."
    assert 50 < tokens < 200, "Onda beta deveria durar entre 50 e 200 tokens."

def test_canal_gama_epifania():
    """Testa uma onda gerada por um prompt hipercomplexo."""
    t = (torch.randn(1, 1, 1152) * 0.5) + 1.5
    energia = torch.norm(t).item()
    tokens = calcular_duracao_onda(energia)
    
    assert identificar_canal_cognitivo(t) == "GAMA_EPIPHANY", "Deveria estar no Canal GAMA."
    assert tokens >= 150, "Onda gama deveria exigir alta duração."

def test_canal_omega_colapso():
    """Testa a superposição lógica (erro do RoPE ou Warp Drive)."""
    # Tensor com valores explosivos
    t = torch.randn(1, 1, 1152) * 50.0 
    energia = torch.norm(t).item()
    tokens = calcular_duracao_onda(energia)
    
    assert identificar_canal_cognitivo(t) == "OMEGA_COLLAPSE", "Falhou em detectar Colapso Termodinâmico."
    assert tokens == 800, "O Colapso força o limite estrutural máximo do Córtex."

def test_integridade_matematica_onda():
    """Garante que a equação não extrapola os limites seguros do buffer."""
    energia_infinita = 999999.0
    energia_nula = 0.0
    
    assert calcular_duracao_onda(energia_infinita) == 800, "Falha de teto (Overflow)."
    assert calcular_duracao_onda(energia_nula) == 50, "Falha de chão (Underflow)."

if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
