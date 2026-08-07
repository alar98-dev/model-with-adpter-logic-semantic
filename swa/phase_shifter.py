import torch
import numpy as np

class PhaseShifter:
    """
    Controlador de Interferência de Hardware para o Domínio SWA.
    Gerencia a transição entre o Estado Real e o Estado Simulado no NVMe.
    """
    def __init__(self, hardware_manager, inertia_lambda=0.95):
        self.hw = hardware_manager
        self._lambda = inertia_lambda  # Fator de Inércia Cognitiva
        self._last_phase_vector = None

    @property
    def entropy(self) -> float:
        """
        Calcula a Entropia de Bit (Hb) entre a Realidade e a Simulação.
        Lê diretamente os offsets 0x0000 e 0x4000 do /dev/sde.
        """
        real = self.hw.read_state("reality")
        sim  = self.hw.read_state("prediction")

        if real is None or sim is None:
            return 0.0

        # Hb = -sum(p * log2(p)) -> Simplificado para divergência vetorial
        cos_sim = torch.nn.functional.cosine_similarity(real.flatten(), sim.flatten(), dim=0)
        # Quanto menor a similaridade, maior a entropia (incerteza)
        return float(1.0 - cos_sim.item())

    @property
    def inertia_weight(self) -> float:
        """
        Retorna o peso de Inércia baseado na estabilidade do fluxo de bits.
        Quanto maior a entropia, mais o sistema se 'tranca' no estado anterior (Inércia).
        """
        # Se a entropia sobe, a inércia aumenta para filtrar o ruído
        return min(0.99, self._lambda + (self.entropy * 0.04))

    @property
    def organic_bias(self) -> torch.Tensor:
        """
        O 'Ajustador de Fase' propriamente dito.
        Realiza a mistura vetorial (Interferência) para as 8 cabeças de atenção.
        Agora usa múltiplos estados simultâneos: Realidade + Predição + Mapa
        """
        # Use multi-state interference for richer bias computation
        interference_vector = self.hw.get_multi_state_interference()

        # Fallback se a interferência falhar
        if interference_vector is None or torch.isnan(interference_vector).any():
            interference_vector = torch.zeros(2560)

        # Aplicação do Cálculo de Inércia (Momento) sobre o vetor de interferência
        if self._last_phase_vector is None:
            self._last_phase_vector = interference_vector

        # A Inércia Cognitiva: V_final = (L * V_old) + ((1-L) * V_new)
        # O mapa de navegação adiciona estabilidade extra
        inertia_with_map = self.inertia_weight * 1.2  # Mapa aumenta inércia
        v_final = (inertia_with_map * self._last_phase_vector) + \
                  ((1 - inertia_with_map) * interference_vector)

        self._last_phase_vector = v_final

        # Redimensiona para o Barramento de 8 Heads [1, 8, 1, 320]
        return v_final.view(1, 8, 1, 320)

    def inject_to_gemma(self):
        """Prepara o bias para o forward pass do modelo."""
        return self.organic_bias