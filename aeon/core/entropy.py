from collections import deque
import math

class AdaptiveEntropyRegulator:
    """
    Regulador de Entropia Adaptativo com Termodinâmica Bayesiana.
    Substitui limiares estáticos por escore padronizado Z-score em janela deslizante (N=50),
    incorporando um piso mínimo de variância (sigma_min = 0.05) para prevenir falsa saturação.
    """
    def __init__(self, window_size: int = 50, z_threshold: float = 2.0, sigma_min: float = 0.05):
        self.window = deque(maxlen=window_size)
        self.z_threshold = z_threshold
        self.sigma_min = sigma_min
        self.eps = 1e-6

    def reset(self) -> None:
        """Limpa a janela deslizante para novo ciclo ou requisição."""
        self.window.clear()

    def update_and_evaluate(self, entropy_val: float) -> tuple[bool, float, str]:
        """
        Alimenta a entropia pontual H_t e retorna:
        (should_fission: bool, z_score: float, decision_arm: str)
        """
        if len(self.window) < 10:
            self.window.append(entropy_val)
            return False, 0.0, "FOLLOW"

        mean_h = sum(self.window) / len(self.window)
        variance = sum((x - mean_h) ** 2 for x in self.window) / (len(self.window) - 1)
        raw_std = math.sqrt(variance)
        
        # Piso de variância efetiva contra repetição sintética de tokens
        effective_std = max(raw_std, self.sigma_min) + self.eps
        
        z_score = (entropy_val - mean_h) / effective_std
        self.window.append(entropy_val)

        if z_score > self.z_threshold:
            return True, z_score, "DIVERGIR"      # Fissão ativada (270M -> 1B)
        elif z_score < -0.5:
            return False, z_score, "FOLLOW"       # Estabilidade semântica
        else:
            return False, z_score, "CONTINUAR"    # Manutenção do fluxo

    def update(self, entropy_val: float) -> tuple[bool, float, str]:
        """Alias para update_and_evaluate."""
        return self.update_and_evaluate(entropy_val)
