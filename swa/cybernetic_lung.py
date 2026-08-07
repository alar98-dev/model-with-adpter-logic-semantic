import sys

class CyberneticLung:
    """
    Órgão Cibernético de controle respiratório (Fole Termodinâmico).
    Mantém o sincronismo do Ritmo Sumério (10-60).
    """
    def __init__(self):
        self.min_breath = 10
        self.max_breath = 60
        self.entropy_multiplier = 50

    def breathe(self, current_entropy, thermodynamic_fatigue):
        """
        Calcula a capacidade pulmonar com base na entropia interna e fadiga externa.
        """
        if thermodynamic_fatigue > 0.1:
            dynamic_calc = int(self.min_breath + (current_entropy * self.entropy_multiplier))
            limite_superior = int(self.max_breath - (self.entropy_multiplier * thermodynamic_fatigue))
            breath_capacity = max(self.min_breath, min(dynamic_calc, limite_superior))
            sys.stdout.write(f"\n🔥 [CYBER-LUNG] Fadiga sistêmica detectada ({thermodynamic_fatigue:.2f}). Pulmão esmagado para {breath_capacity} tokens.\n")
        else:
            dynamic_calc = int(self.min_breath + (current_entropy * self.entropy_multiplier))
            breath_capacity = max(self.min_breath, min(dynamic_calc, self.max_breath))
            
        return breath_capacity
