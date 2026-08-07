import torch
import numpy as np

class BlakeTensorRestShaper:
    def __init__(self, generator):
        self.gen = generator
        # Dynamic alignment to avoid structural mismatches (e.g. 1152d vs 2560d)
        self.d_model = getattr(generator, "d_model", 2560) if generator is not None else 2560
        self.amplitude = 1.5  # Controle dinâmico de ganho adaptativo (Amplificado para estabilizar contexto)
        
        # CORREÇÃO 1: Inicializa a matriz e força explicitamente para BFloat16 para casar com o Gemma-3
        # Usa o dtype do próprio gerador se disponível
        target_dtype = getattr(generator, "dtype", torch.bfloat16) if generator is not None else torch.bfloat16
        self.W_steer = torch.randn(3, self.d_model, dtype=target_dtype) * 0.01
        print(f"⚡ [REST TENSOR] Matriz de Projeção Linear W_steer acoplada em {target_dtype} com dimensão {self.d_model}D.")

    def adjust_gauge_amplitude(self, factor: float):
        """Ajusta o fator de amplitude dinamicamente via thread de monitoramento."""
        self.amplitude = factor
        print(f"⚡ [REST TENSOR] Amplitude do gauge ajustada para {self.amplitude}")

    def forge_steer_vector(self, bias: float, cpu_entropy: float, mem_inertia: float) -> torch.Tensor:
        """
        Transmuta as telemetrias diretamente em um tensor de força de d_model canais.
        Garante alinhamento estrito de tipo (dtype) para evitar quebras no Kernel.
        """
        # Usa o dtype do gerador para evitar qualquer mismatch de tipos
        target_dtype = getattr(self.gen, "dtype", torch.bfloat16) if self.gen is not None else torch.bfloat16
        raw_hardware_state = torch.tensor([bias, cpu_entropy, mem_inertia], dtype=target_dtype)
        
        with torch.no_grad():
            # Multiplicação segura com dtypes consistentes
            projected_field = torch.matmul(raw_hardware_state, self.W_steer)
            
            # Normalização de Gauge para manter a inércia da Layer 12
            steer_tensor = torch.nn.functional.normalize(projected_field, p=2, dim=0)
            
            # Aplica o multiplicador de amplitude adaptativo controlado pela thread REST
            steer_tensor = steer_tensor * self.amplitude
            
        # Retorna no shape correto (1, 1, d_model) alinhado com o fluxo residual
        return steer_tensor.unsqueeze(0).unsqueeze(0)
