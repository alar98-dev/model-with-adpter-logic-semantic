import torch
import torch.nn as nn

class IsometricProjector(nn.Module):
    """
    Projeta metamatéria 640D -> 2560D garantindo conservação isométrica de norma e fase.
    Resolve o problema de Procrustes ortogonal via SVD em FP32 e exporta no dtype final.
    """
    def __init__(self, in_features: int = 640, out_features: int = 2560, device: str = "cpu", dtype=torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Inicialização com matriz ortogonal exata em FP32 para evitar perda de precisão
        w_init = torch.empty(out_features, in_features, device="cpu", dtype=torch.float32)
        nn.init.orthogonal_(w_init)
        
        # Decomposição SVD fina em FP32: W_iso^T * W_iso = I_640
        U, _, Vt = torch.linalg.svd(w_init, full_matrices=False)
        w_iso_fp32 = torch.matmul(U, Vt)  # [2560, 640]
        
        # Registrar buffer no dtype final
        self.register_buffer("W_iso", w_iso_fp32.to(device=device, dtype=dtype))
        self.register_buffer("W_iso_fp32", w_iso_fp32)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """
        m: Tensor [batch_size, 640] ou [640]
        Retorna tensor projetado em 2560D preservando a norma L2 original de m.
        """
        target_dtype = m.dtype
        target_device = m.device
        
        m_calc = m.to(device=self.device, dtype=self.W_iso.dtype)
        
        norm_m = torch.linalg.norm(m_calc, dim=-1, keepdim=True) + 1e-8
        m_unit = m_calc / norm_m
        
        # Projeção no espaço do manifold 2560D
        v_projected = torch.matmul(m_unit, self.W_iso.t())
        
        # Re-ancora o comprimento vetorial original sem escalonamento arbitrário
        res = v_projected * norm_m
        return res.to(device=target_device, dtype=target_dtype)
