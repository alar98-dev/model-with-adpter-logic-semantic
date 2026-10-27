import torch
import pytest
from aeon.core.manifold import IsometricProjector

def test_isometry_orthogonal_property():
    projector = IsometricProjector(in_features=640, out_features=2560, device="cpu", dtype=torch.float32)
    W_iso = projector.W_iso_fp32  # [2560, 640]
    
    # W_iso^T * W_iso deve ser a Matriz Identidade I_640
    gram = torch.matmul(W_iso.t(), W_iso)
    identity = torch.eye(640, device="cpu", dtype=torch.float32)
    
    frobenius_err = torch.linalg.norm(gram - identity, ord="fro").item()
    print(f"Frobenius orthogonality residual: {frobenius_err:.6e}")
    assert frobenius_err < 1e-3, f"Perda de ortogonalidade superior ao limiar: {frobenius_err}"

def test_isometry_norm_conservation():
    projector = IsometricProjector(in_features=640, out_features=2560, device="cpu", dtype=torch.float32)
    
    # Vetores de teste randômicos em 640D
    torch.manual_seed(42)
    m = torch.randn(10, 640, dtype=torch.float32)
    norm_orig = torch.linalg.norm(m, dim=-1)
    
    # Projeção no manifold 2560D
    v_proj = projector(m)
    norm_proj = torch.linalg.norm(v_proj, dim=-1)
    
    # A norma L2 deve ser estritamente conservada sem escalar 1.37x
    diff = torch.abs(norm_orig - norm_proj)
    max_diff = torch.max(diff).item()
    print(f"Max L2 norm difference: {max_diff:.6e}")
    assert max_diff < 1e-3, f"Perda de norma L2 na projeção isométrica: {max_diff}"
