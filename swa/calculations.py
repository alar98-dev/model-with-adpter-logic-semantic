"""Vector math helpers and closed-context representations.

This module provides small utilities for normalization and a function
to produce a compact representation for a "closed" context (a list
of segment vectors). Implementations are CPU-friendly and avoid
in-place device transfers.
"""
import torch


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)


def orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    """Orthonormalize columns of matrix (shape: d x k) using QR.

    Input expected shape: (d, k) or (k, d) depending on caller; caller
    should provide consistent orientation. Returns unitary Q with same
    shape as input (where applicable).
    """
    q, r = torch.linalg.qr(matrix)
    return q


def closed_context_vector(vectors, method: str = 'dynamic', pca_components: int = 1):
    """Produce a single vector representing a closed context.

    Args:
      vectors: iterable of 1-D torch.Tensor (shape (d,) or (1,d)).
      method: aggregation method: 'mean', 'pca'/'svd', or 'dynamic' (default).
              If 'dynamic', automatically chooses 'svd' if semantic variance is
              high (tension), otherwise 'mean'.
      pca_components: number of principal components to consider (only
              used for 'pca'). If >1, returns the first component.

    Returns:
      torch.Tensor shaped (1, d) with L2 normalization, or None if empty.
    """
    vecs = list(vectors)
    if not vecs:
        return None
    # normalize inputs to (n, d)
    x = torch.stack([v.squeeze(0) if v.dim() == 2 and v.size(0) == 1 else v.view(-1) for v in vecs], dim=0)

    if method == 'dynamic':
        if x.size(0) <= 2:
            chosen_method = 'mean'
        else:
            try:
                # Compute pairwise cosine similarities to estimate cognitive tension/variance
                norm_x = l2_normalize(x)
                sim_matrix = torch.matmul(norm_x, norm_x.t())
                n = sim_matrix.size(0)
                mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
                pairwise_sims = sim_matrix[mask]
                
                variance = pairwise_sims.var().item()
                mean_sim = pairwise_sims.mean().item()
                
                if mean_sim < 0.75 or variance > 0.05:
                    chosen_method = 'svd'
                    print(f"🧠 [Sintonia Global] Variância Semântica Elevada (var={variance:.4f}, mean_sim={mean_sim:.4f}). Ativando SVD/PCA dinâmico.")
                else:
                    chosen_method = 'mean'
                    print(f"🧠 [Sintonia Global] Fluidez Semântica Estável (var={variance:.4f}, mean_sim={mean_sim:.4f}). Usando agregação simplificada (Mean).")
            except Exception:
                chosen_method = 'mean'
    else:
        chosen_method = method

    if chosen_method == 'mean':
        out = x.mean(dim=0, keepdim=True)
        return l2_normalize(out)

    if chosen_method in ('pca', 'svd'):
        # center and compute SVD; top right-singular vectors are components
        x_center = x - x.mean(dim=0, keepdim=True)
        try:
            u, s, vh = torch.linalg.svd(x_center, full_matrices=False)
        except RuntimeError:
            # fallback to mean if SVD fails on degenerate input
            out = x.mean(dim=0, keepdim=True)
            return l2_normalize(out)
        # vh shape: (d, d) when full_matrices=False it will be (d, d) or (n, d)
        # choose the first principal direction
        pc = vh[0]
        return l2_normalize(pc.unsqueeze(0))

    # fallback
    out = x.mean(dim=0, keepdim=True)
    return l2_normalize(out)


def topk_similarity_matrix(x: torch.Tensor, k: int = 5):
    """Compute cosine similarity matrix and return top-k indices per row.

    x should be (n, d). Returns (n, k) indices of most similar items (excluding self).
    """
    if x.size(0) <= 1:
        return torch.empty((x.size(0), 0), dtype=torch.long)
    sim = torch.nn.functional.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    # mask diagonal
    sim.fill_diagonal_(-9e9)
    vals, idx = torch.topk(sim, min(k, x.size(0) - 1), dim=-1)
    return idx
