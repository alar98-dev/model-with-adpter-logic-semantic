import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import ctypes
import os
import psutil

def set_phase_affinity(cpu_ids, phase: str = "prep"):
    """Bind process to specific NUMA node CPUs. 
    Node A (Even: [0,2,4,6,8,10,12,14]) -> Vectorizer + SWA (prep phase). 
    Node B (Odd: [1,3,5,7,9,11,13,15]) -> GemmaGenerator (infer phase). 
    Explicit separation prevents Cache Thrashing in Xeon Dual-Socket systems."""

    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(cpu_ids)
        os.nice(-10)
        print(f"Phase {phase} Affinity set to CPUs {cpu_ids} (niceness -10)")
    except Exception as e:
        print(f"Affinity fail ({phase}): {e}")

def lock_memory():
    """Lock process memory (mlockall) to prevent swap on Xeon."""

    MCL_CURRENT = 1
    MCL_FUTURE = 2
    try:
        libc = ctypes.CDLL("libc.so.6")
        ret = libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        if ret != 0:
            err = ctypes.get_errno()
            print(f"mlockall fail: errno {err}")
            return False, f"errno {err}"
        print("mlockall success: RAM locked.")
        return True, "success"
    except Exception as e:
        print(f"mlock fail: {e}")
        return False, str(e)

def optimize_for_xeon(node: int = 0):
    """Dynamic NUMA socket / CPU topology optimization.
    Discovers cores dynamically and sets affinity based on execution phase.
    """
    try:
        import psutil
        total_cpus = psutil.cpu_count() or 4
        if total_cpus >= 16:
            # Dual-Socket Xeon: divide evens and odds to prevent cache thrashing
            node_a = [i for i in range(total_cpus) if i % 2 == 0]
            node_b = [i for i in range(total_cpus) if i % 2 != 0]
        else:
            # Client CPU: split first half and second half
            half = max(2, total_cpus // 2)
            node_a = list(range(0, half))
            node_b = list(range(half, total_cpus))
            
        target_cpus = node_a if node == 0 else node_b
        set_phase_affinity(target_cpus, f"xeon-dynamic-node-{node}")
    except Exception as e:
        print(f"Dynamic Xeon affinity fail: {e}")

def optimize_for_dual_cpu(node_a_cpus = None, node_b_cpus = None):
    """NUMA Dual CPU Dynamic flow alignment. Falls back to dynamic Xeon optimizer."""
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    torch.set_num_threads(8)
    optimize_for_xeon(0)

def optimize_for_single_socket():
    """Mobile / single-socket: thread pool aligned to physical cores dynamically."""
    phys = psutil.cpu_count(logical=False) or max(1, (psutil.cpu_count() or 4) // 2)
    threads = max(1, int(os.environ.get("SWA_INTRAOP_THREADS", str(min(phys, 12)))))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    print(f"SWA single-socket: dynamic threads={threads}")

def optimize_for_intel_120u_pcores():
    """Intel Core 5 120U (hybrid 2 P-Cores + 8 E-Cores).
    Dynamically binds process to high-performance cores to reduce latency.
    """
    try:
        total_cpus = psutil.cpu_count() or 4
        # Standard Intel hybrid: target physical performance cores (usually first 4 threads)
        target_cores = list(range(min(4, max(1, total_cpus // 2))))
        p = psutil.Process(os.getpid())
        p.cpu_affinity(target_cores)
        print(f"Intel dynamic affinity forced to P-cores: {target_cores}")
    except Exception as e:
        print(f"Dynamic hybrid affinity fail: {e}")

    threads = max(1, int(os.environ.get("SWA_INTRAOP_THREADS", "4")))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Intel XPU available for GEMMA_DEVICE=xpu")

    # Garantir que não reste afinidade/numa anterior por configuração externa.



class SemanticWarpAdapter(nn.Module):
    """Semantic Warp Adapter (SWA): d->k GELU k->d with ortho loss, prefix_mode for Gemma v_prefix injection (k->3072). Xeon float32 CPU stable."""

    def __init__(self, d: int, k: int = 128, ortho_lambda: float = 1e-3, use_gating: bool = False, low_rank: int | None = None, prefix_mode: bool = False, d_model: int = 3072, node: int = 0):
        super().__init__()
        lock_memory()
        # intel-120u-xpu: priorize P-cores para reduzir latência do decode.
        optimize_for_intel_120u_pcores()
        

        
        self.d = d
        self.k = k
        self.ortho_lambda = ortho_lambda
        self.use_gating = use_gating
        self.low_rank = low_rank
        self.prefix_mode = prefix_mode
        self.d_model = d_model

        self.prefix_projection = None
        self.bias_proj = None
        if prefix_mode:
            self.prefix_projection = nn.Linear(k, d_model, )
            self.bias_proj = nn.Linear(k, d_model, )

        self.W_a = nn.Linear(d, k, )
        self

        if low_rank and low_rank < k:
            self.W_b_1 = nn.Linear(k, low_rank, bias=False, )
            self.W_b_2 = nn.Linear(low_rank, d, )
            self._use_factorized_b = True
        else:
            self.W_b = nn.Linear(k, d, )
            self._use_factorized_b = False

        if use_gating:
            self.gate = nn.Parameter(torch.ones(k, ))

        self._norm_eps = 1e-6

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # [SWA DTYPE ALIGNMENT] Force input to match adapter weight dtype (Float32)
        v = v.to()
        orig_shape = v.shape
        z = self.latent_gelu(v)
        if self._use_factorized_b:
            x = self.W_b_2(self.W_b_1(z))
        else:
            x = self.W_b(z)
        x = x.view(*orig_shape[:-1], self.d)
        norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(self._norm_eps)
        return x / norm

    def latent_gelu(self, v: torch.Tensor) -> torch.Tensor:
        """GELU(W_a @ v) in shape (batch, k); used by forward and prefix heads."""
        # [SWA DTYPE ALIGNMENT] Force input to match adapter weight dtype (Float32)
        v_flat = v.to().view(-1, self.d)
        z = F.gelu(self.W_a(v_flat))
        if self.use_gating:
            z = z * self.gate
        return z

    def prefix_and_bias(self, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project macro context into Gemma d_model space (prefix + optional bias). Expects v (batch, d)."""
        if self.prefix_projection is None:
            raise ValueError("prefix_mode=False (prefix_projection missing)")
        # [SWA DTYPE ALIGNMENT] Force v to match adapter dtype before latent_gelu
        v = v.to()
        z = self.latent_gelu(v)
        v_prefix = self.prefix_projection(z).squeeze(0)
        if self.bias_proj is not None:
            v_bias = self.bias_proj(z).squeeze(0)
        else:
            v_bias = torch.zeros_like(v_prefix)
        
        # [SWA DTYPE ALIGNMENT] Return in Float32, cast by caller if needed
        return v_prefix, v_bias

    def prefix_project(self, v: torch.Tensor) -> torch.Tensor:
        if self.prefix_projection is None:
            raise ValueError("prefix_mode=False")
        pre, _bias = self.prefix_and_bias(v)
        return pre

    def ortho_loss(self) -> torch.Tensor:
        if self._use_factorized_b:
            Wb = self.W_b_2.weight @ self.W_b_1.weight
        else:
            Wb = self.W_b.weight
        G = Wb.t() @ Wb
        I = torch.eye(G.size(0), device=G.device)
        return self.ortho_lambda * ((G - I) ** 2).sum()

def triplet_loss(a, p, n, margin=0.2, distance="cos"):
    if distance == "cos":
        d = lambda u,v: 1 - F.cosine_similarity(u, v, dim=-1)
    else:
        d = lambda u,v: (u - v).pow(2).sum(dim=-1)
    return F.relu(d(a, p) - d(a, n) + margin).mean()

if __name__ == "__main__":
    m = SemanticWarpAdapter(d=768, k=64, prefix_mode=True, d_model=3072)
    x = torch.randn(4, 768)
    y = m(x)
    print("SWA test OK:", y.shape, "norms:", y.norm(dim=-1))

