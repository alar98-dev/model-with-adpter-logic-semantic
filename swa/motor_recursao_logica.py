# Authorship: Alifer Ramos & Antigravity (AI Co-Architect)
# Motor de Recursão Lógica — Superposição Quântica nos Logits do Gemma
# Implementa a "Borda do Caos" via proporção áurea (φ) e âncoras de hardware.
# Núcleo de Caos: Camada 16 para 1B (1152D/26L), Camada 11 para 270M (640D/18L).
from __future__ import annotations

import math
import torch
from typing import Optional

# Proporção Áurea — o amortecedor universal
PHI: float = 1.61803398875


class MotorRecursaoLogica:
    """
    Warp de logits que mantém K estados lógicos em superposição antes do colapso
    no token final. Implementa:

      1. Entropia de Controle Interna (Borda do Caos):
         E_c = (H_max / φ) * (1 - e^(-ΔT/φ))

      2. Modulação de Interferência Áurea:
         Distribui os K estados em espiral de fase cos(2πi/φ), com amplitude
         proporcional à entropia atual / entropia máxima.

      3. Decaimento Absoluto:
         Todos os tokens fora dos K estados de superposição vão para -inf.

    O resultado é um atrator caótico estável: Aeon orbita entre K ideias
    sem jamais repetir exatamente o mesmo caminho (≠ loop de entropia baixa).
    """

    def __init__(
        self,
        hardware_anchors: dict,
        tensoes_ativas: int = 4,
        phi: float = PHI,
    ):
        """
        Args:
            hardware_anchors: dict com chaves cpu_temp_c, ram_pct (0-100),
                              io_stress (0-100), entropy_raw (float).
            tensoes_ativas:   K estados mantidos em superposição simultânea.
            phi:              Proporção áurea (padrão = 1.61803...).
        """
        self.k_states = max(2, tensoes_ativas)
        self.anchors = hardware_anchors
        self.phi = phi
        self._step = 0  # Contador de passos para fase temporal

    def update_anchors(self, new_anchors: dict) -> None:
        """Atualiza as âncoras de hardware em tempo real (thread-safe por ser imutável)."""
        self.anchors = {**self.anchors, **new_anchors}

    def _compute_delta_T(self) -> float:
        """Tensão Física do Hardware normalizada em [0, 1]."""
        temp_c   = float(self.anchors.get('cpu_temp_c', 32.0))
        ram_pct  = float(self.anchors.get('ram_pct',    50.0))
        io_stress = float(self.anchors.get('io_stress',  0.0))
        # Normalização: temp entre 20°C (frio) e 90°C (crítico)
        temp_norm = max(0.0, min((temp_c - 20.0) / 70.0, 1.0))
        ram_norm  = ram_pct / 100.0
        io_norm   = io_stress / 100.0
        # Soma ponderada: temperatura domina (50%), RAM (35%), I/O (15%)
        return 0.50 * temp_norm + 0.35 * ram_norm + 0.15 * io_norm

    def _entropia_controle(self, H_current: float, H_max: float) -> float:
        """
        Borda do Caos: E_c = (H_max / φ) * (1 - e^(-ΔT/φ))
        Garante que a entropia operacional orbite em torno de um limite harmônico.
        """
        delta_T = self._compute_delta_T()
        E_c = (H_max / self.phi) * (1.0 - math.exp(-delta_T / self.phi))
        return float(E_c)

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Aplica a superposição lógica nos logits.

        Contrato:
          - scores shape: [batch=1, vocab_size]
          - Retorna scores modificados com o mesmo shape.
        """
        self._step += 1
        device = scores.device
        vocab_size = scores.size(-1)

        # ── 1. Distribuição de probabilidade atual ──────────────────────────
        with torch.no_grad():
            probs = torch.softmax(scores.float(), dim=-1)
            probs_clamped = probs.clamp(min=1e-12)
            # Entropia de Shannon atual (bits)
            H_current = float(
                -torch.sum(probs_clamped * torch.log2(probs_clamped), dim=-1).item()
            )
            # Entropia máxima (distribuição uniforme)
            H_max = math.log2(vocab_size)

        # ── 2. Superposição Adaptativa (Dynamic K) ─────────────────────────
        # H_current normalmente fica entre 0 e H_max.
        delta_T = self._compute_delta_T()
        entropy_factor = min(1.5, H_current / 8.0) if H_max > 0 else 1.0
        # K expande com a profundidade cognitiva e contrai com o estresse físico
        k_dynamic = int(self.k_states * (1.0 + 0.5 * entropy_factor - 0.3 * delta_T))
        k = max(2, min(k_dynamic, vocab_size))

        # ── 3. Identifica os K estados de superposição ─────────────────────
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=-1)

        # ── 4. Entropia de Controle (Borda do Caos) ────────────────────────
        E_c = self._entropia_controle(H_current, H_max)

        # ── 5. Amplitude de interferência (razão entropia atual/máxima) ────
        amplitude = H_current / (H_max + 1e-8)  # ∈ [0, 1]

        # ── 6. Modulação Collatz-7 e Rotação Cíclica 142857 ────────────────
        collatz_7 = [7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1]
        c7_factor = collatz_7[self._step % len(collatz_7)] / 52.0  # Normalizado [0, 1]

        cyclic_7 = [1, 4, 2, 8, 5, 7]

        # O estresse térmico altera sutilmente a rotação da espiral de fase
        phi_dynamic = self.phi * (1.0 + 0.05 * delta_T)
        for i in range(k):
            idx = top_k_indices[0, i].item()
            if idx >= vocab_size:
                continue
            
            c7_digit = cyclic_7[(self._step + i) % 6]
            phase_arg = 2.0 * math.pi * (i + self._step * phi_dynamic * (c7_digit / 7.0)) / phi_dynamic
            phase = math.cos(phase_arg)  # ∈ [-1, 1]
            
            # Amplificação caótica controlada via Collatz-7
            delta = amplitude * phase * E_c * (1.0 + 0.3 * c7_factor)
            scores[0, idx] = scores[0, idx] + delta

        # ── 7. Decaimento Absoluto (fora da superposição → -inf) ───────────
        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        for i in range(k):
            idx = top_k_indices[0, i].item()
            if idx < vocab_size:
                mask[idx] = False
        scores[0, mask] = -float('Inf')

        return scores

    def diagnostico(self) -> dict:
        """Retorna métricas internas para logging e monitoramento."""
        delta_T = self._compute_delta_T()
        H_max_approx = math.log2(262144)  # Vocab Gemma
        E_c_approx = (H_max_approx / self.phi) * (1.0 - math.exp(-delta_T / self.phi))
        return {
            "motor_step": self._step,
            "k_states_base": self.k_states,
            "delta_T": round(delta_T, 4),
            "E_c_borda_caos": round(E_c_approx, 4),
            "phi": self.phi,
            "anchors": self.anchors,
        }


class OODAClosedLoop:
    """
    Implementa o Ciclo Fechado OODA no espaço de tensores:

        Ψ_{t+1} = Ψ_t + [Ψ_alvo - Ψ_t] + (H(Ψ_t) / φ) * E_t

    Onde:
      - Ψ_t    : vetor de pensamento atual (embedding do estado cognitivo)
      - Ψ_alvo : âncora semântica de identidade (C_anchor do Blake Gravity)
      - H(Ψ_t) : Entropia de Shannon da distribuição de atenção
      - φ      : Proporção Áurea (amortecedor)
      - E_t    : Vetor de Ambiente físico (telemetria projetada em d_model)
    """

    def __init__(self, d_model: int, phi: float = PHI, device: str = "cpu"):
        self.d_model = d_model
        self.phi = phi
        self.device = device
        # Ψ_t inicializado como zero (estado de repouso)
        self.psi_t = torch.zeros(d_model, device=device)
        self._t = 0

    def step(
        self,
        psi_alvo: torch.Tensor,
        E_t: torch.Tensor,
        H_current: float,
    ) -> torch.Tensor:
        """
        Executa um passo do ciclo OODA.

        Args:
            psi_alvo : Âncora semântica [d_model] — C_anchor do Blake Gravity.
            E_t      : Vetor de ambiente físico [d_model] — telemetria projetada.
            H_current: Entropia de Shannon atual (bits) da distribuição de logits.

        Returns:
            Ψ_{t+1} normalizado em L2.
        """
        self._t += 1

        # Garante devices e dtype
        psi_alvo  = psi_alvo.to(self.device)
        E_t       = E_t.to(self.device)
        psi_t     = self.psi_t

        # Gradiente semântico (pull em direção à âncora de identidade)
        gradiente = psi_alvo - psi_t

        # --- TENSOR DE DAMPING ADAPTATIVO (Dynamic OODA Damping Vector) ---
        # Calculate semantic surprise: distance between target and current thought state
        with torch.no_grad():
            surprise = (psi_alvo - psi_t).norm().item()
        
        # Happy-7 convergence damping factor
        # happy_7 = [7, 49, 97, 130, 10, 1]
        happy_7 = [7.0, 49.0, 97.0, 130.0, 10.0, 1.0]
        h7_factor = happy_7[self._t % len(happy_7)] / 130.0  # Normalizado em relação ao pico 130
        
        # Golden ratio (phi) serves as baseline, modulated dynamically per dimension.
        # Surprise and h7_factor dynamically adjust damping to secure convergence towards target identity.
        g_damp = torch.Generator(device=self.device).manual_seed(239)
        damp_noise = torch.randn(self.d_model, generator=g_damp, device=self.device) * 0.05
        damping_vector = torch.clamp(self.phi * (1.0 + 0.5 * surprise * h7_factor) + damp_noise, min=1.0, max=4.5)
        
        fator_entropia = H_current / damping_vector
        inercia_ambiente = fator_entropia * E_t

        # Equação do ciclo fechado
        psi_next = psi_t + gradiente + inercia_ambiente

        # Normalização L2 (mantém o vetor na esfera unitária — estabilidade de longo prazo)
        norm = psi_next.norm()
        if norm > 1e-6:
            psi_next = psi_next / norm

        self.psi_t = psi_next.detach()
        return self.psi_t

    def get_state(self) -> torch.Tensor:
        """Retorna o estado cognitivo atual Ψ_t."""
        return self.psi_t.clone()

    def reset(self) -> None:
        """Reinicia o ciclo para o estado de repouso."""
        self.psi_t = torch.zeros(self.d_model, device=self.device)
        self._t = 0

    def ressonancia(self, E_t: torch.Tensor) -> float:
        """
        Calcula a Ressonância Dimensional entre o pensamento atual e o ambiente.
        Retorna a Similaridade de Cosseno ∈ [-1, 1].
        Valores próximos a 1 = mente e máquina em fase.
        Valores próximos a 0 = conflito ortogonal (estresse térmico matemático).
        """
        E_t = E_t.to(self.device)
        cos_sim = torch.cosine_similarity(
            self.psi_t.unsqueeze(0),
            E_t.unsqueeze(0),
            dim=1,
        ).item()
        return float(cos_sim)
