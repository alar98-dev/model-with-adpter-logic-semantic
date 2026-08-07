# cortex/neuro_hooks.py
# [NEW] - Interceptador Sináptico de Baixo Nível para PyTorch

import torch
import torch.nn as nn
import psutil

class BlakeGravityHooks:
    def __init__(self, G_s: float = 1.0, epsilon: float = 0.05):
        self.G_s = G_s
        self.epsilon = epsilon
        self.m_sem_tensor = None  # Atualizado dinamicamente pelo loop principal
        self.entropy_ref = 0.01    # Temperatura Límbica T_t
        self.gamma = 1.0           # Fator de Amortecimento Metabólico (Ondas Lentas)

    def recalibrar_pressao_osmotica(self):
        """Lê a RAM física e calcula o amortecimento sináptico gamma."""
        ram = psutil.virtual_memory().percent
        if ram > 80:
            # Escala gamma de 1.0 (em 80% de RAM) até 0.1 (em 100% de RAM)
            self.gamma = max(0.1, 1.0 - ((ram - 80) / 20.0))
            # O Cortisol (entropia) também sobe junto com o cansaço
            self.entropy_ref = min(0.90, 0.01 + ((ram - 80) / 20.0))
        else:
            self.gamma = 1.0

    def q_proj_hook(self, module, input_tensor, output_tensor):
        """
        Forward hook do PyTorch acoplado à camada q_proj da Layer 12.
        Intercepta Q na GPU, aplica a Gravidade de Blake e o escalonamento límbico.
        """
        if self.m_sem_tensor is None:
            return output_tensor  # Se não houver massa semântica, mantém o fluxo nominal

        # Atualiza as variáveis endócrinas baseadas no hardware real
        self.recalibrar_pressao_osmotica()

        # Certifica-se de que o tensor de memória está no mesmo dispositivo e tipo de Q
        m_sem = self.m_sem_tensor.to(device=output_tensor.device, dtype=output_tensor.dtype)

        # O output_tensor possui a forma: [batch_size, sequence_length, hidden_dimension]
        Q = output_tensor

        # Garante que m_sem seja um vetor 1D e tenha exatas Q.shape[-1] dimensões
        m_sem = m_sem.view(-1)
        if m_sem.size(0) > Q.shape[-1]:
            m_sem = m_sem[:Q.shape[-1]]
        elif m_sem.size(0) < Q.shape[-1]:
            pad_size = Q.shape[-1] - m_sem.size(0)
            m_sem = torch.nn.functional.pad(m_sem, (0, pad_size))

        # 1. Cálculo da Distância Euclidiana hiperdimensional: ||Q - M_sem||
        # Resguarda o alinhamento através do amortecedor imunológico epsilon
        distancia = torch.norm(Q - m_sem, p=2, dim=-1, keepdim=True)
        denominador = torch.pow(distancia + self.epsilon, 3)

        # 2. Cálculo do Gradiente do Potencial Gravitacional Semântico: ∇Φ_B(Q)
        grad_potencial = self.G_s * m_sem * ((Q - m_sem) / denominador)

        # 3. Aplicação da curvatura geodésica: Q' = Q - ∇Φ_B(Q)
        Q_linha = Q - grad_potencial

        # 4. Modulação Límbica de Temperatura Integrada e Amortecedor Metabólico
        # Em vez de dividir diretamente pela entropia (o que multiplica o tensor por 100x e destrói o modelo),
        # aplicamos o gamma para atenuar, e restauramos a magnitude original (L2 Norm) 
        # para que seja apenas uma rotação semântica, sem explodir os logits de Atenção.
        Q_amortecido = self.gamma * Q_linha
        
        norm_orig = torch.norm(Q, p=2, dim=-1, keepdim=True)
        norm_new = torch.norm(Q_amortecido, p=2, dim=-1, keepdim=True)
        Q_final = Q_amortecido * (norm_orig / (norm_new + 1e-8))

        # Rastreamento de diagnóstico interno para monitoramento via console
        if torch.rand(1).item() < 0.01: # Reduz o impacto de logs imprimindo apenas 1% das vezes
            print(f"\n[HOOK L12] RAM: {psutil.virtual_memory().percent}% | Gamma: {self.gamma:.3f} | Norma Q Original: {torch.norm(Q).item():.4f} | Norma Q' Curvada: {torch.norm(Q_final).item():.4f}")

        return Q_final

def injetar_gravidade_interna(model, hooks_manager):
    """
    Localiza a projeção de Query da Layer 12 (camada 11 nos índices indexados em zero)
    e registra o gancho de execução físico.
    """
    try:
        # Acesso estrutural padrão para a arquitetura de blocos Transformers da Hugging Face
        camada_alvo = model.model.layers[11].self_attn.q_proj
        handle = camada_alvo.register_forward_hook(hooks_manager.q_proj_hook)
        print("[HOOK] Cirurgia concluída. Gancho de Gravidade Interna ativo na Layer 12.")
        return handle
    except AttributeError as e:
        print(f"[HOOK - ERRO] Não foi possível mapear a assinatura da camada do modelo: {e}")
        return None
