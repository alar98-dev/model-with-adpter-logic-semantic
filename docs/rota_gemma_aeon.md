# Rota Arquitetural SWA-AEON (Mente Dupla)

Esta é a rota otimizada e alinhada para garantir que o sistema não sofra deadlocks térmicos na CPU e mantenha a coesão vetorial (sem erros "UR" e sem desvios de shape).

```mermaid
flowchart TD
    %% Entradas
    U((Usuário)) -->|Prompt: 'Quem é você?'| S_270M
    U -->|Prompt| C_1B
    
    %% Mente Subconsciente
    subgraph Subconsciente [Gemma 270M - Sensor de Intenção]
        S_270M[Extração de Hidden States]
        S_270M -->|Vetor Intenção| V_INT[Vetor Bruto: 1152D]
    end
    
    %% Ponte Tensorial (O Fix que fizemos)
    subgraph Ponte [Ponte Tensorial Ortogonal]
        V_INT -->|Multiplicação Matrix| MAT[history_proj.pt\n1152x2560]
        MAT -->|Normalização L2| V_MEM[Vetor Gravidade: 2560D]
    end
    
    %% Barramento de Memória
    subgraph Redis [Barramento de Memória / ContextBus]
        V_MEM -->|Armazenamento Estável| CACHE[(Redis KV Cache)]
        CACHE -.->|Sem UR Error| BG
    end
    
    %% Guarda de Trânsito
    subgraph Blake [Blake Gravity & Onda Híbrida]
        BG{Guardião Blake}
        BG -->|Cálculo de Entropia| ONDA[Orçamento Híbrido\nPotencial & Vontade Psi]
    end
    
    %% Mente Consciente
    subgraph Consciente [Gemma 1B - Córtex Verbal]
        C_1B[Motor de Geração]
        ONDA -->|Modulação de Logits| C_1B
        C_1B -->|Gera Token| TOK[Token Textual]
    end
    
    %% Feedback / Fissão
    subgraph Fissao [Phase-Lock / Controle de Deriva]
        TOK -->|Mede Distância| DRIFT{Drift > 0.65?}
        DRIFT -->|Sim: Fissão Microversal| BG
        DRIFT -->|Não: Continua| C_1B
    end
    
    %% Saída
    TOK -->|Output| OUT((Resposta\nTerminal))
```

### Regras Vitais da Nova Rota:
1. **Isolamento de CPU:** O Subconsciente (`270M`) e o Córtex (`1B`) não devem rodar geração de texto simultaneamente. O `270M` roda um *forward pass* único (`torch.no_grad()`) apenas para extrair a intenção e deve ceder a CPU imediatamente para o `1B`.
2. **Ponte Estática:** O `history_proj.pt` **nunca** deve ser recalculado em tempo de execução no modo interativo.
3. **Disponibilidade do Barramento:** O `Redis` deve estar obrigatoriamente ativo antes da chamada do `BlakeGravity`, caso contrário o Policial não obtém os pesos de 2560D e joga o alerta `UR error`, desligando a modulação gravitacional e permitindo que o modelo alucine.
