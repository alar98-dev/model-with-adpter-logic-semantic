# Módulo 04: Fissões Sucessivas e Calibração do Gemma

## 1. O que é uma Fissão Semântica?

Em física nuclear, "fissão" é quando um núcleo atômico pesado se divide em fragmentos menores liberando energia.

No nosso sistema cognitivo:
- O pensamento contínuo é uma onda gigante de ideias.
- Em vez de deixar o modelo falar sem parar até encher a tela de texto solto, o sistema monitora a coerência do fluxo.
- Quando o Córtex completa um bloco com sentido próprio ou atinge o limiar de desvio, o motor dispara um corte geométrico chamado **Fissão**.
- O gatilho que causa esse corte é a **Fronteira Semântica** (`semantic_boundary`).

---

## 2. A Fábrica de Pares de Treinamento (`aeon_wave_training_examples`)

O que acontece a cada fissão? Uma foto tomográfica completa é tirada e salva no PostgreSQL:

| Campo Salvo no Banco | Dimensão / Tipo | O que Representa? |
| :--- | :--- | :--- |
| `input_prefix_base64` | **1152 floats** (FP16/FP32) | A "bússola de navegação" que entregamos para o Córtex (Gemma 1B). |
| `sub_embedding_base64` | **640 floats** (FP32 unitário) | A "resposta intuitiva" sintetizada pelo Subcórtex (Gemma 270M). |
| `target_text` | **Texto / Tokens** | As palavras exatas que nasceram da combinação desses dois tensores. |
| `source_event_hash` | **Hash BLAKE3** | O carimbo digital infalsificável que prova a autenticidade e ordem do evento. |

No seu turno ativo (`chat-1793137708413-2`), mais de **235 fotos dessas foram tiradas e armazenadas com sucesso**!

---

## 3. Por que isso "Calibra" o Espaço Latente?

Imagine que você está ensinando um robô a caminhar por uma sala cheia de obstáculos:
- Se você deixar o robô andar uma única vez em linha reta, você só aprendeu sobre aquela linha.
- Se o robô der 235 passos circulares cobrindo cada centímetro quadrado do piso, você terá um **mapa de elevação em altíssima resolução de todo o ambiente**.

Cada fissão adicional:
1. Registra como o modelo transita de um ponto $\mathbf{A}$ para um ponto $\mathbf{B}$ na hiperesfera.
2. Cria pares de entrada/saída prontos para **Ajuste Fino Supervisionado (SFT)** ou **Otimização por Preferência Direta (DPO)**.
3. Permite que o modelo aprenda atalhos futuros: da próxima vez que falarmos sobre o mesmo assunto, o modelo já conhece as curvas do terreno e responde de forma muito mais concisa e direta.
