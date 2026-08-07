"""
generate_triplets.py
─────────────────────────────────────────────────────────────────────────────
Gerador de triplets sintéticos para treino do SemanticWarpAdapter (SWA).

Conceito:
  Cada triplet é composto por:
    • anchor   – um conceito/operação matemática ou semântica de referência
    • positive – o mesmo conceito reescrito (paráfrase semanticamente fiel)
    • negative – algo superficialmente similar mas logicamente OPOSTO ou
                 categoricamente diferente ("hard negative")

Categorias cobertas:
  1. Álgebra Linear / Operações vetoriais
  2. Métricas de similaridade
  3. Transformações do espaço latente
  4. Lógica de ferramentas (tool-schema semantics)
  5. Raciocínio causal
  6. Contexto de sequência / memória

Uso:
  python scripts/generate_triplets.py --output data/train_triplets.jsonl \\
      --samples 2000 --hard-negatives

Saída: arquivo .jsonl onde cada linha é {"anchor":..., "positive":..., "negative":...}
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

# ─────────────────────────────────────── Template bank ──────────────────────

TEMPLATES: List[Tuple[str, str, str]] = [
    # ── Álgebra Linear ───────────────────────────────────────────────────────
    (
        "Calcular a similaridade de cosseno entre os vetores A e B.",
        "Medir a proximidade direcional de A e B pelo produto escalar normalizado.",
        "Somar as magnitudes absolutas dos componentes de A e B.",
    ),
    (
        "Projetar o vetor v no subespaço definido pela base ortonormal U.",
        "Decompor v nos componentes paralelos à base U via produto interno.",
        "Escalar o vetor v uniformemente por um fator constante.",
    ),
    (
        "Aplicar rotação ortogonal no espaço latente usando uma matriz R.",
        "Transformar o tensor com matriz de rotação que preserva a norma L2.",
        "Realizar o achatamento (flatten) das dimensões do tensor latente.",
    ),
    (
        "Extrair o componente principal de um conjunto de vetores via SVD.",
        "Reduzir a dimensionalidade mantendo a variância máxima por decomposição singular.",
        "Calcular a média aritmética simples de todos os elementos do vetor.",
    ),
    (
        "Normalizar o vetor pela sua norma L2.",
        "Dividir cada componente do vetor pelo comprimento euclidiano do vetor.",
        "Multiplicar o vetor pelo seu valor máximo absoluto.",
    ),
    (
        "Calcular o produto externo (outer product) de dois vetores.",
        "Construir uma matriz de rank-1 a partir do produto tensorial de dois vetores.",
        "Computar o produto interno (dot product) escalar de dois vetores.",
    ),
    (
        "Decompor a matriz W em valores singulares (U, S, Vt).",
        "Fatorar W em componentes ortogonais e diagonais usando álgebra linear SVD.",
        "Calcular o determinante da matriz W.",
    ),
    (
        "Orthogonalizar as colunas de uma matriz via decomposição QR.",
        "Produzir uma base ortonormal a partir das colunas da matriz pelo processo de Gram-Schmidt.",
        "Transpor a matriz e multiplicar pela identidade.",
    ),

    # ── Métricas de similaridade ─────────────────────────────────────────────
    (
        "Calcular a distância euclidiana entre dois pontos no espaço vetorial.",
        "Medir o comprimento geométrico do segmento que liga os dois vetores.",
        "Calcular a divergência de Kullback-Leibler entre duas distribuições.",
    ),
    (
        "Usar a distância de Mahalanobis para comparar vetores considerando correlação.",
        "Medir a distância entre pontos ponderada pela covariância da distribuição.",
        "Aplicar a norma L1 (Manhattan) como métrica de distância.",
    ),
    (
        "Calcular a similaridade Jaccard entre dois conjuntos de tokens.",
        "Medir a sobreposição entre dois conjuntos pela intersecção dividida pela união.",
        "Contar a frequência de termos no vocabulário (TF-IDF bruto).",
    ),

    # ── Adaptador / Espaço latente ───────────────────────────────────────────
    (
        "Projetar a embedding de alta dimensão para um subespaço comprimido via W_a.",
        "Comprimir o vetor de entrada para o espaço de bottleneck k-dimensional pelo adaptador local.",
        "Aumentar a dimensionalidade da embedding com uma camada linear expansora.",
    ),
    (
        "Reconstruir o vetor original a partir do espaço comprimido usando W_b.",
        "Decodificar o vetor de bottleneck de volta à dimensão d do modelo base.",
        "Quantizar o vetor comprimido em representação de ponto fixo.",
    ),
    (
        "Aplicar GELU como função de ativação não-linear na camada de projeção.",
        "Introduzir não-linearidade suave no mapeamento linear com a ativação Gaussian Error Linear Unit.",
        "Usar ReLU simples como ativação, zerando valores negativos.",
    ),
    (
        "Penalizar a divergência de W_b^T W_b em relação à identidade (ortho_loss).",
        "Regularizar o adaptador forçando as colunas de W_b a serem mutuamente ortogonais.",
        "Minimizar a norma de Frobenius de W_b diretamente.",
    ),

    # ── Semântica de ferramentas (tool-schema) ──────────────────────────────
    (
        "Chamar a ferramenta `embed_text` para codificar um texto em vetor.",
        "Invocar o endpoint de embedding que converte texto em representação densa.",
        "Executar o parser de tokenização para contar o número de tokens.",
    ),
    (
        "Acionar a ferramenta `compute_triplet_loss` com âncora, positivo e negativo.",
        "Calcular a perda de triplet para ajustar o espaço semântico do adaptador.",
        "Chamar `count_tokens` para verificar o tamanho do contexto.",
    ),
    (
        "Usar `closed_context_vector` para sintetizar a memória do contexto fechado.",
        "Agregar os vetores de segmento acumulados em um único vetor representativo.",
        "Recuperar o vetor do primeiro token do contexto como representação.",
    ),
    (
        "Executar `topk_similarity_matrix` para encontrar os k vizinhos mais próximos.",
        "Construir a matriz de similaridade de cosseno e selecionar os k maiores valores.",
        "Ordenar os vetores por norma decrescente e retornar os k primeiros.",
    ),

    # ── Raciocínio causal ────────────────────────────────────────────────────
    (
        "Aumentar a taxa de aprendizado acelera a convergência do modelo.",
        "Um learning rate maior faz o gradiente descender mais rápido, convergindo em menos epochs.",
        "Aumentar o batch size reduz a variância dos gradientes estocásticos.",
    ),
    (
        "Adicionar regularização L2 reduz o overfitting ao penalizar pesos grandes.",
        "O termo de weight decay contrai os pesos durante o backprop, combatendo memorização.",
        "Aumentar o número de camadas aumenta a capacidade do modelo.",
    ),
    (
        "Hard negatives (negativos difíceis) aceleram a especialização do adaptador.",
        "Exemplos negativos muito próximos do âncora forçam o modelo a aprender fronteiras mais finas.",
        "Negativos aleatórios são suficientes para qualquer tarefa de recuperação semântica.",
    ),

    # ── Sequência / Memória ──────────────────────────────────────────────────
    (
        "O adaptador macro agrega os vetores locais em uma representação global do contexto.",
        "O nível macro do SWA produz um vetor de contexto sintetizando todas as segmentos processados.",
        "O adaptador macro descarta os vetores locais e processa apenas o último evento.",
    ),
    (
        "A memória de segmentos é acumulada a cada evento processado pelo LogicalEngine.",
        "Cada chamada a `process_event` adiciona o vetor local à lista de memória do motor.",
        "A memória é zerada antes de cada evento para evitar interferência contextual.",
    ),
]

# ─────────────────────────────── Hard negatives extras ──────────────────────
# Pares anchor→hard_negative para substituição quando --hard-negatives ativo
HARD_NEGATIVES: dict = {
    "Calcular a similaridade de cosseno entre os vetores A e B.":
        "Calcular a distância euclidiana ao quadrado entre A e B.",
    "Normalizar o vetor pela sua norma L2.":
        "Normalizar o vetor pela sua norma L1 (soma dos valores absolutos).",
    "Extrair o componente principal de um conjunto de vetores via SVD.":
        "Extrair a componente de maior valor absoluto coluna a coluna.",
    "Penalizar a divergência de W_b^T W_b em relação à identidade (ortho_loss).":
        "Penalizar a soma dos quadrados dos pesos de W_b (L2 regularization).",
    "Hard negatives (negativos difíceis) aceleram a especialização do adaptador.":
        "Negativos aleatórios treinam o adaptador de forma equivalente aos hard negatives.",
}

# ──────────────────────────────── Paráfrases extras ─────────────────────────
PARAPHRASE_VARIATIONS = [
    ("do sistema", "do módulo"),
    ("vetor", "tensor unidimensional"),
    ("embedding", "representação densa"),
    ("modelo", "rede neural"),
    ("adaptador", "camada de adaptação"),
    ("calcular", "computar"),
    ("medir", "quantificar"),
    ("aplicar", "executar"),
]


def vary_text(text: str) -> str:
    """Aplica pequenas substituições para diversificar o corpus."""
    t = text
    # aplicar até 2 substituições aleatórias
    subs = random.sample(PARAPHRASE_VARIATIONS, min(2, len(PARAPHRASE_VARIATIONS)))
    for old, new in subs:
        if old in t.lower():
            t = t.replace(old, new, 1)
            break  # uma substituição por vez para manter legibilidade
    return t


def generate_triplets(
    num_samples: int = 500,
    use_hard_negatives: bool = True,
    seed: int = 42,
) -> List[dict]:
    random.seed(seed)
    triplets = []

    for _ in range(num_samples):
        anchor, positive, negative = random.choice(TEMPLATES)

        # aplicar variação nas paráfrases
        if random.random() < 0.5:
            positive = vary_text(positive)
        if random.random() < 0.3:
            anchor = vary_text(anchor)

        # substituir por hard negative se disponível e solicitado
        if use_hard_negatives and anchor in HARD_NEGATIVES:
            if random.random() < 0.6:   # 60% das vezes usa hard negative
                negative = HARD_NEGATIVES[anchor]

        triplets.append({
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        })

    return triplets


def main():
    parser = argparse.ArgumentParser(description="Gerador de triplets sintéticos para SWA")
    parser.add_argument("--output", type=str, default="data/train_triplets.jsonl",
                        help="Caminho do arquivo .jsonl de saída")
    parser.add_argument("--samples", type=int, default=500,
                        help="Número de triplets a gerar")
    parser.add_argument("--hard-negatives", action="store_true", default=True,
                        help="Usar hard negatives quando disponíveis (padrão: True)")
    parser.add_argument("--seed", type=int, default=42, help="Seed aleatória para reprodutibilidade")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_triplets(
        num_samples=args.samples,
        use_hard_negatives=args.hard_negatives,
        seed=args.seed,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ {len(data)} triplets gerados → {out_path}")
    print(f"   Hard negatives ativos: {args.hard_negatives}")
    print(f"   Seed: {args.seed}")
    print(f"\nPara iniciar o treino:")
    print(f"  python -m swa.train_adapter \\")
    print(f"    --model-dir models/gemma2b \\")
    print(f"    --triplets {out_path} \\")
    print(f"    --save-dir swa/checkpoints/gemma2b \\")
    print(f"    --epochs 5 --batch 16 --k-local 64 --k-macro 128")


if __name__ == "__main__":
    main()
