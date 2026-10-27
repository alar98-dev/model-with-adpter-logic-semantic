# Módulo 02: Cosseno Semântico e Velocidade Angular

## 1. O Produto Escalar na Prática

Imagine que temos dois pensamentos representados por vetores unitários de apenas 3 dimensões (para simplificar a visualização):
- Pensamento A ($\mathbf{u}$): $(0.6, 0.8, 0.0)$
- Pensamento B ($\mathbf{v}$): $(0.6, 0.8, 0.0)$

O **Produto Escalar** (dot product) multiplica cada coordenada correspondente e soma tudo:
$$\mathbf{u} \cdot \mathbf{v} = (0.6 \times 0.6) + (0.8 \times 0.8) + (0.0 \times 0.0) = 0.36 + 0.64 + 0.0 = 1.0$$

Como o resultado foi $1.0$, o ângulo entre eles é $0^\circ$ — ou seja, são o mesmo pensamento!

---

## 2. O Cosseno Semântico ($\cos \theta$)

No nosso Subcórtex (640D), a fórmula é a mesma, só que somando 640 produtos:
$$\text{semantic\_cosine} = \cos(\theta) = \sum_{i=1}^{640} u_i \cdot v_i$$

### O que o valor do Cosseno nos diz?

| Valor de $\cos(\theta)$ | Ângulo $\theta$ | Significado Semântico |
| :--- | :--- | :--- |
| **$+1.0$** | $0^\circ$ | **Idêntico**: mesma direção exata. |
| **$+0.96$ a $+0.98$** | $\approx 11^\circ$ a $16^\circ$ | **Irmãos Semânticos**: mesmo tema, pequena variação de palavras (o que vimos no trace!). |
| **$+0.75$** | $\approx 41^\circ$ | **Limiar de Fronteira** (`SEMANTIC_ANCHORED_THRESHOLD`): o limite seguro antes de perder o foco. |
| **$0.0$** | $90^\circ$ | **Ortogonal / Neutro**: assuntos totalmente descorrelacionados (ex.: "Álgebra Linear" e "Receita de Pão"). |
| **$-1.0$** | $180^\circ$ | **Antônimos Perfeitos**: direções opostas na hiperesfera. |

---

## 3. O Passo na Esfera: A Distância Geodésica ($\Delta \theta$)

Quando você viaja de avião de São Paulo para Tóquio, você não cava um túnel em linha reta pelo centro da Terra; você voa acompanhando a curvatura do planeta (a geodésica).

No espaço latente, a distância entre a onda de fissão anterior e a atual é o arco $\Delta \theta$:
$$\Delta \theta = \arccos(\text{semantic\_cosine})$$

### O Cálculo com os Dados Reais do Nosso Sistema:
No log de trace do turno ativo:
$$\cos(\theta) = 0.9733$$
$$\Delta \theta = \arccos(0.9733) \approx 0.2316\text{ radianos}$$

Convertendo radianos para graus:
$$\text{Graus} = 0.2316 \times \frac{180}{\pi} \approx 13.27^\circ$$

> **Conclusão Visual**: A cada 8 segundos (uma onda de fissão), o Gemma dá um "passo" de **$13.27^\circ$** na curvatura da hiperesfera de 640 dimensões!

---

## 4. O Conceito de Novidade (Novelty)

A novidade mede quanto do vetor virou para uma nova direção:
$$\text{Novelty} = 1.0 - \text{semantic\_cosine}$$

Se $\cos(\theta) = 0.9733$:
$$\text{Novelty} = 1.0 - 0.9733 = 0.0267 \quad (2.67\%)$$

Isso significa que **$97.33\%$ da energia mental** do modelo continuou presa ao assunto anterior, e **$2.67\%$ foi novidade exploratória**.
