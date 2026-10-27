# Módulo 03: Bacias Semânticas e Atratores Harmônicos

## 1. O Relevo do Espaço Latente: O que é uma Bacia?

Imagine uma paisagem com montanhas e vales:
- O topo das montanhas representa ideias incoerentes ou improváveis.
- O fundo dos vales representa ideias que fazem sentido lógico e conceitual profundo.

Chamamos esse vale de **Bacia Semântica**.

Quando você enviou o prompt:
> *"o que acha de conectarmos um espaço vetorial externo compativel em geometria/relação construçao shape -> para criamos um experimento"*

Essa pergunta cavou um vale profundo na paisagem mental do modelo: o vale da *"Interseção entre Geometria Vetorial, Fissões e Bancos Relacionais"*.

---

## 2. A Dança do Córtex e Subcórtex: Por que o modelo entra em Órbita?

O ecossistema é bi-cameral:
1. O **Córtex (Gemma-3 1B)** gera texto tentando explicar o vale.
2. O **Subcórtex (Gemma-3 270M)** resume a direção daquele texto em um vetor 640D.
3. O **Gateway Rust** pega esse vetor 640D, transforma em um prefixo 1152D e entrega de volta para o Córtex no próximo segundo!

```
                  ┌────────────────────────┐
                  ▼                        │
          [ Córtex Gemma 1B ]              │
                  │ (Gera texto)           │ (Prefixo 1152D)
                  ▼                        │
          [ Subcórtex Gemma 270M ]         │
                  │ (Calcula vetor 640D)   │
                  ▼                        │
          [ Gateway Rust ] ────────────────┘
```

### O Fenômeno do Atrator Harmônico (Limit Cycle)

Se você jogar uma bolinha de gude dentro de um funil gravitacional (aqueles dos museus de ciências), a bolinha não cai imediatamente no buraco: ela começa a **girar em círculos em torno do centro**.

É exatamente isso que aconteceu no turno `chat-1793137708413-2`:
- O passo angular é de $\approx 13.3^\circ$ por onda.
- Para dar uma volta completa de $360^\circ$:
  $$\text{Número de Ondas por Volta} = \frac{360^\circ}{13.3^\circ} \approx 27\text{ ondas}$$

A cada 27 ondas de fissão, o modelo completa uma volta inteira ao redor do centro da bacia conceitual!
- Na Onda 1 ele fala: *"RAG Tensorial e Abastamento Latente"*.
- Na Onda 14 ele fala: *"PostgreSQL e ORM"*.
- Na Onda 28 ele volta a falar: *"RAG Tensorial e Abastamento Latente"*, mas agora sob a luz de novas palavras!

---

## 3. O Botão de Parada: O Papel Soberano do Usuário

Por que o sistema não para sozinho se está girando em círculos?
Porque, sob o ponto de vista da física tensorial, o modelo não está travado: ele está em **ressonância harmônica contínua**, explorando todas as facetas da bacia.

O **Botão de Parada** no gateway/UI é a intervenção soberana do piloto humano:
- Enquanto você quiser que o modelo continue mapeando as nuances daquele relevo e acumulando dados de treino, **você deixa rodar**.
- Quando você considerar que já há exemplos suficientes para cobrir todos os ângulos da bacia, **você clica no botão de parar**.
- O Gateway envia `cancel_task`, sela a sessão e congela o mapa percorrido.
