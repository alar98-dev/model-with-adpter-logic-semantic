"""Dreamer — Córtex Pré-frontal do AEON.

Camada proativa acima do sistema imune:
  1. Observa o tensor de gravidade + memória imune como feedback
  2. Mantém um modelo interno do mundo (world model)
  3. Simula futuros possíveis para cada ação candidata
  4. Planeja a sequência ótima de ações
  5. Pode inibir respostas imunes imediatas se a simulação
     mostrar consequências piores

Integração:
  - Lê de: immune_memory, gravity_tensor, action_history
  - Escreve em: /var/run/aeon/dreamer.action (plano atual)
  - Inibe: main.py (se dreamer diz "não aja agora")
"""

from .dreamer import Dreamer

__all__ = ["Dreamer"]
