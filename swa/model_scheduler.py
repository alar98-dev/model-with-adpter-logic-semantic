"""Agendador do modelo compartilhado (Gemma 1B).

Regra de prioridade:
  - Vias reativas (chat, introspecção): acquire(reactive=True) — bloqueiam
    até conseguir o modelo (timeout).
  - Loop entrópico de fundo: acquire(reactive=False, timeout=0) — se o modelo
    estiver ocupado por uma via reativa, cede o ciclo imediatamente.

Todas as threads que tocam o modelo compartilhado devem passar por aqui para
garantir fatia de CPU/token controlada.
"""

import threading

_lock = threading.Lock()
_reactive_active = [False]


def acquire(reactive: bool = False, timeout: float = 0.0) -> bool:
    if reactive:
        _reactive_active[0] = True
    if not _lock.acquire(timeout=timeout):
        if reactive:
            _reactive_active[0] = False
        return False
    return True


def release() -> None:
    try:
        _lock.release()
    except RuntimeError:
        pass
    _reactive_active[0] = False


def reactive_active() -> bool:
    return _reactive_active[0]
