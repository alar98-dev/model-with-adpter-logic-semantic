import threading

class AIMDController:
    def __init__(self, initial_window=10.0):
        self._window = initial_window
        self.lock = threading.Lock()
        self.min_window = 1.0
        self.max_window = 100.0
        self.alpha = 2.0  # Incremento aditivo (Aumento)
        self.beta = 0.5   # Fator de decréscimo (Multiplicativo)

    @property
    def window(self):
        with self.lock:
            return self._window

    def on_success(self):
        with self.lock:
            self._window = min(self.max_window, self._window + self.alpha)
            return self._window

    def on_error(self):
        with self.lock:
            self._window = max(self.min_window, self._window * self.beta)
            return self._window

# Instância global para ser compartilhada pelos órgãos
controller = AIMDController()
