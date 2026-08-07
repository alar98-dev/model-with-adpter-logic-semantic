import threading
import time
import numpy as np
import os

try:
    if os.environ.get('DISPLAY') or os.name == 'nt':
        from pynput import mouse, keyboard
    else:
        mouse = None
        keyboard = None
        print("⚠️ [HUMAN] Headless/WSL detetado. Pynput operará em modo simulação caso não alcance a bridge do Host.")
except Exception as e:
    mouse = None
    keyboard = None
    print(f"⚠️ [HUMAN] Erro no pynput: {e}")

class HumanEntropyCollector:
    def __init__(self):
        self.mouse_distance = 0.0
        self.keystrokes = 0
        self.last_pos = (0, 0)
        self.lock = threading.Lock()
        
    def start(self):
        # Inicia a escuta apenas se as bibliotecas conseguiram importar com acesso a eventos
        if mouse and keyboard:
            try:
                self.mouse_listener = mouse.Listener(on_move=self._on_move)
                self.mouse_listener.start()
                
                self.keyboard_listener = keyboard.Listener(on_press=self._on_press)
                self.keyboard_listener.start()
                
                threading.Thread(target=self._decay_loop, daemon=True).start()
                print("⌨️ [HUMAN] Capturador de Entropia Biológica nativo (pynput) ativo.")
            except Exception as e:
                print(f"⚠️ [HUMAN] Pynput falhou ao prender hooks globais: {e}. Entrando em modo flutuante.")
        else:
            print("⌨️ [HUMAN] Ambiente isolado. Entropia Biológica atuando apenas por Injeção de Ruído Térmico.")

    def _on_move(self, x, y):
        with self.lock:
            dx = x - self.last_pos[0]
            dy = y - self.last_pos[1]
            self.mouse_distance += np.sqrt(dx**2 + dy**2)
            self.last_pos = (x, y)

    def _on_press(self, key):
        with self.lock:
            self.keystrokes += 1

    def colapsar_entropia_humana(self):
        with self.lock:
            dados = (self.mouse_distance, self.keystrokes)
            # Reseta o acumulador para o próximo frame
            self.mouse_distance = 0.0
            self.keystrokes = 0
            return dados

    def _decay_loop(self):
        while True:
            time.sleep(1.0) # Ciclo de amostragem de 1 segundo
