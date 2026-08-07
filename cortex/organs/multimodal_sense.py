import cv2
import numpy as np
import json
import threading
import sys
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame

# Fallback robust para PyAudio
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    sys.stdout.write("⚠️  [multimodal_sense] pyaudio não instalado. Fallback para áudio simulado.\n")

class MultimodalSenseOrgan:
    def __init__(self, node_id="swa-multimodal-sense"):
        self.node_id = node_id
        # Captura de vídeo
        self.cap = cv2.VideoCapture(0)
        
        # Inicialização do áudio com fallback resiliente
        self.p = None
        self.stream = None
        self.has_audio_device = False
        
        if HAS_PYAUDIO:
            try:
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(
                    format=pyaudio.paInt16, 
                    channels=1, 
                    rate=44100, 
                    input=True, 
                    frames_per_buffer=1024
                )
                self.has_audio_device = True
                sys.stdout.write("📡 [multimodal_sense] Entrada de áudio física inicializada com sucesso.\n")
            except Exception as e:
                sys.stdout.write(f"⚠️  [multimodal_sense] Erro ao abrir dispositivo de áudio físico ({e}). Fallback para silêncio.\n")
        
        threading.Thread(target=self._sense_loop, daemon=True).start()

    def _sense_loop(self):
        while True:
            # Capture visual (low res to save bandwidth/compute)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (64, 64))
                visual_tensor = frame.flatten().astype(np.float32) / 255.0
            else:
                visual_tensor = np.zeros((64*64*3,), dtype=np.float32)

            # Capture audio (simple RMS amplitude ou fallback silencioso)
            if self.has_audio_device and self.stream is not None:
                try:
                    audio_data = np.frombuffer(self.stream.read(1024, exception_on_overflow=False), dtype=np.int16)
                    audio_tensor = np.array([np.sqrt(np.mean(audio_data**2))], dtype=np.float32)
                except Exception as e:
                    # Se falhar durante a leitura, gera silêncio
                    audio_tensor = np.zeros((1,), dtype=np.float32)
            else:
                audio_tensor = np.zeros((1,), dtype=np.float32)

            # Combine and send to IPC
            combined = np.concatenate([visual_tensor, audio_tensor])
            self._emit_to_bus(combined)

    def _emit_to_bus(self, topology):
        try:
            import socket
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            payload = json.dumps({"type": "SENSORY_INPUT", "node_id": self.node_id}).encode("utf-8")
            sock.sendall(pack_aeon_frame(topology, payload))
            sock.close()
        except Exception as e:
            pass # Silent fail to not disrupt OODA loop

def init_organ():
    return MultimodalSenseOrgan()

if __name__ == "__main__":
    import time
    # Teste de execução rápida local
    print("Iniciando teste do órgão MultimodalSenseOrgan...")
    organ = init_organ()
    time.sleep(2)
    print("Órgão executando em background. Teste concluído.")
