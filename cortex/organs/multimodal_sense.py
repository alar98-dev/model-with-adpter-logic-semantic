import cv2
import pyaudio
import numpy as np
import json
import threading
from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame

class MultimodalSenseOrgan:
    def __init__(self, node_id="swa-multimodal-sense"):
        self.node_id = node_id
        self.cap = cv2.VideoCapture(0)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
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

            # Capture audio (simple RMS amplitude)
            audio_data = np.frombuffer(self.stream.read(1024, exception_on_overflow=False), dtype=np.int16)
            audio_tensor = np.array([np.sqrt(np.mean(audio_data**2))], dtype=np.float32)

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
