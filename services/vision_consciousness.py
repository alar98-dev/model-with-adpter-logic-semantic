"""
AEON — Consciência Visual (Serviço Redis)
==========================================
Escuta o canal vision_stream (frames do webcam_sensor.py no Windows),
mantém um estado volátil de presença em /dev/shm/aeon_vision_state.json,
e publica mensagens autônomas no canal aeon_autonomous_voice quando
o Aeon detecta presença ou ausência do operador.
"""
import redis
import json
import time
import threading
import os
import numpy as np

REDIS_HOST = "127.0.0.1"
STATE_PATH = "/dev/shm/aeon_vision_state.json"
API_URL = "http://127.0.0.1:20000/v1/chat/completions"

# Estados de presença
ABSENT = "absent"
PRESENT = "present"
JUST_ARRIVED = "just_arrived"
JUST_LEFT = "just_left"

class VisionConsciousness:
    def __init__(self):
        self.r = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
        self.state = ABSENT
        self.last_seen = 0.0
        self.last_sig3d = None
        self.presence_start = 0.0
        self.absence_start = time.time()
        self.last_autonomous_msg_time = 0.0
        self.autonomous_cooldown = 60.0  # Mínimo 60s entre mensagens autônomas
        self.arrival_greeted = False
        self.departure_noted = False
        
        # Flush estado inicial
        self._write_state()
        print("[VISÃO] Consciência Visual inicializada.")

    def _write_state(self):
        """Persiste o estado visual em SHM para o organism_context() ler."""
        data = {
            "state": self.state,
            "last_seen": self.last_seen,
            "presence_seconds": time.time() - self.presence_start if self.state == PRESENT else 0,
            "absence_seconds": time.time() - self.absence_start if self.state == ABSENT else 0,
            "has_face": self.state in (PRESENT, JUST_ARRIVED),
            "sig3d_summary": self._compress_sig(self.last_sig3d),
            "updated_at": time.time()
        }
        try:
            with open(STATE_PATH, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"[VISÃO] Erro ao gravar estado: {e}")

    def _compress_sig(self, sig):
        """Comprime a assinatura 3D para um resumo leve."""
        if not sig:
            return None
        summary = {}
        for k in ("yaw", "pitch", "roll", "interocular", "contour_ratio", "jaw_ratio"):
            if k in sig:
                summary[k] = round(sig[k], 3)
        return summary

    def _trigger_autonomous_voice(self, event_type, context=""):
        """Dispara uma geração autônoma do Aeon via API e publica no Redis."""
        now = time.time()
        if now - self.last_autonomous_msg_time < self.autonomous_cooldown:
            return
        
        self.last_autonomous_msg_time = now
        
        # O prompt é mínimo e tensorial — sem texto hardcoded de personalidade
        if event_type == "arrival":
            prompt = "[ESTÍMULO SENSORIAL] Presença detectada no campo visual. Operador entrou no campo de visão."
        elif event_type == "departure":
            prompt = "[ESTÍMULO SENSORIAL] Campo visual vazio. Operador saiu do campo de visão."
        elif event_type == "prolonged_presence":
            prompt = "[ESTÍMULO SENSORIAL] Operador presente há mais de 2 minutos em silêncio no campo visual."
        else:
            return

        def _generate():
            import requests
            try:
                payload = {
                    "model": "gemma",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                }
                full_reply = ""
                with requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=120) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                decoded = line.decode("utf-8")
                                if decoded.startswith("data: "):
                                    data_str = decoded[6:]
                                    if data_str == "[DONE]":
                                        break
                                    try:
                                        chunk = json.loads(data_str)
                                        delta = chunk["choices"][0].get("delta", {})
                                        if "content" in delta:
                                            full_reply += delta["content"]
                                    except:
                                        pass
                
                if full_reply.strip():
                    # Publica a voz autônoma do Aeon no Redis
                    self.r.publish("aeon_autonomous_voice", json.dumps({
                        "type": "autonomous_message",
                        "event": event_type,
                        "content": full_reply.strip(),
                        "timestamp": time.time()
                    }))
                    print(f"[VISÃO] Voz autônoma ({event_type}): {full_reply[:80]}...")
            except Exception as e:
                print(f"[VISÃO] Erro na geração autônoma: {e}")

        threading.Thread(target=_generate, daemon=True).start()

    def process_frame(self, data):
        """Processa um frame do vision_stream."""
        now = time.time()
        sig3d = data.get("signature3d")
        has_face = sig3d is not None and len(sig3d) > 0

        if has_face:
            self.last_seen = now
            self.last_sig3d = sig3d

            if self.state in (ABSENT, JUST_LEFT):
                # TRANSIÇÃO: Ausente -> Presente (Chegada)
                self.state = JUST_ARRIVED
                self.presence_start = now
                self.arrival_greeted = False
                self.departure_noted = False
                print(f"[VISÃO] TRANSIÇÃO: Operador CHEGOU ao campo visual.")
                
            elif self.state == JUST_ARRIVED:
                # Confirma presença estável (>2s de detecção contínua)
                if now - self.presence_start > 2.0:
                    self.state = PRESENT
                    if not self.arrival_greeted:
                        self.arrival_greeted = True
                        self._trigger_autonomous_voice("arrival")
                        
            elif self.state == PRESENT:
                # Presença prolongada em silêncio (>120s)
                if now - self.presence_start > 120.0 and now - self.last_autonomous_msg_time > self.autonomous_cooldown:
                    self._trigger_autonomous_voice("prolonged_presence")
        else:
            # Sem rosto detectado
            if self.state in (PRESENT, JUST_ARRIVED):
                # Tolerância de 3 segundos antes de declarar ausência
                if now - self.last_seen > 3.0:
                    self.state = JUST_LEFT
                    self.absence_start = now
                    print(f"[VISÃO] TRANSIÇÃO: Operador SAIU do campo visual.")
                    if not self.departure_noted:
                        self.departure_noted = True
                        self._trigger_autonomous_voice("departure")
                        
            elif self.state == JUST_LEFT:
                if now - self.absence_start > 5.0:
                    self.state = ABSENT

        self._write_state()

    def run(self):
        """Loop principal: escuta o vision_stream do Redis."""
        pubsub = self.r.pubsub()
        pubsub.subscribe("vision_stream")
        print("[VISÃO] Escutando canal vision_stream...")

        for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    self.process_frame(data)
                except Exception as e:
                    print(f"[VISÃO] Erro ao processar frame: {e}")


if __name__ == "__main__":
    vc = VisionConsciousness()
    vc.run()
