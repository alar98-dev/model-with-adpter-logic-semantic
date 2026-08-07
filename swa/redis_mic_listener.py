import json
import os
import socket
import threading

try:
    import redis
except ImportError:
    redis = None


class RedisMicListener(threading.Thread):
    """Escuta canais Redis do voice-stream e alimenta o córtex auditivo.

    Canais escutados:
      - swa:mic:speech  → texto transcrito do microfone
      - swa:mic:entity  → nova entidade/locutor descoberto

    Para cada mensagem de fala, faz mirror para UDP :9200 (córtex auditivo)
    como redundância ao caminho direto do orquestrador.
    """

    def __init__(self, host="127.0.0.1", port=6379, auditory_cortex_port=9200):
        super().__init__()
        self.daemon = True
        self.redis_host = host
        self.redis_port = port
        self.udp_port = auditory_cortex_port
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        if redis is None:
            print("[RedisMicListener] redis-py não instalado — desabilitado")
            return

        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=0)
            r.ping()
        except Exception as e:
            print(f"[RedisMicListener] Não foi possível conectar ao Redis: {e}")
            return

        pubsub = r.pubsub()
        pubsub.subscribe("swa:mic:speech", "swa:mic:entity")
        print(f"[RedisMicListener] Escutando Redis {self.redis_host}:{self.redis_port} → UDP :{self.udp_port}")

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        for message in pubsub.listen():
            if not self.running:
                break
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
                channel = message["channel"]

                if channel == b"swa:mic:speech":
                    texto = data.get("texto", "")
                    locutor = data.get("locutor", "Desconhecido")
                    if texto and len(texto) > 2:
                        payload = json.dumps({
                            "text": texto,
                            "source": "MICROPHONE",
                            "speaker": locutor,
                        })
                        udp_sock.sendto(payload.encode(), ("127.0.0.1", self.udp_port))

                elif channel == b"swa:mic:entity":
                    locutor = data.get("locutor", "Desconhecido")
                    print(f"\n[RedisMicListener] Nova entidade descoberta: {locutor}")

            except Exception:
                pass

        udp_sock.close()
        pubsub.close()


if __name__ == "__main__":
    listener = RedisMicListener()
    listener.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        listener.stop()
