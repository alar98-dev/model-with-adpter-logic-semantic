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
      - swa:mic:speech  → texto ASR com proveniência e incerteza
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
                        # Espelha o payload integral e normaliza o veredito de
                        # identidade. Descartar campos aqui apagava a janela
                        # temporal da fala e o instante de cada palavra antes de
                        # chegarem ao córtex auditivo.
                        #
                        # O orquestrador publica o score em `score`; o córtex lia
                        # `similarity`, então a accuracy chegava 0.00% mesmo com
                        # o casamento correto. Aceitamos os três nomes para não
                        # depender do produtor.
                        score = data.get("score")
                        if score is None:
                            score = data.get("similarity")
                        if score is None:
                            score = data.get("rag_similarity")
                        mirror = dict(data)
                        mirror.pop("channel", None)
                        payload = json.dumps({
                            **mirror,
                            "text": texto,
                            "source": "AUDIO_INPUT",
                            "speaker": locutor,
                            "is_alifer": bool(data.get("is_alifer", False)),
                            "similarity": float(score or 0.0),
                            "confidence": float(
                                data.get("confidence", score or 0.0) or 0.0
                            ),
                            "energy": data.get("energy"),
                            "timestamp": data.get("timestamp"),
                            "utter_started_at": data.get("utter_started_at"),
                            "utter_ended_at": data.get("utter_ended_at"),
                            "duration_ms": data.get("duration_ms"),
                            "word_timestamps": data.get("word_timestamps") or [],
                            "audio_provenance": data.get("proveniencia_audio") or {
                                "schema_version": 1,
                                "propagation_path": "unknown",
                                "source_class": "unknown",
                                "speaker_attribution": "unverified",
                            },
                        })
                        udp_sock.sendto(payload.encode(), ("127.0.0.1", self.udp_port))

                elif channel == b"swa:mic:entity":
                    locutor = data.get("locutor", "Desconhecido")
                    print(f"\n[RedisMicListener] Nova entidade descoberta: {locutor}")

            except Exception as e:
                print(
                    f"[RedisMicListener] Evento do canal {channel!r} "
                    f"não foi espelhado: {e}")

        udp_sock.close()
        pubsub.close()


if __name__ == "__main__":
    listener = RedisMicListener()
    listener.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        listener.stop()
