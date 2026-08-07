import queue
import threading
import json
import time
import sys

class SensoryOutputDaemon(threading.Thread):
    def __init__(self):
        super().__init__()
        self.q = queue.Queue()
        self.running = True
        self.daemon = True

    def put(self, msg_type, payload):
        self.q.put((msg_type, payload))

    def stop(self):
        self.running = False
        self.q.put(("STOP", None))

    def run(self):
        while self.running:
            try:
                msg_type, payload = self.q.get(timeout=1.0)
                
                if msg_type == "STOP":
                    break
                    
                if msg_type == "STDOUT":
                    sys.stdout.write(payload)
                    sys.stdout.flush()
                    
                elif msg_type == "LOG_AEON":
                    with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                        f.write(payload)
                        
                elif msg_type == "LOG_PURO":
                    with open("/services/model-with-adpter-logic-semantic/swa_PENSAMENTO_PURO.log", "a", encoding="utf-8") as f:
                        f.write(payload)
                        
                elif msg_type == "JSONL":
                    with open("/services/model-with-adpter-logic-semantic/chaos_telemetry.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps(payload) + "\n")
                        
                elif msg_type == "UDP":
                    try:
                        import socket
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.sendto(payload[0].encode('utf-8'), payload[1])
                    except Exception: pass
                    
                elif msg_type == "UDP_BROADCAST":
                    try:
                        import socket
                        import json
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        # payload: (chunk, voice_type, voice_origin, entropy)
                        chunk = payload[0]
                        entropy = payload[3] if len(payload) > 3 else 0.0
                        packet = json.dumps({"type": "thought_stream", "content": chunk, "entropy": float(entropy)}).encode('utf-8')
                        sock.sendto(packet, ("127.0.0.1", 20005))
                    except Exception: pass
                    
                self.q.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                pass
