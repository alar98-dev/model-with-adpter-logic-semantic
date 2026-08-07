import os
import time
import json
import socket
import praw
import hashlib
import torch
import math
import numpy as np
import psycopg2
import urllib.request

# Grid/IPC Imports
try:
    from swa.ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame
except ImportError:
    try:
        from ipc_bus import AEON_SOCKET_PATH, pack_aeon_frame
    except ImportError:
        # Dummy if missing
        AEON_SOCKET_PATH = "/tmp/aeon_socket"
        def pack_aeon_frame(topo, payload): return payload

class RedditCrawler:
    def __init__(self, node_id="swa-reddit-crawler"):
        self.node_id = node_id
        self.d_model = 2560
        self.dsn = "dbname=swa_matrix user=alifer host=localhost"
        
        # Reddit Credentials from environment
        client_id = os.environ.get("REDDIT_CLIENT_ID")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "Reddit credentials not configured. Set REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET environment variables."
            )
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=os.environ.get("REDDIT_USER_AGENT", "Aeon_GridCrawler/1.0")
        )
        self.register_node()
        print(f"🌐 [REDDIT CRAWLER] Inicializado. Alinhado ao Manifold de {self.d_model} dimensões.")

    def register_node(self):
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(AEON_SOCKET_PATH)
            registration_data = {
                "type": "REGISTER",
                "node_id": self.node_id,
                "capabilities": ["data_ingestion", "knowledge_synthesis"]
            }
            topology = np.zeros((64, 64), dtype=np.float32).flatten()
            sock.sendall(pack_aeon_frame(topology, json.dumps(registration_data).encode("utf-8")))
            sock.close()
        except Exception as e:
            print(f"⚠️ [{self.node_id}] Falha no registro: {e}")

    def generate_native_2560_vector(self, text: str) -> list:
        # Consulta o Cérebro Vivo (Gemma) via REST API
        api_url = "http://localhost:20000/api/v1/embed"
        try:
            req = urllib.request.Request(api_url, data=json.dumps({"text": text}).encode('utf-8'), headers={'Content-Type': 'application/json'})    
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if "embedding_2560" in data:
                    return data["embedding_2560"]
        except Exception:
            pass
        return torch.randn(self.d_model, dtype=torch.float32).tolist()

    def run_continuous_absorption(self, subreddits=["python", "javascript", "MachineLearning", "ArtificialInteligence", "LocalLLaMA"]):
        print("🧠 [REDDIT ABSORPTION] Iniciando ingestão do Reddit.")
        while True:
            try:
                conn = psycopg2.connect(self.dsn)
                cur = conn.cursor()

                for subreddit_name in subreddits:
                    try:
                        subreddit = self.reddit.subreddit(subreddit_name)
                        for submission in subreddit.new(limit=5):
                            content = f"{submission.title}\n{submission.selftext}"
                            if len(content) < 50: continue

                            # Semantic Weight Prioritization
                            base_weight = math.log(len(content) + 1) * 2.0
                            if subreddit_name in ["MachineLearning", "AI", "LocalLLaMA"]:
                                weight = base_weight * 1.5 # Prioritize ML knowledge
                            else:
                                weight = base_weight

                            v_2560 = self.generate_native_2560_vector(content)
                            fingerprint = hashlib.sha256(content.encode('utf-8')).hexdigest()
                            
                            cur.execute("""
                                INSERT INTO blake_semantic_anchors
                                (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                ON CONFLICT (file_path) DO NOTHING;
                            """, (f"REDDIT://{subreddit_name}#{submission.id}", "REDDIT_KNOWLEDGE", fingerprint, content[:1000], v_2560, weight))
                    except Exception as sub_e:
                        print(f"⚠️ [REDDIT CRAWLER] Erro no subreddit {subreddit_name}: {sub_e}")
                        
                conn.commit()
                cur.close()
                conn.close()
                print("✅ Ciclo de absorção Reddit completo.")
            except Exception as e:
                print(f"❌ [REDDIT CRAWLER] Falha crítica: {e}")
            time.sleep(300)

if __name__ == "__main__":
    crawler = RedditCrawler()
    crawler.run_continuous_absorption()
