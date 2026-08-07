import sys
import json
import urllib.request
import psycopg2

DB_CONN = "dbname='matriz_vetorial' user='postgres' password='antigravity' host='localhost'"
EMBEDDER_URL = "http://127.0.0.1:21000/embed"

def search(query):
    print(f"🔍 Colapsando onda vetorial para a intenção: '{query}'")
    print("🧠 Consultando serviço Embedder residente (Porta 21000)...")
    
    req = urllib.request.Request(EMBEDDER_URL, data=json.dumps({"text": query}).encode("utf-8"), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            embedding = res_data.get("embedding_640") 
    except Exception as e:
        print(f"❌ Erro ao contatar Embedder: {e}")
        return

    print("🌌 Consultando o pgvector (Memória Longa)...")
    with psycopg2.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
            cur.execute("""
                SELECT file_path, embedding <=> %s::vector AS distance 
                FROM project_tree_embeddings 
                ORDER BY distance ASC 
                LIMIT 5
            """, (vec_str,))
            results = cur.fetchall()
            
    print("\n🧲 [MAGNETISMO VETORIAL] Arquivos mais ressonantes com a busca:")
    for filepath, dist in results:
        print(f"   -> {filepath} (Distância Cosseno: {dist:.4f})")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "arquitetura do sistema SWA e o processamento semântico"
    search(query)
