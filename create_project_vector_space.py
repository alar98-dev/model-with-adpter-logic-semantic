import os
import torch
import psycopg2
import math
import uuid
import datetime
from transformers import AutoModelForCausalLM, AutoProcessor

# Bypass OOM no Intel GPU (código legado do projeto SWA)
if hasattr(torch, "xpu"):
    def patched_mem_get_info(device=None):
        return (256 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
    torch.xpu.mem_get_info = patched_mem_get_info

MODEL_PATH = "/services/model-with-adpter-logic-semantic/models/gemma-3-270m"
DB_CONN = "dbname='matriz_vetorial' user='postgres' password='antigravity' host='localhost'"

def setup_db(dimension):
    print(f"📦 Inicializando tabela vetorial no pgvector com dimensao {dimension}...")
    with psycopg2.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS project_tree_embeddings (
                    id UUID PRIMARY KEY,
                    file_path TEXT UNIQUE,
                    content_chunk TEXT,
                    embedding vector({dimension}),
                    created_at TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS project_tree_idx ON project_tree_embeddings USING hnsw (embedding vector_cosine_ops);
            """)
        conn.commit()

def calculate_entropy(text):
    if not text:
        return 0.0
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum([p * math.log(p) / math.log(2.0) for p in prob])

def main():
    print("🧠 Carregando Modelo Gemma 3 para Visão Mental do Projeto...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16
    )
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    model.to(device)
    
    hidden_size = model.config.hidden_size
    setup_db(hidden_size)

    print("🌳 Iniciando mapeamento universal do sistema (Linux -> Windows)...")
    
    ignored_dirs = {'venv', '.git', '__pycache__', 'models', '.kilo', 'data'}
    anomalies = ('/proc', '/sys', '/dev', '/run')
    
    scan_phases = [
        {"root": "/", "ignore_mnt": True},
        {"root": "/mnt", "ignore_mnt": False}
    ]
    
    for phase in scan_phases:
        for root, dirs, files in os.walk(phase["root"]):
            # Tratamento de Anomalias Vetoriais
            if root.startswith(anomalies):
                print(f"🌌 [ANOMALIA VETORIAL] Singularidade física detectada em: {root}")
                dirs[:] = [] # Interrompe a descida infinita na anomalia
                files = [] # Limpa arquivos para não lê-los fisicamente
                
                try:
                    text_input = f"[ANOMALIA VETORIAL] Colapso de sistema e hardware detectado na singularidade: {root}"
                    inputs = processor(text=text_input, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(**inputs, output_hidden_states=True)
                        embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()
                    
                    with psycopg2.connect(DB_CONN) as conn:
                        with conn.cursor() as cur:
                            vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                            cur.execute(
                                """
                                INSERT INTO project_tree_embeddings (id, file_path, content_chunk, embedding, created_at)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (file_path) DO NOTHING;
                                """,
                                (str(uuid.uuid4()), root, text_input, vec_str, datetime.datetime.now())
                            )
                        conn.commit()
                except Exception as e:
                    pass
                continue
                
            # Filtro de prioridade (Linux primeiro, MNT depois)
            if phase["ignore_mnt"] and root == "/":
                if 'mnt' in dirs:
                    dirs.remove('mnt')
                    
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(('.py', '.md', '.txt', '.json', '.sh')):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        text_input = f"Arquivo: {file_path}\nConteudo:\n{content[:2000]}"
                        inputs = processor(text=text_input, return_tensors="pt").to(device)
                        
                        with torch.no_grad():
                            outputs = model(**inputs, output_hidden_states=True)
                            embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().tolist()
                            
                        with psycopg2.connect(DB_CONN) as conn:
                            with conn.cursor() as cur:
                                vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
                                cur.execute(
                                    """
                                    INSERT INTO project_tree_embeddings (id, file_path, content_chunk, embedding, created_at)
                                    VALUES (%s, %s, %s, %s, %s)
                                    ON CONFLICT (file_path) DO UPDATE 
                                    SET content_chunk = EXCLUDED.content_chunk, embedding = EXCLUDED.embedding, created_at = EXCLUDED.created_at;
                                    """,
                                    (str(uuid.uuid4()), file_path, content[:2000], vec_str, datetime.datetime.now())
                                )
                            conn.commit()
                        print(f"✅ Cristalizado no pgvector: {file_path}")
    
                    except Exception as e:
                        print(f"❌ Erro ao processar {file_path}: {e}")

if __name__ == "__main__":
    main()
