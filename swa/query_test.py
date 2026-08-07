import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import torch

def generate_query_vector(query_text: str) -> list:
    """Gera o mesmo espaço geométrico de 384 dimensões do crawler."""
    hashed = hashlib.sha256(query_text.encode('utf-8')).digest()
    tensor_base = torch.randn(384)
    for i in range(384):
        tensor_base[i] += hashed[i % len(hashed)]
    vector_normalized = torch.nn.functional.normalize(tensor_base, p=2, dim=0)
    return vector_normalized.tolist()

def evaluate_latent_weights(probe_text: str):
    """Executa a busca por similaridade de cosseno no Postgres."""
    v_query = generate_query_vector(probe_text)
    
    # Conexão direta via Socket Unix (removido user=postgres para usar o usuário atual alifer)
    conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql", cursor_factory=RealDictCursor)
    with conn.cursor() as cur:
        # Usando a operação de distância de cosseno do pgvector (1 - (<=>)) para obter a similaridade
        cur.execute("""
            SELECT file_path, concept_group, raw_content,
                   (1 - (embedding_384 <=> %s::vector)) AS semantic_proximity
            FROM blake_semantic_anchors
            ORDER BY embedding_384 <=> %s::vector
            LIMIT 3;
        """, (v_query, v_query))
        
        results = cur.fetchall()
        
    print(f"\n📡 [PROBE] Investigando Ativação Latente para: '{probe_text}'")
    print("-" * 80)
    for row in results:
        print(f"🔹 Proximidade: {row['semantic_proximity']:.4f} | Grupo: {row['concept_group']:<28} | Path: {row['file_path']}")
        print(f"   Trecho: {row['raw_content'].strip()[:120]}...")
        print("-" * 80)
    conn.close()

if __name__ == "__main__":
    # Teste 1: Forçando ativação no subsistema de segurança
    evaluate_latent_weights("EMBED_STREAM_BLAKE_ALGORITHM sha256 security key")
    
    # Teste 2: Forçando ativação na persistência e latência
    evaluate_latent_weights("PostgreSQL Unix Socket connection stability and latency target")
