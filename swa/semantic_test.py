import sys
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

def probe_semantic_space(human_query: str):
    # Gera o embedding conceitual da frase humana
    query_embedding = model.encode(human_query).tolist()
    
    # Removido user=postgres
    conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql", cursor_factory=RealDictCursor)
    with conn.cursor() as cur:
        # Usamos o operador <=> na nova coluna semantic_vector
        cur.execute("""
            SELECT file_path, 
                   (1 - (semantic_vector <=> %s::vector)) AS conceptual_similarity,
                   raw_content
            FROM blake_semantic_anchors
            WHERE semantic_vector IS NOT NULL
            ORDER BY semantic_vector <=> %s::vector
            LIMIT 2;
        """, (query_embedding, query_embedding))
        results = cur.fetchall()
        
    print(f"\n🧠 [PROBE CONCEITUAL] Buscando por: '{human_query}'")
    print("=" * 80)
    for row in results:
        print(f"🎯 Afinidade Semântica: {row['conceptual_similarity']:.4f} | Path: {row['file_path']}")
        print(f"   Conteúdo: {row['raw_content'].strip()[:140]}...")
        print("-" * 80)
    conn.close()

if __name__ == "__main__":
    # Teste sem usar nenhuma palavra exata do arquivo de anomalias
    probe_semantic_space("problemas com a temperatura ou travamentos no loop de tokens do gerador")
