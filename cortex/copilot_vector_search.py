import os
import sys
import torch
import psycopg2
import json
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_PATH = "/services/model-with-adpter-logic-semantic/models/gemma-3-270m"
_DB_PASS = os.environ.get("SWA_DB_PASSWORD", "antigravity")
DB_CONN = f"dbname='matriz_vetorial' user='postgres' password='{_DB_PASS}' host='localhost'"

if hasattr(torch, "xpu"):
    def patched_mem_get_info(device=None):
        return (256 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
    torch.xpu.mem_get_info = patched_mem_get_info

def main():
    if len(sys.argv) < 2:
        return
    query = sys.argv[1]
    
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    model.to(device)
    
    inputs = processor(text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embedding = hidden_states.mean(dim=1).squeeze().tolist()
        
    vec_str = "[" + ",".join(str(x) for x in embedding) + "]"
    
    with psycopg2.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content_chunk, file_path 
                FROM project_tree_embeddings 
                ORDER BY embedding <=> %s::vector ASC 
                LIMIT 3
            """, (vec_str,))
            results = cur.fetchall()
            
    combined_text = []
    for chunk, filepath in results:
        if chunk:
            combined_text.append(f"--- Fonte: {filepath} ---\n{chunk.strip()}")
            
    print("\n\n".join(combined_text))

if __name__ == "__main__":
    main()
