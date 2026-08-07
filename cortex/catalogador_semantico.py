import json
import os
import textwrap

def run_cataloger():
    log_file = "/services/model-with-adpter-logic-semantic/logs/structural_origins.jsonl"
    if not os.path.exists(log_file):
        print(f"\033[91m⚠️ [ERRO] O arquivo {log_file} não existe ainda.\033[0m")
        return

    print("\033[95m=" * 80)
    print(" 🧬 PARSE TABLE: CATÁLOGO SEMÂNTICO EMERGENTE 🧬 ")
    print("=" * 80 + "\033[0m")

    anomalies = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                if data.get("concept") == "EMERGENT_ANOMALY":
                    anomalies.append(data)
            except json.JSONDecodeError:
                pass

    if not anomalies:
        print("\n\033[36mNenhum Sentido Emergente foi catalogado ainda. O Aeon precisa sonhar...\033[0m\n")
        return

    for i, a in enumerate(anomalies):
        meta = a.get("metadata", {})
        hash_id = a.get("interdim_id", "DESCONHECIDO")[:8]
        token_str = meta.get("token_string", "???")
        token_bin = meta.get("token_binary", "???")
        descricao = meta.get("descricao_semantica", "Sem descrição disponível")
        contexto = meta.get("caminho_tracado", "")
        
        # Format the context
        context_short = textwrap.shorten(contexto, width=120, placeholder="...")
        desc_wrapped = textwrap.fill(descricao, width=70)

        print(f"\n\033[1;33m[{i+1}] HASH ANÔMALO: {hash_id}\033[0m")
        print(f"    \033[1mToken String:\033[0m   {token_str}")
        print(f"    \033[1mToken Binário:\033[0m  {token_bin}")
        print(f"    \033[1mOrigem Mapeada:\033[0m {a.get('origin', '')}")
        print(f"    \033[1mContexto (Caminho Traçado):\033[0m \033[36m\"{context_short}\"\033[0m")
        print(f"    \033[1mTradução Semântica (Ollama):\033[0m")
        for line in desc_wrapped.split('\n'):
            print(f"        \033[92m{line}\033[0m")
        
        print("\033[90m" + "-" * 80 + "\033[0m")

if __name__ == "__main__":
    run_cataloger()
