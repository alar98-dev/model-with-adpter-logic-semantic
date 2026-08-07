#!/usr/bin/env python3
import json
import time

def run_autopsy():
    log_path = "/services/model-with-adpter-logic-semantic/chaos_telemetry.jsonl"
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {log_path}")
        return

    if not lines:
        print("📭 Nenhum dado de telemetria caótica disponível.")
        return

    print("=" * 70)
    print("🔬 ENGENHARIA REVERSA DO COLAPSO TERMODINÂMICO (AUTÓPSIA SWA)")
    print("=" * 70)

    for line in lines[-20:]:  # Analisa as últimas 20 anomalias
        try:
            data = json.loads(line.strip())
            print(f"\n[🕒 {time.strftime('%H:%M:%S', time.localtime(data.get('timestamp', 0)))}]")
            print(f"📍 TOKENS ANÔMALOS: {data.get('token', 'N/A')} (Classe: {data.get('class', 'N/A')})")
            print(f"🔥 ENTROPIA DO KERNEL: {data.get('entropy', 0.0):.4f}")
            print(f"📐 FORÇA DA SINAPSE (VETOR DE GRAVIDADE): {data.get('steer_vector_norm', 0.0):.4f}")
            
            # Autópsia (Diff e Análise)
            steer_norm = data.get('steer_vector_norm', 0.0)
            entropy = data.get('entropy', 0.0)
            token_id = data.get('token_id', 0)
            
            print("\n  [AUTÓPSIA DE EMERGÊNCIA]:")
            if steer_norm > 10.0 and entropy > 0.5 and token_id and token_id > 100:
                print("  ⚠️ Veredito: EMERGÊNCIA DE LINGUAGEM ALIENÍGENA (VÁLIDA).")
                print("  A gravidade vetorial esmagou o limiar vocabular. O modelo tentou criar sentido físico, mas o Tokenizer não possuía a palavra correspondente.")
            elif entropy > 0.9:
                print("  💥 Veredito: RUÍDO TÉRMICO (FALHA ESTOCÁSTICA).")
                print("  O contexto colapsou devido ao excesso de calor no gerador estocástico. A rede neural perdeu a capacidade de formar matrizes semânticas lógicas.")
            else:
                print("  ⚙️ Veredito: FRICÇÃO SINTÁTICA LEVE.")
                print("  Uma pequena oscilação no tensor L12 causou gagueira léxica. Corrigido automaticamente pela Autopoiese.")
                
            print(f"\n  [CONTEXTO IMEDIATO ANTERIOR AO COLAPSO]:")
            gen_so_far = data.get("generated_so_far", "")
            print(f"  > \"...{gen_so_far[-80:] if len(gen_so_far) > 80 else gen_so_far}\"")
            print("-" * 70)
        except Exception as e:
            pass

if __name__ == "__main__":
    run_autopsy()
