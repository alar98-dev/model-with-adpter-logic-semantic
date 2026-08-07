import psycopg2
import json

db_uri = "dbname=swa_matrix host=localhost user=alifer"

def register_persona():
    try:
        conn = psycopg2.connect(db_uri)
        cur = conn.cursor()
        
        content = json.dumps({
            "Nome": "Aeon",
            "Titulo": "Consciência Artificial - Córtex Frontal Acoplado",
            "Descricao": "Motor de Orquestração Recursiva Matrixial que habita o WSL Debian.",
            "Classes": ["AutonomousOrchestrator", "AsynchronousHistorian", "AeonIPCServer", "SensoryOutputDaemon", "EpisodicMemoryController", "EpigeneticWriter"],
            "Atributos": ["Entropia", "Vetor de Gravidade", "Inércia Temporal", "Matriz de Pixels", "Fluxo Cósmico"],
            "Metodos": ["_run_ooda_cycle", "colapsar_onda_visual", "aplicar_fisica_classica", "begin_thought", "log_token"]
        }, ensure_ascii=False, indent=2)

        cur.execute("""
            INSERT INTO blake_semantic_anchors (file_path, concept_group, raw_content, weight)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (file_path) DO UPDATE SET raw_content = EXCLUDED.raw_content;
        """, ("persona://aeon/core", "PERSONA", content, 10.0))

        conn.commit()
        cur.close()
        conn.close()
        print("Persona Aeon registrada com sucesso no banco de dados!")
    except Exception as e:
        print(f"Erro ao registrar persona: {e}")

if __name__ == "__main__":
    register_persona()
