import sys
import time
import torch
import psycopg2
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SWAWeightMutator(FileSystemEventHandler):
    def __init__(self, dsn="dbname=swa_matrix user=alifer"):
        try:
            self.conn = psycopg2.connect(dsn)
            self.cursor = self.conn.cursor()
            print("📡 [BROADCAST MUTATOR] Canal de escuta acoplado ao FileSystem.")
        except Exception as e:
            print(f"❌ Falha ao conectar ao Postgres: {e}")
            sys.exit(1)

    def on_modified(self, event):
        if event.is_directory or not event.src_path.endswith('.py'):
            return
            
        print(f"⚡ [ALTERAÇÃO FÍSICA] Detectada em: {event.src_path}")
        try:
            # Simula a leitura da nova massa binária e gera um ruído de decaimento térmico
            with open(event.src_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Mutação matemática: Cria um vetor de perturbação dinâmico baseado na massa do arquivo
            seed = len(content) % 1000
            torch.manual_seed(seed)
            mutated_tensor = torch.randn(2560, dtype=torch.float32) * 0.05
            vec_list = mutated_tensor.numpy().tolist()

            # Transmite o sinal via atualização direta na tabela blake_semantic_anchors
            query = """
                UPDATE blake_semantic_anchors 
                SET embedding_2560 = embedding_2560 + %s::vector
                WHERE file_path = %s;
            """
            self.cursor.execute(query, (vec_list, event.src_path))
            self.conn.commit()
            print(f"🌀 [BROADCAST EVENTUAL] Massa vetorial atualizada no Postgres para equilibrar torque.")
            
        except Exception as e:
            self.conn.rollback()
            print(f"❌ Falha na transmissão do sinal de broadcast: {e}")

if __name__ == "__main__":
    path = "/services/model-with-adpter-logic-semantic"
    event_handler = SWAWeightMutator()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"👀 Monitorando alterações em {path}...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
