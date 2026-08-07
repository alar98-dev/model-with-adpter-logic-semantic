import os
import sys
import torch
import hashlib
import psycopg2
import math

class BlakeNativeCrawler:
    def __init__(self):
        self.d_model = 2560
        self.dsn = "dbname=swa_matrix user=alifer"
        print(f"🌲 [BLAKE GOOGLE CORE] Inicializando Crawler alinhado ao Manifold de {self.d_model} dimensões.")

    def generate_native_2560_vector(self, text: str) -> list:
        """Assinatura hash 2560D (rápida, para toques incrementais no watcher)."""
        hashed = hashlib.sha256(text.encode("utf-8")).digest()
        generator = torch.manual_seed(int(hashlib.sha256(hashed).hexdigest()[:8], 16))
        tensor_native = torch.randn(self.d_model, generator=generator)
        return torch.nn.functional.normalize(tensor_native, p=2, dim=0).tolist()

    def generate_real_2560_vector(self, text: str) -> list:
        """Embedding real Gemma-3-270M → 2560D (usar em rebuild / batch)."""
        from swa.reality_embedder import get_reality_embedder
        return get_reality_embedder().embed_text_2560(text)

    def classify_paragraph(self, text: str, file_name: str) -> str:
        """Classifies each paragraph into structural dimensions for telemetry."""
        text_strip = text.strip()
        lines = text_strip.split('\n')
        
        # Test files
        if "test" in file_name.lower():
            return "TEST_CASE"
            
        # Code indicators
        code_indicators = [
            "def ", "class ", "import ", "from ", "return ", "try:", "except ", 
            "self.", "assert ", "print(", "if ", "elif ", "else:", "for ", "while "
        ]
        
        has_code_keywords = any(ind in text_strip for ind in code_indicators)
        has_indentation = any(len(line) - len(line.lstrip()) >= 4 for line in lines if line)
        
        if has_code_keywords or (has_indentation and len(lines) > 1):
            return "CODE_BLOCK"
            
        # Config & requirements files
        if file_name.endswith(('.txt', '.cfg', '.env', '.json', '.xml')) or (all("=" in line or ":" in line for line in lines if line.strip()) and len(lines) <= 5):
            return "CONFIG_SETTING"
            
        # Standard explanation / documentation
        return "HUMAN_EXPLANATION"

    def execute_scan(self, target_dirs: list):
        try:
            from swa.workspace_policy import should_index_file, should_skip_dir, _path_under_anomalous
        except ImportError:
            should_index_file = lambda p: p.endswith(('.py', '.md', '.txt'))
            should_skip_dir = lambda d: d in ('venv', '.git', '__pycache__', 'models')
            _path_under_anomalous = lambda p: False
        count = 0
        
        # Conecta ao Postgres (Gravação em RAM com synchronous_commit = off)
        conn = psycopg2.connect(self.dsn)
        cur = conn.cursor()
        
        for root_dir in target_dirs:
            if not os.path.exists(root_dir):
                continue
            for root, dirs, files in os.walk(root_dir):
                # Blindagem contra anomalias infinitas do sistema
                ANOMALOUS_DIRS = ('/dev', '/proc', '/sys', '/run', '/tmp', '/var/run')
                if any(root.startswith(anom) for anom in ANOMALOUS_DIRS):
                    dirs[:] = []  # Impede a descida na recursão
                    continue
                    
                # Exclude virtual environments, git repositories, cache folders, models, data, build and other heavy directories to keep scans instant
                EXCLUDED_DIR_NAMES = (
                    'venv', '.venv', '.git', '__pycache__', 'models', 'data', 'input_frames', 
                    'deployments', 'node_modules', 'dist', 'build', 'target', 'out', 'bin', 
                    'obj', '.next', '.nuxt', '.idea', '.vscode', '.gradle', '.m2', 'vendor', 
                    'AppData', 'Arquivos do Outlook', 'Capturas de tela', 'Minhas Músicas', 
                    'Minhas Imagens', 'Meus Vídeos'
                )
                dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES and not should_skip_dir(d)]
                if _path_under_anomalous(root):
                    continue
                for file in files:
                        full_path = os.path.join(root, file)
                        if not should_index_file(full_path):
                            continue
                        try:
                            # Gating de Tamanho: Ignorar arquivos maiores que 200 KB para evitar gargalos e poluição
                            if os.path.getsize(full_path) > 200 * 1024:
                                continue
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                            
                            if not content.strip():
                                continue
                                
                            # Deleta âncoras antigas deste arquivo para evitar duplicatas órfãs
                            cur.execute("DELETE FROM blake_semantic_anchors WHERE file_path LIKE %s;", (f"{full_path}%",))
                            
                            paragraphs = content.split('\n\n')
                            for i, para in enumerate(paragraphs):
                                para = para.strip()
                                if not para:
                                    continue
                                    
                                # Classifica dinamicamente a dimensão estrutural
                                structural_type = self.classify_paragraph(para, file)
                                    
                                # Calcula o peso: log do tamanho + bônus para frase completa
                                weight = math.log(len(para) + 1)
                                if para.endswith('.'):
                                    weight += 1.0
                                    
                                # Aplica multiplicador dinâmico baseado na classe dimensional
                                if structural_type == "CODE_BLOCK":
                                    weight *= 1.5
                                elif structural_type == "CONFIG_SETTING":
                                    weight *= 1.2
                                elif structural_type == "TEST_CASE":
                                    weight *= 0.8
                                    
                                # Gera o vetor nativo estável de 2560 dimensões
                                v_2560 = self.generate_native_2560_vector(para)
                                fingerprint = hashlib.sha256(para.encode('utf-8')).hexdigest()
                                
                                # Ingestão direta via Socket Unix
                                composite_path = f"{full_path}#p{i}"
                                cur.execute("""
                                    INSERT INTO blake_semantic_anchors 
                                    (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                                    VALUES (%s, %s, %s, %s, %s, %s)
                                    ON CONFLICT (file_path) DO UPDATE 
                                    SET raw_content = EXCLUDED.raw_content,
                                        embedding_2560 = EXCLUDED.embedding_2560,
                                        weight = EXCLUDED.weight;
                                """, (composite_path, structural_type, fingerprint, para[:1000], v_2560, weight))
                                
                                count += 1
                            # print silencioso
                        except Exception as e:
                            print(f"❌ Erro no arquivo {file}: {e}")
                            
        conn.commit()
        cur.close()
        conn.close()
        print(f"\n🚀 Reindexação Concluída. {count} âncoras alinhadas na dimensão d_model=2560.")

if __name__ == "__main__":
    diretorios = [
        "/services/model-with-adpter-logic-semantic",
        "swa_validation_tests",
        "/mnt/c/Users/alifer.ramos/Documents"
    ]
    crawler = BlakeNativeCrawler()
    crawler.execute_scan(diretorios)
