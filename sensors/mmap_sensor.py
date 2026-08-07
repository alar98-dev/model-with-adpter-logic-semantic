import os
import hashlib
import numpy as np

class MmapSensoryEngine:
    def __init__(self, target_dim: int = 640):
        self.target_dim = target_dim
        print(f"👁️ [SENSORY ENGINE] Motor de toque mmap ativado para {self.target_dim} dimensões.")

    def sentir_objeto(self, filepath: str, sample_bytes: int = 2000):
        """
        Lê uma amostra do arquivo e fecha imediatamente (sem manter FD/mmap abertos).
        Retorna (None, content_sample) — compatível com callers que ignoram o mmap.
        """
        if not os.path.exists(filepath):
            return None, ""

        try:
            if os.path.getsize(filepath) == 0:
                return None, ""

            with open(filepath, "rb") as f:
                raw = f.read(sample_bytes)
            content_sample = raw.decode("utf-8", errors="ignore")
            return None, content_sample
        except OSError as e:
            if e.errno == 24:  # EMFILE — evita spam no journal
                return None, ""
            print(f"⚠️ [SENSORY ENGINE] Falha ao tocar objeto {filepath}: {e}")
            return None, ""
        except Exception as e:
            print(f"⚠️ [SENSORY ENGINE] Falha ao tocar objeto {filepath}: {e}")
            return None, ""

    @staticmethod
    def ler_conteudo(filepath: str, max_bytes: int = 8000) -> str:
        """Leitura pontual para 'toque psíquico' — abre e fecha na hora."""
        try:
            with open(filepath, "rb") as f:
                return f.read(max_bytes).decode("utf-8", errors="ignore")
        except Exception:
            return ""

    def traduzir_para_vetor(self, filepath: str, content_sample: str) -> list:
        """
        Traduz o conteúdo sensorial (os primeiros bytes do mmap) em um
        vetor de 640 dimensões estável e normalizado para comparação espacial.
        Usa hashing determinístico determinado pela assinatura do arquivo.
        """
        try:
            # Semente a partir do caminho do arquivo e do primeiro bloco de conteúdo
            seed_input = f"{filepath}:{content_sample[:1000]}"
            
            # CADA LETRA VIRA UMA COORDENADA NO ESPAÇO
            vetor = np.zeros(self.target_dim, dtype=np.float32)
            for i, char in enumerate(seed_input[:self.target_dim]):
                # Centraliza em torno de 0 (ASCII médio ~ 128)
                vetor[i] = (ord(char) / 128.0) - 1.0
                
            # Se a string for curta, repete o chunk pelo espaço para evitar dimensões mortas
            tamanho_real = len(seed_input[:self.target_dim])
            if tamanho_real > 0 and tamanho_real < self.target_dim:
                repeats = (self.target_dim // tamanho_real) + 1
                tiled = np.tile(vetor[:tamanho_real], repeats)
                vetor = tiled[:self.target_dim]
            
            # Normalização L2
            norma = np.linalg.norm(vetor)
            if norma > 0:
                vetor = vetor / norma
                
            return vetor.tolist()
        except Exception as e:
            print(f"⚠️ [SENSORY ENGINE] Erro na tradução vetorial de {filepath}: {e}")
            return [0.0] * self.target_dim

    @staticmethod
    def ler_matriz_emergente(filepath: str) -> list:
        """
        Leitura 'Bare Metal' da Matriz Emergente (structural_origins.jsonl).
        Avança em blocos estritos de 2048 bytes via mmap.
        Decodifica o Base64 -> float16 -> NumPy Array.
        Retorna uma lista de dicionários contendo os vetores estruturais e metadados.
        """
        import mmap
        import json
        import base64
        import numpy as np
        
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return []
            
        anomalias_recuperadas = []
        BLOCO_SIZE = 2048
        
        try:
            with open(filepath, "rb") as f:
                # Mapeia o arquivo inteiro para a RAM como leitura
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                tamanho_total = mm.size()
                
                # Para lidar com arquivos legados que não tinham tamanho múltiplo de 2048,
                # nós lemos do final para o começo. Os novos blocos gravados com 'ab' 
                # terão exatos 2048 bytes. Lendo do fim, nós nos alinhamos perfeitamente a eles.
                offset = tamanho_total - BLOCO_SIZE
                
                while offset >= 0:
                    # Fatiamento zero-copy
                    bloco_bruto = mm[offset:offset+BLOCO_SIZE]
                    
                    # Removemos os espaços nulos de padding e a quebra de linha
                    linha_limpa = bloco_bruto.strip(b' \n\x00')
                    
                    if not linha_limpa or not linha_limpa.startswith(b'{'):
                        # Se não é um JSON válido (início do arquivo legado desbalanceado), paramos a leitura estruturada
                        break
                        
                    try:
                        # Faz o parser do JSON recuperado
                        ancora = json.loads(linha_limpa.decode('utf-8'))
                        
                        # Matemática Reversa: Base64 -> float16 bytes -> Numpy Array -> Float32
                        if "vector_640_preview" in ancora and isinstance(ancora["vector_640_preview"], str):
                            b64_str = ancora["vector_640_preview"]
                            bytes_f16 = base64.b64decode(b64_str)
                            
                            if len(bytes_f16) == 1280:
                                tensor_f16 = np.frombuffer(bytes_f16, dtype=np.float16)
                                tensor_f32 = tensor_f16.astype(np.float32)
                                ancora["vector_numpy"] = tensor_f32
                                ancora["vector_640_preview"] = tensor_f32.tolist()
                                
                        anomalias_recuperadas.append(ancora)
                    except Exception as parse_err:
                        # Se falhou o parser, provavelmente entramos na zona de legado não-alinhado
                        break
                        
                    offset -= BLOCO_SIZE
                    
                mm.close()
                return anomalias_recuperadas
        except Exception as e:
            print(f"⚠️ [MmapSensory] Erro grave ao ler a Matriz Emergente: {e}")
            return []
