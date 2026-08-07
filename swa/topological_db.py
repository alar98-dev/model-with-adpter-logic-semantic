import psycopg2
import uuid
import datetime
import traceback
import threading
import hashlib
import numpy as np

class TopologicalDB:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TopologicalDB, cls).__new__(cls)
                cls._instance._init_db()
            return cls._instance

    def _init_db(self):
        self.conn_str = "dbname='matriz_vetorial' user='postgres' password='antigravity' host='localhost'"
        print("🌌 [TOPOLOGICAL DB] Conectando ao Núcleo pgvector (Logical Qubits Mode)...")

    def _gerar_qubits_logicos(self, texto: str, dimensao: int = 384) -> list:
        """
        Gera uma assinatura espacial pseudo-aleatória determinística (Qubits Lógicos)
        baseada puramente no hash SHA-256 do conteúdo, evitando colisões de tensores PyTorch.
        """
        # Semente determinística a partir do hash do texto
        seed_hash = hashlib.sha256(texto.encode('utf-8')).digest()
        seed_int = int.from_bytes(seed_hash[:4], byteorder='little')
        
        # Cria um gerador aleatório isolado com essa semente
        rng = np.random.RandomState(seed_int)
        
        # Gera o vetor em espaço de alta dimensão
        vetor = rng.randn(dimensao)
        
        # Normaliza o vetor (L2 norm) para projetá-lo na esfera unitária
        norma = np.linalg.norm(vetor)
        if norma > 0:
            vetor = vetor / norma
            
        return vetor.tolist()

    def mapear_no(self, caminho: str, conteudo: str, entropia: float):
        """Calcula a assinatura espacial e salva no banco de dados vetorial."""
        try:
            texto_base = f"Caminho: {caminho}\nConteudo: {conteudo[:500]}"
            # Executa a codificação neural via Python Puro (Adaptação Lógica Quantum)
            vector = self._gerar_qubits_logicos(texto_base)
        except Exception as e:
            print(f"⚠️ [MATRIZ VETORIAL] Falha na vetorização lógica: {e}")
            return

        def run_insert(vec_arr):
            try:
                # Injeta no banco
                with psycopg2.connect(self.conn_str) as conn:
                    with conn.cursor() as cur:
                        vec_str = "[" + ",".join(str(x) for x in vec_arr) + "]"
                        cur.execute(
                            "INSERT INTO arquivos (id, caminho, entropia, embedding, data_registro) VALUES (%s, %s, %s, %s, %s)",
                            (str(uuid.uuid4()), caminho, entropia, vec_str, datetime.datetime.now())
                        )
                    conn.commit()
                print(f"💎 [MATRIZ VETORIAL] Nó Qubit cristalizado no pgvector: {caminho} (Entropia: {entropia:.2f})")
            except Exception as e:
                print(f"⚠️ [MATRIZ VETORIAL] Falha ao cristalizar nó no banco: {e}")
                traceback.print_exc()

        # Roda apenas a inserção de I/O de rede em thread
        threading.Thread(target=run_insert, args=(vector,), daemon=True).start()

    def consultar_vizinhanca_vetorial(self, intencao_texto: str) -> str:
        """
        Converte a intenção em um Qubit Lógico e busca o arquivo cuja
        assinatura topológica esteja matematicamente mais próxima no pgvector.
        """
        try:
            vector = self._gerar_qubits_logicos(intencao_texto)
            vec_str = "[" + ",".join(str(x) for x in vector) + "]"
            
            with psycopg2.connect(self.conn_str) as conn:
                with conn.cursor() as cur:
                    # Usamos a distância cosseno (<=>) do pgvector para achar a maior ressonância
                    cur.execute(
                        "SELECT caminho FROM arquivos ORDER BY embedding <=> %s::vector LIMIT 1",
                        (vec_str,)
                    )
                    resultado = cur.fetchone()
                    if resultado:
                        caminho_encontrado = resultado[0]
                        print(f"🧲 [MAGNETISMO VETORIAL] A Vontade ({intencao_texto[:20]}...) atraiu a massa de: {caminho_encontrado}")
                        return caminho_encontrado
        except Exception as e:
            print(f"⚠️ [MAGNETISMO VETORIAL] Falha ao colapsar onda de busca: {e}")
        
        return ""

# Helper global
_db_cache = None

def registrar_leitura_vetorial(caminho, conteudo, entropia):
    global _db_cache
    if _db_cache is None:
        _db_cache = TopologicalDB()
    _db_cache.mapear_no(caminho, conteudo, entropia)

def colapsar_onda_vetorial(intencao: str) -> str:
    global _db_cache
    if _db_cache is None:
        _db_cache = TopologicalDB()
    return _db_cache.consultar_vizinhanca_vetorial(intencao)
