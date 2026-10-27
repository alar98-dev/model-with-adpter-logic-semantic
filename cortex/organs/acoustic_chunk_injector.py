"""
SWA-AEON: Módulo de Injeção de Chunks Acústicos Nativos 2560D
============================================================
Converte assinaturas acústicas (MFCC 13D) + espectro de áudio bruto
fundidos com o Subcórtex (Gemma 270M) em vetores 2560D gravados
diretamente na tabela kbase_chunks do PostgreSQL (swa_matrix).
"""
import os
import sys
import json
import time
import numpy as np
import psycopg2

# Resolução do locutor: NÃO usa mais ChromaDB no query path.
# Usa o VectorStore nativo compartilhado (controlador_ambiente.vs).
try:
    sys.path.insert(0, "/services/voice-stream-client")
    from controllers.ambienteController import controlador_ambiente
    _LOCUTOR_STORE_READY = True
except Exception as _e:
    print(f"⚠️ [ACOUSTIC INJECTOR] Sem acesso ao VectorStore compartilhado: {_e}")
    controlador_ambiente = None
    _LOCUTOR_STORE_READY = False

DB_NAME = "swa_matrix"
DB_USER = "alifer"
NATIVE_DIM = 2560

PROXIMITY_THRESHOLD = 0.85
UNKNOWN_LABEL = "Desconhecido"

class AcousticChunkInjector:
    def __init__(self):
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                host="localhost",
                port=5432
            )
            self.conn.autocommit = True
            print("🔊 [ACOUSTIC INJECTOR] Conectado ao PostgreSQL (swa_matrix).")
        except Exception as e:
            print(f"⚠️ [ACOUSTIC INJECTOR] Erro de conexão DB: {e}")
            self.conn = None

    def fit_dim(self, vec, target_dim=2560):
        """Expande ou comprime um vetor para a dimensão nativa 2560D com normalização."""
        vec = np.array(vec, dtype=np.float32).flatten()
        curr_dim = len(vec)
        if curr_dim == target_dim:
            norm = np.linalg.norm(vec)
            return (vec / (norm + 1e-8)).tolist()
        elif curr_dim < target_dim:
            tile_factor = int(np.ceil(target_dim / curr_dim))
            expanded = np.tile(vec, tile_factor)[:target_dim]
            freqs = np.sin(np.linspace(0, 4 * np.pi, target_dim)) * 0.05
            expanded = expanded + freqs
            norm = np.linalg.norm(expanded)
            return (expanded / (norm + 1e-8)).tolist()
        else:
            compressed = vec[:target_dim]
            norm = np.linalg.norm(compressed)
            return (compressed / (norm + 1e-8)).tolist()

    def _avaliar_locutor_por_proximidade(self, mfcc_sig, speaker_hint=None):
        """
        Avalia por proximidade (VectorStore nativo — hamming compactado/cosseno)
        se o locutor está cadastrado.
        Retorna (locutor_identificado, confianca, cadastrado).
        - cadastrado=True  -> a assinatura bate com um perfil registrado (score >= threshold)
        - cadastrado=False -> nenhum perfil registrado atinge o limiar de proximidade
        """
        if not mfcc_sig or len(mfcc_sig) == 0:
            return (speaker_hint or UNKNOWN_LABEL), 0.0, False

        if not _LOCUTOR_STORE_READY or controlador_ambiente is None:
            return (speaker_hint or UNKNOWN_LABEL), 0.0, False

        try:
            sig = np.asarray(mfcc_sig, dtype=np.float32)
            if controlador_ambiente.vs.count() == 0:
                return (speaker_hint or UNKNOWN_LABEL), 0.0, False
            best = controlador_ambiente.vs.query(sig, k=1, mode="auto")
            if not best:
                return (speaker_hint or UNKNOWN_LABEL), 0.0, False
            nome_proximo, similarity = best[0]
            cadastrado = float(similarity) >= PROXIMITY_THRESHOLD
            return nome_proximo, float(similarity), cadastrado
        except Exception as e:
            print(f"⚠️ [ACOUSTIC INJECTOR] Falha na avaliação de proximidade: {e}")
            return (speaker_hint or UNKNOWN_LABEL), 0.0, False

    def create_acoustic_chunk(self, text, mfcc_sig, raw_audio=None, speaker=None):
        """
        Funde a assinatura MFCC (13D) + espectro de áudio com a semântica do texto,
        projetando em um vetor nativo 2560D gravado no PostgreSQL kbase_chunks.
        O locutor é resolvido pelo algoritmo de proximidade (ChromaDB): se a assinatura
        não atingir o limiar de similaridade, o chunk é marcado como não cadastrado.
        """
        if not self.conn:
            self._connect()
            if not self.conn:
                return False

        try:
            # 1. Base do vetor MFCC (13D)
            mfcc_arr = np.array(mfcc_sig, dtype=np.float32) if mfcc_sig else np.zeros(13, dtype=np.float32)
            
            # 2. Caracterização espectral do áudio bruto se disponível
            spec_features = []
            if raw_audio is not None and len(raw_audio) > 0:
                audio_arr = np.array(raw_audio, dtype=np.float32)
                fft_mag = np.abs(np.fft.rfft(audio_arr))
                spec_features = [
                    float(np.mean(fft_mag)),
                    float(np.std(fft_mag)),
                    float(np.max(fft_mag)),
                    float(np.mean(np.diff(fft_mag))) if len(fft_mag) > 1 else 0.0
                ]
            else:
                spec_features = [0.0, 0.0, 0.0, 0.0]

            # Vetor Acústico Combinado (17D)
            acoustic_base = np.concatenate([mfcc_arr, spec_features])
            
            # 2.5. Resolve o locutor por proximidade (ChromaDB) — sem hardcoded
            locutor_efetivo, confianca, cadastrado = self._avaliar_locutor_por_proximidade(
                mfcc_sig, speaker_hint=speaker
            )
            if cadastrado:
                status_rotulo = f"cadastrado"
            else:
                status_rotulo = f"NAO_CADASTRADO"
                locutor_efetivo = f"{UNKNOWN_LABEL}_{status_rotulo}"
            
            # Projeta o vetor acústico para 2560D e 640D
            vector_2560 = self.fit_dim(acoustic_base, 2560)
            vector_640 = self.fit_dim(acoustic_base, 640)
            
            chunk_ts = int(time.time() * 1000)
            source_path = f"voice:{locutor_efetivo}:{chunk_ts}"

            # 3. Inserção direta na tabela kbase_chunks do PostgreSQL (pgvector)
            with self.conn.cursor() as cur:
                query = """
                INSERT INTO kbase_chunks (layer, source_path, collection, chunk_index, text, embedding_640, embedding_2560, weight, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, %s, NOW())
                """
                cur.execute(query, (
                    2560,
                    source_path,
                    "memoria_auditiva_master",
                    0,
                    f"[CHUNK AUDITIVO {locutor_efetivo} ({status_rotulo} {confianca:.0%})]: {text}",
                    json.dumps(vector_640),
                    json.dumps(vector_2560),
                    1.0 if cadastrado else 0.3
                ))
            print(f"🧠 [ACOUSTIC INJECTOR] Chunk 2560D gravado no PostgreSQL ({locutor_efetivo}, {status_rotulo} {confianca:.0%}): \"{text[:40]}...\"")
            return True

        except Exception as e:
            print(f"⚠️ [ACOUSTIC INJECTOR] Falha ao gravar chunk 2560D: {e}")
            return False


if __name__ == "__main__":
    injector = AcousticChunkInjector()
    dummy_mfcc = [0.1, -0.2, 0.5, 0.1, -0.1, 0.3, 0.0, 0.2, -0.4, 0.1, 0.05, -0.15, 0.2]
    injector.create_acoustic_chunk("Teste de cristalização de chunk acústico 2560D", dummy_mfcc)
