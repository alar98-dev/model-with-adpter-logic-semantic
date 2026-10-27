import psycopg2
from psycopg2.extras import RealDictCursor
import hashlib
import time
import random
from typing import Dict, List, Any, Optional
import psutil
import numpy as np
import torch



class BaseSWAController:
    """Master controller class for vector state management in PostgreSQL/RAM.

    Implements a robust dual-mode design:
    - Main: Leverages PostgreSQL and pgvector for persistent relational vector searches.
    - Fallback: Uses a static in-memory dictionary database (_RAM_DATABASE) with
      NumPy-backed cosine similarity calculations when PostgreSQL is unreachable.
    """

    # Static in-memory database acting as full database fallback
    _RAM_DATABASE: Dict[str, List[Dict[str, Any]]] = {
        "blake_semantic_anchors": [],
        "swa_episodic_episodes": [],
        "swa_cpu_scheduler_logs": [],
        "swa_cpu_affinity_logs": [],
        "swa_memory_page_logs": [],
        "matrix_system_state": [],
        "swa_eventual_embeddings": [],
    }

    def __init__(self, db_uri: str = "dbname=swa_matrix host=/var/run/postgresql user=alifer"):
        self.db_uri = db_uri
        self._db_active: Optional[bool] = None

    def _get_connection(self):
        """Attempts to connect to PostgreSQL.

        Sets internal flag and raises standard OperationalError if unavailable.
        """
        try:
            conn = psycopg2.connect(self.db_uri, cursor_factory=RealDictCursor)
            self._db_active = True
            return conn
        except Exception as e:
            self._db_active = False
            raise psycopg2.OperationalError(f"Database connection failed: {e}")

    def is_db_active(self) -> bool:
        """Verifies PostgreSQL connectivity dynamically."""
        if self._db_active is None:
            try:
                conn = self._get_connection()
                conn.close()
                self._db_active = True
            except Exception:
                self._db_active = False
        return self._db_active

    def compute_state_hash(self, data_str: str) -> str:
        """Computes the SWA_HASH_MASTER for protection against identity loss."""
        return hashlib.sha256(data_str.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """NumPy-backed Cosine Similarity calculation for RAM-fallback search."""
        arr_a = np.array(a, dtype=np.float32)
        arr_b = np.array(b, dtype=np.float32)
        dot = np.dot(arr_a, arr_b)
        norm_a = np.linalg.norm(arr_a)
        norm_b = np.linalg.norm(arr_b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(dot / (norm_a * norm_b))


# ==============================================================================
# PROPOSTA EXECUTÁVEL DAS 26 CLASSES CONTROLADORAS ( MAPEAMENTO DO KERNEL )
# ==============================================================================

# --- BLOCO 1: MEMÓRIA ---


class MemoryPageController(BaseSWAController):
    """Monitors dynamic memory allocations and page indexes, logging them semantically."""

    def log_memory_pages(self) -> Dict[str, Any]:
        """Obtains dynamic host RAM pages specs and indexes the signature."""
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        used_gb = mem.used / (1024**3)
        percent = mem.percent

        sig_data = f"RAM:{total_gb:.1f}GB:{percent:.1f}%"
        signature = self.compute_state_hash(sig_data)

        payload = {
            "total_ram_gb": float(f"{total_gb:.2f}"),
            "used_ram_gb": float(f"{used_gb:.2f}"),
            "percent_used": float(percent),
            "semantic_state_signature": signature,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO swa_memory_page_logs 
                            (total_ram_gb, used_ram_gb, percent_used, semantic_state_signature)
                            VALUES (%s, %s, %s, %s);
                        """,
                            (total_gb, used_gb, percent, signature),
                        )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE["swa_memory_page_logs"].append(payload)
        return payload


class CacheManifoldController(BaseSWAController):
    """Manages high-performance CPU L1/L2/L3 cache metrics representation."""

    pass


class VirtualSwapController(BaseSWAController):
    """Manages virtual swapping cycles logs."""

    pass


class BufferPoolController(BaseSWAController):
    """Indexes semantic buffers and locks."""

    pass


class InertiaRegistryController(BaseSWAController):
    """Tracks cumulative inertia resistances."""

    pass


# --- BLOCO 2: CPU ---


class CpuAffinityController(BaseSWAController):
    """Maps physical CPU core affinity constraints into a semantic trace vector."""

    def save_affinity_state(self, core_id: int, logical_processors: List[int]) -> Dict[str, Any]:
        """Binds CPU core affinity bindings with a 2560d semantic trace."""
        proc_str = ",".join(map(str, logical_processors))

        # Generate a stable 2560d vector trace representing this affinity binding
        np.random.seed(core_id + len(logical_processors))
        trace_vector = np.random.randn(2560)
        # L2 Normalize
        trace_vector = trace_vector / np.linalg.norm(trace_vector)
        trace_list = trace_vector.tolist()

        payload = {
            "core_id": core_id,
            "logical_processors": proc_str,
            "semantic_trace": trace_list,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO swa_cpu_affinity_logs (core_id, logical_processors, semantic_trace)
                            VALUES (%s, %s, %s);
                        """,
                            (core_id, proc_str, trace_list),
                        )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE["swa_cpu_affinity_logs"].append(payload)
        return payload


class PCoreSchedulerController(BaseSWAController):
    """Tracks and registers dynamic P-Core scheduling diagnostics."""

    def log_scheduler_state(
        self,
        utilization_avg: float,
        temp_avg: float,
        active_cores: int,
        throttled_cores: int,
        summary: str,
    ) -> Dict[str, Any]:
        """Persists consolidated topology measurements semantically."""
        payload = {
            "utilization_average": utilization_avg,
            "temperature_average": temp_avg,
            "active_cores": active_cores,
            "throttled_cores": throttled_cores,
            "diagnostic_summary": summary,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO swa_cpu_scheduler_logs 
                            (utilization_average, temperature_average, active_cores, throttled_cores, diagnostic_summary)
                            VALUES (%s, %s, %s, %s, %s);
                        """,
                            (utilization_avg, temp_avg, active_cores, throttled_cores, summary),
                        )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE["swa_cpu_scheduler_logs"].append(payload)
        return payload


class IntraOpLoadController(BaseSWAController):
    """Manages PyTorch intra-op threads allocation logging."""

    pass


class QuantumClockController(BaseSWAController):
    """Tracks semantic cycle timers."""

    pass


class ThreadConcurrenceController(BaseSWAController):
    """Indices and logs thread collisions."""

    pass


# --- BLOCO 3: BARRAMENTO I/O ---


class UnixSocketController(BaseSWAController):
    """Monitors live Unix Domain Socket interactions."""

    pass


class DeviceHalController(BaseSWAController):
    """Telemetry register for physical system sensors."""

    pass


class PostgresBridgeController(BaseSWAController):
    """Validates PostgreSQL connectivity, extensions, and schema migrations."""

    def verify_schema_and_migrations(self) -> bool:
        """Ensures vector extensions are enabled and tables/columns match schema."""
        if not self.is_db_active():
            print("⚠️ PostgresBridge: DB is inactive. Verification bypassed (operating in RAM mode).")
            return False

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Enable extension
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                    # 2. Add weight column if not present in anchors
                    cur.execute(
                        """
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name='blake_semantic_anchors' AND column_name='weight';
                    """
                    )
                    if not cur.fetchone():
                        cur.execute("ALTER TABLE blake_semantic_anchors ADD COLUMN weight REAL DEFAULT 1.0;")

                    # 3. Create episodic table if not present
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_episodic_episodes (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            concept_group VARCHAR(64) NOT NULL,
                            content TEXT NOT NULL,
                            embedding_2560 vector(2560) NOT NULL,
                            weight REAL DEFAULT 1.0
                        );
                    """
                    )

                    # 4. Create CPU affinity table if not present
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_cpu_affinity_logs (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            core_id INTEGER NOT NULL,
                            logical_processors TEXT NOT NULL,
                            semantic_trace vector(2560)
                        );
                    """
                    )

                    # 5. Create CPU scheduler table if not present
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_cpu_scheduler_logs (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            utilization_average REAL NOT NULL,
                            temperature_average REAL NOT NULL,
                            active_cores INTEGER NOT NULL,
                            throttled_cores INTEGER NOT NULL,
                            diagnostic_summary TEXT
                        );
                    """
                    )

                    # 6. Create RAM page table if not present
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_memory_page_logs (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            total_ram_gb REAL NOT NULL,
                            used_ram_gb REAL NOT NULL,
                            percent_used REAL NOT NULL,
                            semantic_state_signature VARCHAR(64)
                        );
                    """
                    )

                    # 7. Create eventual embeddings relational table if not present
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_eventual_embeddings (
                            id SERIAL PRIMARY KEY,
                            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            state_id INTEGER REFERENCES matrix_system_state(id) ON DELETE CASCADE,
                            cpu_log_id INTEGER REFERENCES swa_cpu_scheduler_logs(id) ON DELETE SET NULL,
                            mem_log_id INTEGER REFERENCES swa_memory_page_logs(id) ON DELETE SET NULL,
                            eventual_bias_2560 vector(2560) NOT NULL,
                            phase_shift_angle REAL NOT NULL,
                            signature_hash VARCHAR(64) NOT NULL
                        );
                    """
                    )

                    # 8. Tabela de Referência de Erros Normalizados (Apenas para Consulta Humana)
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_error_reference (
                            error_code VARCHAR(16) PRIMARY KEY,
                            error_class VARCHAR(64) NOT NULL,
                            log_pattern TEXT NOT NULL,
                            physical_explanation TEXT NOT NULL,
                            auto_mitigation TEXT NOT NULL,
                            action_needed TEXT NOT NULL
                        );
                    """
                    )

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_projection_registry (
                            name VARCHAR(96) PRIMARY KEY,
                            version INTEGER NOT NULL DEFAULT 1,
                            source_dim INTEGER NOT NULL,
                            target_dim INTEGER NOT NULL,
                            artifact_path TEXT,
                            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        );
                        """
                    )
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS swa_narrative_events (
                            id BIGSERIAL PRIMARY KEY,
                            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            wave INTEGER NOT NULL DEFAULT 0,
                            event_type VARCHAR(32) NOT NULL,
                            summary TEXT NOT NULL,
                            entities JSONB NOT NULL DEFAULT '[]'::jsonb,
                            objectives JSONB NOT NULL DEFAULT '[]'::jsonb,
                            time_context TEXT NOT NULL DEFAULT '',
                            location TEXT NOT NULL DEFAULT ''
                        );
                        """
                    )

                    cur.execute("SELECT COUNT(*) FROM swa_error_reference;")
                    count_row = cur.fetchone()
                    count_val = count_row['count'] if isinstance(count_row, dict) else count_row[0]
                    if count_val == 0:
                        errors_to_insert = [
                            ("SWA-01", "XPU / GPU", "torch.OutOfMemoryError: XPU out of memory", 
                             "Limite de janela de alocação de memória gráfica compartilhada sob WSL2/Hyper-V na GPU integrada Intel.",
                             "Purga VRAM/RAM residual e aciona automaticamente o fallback estável para CPU de alta performance.",
                             "Nenhuma. Comportamento virtualizado padrão esperado."),
                            ("SWA-02", "XPU Driver", "Não foi possível ler vendor/device id", 
                             "WSL2 carece de privilégios ou mapeamento PCI direto para ler vendor/device id via torch.xpu.",
                             "Transiciona automaticamente para leitura de dispositivo lógico genérico Level Zero.",
                             "Nenhuma. Limitação inofensiva e tratada pelo kernel."),
                            ("SWA-03", "Hardware Bus", "Vacuum device not available (/dev/sde)", 
                             "O barramento físico ou ponto de montagem do vácuo (/dev/sde) está desconectado.",
                             "Transiciona para RAM-only mode e computa bias dinâmico por telemetria sintética.",
                             "Mapeie o USB ou ative a montagem do disco se utilizar dispositivo físico."),
                            ("SWA-04", "Thermal HAL", "psutil.sensors_temperatures() -> empty", 
                             "Falta de bindings de sensores térmicos ACPI virtuais no kernel do WSL2.",
                             "Usa valor de fallback nominal de 40.0°C ou Host API para evitar falhas no loop térmico.",
                             "Nenhuma. Evita a quebra do monitoramento de temperatura."),
                            ("SWA-05", "Cognitivo", "🌊 [DISSIPAÇÃO CRÍTICA] Saturação persistente", 
                             "Saturação de repetição de contexto (Cosine Similarity > 0.95 entre Camadas 6 e 12).",
                             "Esvazia o KV Cache, poda o histórico latente de tokens e reseta o contador de saturação para 0.",
                             "Nenhuma. Restaura instantaneamente o fluxo cognitivo de inferência."),
                            ("SWA-06", "CPU Affinity", "Afinidade de Threads Ativa: E-Cores", 
                             "Exposição de afinidade livre permitindo ao sistema relegar geração para Efficient Cores (E-Cores).",
                             "Trava a inferência estritamente nos Performance Cores [0, 1, 2, 3] e limita OpenMP a 4 threads.",
                             "Nenhuma. Restaura latência de microssegundos e performance de silício."),
                            ("SWA-07", "Database", "psycopg2.OperationalError: connection failed", 
                             "O servidor Postgres está inativo ou o banco swa_matrix não foi provisionado.",
                             "Transiciona dinamicamente para tabelas locais SQLite ou banco interno em memória RAM.",
                             "Inicie o Postgres usando 'sudo service postgresql start' para habilitar logs relacionais.")
                        ]
                        for err in errors_to_insert:
                            cur.execute(
                                """
                                INSERT INTO swa_error_reference 
                                (error_code, error_class, log_pattern, physical_explanation, auto_mitigation, action_needed)
                                VALUES (%s, %s, %s, %s, %s, %s);
                            """, err)

                conn.commit()
            print("✅ PostgresBridge: Schema migrations checked and updated successfully.")
            return True
        except Exception as e:
            print(f"❌ PostgresBridge error during migrations: {e}")
            return False


class FileCrawlerController(BaseSWAController):
    """Registers file crawls and anchors into PGVector or RAM."""

    def register_file(self, path: str, group: str, text: str, vector_384: List[float]) -> bool:
        h = self.compute_state_hash(path + text[:100])
        payload = {
            "file_path": path,
            "concept_group": group,
            "signature_hash": h,
            "raw_content": text[:500],
            "embedding_384": vector_384,
            "weight": 1.0,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO blake_semantic_anchors (file_path, concept_group, signature_hash, raw_content, embedding_384)
                            VALUES (%s, %s, %s, %s, %s) ON CONFLICT (file_path) DO NOTHING;
                        """,
                            (path, group, h, text[:500], vector_384),
                        )
                    conn.commit()
                return True
            except Exception:
                pass

        self._RAM_DATABASE["blake_semantic_anchors"].append(payload)
        return True


class MatrixStreamController(BaseSWAController):
    """Monitors live tensor weight ingestion flows."""

    pass


# --- BLOCO 4: SEGURANÇA E CRIPTOGRAFIA ---


class BlakeHashMasterController(BaseSWAController):
    """Secures structural state hashing routines."""

    pass


class FieldSteeringController(BaseSWAController):
    """Ensures steer boundaries are maintained securely."""

    pass


class HardeningShieldController(BaseSWAController):
    """Prevents system token manipulation anomalies."""

    pass


class TokenAnnihilationController(BaseSWAController):
    """Handles immediate model execution termination loops."""

    pass


class EntropyBiasController(BaseSWAController):
    """Secures entropy adjustments logs."""

    pass


# --- BLOCO 5: PERSISTÊNCIA COMPORTAMENTAL ---


class EpisodicMemoryController(BaseSWAController):
    """Cognitive controller executing semantic episodic search via PGVector or RAM fallback."""

    def store_episode(
        self, content: str = "", embedding_2560: Optional[List[float]] = None,
        concept_group: str = "GENERAL", weight: float = 1.0,
        **kwargs,
    ) -> bool:
        """Stores a structured sense episode."""
        # Also accept the split-controller's positional order and keyword
        # spelling, preserving existing callers.
        if isinstance(concept_group, (list, tuple)) and isinstance(embedding_2560, str):
            content, embedding_2560, concept_group = embedding_2560, concept_group, content
        if embedding_2560 is None:
            embedding_2560 = kwargs.pop("embedding", None)
        if embedding_2560 is None:
            raise ValueError("embedding is required")
        from swa.dimensions import to_memory
        embedding_2560 = to_memory(embedding_2560).tolist()
        payload = {
            "concept_group": concept_group,
            "content": content,
            "embedding_2560": embedding_2560,
            "weight": weight,
            "timestamp": time.time(),
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO swa_episodic_episodes (concept_group, content, embedding_2560, weight)
                            VALUES (%s, %s, %s, %s);
                        """,
                            (concept_group, content, embedding_2560, weight),
                        )
                    conn.commit()
                return True
            except Exception as exc:
                print(f"[episodic] store failed; RAM fallback: {exc}")
                pass

        self._RAM_DATABASE["swa_episodic_episodes"].append(payload)
        return True

    def query_similar_episodes(self, query_embedding: List[float], limit: int = 5,
                               categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Executes PGVector Cosine similarity semantic search with NumPy fallback."""
        from swa.dimensions import to_memory
        query_embedding = to_memory(query_embedding).tolist()
        if self.is_db_active():
            try:
                # PGVector distance operator: <=> represents Cosine Distance.
                # Cosine Similarity = 1 - Cosine Distance.
                # We sort ascending by distance to find closest matches.
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT concept_group, content, weight, (1 - (embedding_2560 <=> %s::vector)) AS similarity
                            FROM swa_episodic_episodes
                            WHERE (%s IS NULL OR concept_group = ANY(%s))
                            ORDER BY embedding_2560 <=> %s::vector ASC
                            LIMIT %s;
                        """,
                            (query_embedding, categories, categories, query_embedding, limit),
                        )
                        rows = list(cur.fetchall())
                        if categories:
                            rows = [r for r in rows if r["concept_group"] in set(categories)]
                        return rows
            except Exception as e:
                print(f"⚠️ PGVector search failed ({e}). Bypassing to RAM fallback.")

        # Fallback NumPy Cosine Similarity calculation
        episodes = self._RAM_DATABASE["swa_episodic_episodes"]
        scored_episodes = []

        for ep in episodes:
            sim = self.cosine_similarity(query_embedding, ep["embedding_2560"])
            scored_episodes.append(
                {
                    "concept_group": ep["concept_group"],
                    "content": ep["content"],
                    "weight": ep["weight"],
                    "similarity": sim,
                }
            )

        # Sort descending by similarity score
        scored_episodes.sort(key=lambda x: x["similarity"], reverse=True)
        if categories:
            allowed = set(categories)
            scored_episodes = [row for row in scored_episodes if row["concept_group"] in allowed]
        return scored_episodes[:limit]

    search_similar = query_similar_episodes


class LatentSignatureController(BaseSWAController):
    """Tracks latent drift state signatures."""

    pass


class SequenceLenController(BaseSWAController):
    """Records context window size metrics."""

    pass


class DynamicTemperatureController(BaseSWAController):
    """Records dynamic steer temperatures."""

    pass


class PhaseShifterController(BaseSWAController):
    """Stores steering phase shift event trajectories semantically."""

    def log_phase_shift(self, shift_angle: float, entropy: float, inertia: float) -> Dict[str, Any]:
        """Indexes dynamic shift parameters semantically."""
        payload = {
            "shift_angle": shift_angle,
            "entropy": entropy,
            "inertia": inertia,
            "timestamp": time.time(),
        }

        # Uses matrix_system_state table to index state
        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        state_data = f"{shift_angle}-{entropy}-{inertia}"
                        hash_master = self.compute_state_hash(state_data)
                        cur.execute(
                            """
                            INSERT INTO matrix_system_state (current_bias, cpu_entropy, mem_inertia, stream_hash_master)
                            VALUES (%s, %s, %s, %s);
                        """,
                            (shift_angle, entropy, inertia, hash_master),
                        )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE["matrix_system_state"].append(payload)
        return payload


class MatrixStateController(BaseSWAController):
    """Monitors total state updates and outputs Master identity hashes."""

    def update_hardware_state(self, bias: float, cpu: float, mem: float) -> str:
        """Gera o estado físico real no banco e retorna o Hash de Identidade."""
        state_data = f"{bias}-{cpu}-{mem}"
        hash_master = self.compute_state_hash(state_data)

        payload = {
            "current_bias": bias,
            "cpu_entropy": cpu,
            "mem_inertia": mem,
            "stream_hash_master": hash_master,
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO matrix_system_state (current_bias, cpu_entropy, mem_inertia, stream_hash_master)
                            VALUES (%s, %s, %s, %s);
                        """,
                            (bias, cpu, mem, hash_master),
                        )
                    conn.commit()
            except Exception:
                pass

        self._RAM_DATABASE["matrix_system_state"].append(payload)
        return hash_master


class EventualEmbeddingController(BaseSWAController):
    """Manages dynamic relationships between steering embeddings and hardware logs in PGVector."""

    def save_eventual_embedding_relation(
        self,
        state_id: int,
        cpu_log_id: Optional[int],
        mem_log_id: Optional[int],
        eventual_bias_2560: List[float],
        phase_shift_angle: float
    ) -> Dict[str, Any]:
        """Saves a linked cognitive steering embedding linked to hardware telemetry."""
        raw_data = f"{state_id}-{cpu_log_id}-{mem_log_id}-{phase_shift_angle}"
        sig_hash = self.compute_state_hash(raw_data)

        payload = {
            "state_id": state_id,
            "cpu_log_id": cpu_log_id,
            "mem_log_id": mem_log_id,
            "eventual_bias_2560": eventual_bias_2560,
            "phase_shift_angle": phase_shift_angle,
            "signature_hash": sig_hash,
            "timestamp": time.time(),
        }

        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO swa_eventual_embeddings 
                            (state_id, cpu_log_id, mem_log_id, eventual_bias_2560, phase_shift_angle, signature_hash)
                            VALUES (%s, %s, %s, %s, %s, %s);
                        """,
                            (state_id, cpu_log_id, mem_log_id, eventual_bias_2560, phase_shift_angle, sig_hash),
                        )
                    conn.commit()
            except Exception as e:
                print(f"⚠️ Failed to write relational embedding to DB ({e}). Saving to RAM.")

        self._RAM_DATABASE["swa_eventual_embeddings"].append(payload)
        return payload

    def query_eventual_embeddings_by_load(
        self,
        query_embedding: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Queries dynamic eventual embeddings using cosine similarity on PGVector/RAM."""
        if self.is_db_active():
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT state_id, cpu_log_id, mem_log_id, phase_shift_angle, signature_hash, 
                                   (1 - (eventual_bias_2560 <=> %s::vector)) AS similarity
                            FROM swa_eventual_embeddings
                            ORDER BY eventual_bias_2560 <=> %s::vector ASC
                            LIMIT %s;
                        """,
                            (query_embedding, query_embedding, limit),
                        )
                        return list(cur.fetchall())
            except Exception as e:
                print(f"⚠️ PGVector eventual search failed ({e}). Falling back to RAM.")

        # Fallback NumPy Cosine Similarity calculation
        records = self._RAM_DATABASE["swa_eventual_embeddings"]
        scored_records = []

        for rec in records:
            sim = self.cosine_similarity(query_embedding, rec["eventual_bias_2560"])
            scored_records.append(
                {
                    "state_id": rec["state_id"],
                    "cpu_log_id": rec["cpu_log_id"],
                    "mem_log_id": rec["mem_log_id"],
                    "phase_shift_angle": rec["phase_shift_angle"],
                    "signature_hash": rec["signature_hash"],
                    "similarity": sim,
                }
            )

        scored_records.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_records[:limit]


class DBAsyncWorker:
    """High-performance Producer-Consumer thread worker with OOM protection (drop oldest).

    Asynchronously flushes telemetry logs and high-dimensional eventual embeddings in bulk
    to PostgreSQL, preventing any inference loop jitter or database I/O blockage.
    """

    def __init__(
        self,
        state_controller: MatrixStateController,
        pcore_controller: PCoreSchedulerController,
        mem_page_controller: MemoryPageController,
        eventual_embed_controller: EventualEmbeddingController,
        max_queue_size: int = 5000,
        batch_size: int = 100,
    ):
        import queue
        self.state_controller = state_controller
        self.pcore_controller = pcore_controller
        self.mem_page_controller = mem_page_controller
        self.eventual_embed_controller = eventual_embed_controller
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.worker_thread = None
        self.last_partition_maintenance = 0.0

    def start(self):
        import threading
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def enqueue_telemetry(self, payload: Dict[str, Any]):
        """Non-blocking insertion with Drop Oldest policy when queue is full."""
        import queue
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(payload)
            except queue.Empty:
                pass

    def _worker_loop(self):
        import queue
        import psutil

        # Pin DB worker thread strictly to E-Cores (logical CPUs 4..11) to isolate P-Cores
        try:
            p = psutil.Process()
            cpu_count = psutil.cpu_count(logical=True)
            if cpu_count >= 12:
                p.cpu_affinity(list(range(4, min(12, cpu_count))))
            else:
                p.cpu_affinity(list(range(max(0, cpu_count - 4), cpu_count)))
        except Exception:
            pass

        batch = []
        last_flush_time = time.time()

        while self.running or not self.queue.empty():
            try:
                item = self.queue.get(timeout=1.0)
                batch.append(item)
                self.queue.task_done()
            except queue.Empty:
                pass

            current_time = time.time()
            if len(batch) >= self.batch_size or (current_time - last_flush_time >= 3.0 and batch):
                self._flush_batch(batch)
                batch.clear()
                last_flush_time = current_time

    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """Flushes a batch of telemetry data in a single transactional block to DB and RAM fallbacks."""
        # 0. Periodically run partition maintenance (hourly)
        current_time = time.time()
        if self.state_controller.is_db_active() and (current_time - self.last_partition_maintenance >= 3600.0 or self.last_partition_maintenance == 0.0):
            try:
                conn = self.state_controller._get_connection()
                with conn:
                    self._maintain_partitions(conn)
                conn.close()
                self.last_partition_maintenance = current_time
            except Exception as ex:
                import sys
                sys.stderr.write(f"⚠️ Table partitioning maintenance failed: {ex}\n")
                sys.stderr.flush()

        # 1. Update in-memory RAM fallback databases
        for item in batch:
            p = item["pcore"]
            m = item["mem"]
            s = item["state"]
            e = item["eventual_embedding"]

            self.pcore_controller._RAM_DATABASE["swa_cpu_scheduler_logs"].append({
                "utilization_average": p["utilization_avg"],
                "temperature_average": p["temp_avg"],
                "active_cores": p["active_cores"],
                "throttled_cores": p["throttled_cores"],
                "diagnostic_summary": p["summary"],
            })
            self.mem_page_controller._RAM_DATABASE["swa_memory_page_logs"].append({
                "total_ram_gb": m["total_ram_gb"],
                "used_ram_gb": m["used_ram_gb"],
                "percent_used": m["percent_used"],
                "semantic_state_signature": m["signature"],
            })
            self.state_controller._RAM_DATABASE["matrix_system_state"].append({
                "current_bias": s["bias"],
                "cpu_entropy": s["cpu"],
                "mem_inertia": s["mem"],
                "stream_hash_master": s["hash_master"],
            })
            self.eventual_embed_controller._RAM_DATABASE["swa_eventual_embeddings"].append({
                "state_id": len(self.state_controller._RAM_DATABASE["matrix_system_state"]),
                "cpu_log_id": len(self.pcore_controller._RAM_DATABASE["swa_cpu_scheduler_logs"]),
                "mem_log_id": len(self.mem_page_controller._RAM_DATABASE["swa_memory_page_logs"]),
                "eventual_bias_2560": e["eventual_bias_2560"],
                "phase_shift_angle": e["phase_shift_angle"],
                "signature_hash": e["sig_hash"],
                "timestamp": time.time(),
            })

        # 2. Bulk insert to active PostgreSQL in a single commit block
        if self.state_controller.is_db_active():
            try:
                conn = self.state_controller._get_connection()
                with conn:
                    with conn.cursor() as cur:
                        for item in batch:
                            p = item["pcore"]
                            cur.execute("""
                                INSERT INTO swa_cpu_scheduler_logs 
                                (utilization_average, temperature_average, active_cores, throttled_cores, diagnostic_summary)
                                VALUES (%s, %s, %s, %s, %s) RETURNING id;
                            """, (p["utilization_avg"], p["temp_avg"], p["active_cores"], p["throttled_cores"], p["summary"]))
                            cpu_log_id = cur.fetchone()["id"]

                            m = item["mem"]
                            cur.execute("""
                                INSERT INTO swa_memory_page_logs 
                                (total_ram_gb, used_ram_gb, percent_used, semantic_state_signature)
                                VALUES (%s, %s, %s, %s) RETURNING id;
                            """, (m["total_ram_gb"], m["used_ram_gb"], m["percent_used"], m["signature"]))
                            mem_log_id = cur.fetchone()["id"]

                            s = item["state"]
                            cur.execute("""
                                INSERT INTO matrix_system_state (current_bias, cpu_entropy, mem_inertia, stream_hash_master)
                                VALUES (%s, %s, %s, %s) RETURNING id;
                            """, (s["bias"], s["cpu"], s["mem"], s["hash_master"]))
                            state_id = cur.fetchone()["id"]

                            e = item["eventual_embedding"]
                            cur.execute("""
                                INSERT INTO swa_eventual_embeddings 
                                (state_id, cpu_log_id, mem_log_id, eventual_bias_2560, phase_shift_angle, signature_hash)
                                VALUES (%s, %s, %s, %s, %s, %s);
                            """, (state_id, cpu_log_id, mem_log_id, e["eventual_bias_2560"], e["phase_shift_angle"], e["sig_hash"]))
                conn.close()
            except Exception as ex:
                import sys
                sys.stderr.write(f"⚠️ Asynchronous write to PostgreSQL failed: {ex}\n")
                sys.stderr.flush()

    def _maintain_partitions(self, conn):
        """Dynamic hourly partitioning maintenance: creates future partitions and drops expired (>24h)."""
        import datetime
        now = datetime.datetime.now()

        # 1. Create partitions for current and next 2 hours
        for offset in [0, 1, 2]:
            dt = now + datetime.timedelta(hours=offset)
            partition_suffix = dt.strftime("%Y%m%d_%H")

            start_str = dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
            end_dt = dt.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")

            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS swa_cpu_scheduler_logs_p{partition_suffix}
                    PARTITION OF swa_cpu_scheduler_logs
                    FOR VALUES FROM ('{start_str}') TO ('{end_str}');
                """)
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS swa_memory_page_logs_p{partition_suffix}
                    PARTITION OF swa_memory_page_logs
                    FOR VALUES FROM ('{start_str}') TO ('{end_str}');
                """)

        # 2. Drop partitions older than 24 hours
        limit_dt = now - datetime.timedelta(hours=24)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT child.relname AS partition_name
                FROM pg_inherits
                JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
                JOIN pg_class child ON pg_inherits.inhrelid = child.oid
                WHERE parent.relname IN ('swa_cpu_scheduler_logs', 'swa_memory_page_logs');
            """)
            partitions = cur.fetchall()
            for p in partitions:
                pname = p["partition_name"]
                try:
                    # pname is like "swa_cpu_scheduler_logs_p20260519_10"
                    suffix = pname.split("_p")[-1]
                    p_dt = datetime.datetime.strptime(suffix, "%Y%m%d_%H")
                    if p_dt < limit_dt:
                        cur.execute(f"DROP TABLE IF EXISTS {pname};")
                except Exception:
                    pass

    def stop(self):
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()


class ProceduralProjector:
    """High-performance affine tensor projector utilizing PyTorch C++ back-end (addmv).

    Executes y = alpha * (W * x) + b inline to avoid memory allocation or GC triggers.
    """

    def __init__(self, input_dim: int = 32, output_dim: int = 2560, device: str = "cpu"):
        import torch
        self.device = device
        # Static allocations on initialization
        self.W = torch.randn(output_dim, input_dim, device=device)
        self.b = torch.full((output_dim,), 0.001, device=device)
        self.y_buffer = torch.zeros(output_dim, device=device)

    def project_inline(self, x_telemetry: torch.Tensor, alpha: float) -> torch.Tensor:
        """Executes inline GEMV: y = alpha * (W * x) + b."""
        import torch
        with torch.no_grad():
            torch.addmv(input=self.b, mat=self.W, vec=x_telemetry, alpha=alpha, beta=1.0, out=self.y_buffer)
            return self.y_buffer


class AsyncStructuralGeometryLogger:
    """Asynchronous, non-blocking telemetry and structural analysis logger.

    Decouples file I/O from the critical inference loop using a thread-safe queue.
    """
    def __init__(self, log_path: str = "logs/structural_geometry.log"):
        import os
        import queue
        import threading
        
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.queue = queue.Queue()
        self.running = True
        
        # Start background flusher thread
        self.worker = threading.Thread(target=self._flush_loop, daemon=True, name="SWA-AsyncLogger")
        self.worker.start()
        
        # Write header
        self.log_raw("=== INICIALIZAÇÃO DO MOTOR GEOMÉTRICO CONCORRENTE ===")
        
    def log_metrics(self, projection_time_ns: int, cosine_sim: float, is_purged: bool, temp_c: float, cpu_affinity: list):
        """Non-blocking queue push for critical thread metrics."""
        import time
        payload = {
            "timestamp": time.time(),
            "proj_ns": projection_time_ns,
            "cosine": cosine_sim,
            "purged": is_purged,
            "temp": temp_c,
            "affinity": cpu_affinity
        }
        try:
            self.queue.put_nowait(payload)
        except Exception:
            pass # Prevent queue crashes from impacting execution
            
    def log_raw(self, message: str):
        """Log a raw string message asynchronously."""
        import time
        try:
            self.queue.put_nowait({"timestamp": time.time(), "raw_msg": message})
        except Exception:
            pass
            
    def _flush_loop(self):
        import time
        import queue
        while self.running or not self.queue.empty():
            try:
                entries = []
                # Fetch all available logs up to a batch size of 100
                while len(entries) < 100:
                    try:
                        # If we have no entries, block for a short while, otherwise grab instantly
                        block = len(entries) == 0
                        entry = self.queue.get(block=block, timeout=0.1)
                        entries.append(entry)
                        self.queue.task_done()
                    except queue.Empty:
                        break
                        
                if entries:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        for entry in entries:
                            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry["timestamp"]))
                            ms = int((entry["timestamp"] - int(entry["timestamp"])) * 1000)
                            if "raw_msg" in entry:
                                f.write(f"[{ts}.{ms:03d}] {entry['raw_msg']}\n")
                            else:
                                f.write(
                                    f"[{ts}.{ms:03d}] PROJ_LATENCY: {entry['proj_ns']/1000:.2f}µs | "
                                    f"COS_SIM: {entry['cosine']:.4f} | "
                                    f"KV_PURGED: {entry['purged']} | "
                                    f"TEMP: {entry['temp']:.1f}°C | "
                                    f"AFFINITY: {entry['affinity']}\n"
                                )
                    time.sleep(0.05) # Prevent high CPU usage on the logging thread
            except Exception as e:
                import sys
                print(f"⚠️ [AsyncLogger Error]: {e}", file=sys.stderr)
                time.sleep(0.5)
                
    def stop(self):
        self.running = False
        try:
            self.worker.join(timeout=1.0)
        except Exception:
            pass
