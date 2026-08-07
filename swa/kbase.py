"""Motor de Consulta K-Base — Busca Híbrida Real (dense + sparse) em multicamadas.

Padrão adotado (documento K-Base):

  • Vetores densos  : Gemma-3-270M (640D nativo, mean-pool) — espaço semântico
                       real e indexável (pgvector limita índices a 2000D;
                       por isso o HNSW antigo de 2560D era INVALID).
  • Vetores esparsos: tsvector/ts_rank do Postgres (contraparte TF-IDF/BM25),
                       config 'simple' (amigável a código/siglas).
  • Multicamada     : [100, 250, 500, 1000] — micro-contextos de precisão e
                       macro-contextos argumentativos na mesma busca.
  • RRF             : Reciprocal Rank Fusion — 1/(k + rank_denso) + 1/(k + rank_esparso),
                       k=60, funde ranks de espaços não-normalizáveis.
  • Great Filter    : deduplicação por source_path via Set (O(1)), consolidando
                       o top-K cruzando camadas/coleções.

Tabela: kbase_chunks (pgvector). Índices: HNSW (denso), GIN (esparso),
btree collection/layer.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from typing import Optional

import psycopg2

PG_DSN = os.environ.get("SWA_PG_DSN", "dbname=swa_matrix user=alifer")
EMBEDDER_BASE = os.environ.get("SWA_EMBEDDER_URL", "http://127.0.0.1:21000")
EMBEDDER_TIMEOUT = int(os.environ.get("SWA_EMBEDDER_TIMEOUT", "15"))

LAYERS = [100, 250, 500, 1000]
LAYER_OVERLAP = 0.15
RRF_K = 60
MIN_CHUNK = 40


def _windows(text: str, layer: int) -> list[str]:
    """Fatia o texto em janelas de `layer` caracteres com sobreposição de 15%."""
    text = (text or "").strip()
    if len(text) <= layer:
        return [text] if len(text) >= MIN_CHUNK else []
    step = max(1, int(layer * (1.0 - LAYER_OVERLAP)))
    out = []
    start = 0
    while start < len(text):
        window = text[start : start + layer]
        if window and len(window) >= MIN_CHUNK:
            out.append(window)
        start += step
    return out


def split_layers(text: str) -> dict[int, list[str]]:
    """Retorna {layer: [chunks]} para todas as camadas."""
    return {layer: _windows(text, layer) for layer in LAYERS}


class EmbedClient:
    """Cliente do serviço de embedding real (Gemma-3-270M → 2560D)."""

    def __init__(self, base: str = EMBEDDER_BASE, timeout: int = EMBEDDER_TIMEOUT):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self._ready: Optional[bool] = None

    def _post(self, path: str, payload: dict) -> Optional[dict]:
        try:
            req = urllib.request.Request(
                self.base + path,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def ready(self) -> bool:
        if self._ready is not None:
            return self._ready
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                self._ready = bool(data.get("ready"))
        except Exception:
            self._ready = False
        return self._ready

    def embed(self, text: str) -> Optional[list]:
        if not self.ready():
            return None
        data = self._post("/embed", {"text": (text or " ")[:8000]})
        if not data:
            return None
        return data.get("embedding_640") or data.get("embedding")

    def embed_pair(self, text: str) -> Optional[tuple]:
        """Retorna (v_640, v_2560) — espaço indexável + espaço gold."""
        if not self.ready():
            return None
        data = self._post("/embed", {"text": (text or " ")[:8000]})
        if not data:
            return None
        return (data.get("embedding_640"), data.get("embedding"))

    def embed_batch(self, texts: list[str]) -> Optional[list]:
        if not texts:
            return []
        if not self.ready():
            return None
        data = self._post("/embed_batch", {"texts": [(t or " ")[:8000] for t in texts]})
        if data and "embeddings_640" in data:
            return data["embeddings_640"]
        if data and "embeddings" in data:
            return data["embeddings"]
        return None

    def embed_batch_pairs(self, texts: list[str]) -> Optional[list]:
        """Retorna [(v_640, v_2560), ...] — para ingestão em lote."""
        if not texts:
            return []
        if not self.ready():
            return None
        data = self._post("/embed_batch", {"texts": [(t or " ")[:8000] for t in texts]})
        if not data:
            return None
        v640 = data.get("embeddings_640") or []
        v2560 = data.get("embeddings") or []
        return list(zip(v640, v2560)) or None


class KBaseEngine:
    """Motor de consulta híbrida (dense + sparse + RRF + multicamada + Great Filter)."""

    def __init__(self, embed_client: Optional[EmbedClient] = None, enabled: bool = True):
        self.client = embed_client or EmbedClient()
        self.enabled = bool(enabled)
        self.last_status = {
            "state": "standby",
            "collections": 0,
            "chunks": 0,
            "last_query": "",
            "hits": 0,
            "elapsed_ms": 0.0,
            "fusion": "rrf",
            "layers": list(LAYERS),
        }
        self._lock = threading.Lock()
        self._cache: dict[str, list] = {}
        self._cache_order: list[str] = []
        self._cache_max = 32
        self.ensure_schema()

    # ---- schema ----
    def ensure_schema(self) -> None:
        try:
            conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute("SET statement_timeout = 120000")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kbase_chunks (
                        id serial PRIMARY KEY,
                        layer int NOT NULL,
                        source_path text NOT NULL,
                        collection text NOT NULL,
                        chunk_index int NOT NULL DEFAULT 0,
                        text text NOT NULL,
                        embedding_640 vector(640),
                        embedding_2560 vector(2560),
                        tokens_tsv tsvector,
                        weight real NOT NULL DEFAULT 1.0,
                        created_at timestamp NOT NULL DEFAULT now(),
                        UNIQUE (source_path, layer, chunk_index)
                    )
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS kbase_chunks_col_idx ON kbase_chunks (collection)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS kbase_chunks_layer_idx ON kbase_chunks (layer)"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS kbase_chunks_tsv_idx ON kbase_chunks USING gin (tokens_tsv)"
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS kbase_chunks_hnsw_640
                    ON kbase_chunks USING hnsw (embedding_640 vector_cosine_ops)
                    """
                )
                conn.commit()
                cur.close()
            finally:
                conn.close()
            self._refresh_stats()
        except Exception as e:
            self.enabled = False
            self.last_status["state"] = f"erro: {e}"

    def _refresh_stats(self) -> None:
        try:
            conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT count(*), count(DISTINCT collection) FROM kbase_chunks"
                )
                row = cur.fetchone()
                self.last_status["chunks"] = row[0] or 0
                self.last_status["collections"] = row[1] or 0
                cur.close()
            finally:
                conn.close()
        except Exception:
            pass

    # ---- ingestão (multicamada) ----
    def ingest_source(
        self,
        source_path: str,
        collection: str,
        text: str,
        weight: float = 1.0,
        embedder=None,
    ) -> int:
        """Divide o texto nas 4 camadas e ingere cada chunk.

        embedder(chunk_text) → tuple (v_640, v_2560) | None; se omitido, usa o
        EmbedClient. O vetor da fonte é reutilizado em todas as camadas
        (1 embedding por fonte; a camada restringe o texto devolvido).
        """
        if not self.enabled or not text:
            return 0
        pair = None
        if embedder is not None:
            try:
                pair = embedder(text[:8000])
            except Exception:
                pair = None
        if pair is None:
            pair = self._embed_pair(text[:8000])
        if pair is None:
            return 0
        emb_by_layer = {layer: pair for layer in LAYERS}
        return self._upsert_chunks(source_path, collection, text, emb_by_layer, weight)

    def _embed_pair(self, text: str):
        try:
            return self.client.embed_pair(text)
        except Exception:
            return None

    def _upsert_chunks(
        self,
        source_path: str,
        collection: str,
        text: str,
        emb_by_layer: dict[int, tuple],
        weight: float,
    ) -> int:
        try:
            conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute("SET statement_timeout = 120000")
                n = 0
                for layer, chunks in split_layers(text).items():
                    v640, v2560 = emb_by_layer[layer] if emb_by_layer.get(layer) else (None, None)
                    if not chunks:
                        continue
                    params = []
                    for idx, chunk in enumerate(chunks):
                        params.append(
                            (
                                layer,
                                source_path,
                                collection,
                                idx,
                                chunk,
                                (v640.tolist() if hasattr(v640, "tolist") else v640),
                                (v2560.tolist() if hasattr(v2560, "tolist") else v2560),
                                chunk,
                                weight,
                            )
                        )
                    cur.executemany(
                        """
                        INSERT INTO kbase_chunks
                            (layer, source_path, collection, chunk_index, text, embedding_640, embedding_2560, tokens_tsv, weight)
                        VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector,
                                to_tsvector('simple', %s), %s)
                        ON CONFLICT (source_path, layer, chunk_index) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding_640 = EXCLUDED.embedding_640,
                            embedding_2560 = EXCLUDED.embedding_2560,
                            tokens_tsv = EXCLUDED.tokens_tsv,
                            weight = EXCLUDED.weight
                        """,
                        params,
                    )
                    n += len(params)
                conn.commit()
                cur.close()
                return n
            finally:
                conn.close()
        except Exception:
            return 0

    # ---- busca ----
    def _dense_top(self, vec: list, top_k: int, collection: Optional[str] = None) -> list:
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 60000")
            sql = """
                SELECT id, text, collection, layer, source_path, weight,
                       1 - (embedding_640 <=> %s::vector) AS sim
                FROM kbase_chunks
                WHERE embedding_640 IS NOT NULL
            """
            params: list = [vec]
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            sql += " ORDER BY embedding_640 <=> %s::vector LIMIT %s"
            params += [vec, top_k]
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return [{"id": r[0], "text": r[1], "collection": r[2], "layer": r[3], "source_path": r[4], "weight": float(r[5]), "sim": float(r[6])} for r in rows]
        finally:
            conn.close()

    def _sparse_top(self, query: str, top_k: int, collection: Optional[str] = None) -> list:
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 60000")
            sql = """
                SELECT id, text, collection, layer, source_path, weight,
                       ts_rank(tokens_tsv, websearch_to_tsquery('simple', %s), 32) AS sim
                FROM kbase_chunks
                WHERE tokens_tsv @@ websearch_to_tsquery('simple', %s)
            """
            params: list = [query, query]
            if collection:
                sql += " AND collection = %s"
                params.append(collection)
            sql += " ORDER BY sim DESC LIMIT %s"
            params.append(top_k)
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return [{"id": r[0], "text": r[1], "collection": r[2], "layer": r[3], "source_path": r[4], "weight": float(r[5]), "sim": float(r[6])} for r in rows]
        finally:
            conn.close()

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
        fused: dict[int, dict] = {}
        for ranked in ranked_lists:
            for pos, item in enumerate(ranked):
                d = fused.setdefault(item["id"], dict(item))
                w = float(d.get("weight", 1.0) or 1.0)
                # weight como fator log-compressivo (nudge 0.5x–3.5x), não dominante
                w_rrf = max(0.5, 1.0 + math.log(w)) if w > 0 else 1.0
                d["score"] = d.get("score", 0.0) + w_rrf * (1.0 / (k + pos + 1))
        return sorted(fused.values(), key=lambda x: x["score"], reverse=True)

    @staticmethod
    def _great_filter(results: list[dict], top_k: int) -> list[dict]:
        """The Great Filter — deduplica por source_path em O(1), mantendo o melhor score."""
        seen: set[str] = set()
        kept: list[dict] = []
        for r in results:
            if r["source_path"] in seen:
                continue
            seen.add(r["source_path"])
            kept.append(r)
            if len(kept) >= top_k:
                break
        return kept

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        layers: Optional[list] = None,
        collections: Optional[list[str]] = None,
    ) -> list[dict]:
        """Busca híbrida multicamada → RRF → Great Filter."""
        if not self.enabled or not (query or "").strip():
            return []
        q = query.strip()
        cached = self._cache_get(q)
        if cached is not None:
            self.last_status["hits"] = len(cached)
            return cached
        t0 = time.perf_counter()
        vec = self.client.embed(q)
        if vec is None:
            self.last_status["state"] = "sem embedder"
            self.last_status["last_query"] = q
            return []
        layers = layers or LAYERS
        dense_all: list[dict] = []
        sparse_all: list[dict] = []
        for col in (collections or [None]):
            dense_all += self._dense_top(vec, top_k=top_k * 4, collection=col)
            sparse_all += self._sparse_top(q, top_k=top_k * 4, collection=col)
        for item in dense_all + sparse_all:
            if item["layer"] not in layers:
                item["score"] = -1.0
        fused = self._rrf_fuse(dense_all, sparse_all)
        filtered = self._great_filter([r for r in fused if r.get("score", 0) > 0], top_k=top_k)
        self.last_status["state"] = "active"
        self.last_status["last_query"] = q
        self.last_status["hits"] = len(filtered)
        self.last_status["layers"] = list(layers)
        self.last_status["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self._cache_put(q, filtered)
        return filtered

    def search_collection(self, collection: str, query: str, top_k: int = 5) -> list[dict]:
        """Busca dedicada a uma coleção (fan-out interno com dedup)."""
        return self.retrieve(query, top_k=top_k, collections=[collection])

    def collections(self) -> list[str]:
        try:
            conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT collection FROM kbase_chunks ORDER BY 1")
                rows = [r[0] for r in cur.fetchall()]
                cur.close()
                return rows
            finally:
                conn.close()
        except Exception:
            return []

    # ---- cache ----
    def _cache_get(self, key: str):
        with self._lock:
            return self._cache.get(key)

    def _cache_put(self, key: str, value: list) -> None:
        with self._lock:
            if key in self._cache:
                self._cache_order.remove(key)
            self._cache[key] = value
            self._cache_order.append(key)
            while len(self._cache_order) > self._cache_max:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
