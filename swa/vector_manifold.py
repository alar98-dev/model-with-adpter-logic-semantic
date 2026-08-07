"""
Manifold vetorial compartilhado — reescreve a árvore física em espaço de referência único.

Camadas:
  1. Árvore física (paths Debian / projeto / sabedoria agentes)
  2. pgvector (blake_semantic_anchors, 2560D) — verdade semântica
  3. SHM /dev/shm/swa_vector_manifold — índice rápido interdimensional (lookup O(1) por id)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

SHM_MANIFOLD_DIR = Path("/dev/shm/swa_vector_manifold")
MANIFEST_FILE = SHM_MANIFOLD_DIR / "manifest.json"
TREE_FILE = SHM_MANIFOLD_DIR / "reality_tree.jsonl"
ID_INDEX_FILE = SHM_MANIFOLD_DIR / "interdim_index.json"


@dataclass
class RealityNode:
    physical_path: str
    interdim_id: str
    pg_address: str
    parent_path: str
    zone: str
    depth: int
    concept: str
    mtime: float
    size_bytes: int


def interdim_address(physical_path: str, content_fingerprint: str) -> tuple[str, str]:
    """Retorna (interdim_id, pg_address mmap://id#path)."""
    interdim_id = hashlib.blake2b(
        f"{physical_path}:manifold:{content_fingerprint}".encode("utf-8"),
        digest_size=16,
    ).hexdigest()
    pg_address = f"mmap://{interdim_id}#{physical_path}"
    return interdim_id, pg_address


class VectorManifold:
    def __init__(self, bridge=None):
        self.bridge = bridge
        SHM_MANIFOLD_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _classify_sample(sample: str, filename: str) -> str:
        if "agent-transcripts" in filename or sample.strip().startswith("{"):
            return "AGENT_WISDOM"
        if any(k in sample for k in ("def ", "class ", "import ", "return ")):
            return "CODE_BLOCK"
        return "HUMAN_EXPLANATION"

    def collect_tree(self, roots: list[str] | None = None) -> list[RealityNode]:
        from swa.workspace_policy import iter_indexable_files, classify_zone

        nodes: list[RealityNode] = []

        for filepath in iter_indexable_files(roots):
            try:
                st = os.stat(filepath)
            except OSError:
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(4000)
            except OSError:
                continue
            if not sample.strip():
                continue

            fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
            interdim_id, pg_address = interdim_address(filepath, fingerprint)
            parent = str(Path(filepath).parent)
            concept = self._classify_sample(sample[:2000], Path(filepath).name)

            nodes.append(
                RealityNode(
                    physical_path=filepath,
                    interdim_id=interdim_id,
                    pg_address=pg_address,
                    parent_path=parent,
                    zone=classify_zone(filepath),
                    depth=len(Path(filepath).parts),
                    concept=concept,
                    mtime=st.st_mtime,
                    size_bytes=st.st_size,
                )
            )
        return nodes

    def sync_to_pgvector(self, nodes: list[RealityNode], batch_log_every: int = 25) -> int:
        if self.bridge is None:
            from swa.quantum_bridge import QuantumPhaseBridge
            self.bridge = QuantumPhaseBridge()
        if self.bridge.conn is None:
            return 0

        from swa.reality_embedder import get_reality_embedder
        embedder = get_reality_embedder()
        embedder.ensure_loaded()
        print(f"🧬 [MANIFOLD] Fonte de embedding: {embedder.embedding_source_label()}")

        ok = 0
        for i, node in enumerate(nodes):
            try:
                with open(node.physical_path, "r", encoding="utf-8", errors="ignore") as f:
                    sample = f.read(4000)
                v_2560 = embedder.embed_text_2560(sample)
                interdim = self._upsert_with_vector(node, sample, v_2560)
                if interdim:
                    ok += 1
                if batch_log_every and (i + 1) % batch_log_every == 0:
                    print(f"  … {i + 1}/{len(nodes)} âncoras reais no pgvector")
            except Exception:
                continue
        if hasattr(self.bridge, "_cached_L"):
            self.bridge._cached_L = None
        return ok

    def _upsert_with_vector(self, node: RealityNode, sample: str, v_2560: list) -> str | None:
        if self.bridge is None or self.bridge.conn is None:
            return None
        import hashlib
        import math
        from swa.file_crawler import BlakeNativeCrawler

        crawler = BlakeNativeCrawler()
        structural_type = crawler.classify_paragraph(sample, node.physical_path)
        fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
        interdim_id = hashlib.blake2b(
            f"{node.physical_path}:manifold:{fingerprint}".encode("utf-8"), digest_size=16
        ).hexdigest()
        composite_path = f"mmap://{interdim_id}#{node.physical_path}"
        weight = math.log(len(sample) + 1)
        if structural_type == "CODE_BLOCK":
            weight *= 1.5
        try:
            self.bridge.cursor.execute(
                """
                INSERT INTO blake_semantic_anchors
                (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE
                SET raw_content = EXCLUDED.raw_content,
                    embedding_2560 = EXCLUDED.embedding_2560,
                    weight = EXCLUDED.weight,
                    concept_group = EXCLUDED.concept_group,
                    signature_hash = EXCLUDED.signature_hash;
                """,
                (
                    composite_path,
                    structural_type,
                    interdim_id,
                    sample[:1000],
                    v_2560,
                    weight,
                ),
            )
            self.bridge.conn.commit()
            return interdim_id
        except Exception:
            try:
                self.bridge.conn.rollback()
            except Exception:
                pass
            return None

    def write_shared_manifold(self, nodes: list[RealityNode], roots: list[str]) -> dict:
        """Persiste árvore + índice interdimensional em SHM (espaço compartilhado)."""
        by_id = {n.interdim_id: asdict(n) for n in nodes}
        by_path = {n.physical_path: n.interdim_id for n in nodes}

        # Árvore por zona (para navegação tipo tree folder)
        zones: dict[str, list[str]] = {}
        for n in nodes:
            zones.setdefault(n.zone, []).append(n.physical_path)

        try:
            from swa.reality_embedder import get_reality_embedder
            emb_src = get_reality_embedder().embedding_source_label()
        except Exception:
            emb_src = "unknown"

        manifest = {
            "version": 2,
            "rebuilt_at": time.time(),
            "rebuilt_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding_source": emb_src,
            "roots": roots,
            "node_count": len(nodes),
            "zones": {z: len(paths) for z, paths in zones.items()},
            "shm_dir": str(SHM_MANIFOLD_DIR),
            "tree_file": str(TREE_FILE),
            "id_index_file": str(ID_INDEX_FILE),
        }

        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        with open(TREE_FILE, "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(asdict(node), ensure_ascii=False) + "\n")

        with open(ID_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump({"by_id": by_id, "by_path": by_path}, f, ensure_ascii=False)

        return manifest

    def rebuild_reality(self, roots: list[str] | None = None) -> dict:
        from swa.workspace_policy import get_watch_roots

        roots = roots or get_watch_roots()
        print(f"🌳 [MANIFOLD] Reescrevendo realidade — {len(roots)} zonas raiz")
        nodes = self.collect_tree(roots)
        print(f"🌳 [MANIFOLD] {len(nodes)} nós na árvore física")
        pg_ok = self.sync_to_pgvector(nodes)
        print(f"🌐 [MANIFOLD] {pg_ok} âncoras alinhadas ao pgvector")
        manifest = self.write_shared_manifold(nodes, roots)
        print(
            f"✅ [MANIFOLD] Espaço compartilhado: {SHM_MANIFOLD_DIR} "
            f"({manifest['node_count']} refs interdimensionais)"
        )
        return manifest

    @staticmethod
    def load_manifest() -> dict | None:
        if not MANIFEST_FILE.is_file():
            return None
        with open(MANIFEST_FILE, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def resolve_interdim(interdim_id: str) -> RealityNode | None:
        if not ID_INDEX_FILE.is_file():
            return None
        with open(ID_INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("by_id", {}).get(interdim_id)
        if not raw:
            return None
        return RealityNode(**raw)

    @staticmethod
    def resolve_path(physical_path: str) -> str | None:
        if not ID_INDEX_FILE.is_file():
            return None
        with open(ID_INDEX_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("by_path", {}).get(physical_path)

    @staticmethod
    def iter_zone(zone: str) -> Iterator[RealityNode]:
        if not TREE_FILE.is_file():
            return
        with open(TREE_FILE, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                if raw.get("zone") == zone:
                    yield RealityNode(**raw)
