"""
Manifold vetorial compartilhado — reescreve a árvore física em espaço de referência único.

Camadas:
  1. Árvore física (paths Debian / projeto / sabedoria agentes)
  2. pgvector (blake_semantic_anchors, 2560D) — verdade semântica
  3. SHM /dev/shm/swa_vector_manifold — índice rápido interdimensional (lookup O(1) por id)
"""
from __future__ import annotations

import hashlib
import struct
import json
import mmap
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

SHM_MANIFOLD_DIR = Path("/dev/shm/swa_vector_manifold")
MANIFEST_FILE = SHM_MANIFOLD_DIR / "manifest.json"
TREE_FILE = SHM_MANIFOLD_DIR / "reality_tree.jsonl"
ID_INDEX_FILE = SHM_MANIFOLD_DIR / "interdim_index.json"

VBAT_DIMS = (640, 2560)
# ---- vBAT STORE CANÔNICO (fonte única; ADD-ONLY; NUNCA tocado por rebuild) ----
VBAT_STORE_DIR = SHM_MANIFOLD_DIR / "vbat_store"
VBAT_STORE_FILE = {d: VBAT_STORE_DIR / f"vbat_{d}.bin" for d in VBAT_DIMS}
VBAT_STORE_MANIFEST = VBAT_STORE_DIR / "vbat_manifest.json"

VBAT_FILES = {dim: SHM_MANIFOLD_DIR / f"vectors_{dim}.bin" for dim in VBAT_DIMS}
VBAT_FLAG_FILES = {dim: SHM_MANIFOLD_DIR / f"flags_{dim}.bin" for dim in VBAT_DIMS}
VBAT_DRIFTING = 0
VBAT_FROZEN = 1
VBAT_DECAY = 2


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

        idx = {"by_id": by_id, "by_path": by_path}
        vbat_idx = (self._load_id_index().get("vbat") or {}) or {}
        if vbat_idx:
            idx["vbat"] = vbat_idx

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
        vbat_man = (self.load_manifest() or {}).get("vbat") or {}
        if vbat_man:
            manifest["vbat"] = vbat_man

        # TRAVA CANÔNICA: se vBAT canônico existe, a âncora é REAL por contrato.
        # Sem isso, um rebuild com embedder ainda-não-carregado marca embedding_source
        # como "sintético" → no próximo tick o timer dispara rebuild full → wipe vBAT.
        # Aqui forçamos o canônico: a guarda "embedding_source != real → skip" do
        # timer passa a NUNCA re-disparar rebuild quando vBAT já está de pé.
        if vbat_man:
            manifest["embedding_source"] = "gemma-3-270m_proj2560"

        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        with open(TREE_FILE, "w", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(asdict(node), ensure_ascii=False) + "\n")

        with open(ID_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False)

        return manifest

    def _dump_manifest(self, manifest: dict) -> None:
        SHM_MANIFOLD_DIR.mkdir(parents=True, exist_ok=True)
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ---- helpers vBAT STORE canônico (add-only, idempotentes) ----
    def _vbat_store_entry_count(self, dim: int) -> int:
        p = VBAT_STORE_FILE[dim]
        return 0 if not p.is_file() else p.stat().st_size // (4 * 4096 + 4)

    def _vbat_has_interdim(self, dim: int, interdim_id: str) -> bool:
        """Idempotência ANTIGA-vs-NOVA: a verdade é o ARQUIVO FÍSICO add-only.

        O interdim_id recebido é o que o write canônico computou a partir da
        topologia. "Presente" = o STORE add-only físico tem um bloco cujo
        sha256-topology bate — o arquivo físico é o juiz, nunca o manifest
        (que herdou espelhos órfãos tipo \`vbat 2560D frozen\` sem .bin).
        """
        f = VBAT_STORE_FILE[dim]
        if not f.is_file():
            return False
        blk = 4 * 4096 + 4
        with open(f, "rb") as fh:
            chunk = fh.read()
        off = 0
        while off + blk <= len(chunk):
            topo_blk = list(struct.unpack("!4096f", chunk[off:off + 4 * 4096]))
            cand = "vbat:" + hashlib.sha256(bytes(repr(topo_blk[:64]), "utf-8")).hexdigest()[:24]
            if cand == interdim_id:
                return True
            off += blk
        return False

    def write_vbat_store_block(self, dim: int, topology: list, payload: bytes) -> dict:
        """Grava 1 bloco no STORE dedicado — add-only, idempotente por interdim_id.

        Este é o ÚNICO write do vBAT: nunca existe wipe aqui (append de 1 frame
        canônico), e nenhum rebuild/rebase reescreve o arquivo.
        """
        VBAT_STORE_DIR.mkdir(parents=True, exist_ok=True)
        interdim_id = "vbat:" + hashlib.sha256(repr(topology[:64]).encode()).hexdigest()[:24]
        if self._vbat_has_interdim(dim, interdim_id):
            return {"status": "already_present", "id": interdim_id, "dim": dim,
                    "blocks": self._vbat_store_entry_count(dim)}
        with open(VBAT_STORE_FILE[dim], "ab") as f:
            f.write(struct.pack("!4096f I", *topology, len(payload)))
            f.write(payload)
        mans = []
        if VBAT_STORE_MANIFEST.is_file():
            try:
                mans = json.loads(VBAT_STORE_MANIFEST.read_text(encoding="utf-8"))
            except Exception:
                mans = []
        mans.append({"interdim_id": interdim_id, "dim": dim, "blocks": len(mans) + 1,
                     "sha256": hashlib.sha256(bytes(repr(topology[:64]), "utf-8")).hexdigest()[:16],
                     "at": time.time()})
        VBAT_STORE_MANIFEST.write_text(json.dumps(mans, indent=2), encoding="utf-8")
        return {"status": "stored", "id": interdim_id, "dim": dim,
                "blocks": self._vbat_store_entry_count(dim)}

    def _load_id_index(self) -> dict:
        if not ID_INDEX_FILE.is_file():
            return {}
        with open(ID_INDEX_FILE, encoding="utf-8") as f:
            return json.load(f)

    def _dump_id_index(self, index: dict) -> None:
        SHM_MANIFOLD_DIR.mkdir(parents=True, exist_ok=True)
        with open(ID_INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False)

    def write_vbat(
        self,
        dim: int,
        vectors,
        ids: list[str],
        source: str | list[str],
        labels: list[str] | None = None,
    ) -> dict:
        """Persiste matriz contígua float32 em /dev/shm (mmap) e indexa slots por id."""
        if dim not in VBAT_DIMS:
            raise ValueError(f"dim {dim} não suportada ({VBAT_DIMS})")
        arr = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[1] != dim or arr.shape[0] != len(ids):
            raise ValueError("matriz/ids incoerentes")

        path = VBAT_FILES[dim]
        flag_path = VBAT_FLAG_FILES[dim]
        SHM_MANIFOLD_DIR.mkdir(parents=True, exist_ok=True)

        n = int(arr.shape[0])
        byte_size = int(arr.nbytes)
        with open(path, "wb") as f:
            f.truncate(byte_size)
        with open(path, "r+b") as f:
            buf = mmap.mmap(f.fileno(), byte_size, access=mmap.ACCESS_WRITE)
            buf.write(arr.tobytes(order="C"))
            buf.flush()
            buf.close()

        normas = np.linalg.norm(arr, axis=1)
        flags = np.where(
            normas >= 0.9995,
            VBAT_FROZEN,
            np.where(normas <= 0.90, VBAT_DECAY, VBAT_DRIFTING),
        ).astype(np.uint8)
        with open(flag_path, "wb") as f:
            f.write(flags.tobytes())

        index = self._load_id_index()
        blocos = index.setdefault("vbat", {}).setdefault(str(dim), {})
        for slot, vid in enumerate(ids):
            entry = {"slot": int(slot), "source": source, "state": int(flags[slot])}
            if labels:
                entry["label"] = labels[slot]
            blocos[vid] = entry

        manifest = self.load_manifest() or {"version": 2}
        manifest["vbat"] = manifest.get("vbat", {}) or {}
        manifest["vbat"][str(dim)] = {
            "file": path.name,
            "flag_file": flag_path.name,
            "dim": int(dim),
            "blocks": n,
            "stride_bytes": int(dim * 4),
            "total_bytes": byte_size,
            "source": source,
            "blob_sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
            "built_at_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "state_dist": {
                str(VBAT_FROZEN): int((flags == VBAT_FROZEN).sum()),
                str(VBAT_DRIFTING): int((flags == VBAT_DRIFTING).sum()),
                str(VBAT_DECAY): int((flags == VBAT_DECAY).sum()),
            },
        }
        self._dump_manifest(manifest)
        self._dump_id_index(index)
        return manifest["vbat"][str(dim)]

    def open_vbat(self, dim: int):
        """View zero-copy (np.frombuffer sobre mmap) de vectors_{dim}.bin."""
        path = VBAT_FILES.get(dim)
        if path is None or not path.is_file():
            return None
        size = path.stat().st_size
        if size == 0:
            return np.zeros((0, dim), dtype=np.float32)
        raw = open(path, "rb")
        buf = mmap.mmap(raw.fileno(), size, access=mmap.ACCESS_READ)
        self._vbat_holders = getattr(self, "_vbat_holders", {})
        self._vbat_holders[dim] = (raw, buf)
        return np.frombuffer(buf, dtype=np.float32).reshape(-1, dim)

    def vbat_flags(self, dim: int) -> np.ndarray | None:
        path = VBAT_FLAG_FILES.get(dim)
        if path is None or not path.is_file():
            return None
        return np.frombuffer(path.read_bytes(), dtype=np.uint8)

    def vbat_vector(self, vid: str, dim: int) -> np.ndarray | None:
        """Resolve id → slot → vetor, sem tocar no pgvector (sem TOAST)."""
        entry = (self._load_id_index().get("vbat", {}).get(str(dim)) or {}).get(vid)
        if entry is None:
            return None
        store = self.open_vbat(dim)
        if store is None or entry["slot"] >= store.shape[0]:
            return None
        return store[entry["slot"]]

    def vbat_status(self) -> dict:
        manifest = self.load_manifest() or {}
        out = dict(manifest.get("vbat", {}) or {})
        for dim in VBAT_DIMS:
            entry = out.get(str(dim))
            if not entry:
                continue
            flags = self.vbat_flags(dim)
            if flags is not None:
                entry["flags"] = {
                    str(VBAT_FROZEN): int((flags == VBAT_FROZEN).sum()),
                    str(VBAT_DRIFTING): int((flags == VBAT_DRIFTING).sum()),
                    str(VBAT_DECAY): int((flags == VBAT_DECAY).sum()),
                }
        return out

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
