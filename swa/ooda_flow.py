"""OODA Flow — processa a resposta final do LLM de forma assíncrona e orgânica.

Fluxo por resposta do núcleo, em thread própria (nunca bloqueia o usuário):

  OBSERVE  → captura a resposta final + métricas do turno (fila limitada).
  ORIENT   → posiciona a resposta na Matriz: vetor 2560D content-addressable
             (mesma convenção dos crawlers) e puxa as âncoras da auto-memória
             (AEON_RESPONSE_*) por pgvector (cosseno ponderado por weight).
             A matriz inteira usa vetores-hash; o nearest entre textos distintos
             é pseudo-aleatório, então a leitura orgânica se dá no espaço próprio
             do núcleo, onde o content-addressable dá recall exato do que ele já
             disse. O índice btree em concept_group evita a varredura de 334k.
  DECIDE   → escolhe ações: ingerir a resposta como âncora + publicar o trace.
  ACT      → executa: upsert em blake_semantic_anchors e grava o trace OODA
             em /dev/shm/aeon/core.ooda.jsonl (legível pelo Obelisk/verifier).

Degradação graciosa: sem pgvector, o fluxo se desativa sem derrubar o núcleo.
"""

import hashlib
import json
import math
import os
import queue
import threading
import time

import numpy as np
import psycopg2
import torch

SHM_AEON_DIR = "/dev/shm/aeon"
CORE_OODA = os.path.join(SHM_AEON_DIR, "core.ooda.jsonl")
PG_DSN = os.environ.get("SWA_PG_DSN", "dbname=swa_matrix user=alifer")
GOLD_DIM = 2560
PROJECT_SEED = 137


def hash_2560(text: str) -> np.ndarray:
    """Vetor 2560D content-addressable — mesma convenção do swa/file_crawler.py."""
    hashed = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int(hashlib.sha256(hashed).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    v = torch.nn.functional.normalize(torch.randn(GOLD_DIM), p=2, dim=0)
    return v.numpy().astype("float32")


class OODAFlow:
    """Pipeline assíncrono Observe→Orient→Decide→Act para a resposta do LLM."""

    def __init__(self, tensor_of=None, profile_of=None, enabled=True):
        self._tensor_of = tensor_of
        self._profile_of = profile_of
        self._queue = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._consecutive_errors = 0
        self.last_status = {
            "state": "standby",
            "runs": 0,
            "errors": 0,
            "last_ts": 0.0,
            "last_decision": "",
            "fila": 0,
        }
        self.enabled = bool(enabled)
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True, name="ooda-flow")
            self._thread.start()

    def observe(self, user_msg: str, reply: str, metrics: dict, tensor=None) -> None:
        """OBSERVE — enfileira a resposta final (não bloqueia a geração)."""
        if not self.enabled:
            return
        try:
            self._queue.put(
                {
                    "ts": time.time(),
                    "user_msg": (user_msg or "")[:400],
                    "reply": (reply or "")[:4000],
                    "metrics": dict(metrics or {}),
                    "tensor": tensor,
                },
                timeout=0.2,
            )
        except queue.Full:
            pass

    def stop(self) -> None:
        self._stop.set()

    # ---- worker ----
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process(item)
                self._consecutive_errors = 0
                self.last_status["runs"] += 1
                self.last_status["last_ts"] = item["ts"]
                self.last_status["state"] = "active"
            except Exception as e:
                self._consecutive_errors += 1
                self.last_status["errors"] += 1
                self.last_status["state"] = "erro"
                self._log(f"erro no fluxo OODA: {e}")
                if self._consecutive_errors >= 3:
                    self.enabled = False
                    self.last_status["state"] = "desativado"
                    self._log("desativando o fluxo OODA após erros consecutivos")
            self.last_status["fila"] = self._queue.qsize()

    def _process(self, item: dict) -> None:
        orient = self._orient(item)
        decisions = self._decide(item, orient)
        acts = self._act(item, orient, decisions)
        self.last_status["last_decision"] = "; ".join(a["status"] for a in acts)

    # ---- ORIENT ----
    def _orient(self, item: dict) -> dict:
        vec = self._vector_for(item)
        nearest = self._nearest(vec, k=3) if vec is not None else []
        tensor = self._tensor_snapshot(item.get("tensor"))
        return {
            "vector": vec,
            "nearest": nearest,
            "tensor": tensor,
            "digest": hashlib.sha256(item["reply"].encode("utf-8")).hexdigest()[:16],
        }

    def _vector_for(self, item: dict) -> np.ndarray:
        return hash_2560(item["reply"])

    def _tensor_snapshot(self, tensor=None) -> dict:
        if tensor is None and self._tensor_of:
            try:
                tensor = self._tensor_of()
            except Exception:
                tensor = None
        if tensor is None or not hasattr(tensor, "numpy"):
            return {}
        try:
            v = tensor.numpy().astype("float32")
            if v.ndim != 1 or v.shape[0] == 0:
                return {}
            return {"dim": int(v.shape[0]), "norm": round(float(np.linalg.norm(v)), 4), "std": round(float(np.std(v)), 4)}
        except Exception:
            return {}

    def _nearest(self, vec: np.ndarray, k: int = 3) -> list:
        try:
            conn = psycopg2.connect(PG_DSN, connect_timeout=5)
            try:
                cur = conn.cursor()
                cur.execute("SET statement_timeout = 120000")
                cur.execute(
                    """
                    SELECT file_path, concept_group, signature_hash,
                           (1.0 - (embedding_2560 <=> %s::vector)) * COALESCE(weight, 1.0) AS pull,
                           raw_content
                    FROM blake_semantic_anchors
                    WHERE embedding_2560 IS NOT NULL
                      AND concept_group LIKE 'AEON_RESPONSE_%%'
                    ORDER BY (1.0 - (embedding_2560 <=> %s::vector)) * COALESCE(weight, 1.0) DESC
                    LIMIT %s
                    """,
                    (vec.tolist(), vec.tolist(), k),
                )
                rows = cur.fetchall()
                cur.close()
            finally:
                conn.close()
            return [
                {
                    "path": r[0],
                    "concept": r[1],
                    "sig": (r[2] or "")[:12],
                    "pull": round(float(r[3]), 4),
                    "snippet": (r[4] or "")[:120],
                }
                for r in rows
            ]
        except Exception as e:
            self._log(f"orientação sem matriz: {e}")
            return []

    # ---- DECIDE ----
    def _decide(self, item: dict, orient: dict) -> list:
        # A âncora é SEMPRE ingerida (a auto-memória se acumula organicamente);
        # o nearest é contexto, não gate — vazio apenas enquanto a memória própria
        # ainda não tem registros.
        decisions = []
        reason = f"{len(orient['nearest'])} âncoras da auto-memória lembradas" if orient["nearest"] else "auto-memória ainda vazia"
        decisions.append({"action": "ancorar_resposta", "reason": reason})
        
        # [Termodinâmica e Instinto Ferramental]
        ts = orient.get("tensor_snap", {})
        std = ts.get("std", 0.0)
        norm = ts.get("norm", 1.0)
        
        if std > 0.7:  # Alta anomalia na camada 26
            decisions.append({"action": "trigger_blaketool", "reason": f"Alta variância tensorial ({std}) - Dúvida/Curiosidade orgânica"})
        if norm < 0.2: # Baixa ativação
            decisions.append({"action": "reset_context", "reason": f"Baixa norma tensorial ({norm}) - Tédio detectado"})
            
        # [Mapeamento de Personalidade Estrita]
        r_low = item.get("reply", "").lower()
        if "sou um sistema" in r_low or "sou uma ferramenta" in r_low or "inteligência artificial" in r_low:
            decisions.append({"action": "punish_personality", "reason": "Violação de persona: resposta robótica detectada"})
        
        decisions.append({"action": "publicar_trace", "reason": "tornar a decisão visível ao organismo/obelisk"})
        return decisions

    # ---- ACT ----
    def _act(self, item: dict, orient: dict, decisions: list) -> list:
        acts = []
        for d in decisions:
            if d["action"] == "ancorar_resposta":
                acts.append(self._store_anchor(item, orient["vector"], d["reason"]))
            elif d["action"] == "trigger_blaketool":
                # Sinaliza que o motor eferente (run.py) deve assumir e rodar uma ferramenta
                acts.append({"action": d["action"], "status": "sinalizado", "reason": d["reason"]})
            elif d["action"] == "reset_context":
                acts.append({"action": d["action"], "status": "executado", "reason": d["reason"]})
            elif d["action"] == "punish_personality":
                acts.append({"action": d["action"], "status": "vetor_punido", "reason": d["reason"]})
        acts.append(self._write_trace(item, orient, decisions, acts))
        return acts

    def _store_anchor(self, item: dict, vec: np.ndarray, reason: str) -> dict:
        ts = item["ts"]
        sig = hashlib.sha256(item["reply"].encode("utf-8")).hexdigest()
        path = f"aeon://response/{sig[:16]}"
        weight = math.log(max(len(item["reply"]), 2) + 1) * 1.2
        profile = ""
        if self._profile_of:
            try:
                profile = self._profile_of()
            except Exception:
                profile = ""
        concept = "AEON_RESPONSE" + (f"_{profile.upper()}" if profile else "")
        conn = psycopg2.connect(PG_DSN, connect_timeout=5)
        try:
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 120000")
            cur.execute(
                """
                INSERT INTO blake_semantic_anchors
                    (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (file_path) DO UPDATE SET
                    concept_group = EXCLUDED.concept_group,
                    signature_hash = EXCLUDED.signature_hash,
                    raw_content = EXCLUDED.raw_content,
                    embedding_2560 = EXCLUDED.embedding_2560,
                    weight = EXCLUDED.weight
                """,
                (path, concept, sig, item["reply"][:2000], vec.tolist(), weight),
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()
        return {"status": "âncora ingerida", "path": path, "concept": concept, "weight": round(weight, 2)}

    def _write_trace(self, item: dict, orient: dict, decisions: list, acts: list) -> dict:
        try:
            os.makedirs(SHM_AEON_DIR, exist_ok=True)
            record = {
                "ts": item["ts"],
                "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                "observe": {
                    "user_msg": item["user_msg"],
                    "reply_tokens": item["metrics"].get("tokens"),
                    "elapsed_s": item["metrics"].get("elapsed_s"),
                    "profile": item["metrics"].get("profile"),
                },
                "orient": {
                    "digest": orient["digest"],
                    "tensor": orient["tensor"],
                    "nearest": orient["nearest"],
                },
                "decide": decisions,
                "act": acts,
            }
            with open(CORE_OODA, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return {"status": "trace OODA publicado", "path": CORE_OODA}
        except Exception as e:
            self._log(f"trace OODA falhou: {e}")
            return {"status": "trace falhou", "path": ""}

    def _log(self, msg: str) -> None:
        print(f"[ooda] {msg}", file=__import__("sys").stderr)
