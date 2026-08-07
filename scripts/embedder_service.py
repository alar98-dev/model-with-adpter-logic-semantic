#!/usr/bin/env python3
"""Serviço de embedding real — Gemma-3-270M (640D nativo → 2560D gold).

Endpoints:
  GET  /health        → {ready, native_dim, gold_dim, source}
  POST /embed         {text}          → {embedding_640, embedding}
  POST /embed_batch   {texts: [...]}  → {embeddings_640, embeddings}

O espaço 640D nativo é o indexável (pgvector limita índices a 2000D) e é o
mesmo usado pelo motor K-Base. O 2560D gold (projeção seed 137) fica
disponível para compatibilidade/tensor.

Uso: python3 scripts/embedder_service.py [--port 21000]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/home/alifer/swa-aeon"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F

from swa.reality_embedder import native_to_gold

_model = None
_processor = None
_NATIVE = 640
_GOLD = 2560
_ready = False
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def _ensure_loaded() -> None:
    global _model, _processor, _ready
    if _ready:
        return
    with _load_lock:
        if _ready:
            return
        from transformers import AutoModelForCausalLM, AutoProcessor
        path = "/home/alifer/swa-aeon/models/gemma-3-270m"
        _processor = AutoProcessor.from_pretrained(path)
        _model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
        _model.to("cpu").eval()
        _ready = True


def _embed_vecs(texts: list[str]) -> list[tuple[list, list]]:
    """Retorna [(v_640, v_2560)] por texto, L2-normalizados.

    Trunca em 512 tokens/texto (mean-pool satura; suficiente para RAG).
    """
    _ensure_loaded()
    out = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk_texts = [t[:1000] for t in texts[i : i + batch]]
        toks = _processor(text=chunk_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with _infer_lock:
            with torch.no_grad():
                out_model = _model(**toks, output_hidden_states=True)
        hs = out_model.hidden_states[-1].float()
        emb = hs.mean(dim=1)
        v640 = F.normalize(emb, p=2, dim=1)
        for row in range(emb.shape[0]):
            g = native_to_gold(emb[row])
            out.append((v640[row].cpu().tolist(), g.cpu().tolist()))
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {
                "ready": _ready,
                "native_dim": _NATIVE,
                "gold_dim": _GOLD,
                "source": "gemma-3-270m_proj2560" if _ready else "carregando",
            })
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            self._send(400, {"error": "json inválido"})
            return
        try:
            if self.path == "/embed":
                text = str(data.get("text", ""))
                v640, v2560 = _embed_vecs([text])[0]
                self._send(200, {"embedding_640": v640, "embedding": v2560, "dim": _NATIVE, "gold_dim": _GOLD})
            elif self.path == "/embed_batch":
                texts = [str(t) for t in data.get("texts", [])]
                pairs = _embed_vecs(texts)
                self._send(200, {"embeddings_640": [p[0] for p in pairs], "embeddings": [p[1] for p in pairs]})
            else:
                self._send(404, {"error": "not found"})
        except Exception as e:
            self._send(500, {"error": f"inferência falhou: {e}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=21000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    t = threading.Thread(target=_ensure_loaded, daemon=True)
    t.start()
    print(f"🧠 [EMBEDDER] Carregando Gemma-3-270M ({_NATIVE}D nativo)…", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"✅ [EMBEDDER] http://{args.host}:{args.port} | aguardando modelo…", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[EMBEDDER] encerrando.")
        sys.exit(0)


if __name__ == "__main__":
    main()
