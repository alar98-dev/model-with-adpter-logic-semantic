"""Ping às rotas públicas mínimas do SWA (chat, stream, process).

Rotas de debug só existem com SWA_EXPOSE_DEBUG_ROUTES=true no servidor.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("SWA_BASE_URL", "http://localhost:20000").rstrip("/")

CORE = {
    "chat": "/api/v1/chat",
    "stream": "/api/v1/stream",
    "process": "/api/v1/process",
}

DEBUG = {
    "self_check": "/api/v1/utils/self_check",
    "internal_runtime": "/api/v1/internal/runtime",
}


def _post(path: str, body: dict) -> tuple[int, str]:
    url = BASE_URL + path
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.getcode(), resp.read().decode("utf-8", errors="replace")[:500]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")[:500]


def main() -> int:
    print("BASE_URL", BASE_URL)
    code, body = _post(CORE["chat"], {"prompt": "ping", "max_new_tokens": 4})
    print("chat", code, body[:200])
    if code != 200:
        return 1
    code2, _ = _post(CORE["process"], {"texts": ["a", "b"]})
    print("process", code2)
    code3, _ = _post(DEBUG["self_check"], {})
    print("self_check (404 se debug off)", code3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
