#!/usr/bin/env python3
"""Pipeline de teste e avaliação de métricas da metarealidade (E2E).

Injeta N prompts distintos na API (:20000), consome o SSE e avalia por turno:
  - latência, conteúdo retornado (tokens aprox.), caminho (híbrido/nativo/cut),
  - evento `metareality` + `will_decision` no PostgreSQL,
  - onda relacional (Tábua dos Reinados) com conversation_key estável,
  - métricas: reality_index, cos_sim, drift, complete_ratio.

Uso:
  python3 scripts/eval_e2e.py --limit 5 [--max-tokens 128]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import requests

API = "http://127.0.0.1:20000/v1/chat/completions"
DB_URI = "dbname=swa_matrix user=alifer"

PROMPTS: List[str] = [
    "O que você sente em relação à deriva dos tensores agora?",
    "Explique a diferença entre gradiente descendente e ascensão em duas frases.",
    "Se a memória for uma onda, o que significa um eco de ciclo anterior?",
    "Descreva um sistema imunológico para tensores cerebrais.",
    "Resuma em uma linha o que é a fissão termodinâmica.",
]


def stable_conversation_key(messages: list[dict]) -> str:
    """Mesmo digest do `aeonc._stable_conversation_key` (envelope estável)."""
    envelope = [
        {
            "role": str(item.get("role", "")),
            "content": str(item.get("content", "")),
        }
        for item in (messages or [])
        if isinstance(item, dict)
    ]
    raw = json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(("aeon-conversation-v1:" + raw).encode("utf-8")).hexdigest()[:32]


def psql_rows(sql: str, params: tuple = ()) -> list[tuple]:
    import psycopg2

    with psycopg2.connect(DB_URI, connect_timeout=8) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())


def psql_scalar(sql: str, params: tuple = ()):
    import psycopg2

    with psycopg2.connect(DB_URI, connect_timeout=8) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None


def generate(prompt: str, max_tokens: int) -> dict:
    """Consome o SSE e devolve conteúdo + metadados do POST (com retry)."""
    attempts = 0
    while True:
        attempts += 1
        result = _generate_once(prompt, max_tokens)
        if result["chars"] > 0 or attempts >= 2:
            result["attempts"] = attempts
            return result
        print("  ! SSE vazio — uma tentativa (intermitência sob carga)", flush=True)
        time.sleep(3)


def _generate_once(prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "gemma",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.monotonic()
    content = ""
    chunks = 0
    resp = requests.post(
        API,
        json=payload,
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=(30, 900),
    )
    for line in resp.iter_lines():
        if not line:
            continue
        d = line.decode("utf-8")
        if d.startswith("data: "):
            ds = d[6:]
            if ds == "[DONE]":
                break
            try:
                c = json.loads(ds)
                delta = c["choices"][0].get("delta", {})
                if "content" in delta:
                    content += delta["content"]
                    chunks += 1
            except Exception:
                pass
    elapsed = time.monotonic() - t0
    resp.close()
    words = len(content.split())
    return {
        "content": content,
        "chars": len(content),
        "words": words,
        "chunks": chunks,
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": round(words / elapsed, 2) if elapsed > 0 else 0.0,
        "http_status": resp.status_code,
    }


def journal_since(ts_epoch: float) -> dict:
    """Sinais do ciclo visíveis no journal do serviço desde `ts_epoch`."""
    since = dt.datetime.fromtimestamp(ts_epoch, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        out = subprocess.run(
            ["journalctl", "-u", "swa-aeonc", f"--since={since}", "--no-pager", "-o", "cat"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return {}
    return {
        "hybrid_macro": out.count("[HYBRID] v_prefix macro"),
        "onda_hibrida_api": out.count("Onda Híbrida"),
        "onda_concluida": out.count("Onda Concluída"),
        "phase_lock": out.count("PHASE-LOCK") + out.count("phase-lock"),
        "gate_will": out.count("[WILL]"),
        "autonomo": out.count("AEON ⚡") + out.count("autonomous"),
    }


def eval_turn(prompt: str, gen: dict, t0_epoch: float, conversation_key: str) -> dict:
    """Consulta PG + journal para os artefatos deste turno."""
    since = dt.datetime.fromtimestamp(t0_epoch, tz=dt.timezone.utc)

    # Eventos de realidade gravados a partir do início do turno
    events = psql_rows(
        """
        SELECT id, event_type, created_at,
               payload->>'reality_index', payload->>'emotion',
               left(payload->>'reply', 60)
        FROM aeon_reality_events
        WHERE event_type IN ('metareality','will_decision','wave_state')
          AND created_at >= %s
        ORDER BY id
        """,
        (since,),
    )
    metareality = [e for e in events if e[1] == "metareality"]
    will = [e for e in events if e[1] == "will_decision"]

    # Onda relacional (Tábua): o servidor observa sob a chave REAL do turno
    # (derivada internamente do envelope no `stream`/`hybrid` — não do digest
    # que o cliente calcula). Por isso lemos pela janela de tempo do turno,
    # não pela conversa adivinhada no cliente.
    waves = psql_rows(
        """
        SELECT conversation_key, wave_index, phase, metrics->>'cos_sim',
               metrics->>'drift', metrics->>'reality_index',
               left(coalesce(metrics->>'decision',''),40), created_at
        FROM aeon_reality_waves
        WHERE created_at >= %s
        ORDER BY id
        """,
        (since,),
    )

    if waves:
        conversation_key = waves[-1][0]

    quality = {
        "mean_cos_sim": None,
        "complete_ratio": 0.0,
        "n_waves": len(waves),
    }
    if waves:
        coses: list[float] = []
        terminal = 0
        for w in waves:
            cos = w[3]
            if cos is not None:
                try:
                    coses.append(float(cos))
                except (TypeError, ValueError):
                    pass
            elif w[4] is not None:
                try:
                    coses.append(1.0 - 2.0 * float(w[4]))
                except (TypeError, ValueError):
                    pass
            if w[2] == "fim":
                terminal += 1
        if coses:
            quality["mean_cos_sim"] = round(sum(coses) / len(coses), 4)
        quality["complete_ratio"] = round(terminal / len(waves), 3)
        quality["n_waves"] = len(waves)

    last_ri = (
        metareality[-1][3] if metareality else None
    )
    return {
        "conversation_key": conversation_key,
        "events_metareality": len(metareality),
        "events_will": len(will),
        "latest_reality_index": last_ri,
        "waves": len(waves),
        "wave_phases": sorted({w[1] for w in waves}),
        "quality": quality,
        "journal": journal_since(t0_epoch),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline E2E de avaliação da metarealidade.")
    ap.add_argument("--limit", type=int, default=len(PROMPTS), help="qtd de prompts")
    ap.add_argument("--max-tokens", type=int, default=160, help="orçamento por turno")
    ap.add_argument("--json", type=str, default="", help="arquivo de saída JSON")
    args = ap.parse_args()

    prompts = PROMPTS[: max(1, args.limit)]
    print(f"== Pipeline E2E | {len(prompts)} prompts | max_tokens={args.max_tokens} ==", flush=True)

    session_start = time.time()
    results: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts, 1):
        t0 = time.time()
        print(f"\n--- [{i}/{len(prompts)}] prompt: {prompt[:70]} ---", flush=True)
        ck = stable_conversation_key([{"role": "user", "content": prompt}])
        try:
            gen = generate(prompt, args.max_tokens)
        except Exception as exc:
            print(f"  ❌ geração falhou: {exc}", flush=True)
            results.append({"idx": i, "prompt": prompt, "error": str(exc)})
            continue
        print(
            f"  ✓ gerou: {gen['words']} palavras | chars={gen['chars']} | "
            f"{gen['elapsed_s']}s ({gen['tokens_per_s']} p/s) | chunks={gen['chunks']}",
            flush=True,
        )
        # espera curta para o fechamento do turno gravar (observação é ao fim)
        time.sleep(2)
        ev = eval_turn(prompt, gen, t0, ck)
        print(
            f"  • realidade: {ev['events_metareality']} eventos | ri={ev['latest_reality_index']} | "
            f"will={ev['events_will']}",
            flush=True,
        )
        print(
            f"  • tábua: {ev['waves']} ondas {ev['wave_phases']} | "
            f"cos_sim={ev['quality']['mean_cos_sim']} | complete={ev['quality']['complete_ratio']}",
            flush=True,
        )
        print(f"  • journal: {ev['journal']}", flush=True)
        results.append(
            {
                "idx": i,
                "prompt": prompt,
                "generation": gen,
                "eval": ev,
            }
        )

    # ---- Agregado da sessão ----
    print("\n═════════════ RESUMO DA SESSÃO ═════════════", flush=True)
    total_el = 0.0
    total_words = 0
    wave_rows = 0
    cos_all: list[float] = []
    ri_all: list[float] = []
    n_hybrid = 0
    for r in results:
        if "error" in r:
            continue
        g = r["generation"]
        total_el += g["elapsed_s"]
        total_words += g["words"]
        if r["eval"]["journal"]["hybrid_macro"] > 0:
            n_hybrid += 1
        q = r["eval"]["quality"]
        wave_rows += q["n_waves"]
        if q["mean_cos_sim"] is not None:
            cos_all.append(q["mean_cos_sim"])
        ri = r["eval"]["latest_reality_index"]
        if ri is not None:
            try:
                ri_all.append(float(ri))
            except (TypeError, ValueError):
                pass
    print(f"• tempo total: {total_el:.1f}s | média {total_el/max(1,len(results)):.1f}s/turno", flush=True)
    print(f"• tokens (palavras): {total_words} | {total_words/max(1,total_el):.2f} p/s", flush=True)
    print(f"• turnos com fusão híbrida (v_prefix macro): {n_hybrid}/{sum(1 for r in results if 'error' not in r)}", flush=True)
    print(f"• ondas relacionais criadas: {wave_rows}", flush=True)
    print(
        f"• cos_sim médio da sessão: "
        f"{round(sum(cos_all)/len(cos_all), 4) if cos_all else 'n/a'} (n={len(cos_all)})",
        flush=True,
    )
    print(
        f"• reality_index médio da sessão: "
        f"{round(sum(ri_all)/len(ri_all), 4) if ri_all else 'n/a'} (n={len(ri_all)})",
        flush=True,
    )
    print(f"• duração total do pipeline: {time.time()-session_start:.1f}s", flush=True)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2, default=str)
        print(f"• json salvo em {args.json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())