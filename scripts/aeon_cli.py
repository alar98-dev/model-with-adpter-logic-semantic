#!/usr/bin/env python3
"""Terminal client for the already running Aeon API."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _event_label(chunk: str) -> str | None:
    if "Onda Híbrida" in chunk:
        return "ONDA_INICIO"
    if "FISSÃO" in chunk:
        return "FISSAO"
    if "Onda Concluída" in chunk:
        return "ONDA_FIM"
    return None


def _render_particle_grid(step: int, phase: str, token_estimate: int) -> str:
    """Render a deterministic 16x4 terminal field from live stream telemetry."""
    levels = " .:-=+*#%@"
    phase_gain = {
        "INFERINDO": 0.65,
        "ONDA_INICIO": 0.80,
        "FISSAO": 1.00,
        "ONDA_FIM": 0.35,
    }.get(phase, 0.65)
    cells = []
    for index in range(64):
        oscillation = (math.sin(step * 0.22 + index * 0.71) + 1.0) / 2.0
        pulse = min(1.0, token_estimate / 256.0)
        value = min(0.999, (0.15 + 0.55 * pulse + 0.30 * oscillation) * phase_gain)
        cells.append(levels[int(value * (len(levels) - 1))])
    rows = ["".join(cells[row * 16:(row + 1) * 16]) for row in range(4)]
    return "╭" + "─" * 16 + "╮\n" + "\n".join(
        f"│{row}│" for row in rows
    ) + "\n╰" + "─" * 16 + "╯"


class LiveOrchestrator:
    """Client-side event orchestrator for safe real-time CLI telemetry."""

    def __init__(self) -> None:
        self.phase = "INICIO"
        self.wave = 0
        self.tokens = 0
        self.started = time.monotonic()

    def consume(self, text: str) -> None:
        self.tokens += len(re.findall(r"\S+", text))
        if "FISSÃO" in text:
            self.wave += 1
            self.phase = "FISSAO"
        elif "Onda Concluída" in text:
            self.phase = "FIM"
        elif self.tokens > 24 and self.phase == "INICIO":
            self.phase = "MEIO"

    def status(self, received: int) -> str:
        elapsed = time.monotonic() - self.started
        return (
            f"[AEON | {self.phase} | onda={self.wave} | "
            f"~{self.tokens} tok | {elapsed:.1f}s | {received} B]"
        )


def stream_prompt(base_url: str, prompt: str) -> int:
    request = Request(
        f"{base_url.rstrip('/')}/api/v1/chat",
        data=json.dumps(
            {"prompt": prompt, "telemetry": True}, ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        started = time.monotonic()
        received = 0
        token_estimate = 0
        orchestrator = LiveOrchestrator()
        last_event = "INICIO"
        print("\033[90m[AEON | ORQUESTAÇÃO | INFERINDO]\033[0m", file=sys.stderr)
        print(_render_particle_grid(0, "INFERINDO", 0), file=sys.stderr)
        with urlopen(request, timeout=600) as response:
            buffer = ""
            for chunk in iter(lambda: response.read(4096), b""):
                buffer += chunk.decode("utf-8", errors="replace")
                frames = buffer.split("\n\n")
                buffer = frames.pop()
                for frame in frames:
                    data_line = next(
                        (line[6:] for line in frame.splitlines()
                         if line.startswith("data: ")),
                        None,
                    )
                    if data_line is None:
                        continue
                    try:
                        event_data = json.loads(data_line)
                    except json.JSONDecodeError:
                        continue
                    text = str(event_data.get("text", ""))
                    event_phase = str(event_data.get("phase", ""))
                    if event_phase == "fissao":
                        text_for_state = "FISSÃO"
                    elif event_phase == "inicio":
                        text_for_state = "Onda Híbrida"
                    elif event_phase == "fim":
                        text_for_state = "Onda Concluída"
                    else:
                        text_for_state = text
                    received += len(text.encode("utf-8"))
                    token_estimate += len(re.findall(r"\S+", text))
                    orchestrator.consume(text_for_state)
                    event = _event_label(text_for_state)
                    if event:
                        last_event = event
                    if orchestrator.phase == "FISSAO":
                        last_event = "FISSAO"
                    elif orchestrator.phase == "FIM":
                        last_event = "ONDA_FIM"
                    print("\033[6A", end="", file=sys.stderr)
                    print(
                        _render_particle_grid(
                            token_estimate, last_event, orchestrator.tokens
                        ),
                        file=sys.stderr,
                    )
                    print(
                        f"\r\033[90m{orchestrator.status(received)}\033[0m",
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                    sys.stdout.write(text)
                    sys.stdout.flush()
        print(
            f"\r\033[92m[AEON | CONCLUÍDO | onda={orchestrator.wave} | "
            f"~{orchestrator.tokens} tok | "
            f"{time.monotonic() - started:.1f}s]\033[0m",
            file=sys.stderr,
        )
        sys.stdout.write("\n")
        return 0
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"[aeon-cli] API HTTP {exc.code}: {detail}", file=sys.stderr)
    except (URLError, TimeoutError) as exc:
        print(f"[aeon-cli] conexão indisponível: {exc}", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Conversa com o Aeon já carregado, sem iniciar outro modelo."
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:20000", help="URL base da API"
    )
    args = parser.parse_args()

    print("Aeon CLI conectado. Digite /exit para sair.")
    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/exit", "/quit"}:
            return 0
        stream_prompt(args.url, prompt)


if __name__ == "__main__":
    raise SystemExit(main())
