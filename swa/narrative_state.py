"""Persistent narrative state for long, multi-wave generation."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Iterable, Optional

from swa.paths import project_path


class NarrativeState:
    """Small append-only journal with a bounded, recoverable narrative view."""

    VERSION = 1

    def __init__(self, path: Optional[str] = None, *, max_events: int = 128):
        self._logger = logging.getLogger("swa.narrative")
        self.path = path or os.environ.get(
            "SWA_NARRATIVE_STATE_PATH",
            str(project_path("narrative_state.jsonl")),
        )
        self.max_events = max(8, int(max_events))
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        self._logger.warning("ignoring malformed narrative record: %s", exc)
                        continue
                    if isinstance(record, dict):
                        self._events.append(record)
            self._events = self._events[-self.max_events :]
        except OSError:
            return

    def append(self, kind: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        record = {
            "version": self.VERSION,
            "ts": time.time(),
            "kind": str(kind),
            "payload": dict(payload or {}),
        }
        with self._lock:
            self._events.append(record)
            self._events = self._events[-self.max_events :]
            try:
                parent = os.path.dirname(self.path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                # The in-memory state remains usable when persistence is unavailable.
                self._logger.warning("narrative persistence unavailable at %s: %s", self.path, exc)
        return record

    def record_turn(
        self,
        *,
        user: str,
        reply: str,
        wave: int = 0,
        summary: str = "",
        entities: Optional[Iterable[str]] = None,
        objectives: Optional[Iterable[str]] = None,
        time_context: str = "",
        location: str = "",
    ) -> dict[str, Any]:
        inferred_entities, inferred_objectives, inferred_time, inferred_location = self.infer_structure(
            f"{user}\n{reply}"
        )
        return self.append(
            "wave",
            {
                "user": (user or "")[-2000:],
                "reply": (reply or "")[-4000:],
                "wave": int(wave),
                "summary": summary or (reply or "")[-600:],
                "entities": list(entities or inferred_entities),
                "objectives": list(objectives or inferred_objectives),
                "time": time_context or inferred_time,
                "location": location or inferred_location,
            },
        )

    @staticmethod
    def infer_structure(text: str) -> tuple[list[str], list[str], str, str]:
        """Extract lightweight narrative slots without introducing an NLP dependency."""
        text = text or ""
        entities = list(dict.fromkeys(re.findall(r"\b[A-ZÁÀÃÉÊÍÓÔÕÚÜ][\wÁÀÃÉÊÍÓÔÕÚÜ-]{2,}", text)))[:8]
        objective_matches = re.findall(
            r"(?i)(?:objetivo|meta|precisa|deve|deverá|para)\s+([^.!?\n]{3,100})", text
        )
        objectives = list(dict.fromkeys(v.strip(" ,:;") for v in objective_matches))[:5]
        time_match = re.search(
            r"(?i)\b(agora|hoje|amanhã|ontem|amanhecer|manhã|tarde|noite|século|dia \d+)\b", text
        )
        location_match = re.search(
            r"(?i)\b(?:em|no|na|nos|nas)\s+([A-Za-zÀ-ÿ][\wÀ-ÿ-]{2,}(?:\s+[A-Za-zÀ-ÿ][\wÀ-ÿ-]{2,})?)", text
        )
        return entities, objectives, time_match.group(1) if time_match else "", (
            location_match.group(1) if location_match else ""
        )

    def record_event(
        self,
        event_type: str,
        summary: str,
        *,
        wave: int = 0,
        entities: Optional[Iterable[str]] = None,
        objectives: Optional[Iterable[str]] = None,
        time_context: str = "",
        location: str = "",
    ) -> dict[str, Any]:
        """Persist a structured event independently of a user/assistant turn."""
        return self.append(
            event_type,
            {
                "summary": summary,
                "wave": int(wave),
                "entities": list(entities or []),
                "objectives": list(objectives or []),
                "time": time_context,
                "location": location,
            },
        )

    def context(self, *, limit: int = 6) -> str:
        with self._lock:
            records = [r for r in self._events if isinstance(r.get("payload"), dict)][-limit:]
        if not records:
            return ""
        lines = ["[ESTADO NARRATIVO PERSISTENTE]"]
        for record in records:
            data = record.get("payload", {})
            summary = data.get("summary") or data.get("reply", "")
            line = f"- onda {data.get('wave', 0)}: {summary}"
            if data.get("entities"):
                line += f" | entidades: {', '.join(data['entities'])}"
            if data.get("objectives"):
                line += f" | objetivos: {', '.join(data['objectives'])}"
            if data.get("time") or data.get("location"):
                line += f" | tempo/local: {data.get('time', '')} {data.get('location', '')}".rstrip()
            lines.append(line)
        return "\n".join(lines)

    def recover(self, *, limit: int = 6) -> dict[str, Any]:
        with self._lock:
            records = [r for r in self._events if isinstance(r.get("payload"), dict)][-limit:]
        entities: list[str] = []
        objectives: list[str] = []
        for record in records:
            data = record.get("payload", {})
            entities.extend(str(v) for v in data.get("entities", []))
            objectives.extend(str(v) for v in data.get("objectives", []))
        return {
            "events": records,
            "entities": list(dict.fromkeys(entities)),
            "objectives": list(dict.fromkeys(objectives)),
            "context": self.context(limit=limit),
        }
