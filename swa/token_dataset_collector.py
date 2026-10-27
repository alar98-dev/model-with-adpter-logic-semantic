import os
import sys
import json
import time
import socket
import struct
import threading
import sqlite3
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from swa.paths import project_path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from swa.context_bus import ContextBus, Role, Flag
    HAS_CONTEXT_BUS = True
except ImportError:
    HAS_CONTEXT_BUS = False

DATASET_DIR = os.environ.get(
    "TOKEN_DATASET_DIR",
    str(project_path("models", "token_dataset"))
)
OPENCODE_DB = os.path.expanduser("~/.local/share/opencode/opencode.db")
UDP_PORT = int(os.environ.get("TOKEN_COLLECTOR_UDP_PORT", "9098"))
UDP_HOST = os.environ.get("TOKEN_COLLECTOR_HOST", "127.0.0.1")
BLAKE_SALT = hashlib.sha256(
    os.environ.get("BLAKE_SALT", "aeon-decision-trace-v1").encode()
).digest()

os.makedirs(DATASET_DIR, exist_ok=True)

PartType = Literal[
    "user_text",
    "assistant_text",
    "reasoning",
    "tool_call",
    "tool_result",
    "step_control",
    "patch",
    "compaction",
]

PART_TYPE_MAP = {
    ("user", "text"): "user_text",
    ("assistant", "text"): "assistant_text",
    ("assistant", "reasoning"): "reasoning",
    ("assistant", "tool"): "tool_call",
    ("assistant", "step-start"): "step_control",
    ("assistant", "step-finish"): "step_control",
    ("assistant", "compaction"): "compaction",
    ("assistant", "patch"): "patch",
    ("user", "compaction"): "compaction",
}


def classify_part_type(role: str, part_data: dict) -> PartType:
    ptype = part_data.get("type", "?")
    key = (role, ptype)
    return PART_TYPE_MAP.get(key, f"unknown_{role}_{ptype}")


def blake3_digest(text: str) -> str:
    try:
        import blake3
        h = blake3.blake3(key=BLAKE_SALT)
        h.update(text.encode("utf-8"))
        return h.hexdigest()
    except ImportError:
        return hashlib.sha256(BLAKE_SALT + text.encode("utf-8")).hexdigest()


def semantic_signature(part: dict, context_parts: list) -> list[float]:
    """6-eixo Blake Gravity para uma parte da conversa."""
    text = part.get("text", "")
    ptype = part.get("part_type", "unknown")
    tokens = part.get("tokens_total", 0)

    ex0 = len(text) / 8192.0
    ex1 = (tokens % 1024) / 1024.0
    h = blake3_digest(text)
    ex2 = int(h[:2], 16) / 255.0
    ex3 = sum(ord(c) for c in h[:4]) / 1020.0
    depth = len(context_parts)
    depth = depth / 256.0 if depth > 256 else depth / 256.0
    ex4 = depth
    type_seed = abs(hash(ptype)) % 1000 / 1000.0
    ex5 = type_seed

    magnitude = sum(abs(v) for v in [ex0, ex1, ex2, ex3, ex4, ex5])
    if magnitude > 0:
        norm = [v / magnitude for v in [ex0, ex1, ex2, ex3, ex4, ex5]]
    else:
        norm = [0.0] * 6

    return norm


def classify_trace(tensor: list[float]) -> str:
    ex0, ex1, ex2, ex3, ex4, ex5 = tensor[:6]
    if ex5 > 0.3 and ex0 > 0.2:
        return "decision_point"
    if ex2 > 0.3:
        return "tool_execution"
    if ex1 > 0.2:
        return "reasoning"
    if ex0 < 0.02:
        return "control_flow"
    if ex5 > 0.2:
        return "text_response"
    return "intermediate"


def compute_decision_chain(conversation: list) -> dict:
    """Mapeia decisões: USER → REASONING → TOOL → TOOL-RESULT → TEXT."""
    chains = []
    current = None
    for part in conversation:
        pt = part["part_type"]
        if pt == "user_text":
            current = {"trigger": part["text"][:100], "steps": [], "final": ""}
        elif pt == "tool_call":
            if current is not None:
                current["steps"].append({
                    "type": "tool",
                    "tool": part.get("tool_name", "?"),
                    "blake": part["blake_signature"]
                })
        elif pt == "assistant_text":
            if current is not None:
                current["final"] = part["text"][:100]
                chains.append(current)
                current = None
        elif pt == "reasoning":
            if current is not None:
                current["steps"].append({
                    "type": "reasoning",
                    "blake": part["blake_signature"]
                })
    if current is not None:
        chains.append(current)
    return {
        "chain_count": len(chains),
        "chains": chains,
    }


def compute_semantic_geometry(conversation: list) -> dict:
    """Tensor de geometria semântica completa da sessão."""
    if not conversation:
        return {"error": "empty"}
    type_counts = {}
    total_tokens = 0
    for p in conversation:
        pt = p["part_type"]
        type_counts[pt] = type_counts.get(pt, 0) + 1
        total_tokens += p.get("tokens_total", 0) or 0
    tensors = [p["blake_signature"] for p in conversation if p.get("blake_signature")]
    if tensors:
        centroid = [sum(e) / len(tensors) for e in zip(*tensors)]
    else:
        centroid = [0.0] * 6
    span = [
        max(e) - min(e) if tensors else 0.0
        for e in zip(*tensors)
    ] if tensors else [0.0] * 6
    return {
        "part_count": len(conversation),
        "type_distribution": type_counts,
        "total_tokens": total_tokens,
        "centroid": centroid,
        "span": span,
        "classification": classify_trace(centroid),
    }


class TokenDatasetCollector:
    def __init__(self, dataset_dir: str = DATASET_DIR):
        self.dataset_dir = dataset_dir
        self._lock = threading.Lock()
        self._sessions_file = os.path.join(dataset_dir, "sessions.jsonl")
        self._parts_file = os.path.join(dataset_dir, "parts.jsonl")
        self._geometry_file = os.path.join(dataset_dir, "geometry.jsonl")
        self._opencode_file = os.path.join(dataset_dir, "opencode_sessions.jsonl")
        self._telemetry_file = os.path.join(dataset_dir, "telemetry.jsonl")
        self._sessions_cache = set()
        self._bus = None
        self._running = False
        if HAS_CONTEXT_BUS:
            try:
                self._bus = ContextBus()
                self._bus.register_agent("token-collector")
            except Exception:
                pass

    def ingest_token(self, agent: str, text: str, role: str = "user",
                     session_id: str = "", metadata: dict = None) -> dict:
        entry = {
            "agent": agent,
            "role": role,
            "text": text,
            "session_id": session_id or "",
            "timestamp": time.time(),
            "datetime": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "token_len": len(text.encode("utf-8")),
            "blake": blake3_digest(text),
        }
        self._write_entry(self._sessions_file, entry)
        if self._bus:
            try:
                role_map = {"user": Role.USER, "assistant": Role.ASSISTANT,
                            "system": Role.SYSTEM, "intercept": Role.INTERCEPT,
                            "inject": Role.INJECTED}
                bus_role = role_map.get(role, Role.USER)
                self._bus.write_token(text[:192], role=bus_role)
            except Exception:
                pass
        return {"seq": entry["timestamp"], "stored": True}

    def ingest_batch(self, agent: str, messages: list,
                     session_id: str = "") -> dict:
        count = 0
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", msg.get("text", ""))
            if text.strip():
                self.ingest_token(agent, text, role, session_id)
                count += 1
        return {"ingested": count}

    def collect_opencode_sessions(self, db_path: str = OPENCODE_DB,
                                  limit: int = 5) -> dict:
        if not os.path.exists(db_path):
            return {"error": f"OpenCode DB nao encontrado: {db_path}"}
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
        except Exception as e:
            return {"error": str(e)}
        stats = {"sessions": 0, "parts": 0, "tokens_input": 0, "tokens_output": 0}
        cur.execute("""
            SELECT s.id, s.title, s.model, s.agent, s.tokens_input, s.tokens_output,
                   s.tokens_reasoning, s.cost, s.time_created, s.time_updated
            FROM session s
            WHERE s.tokens_input > 0
            ORDER BY s.time_updated DESC
            LIMIT ?
        """, (limit,))
        sessions = cur.fetchall()
        for s in sessions:
            sid = s["id"]
            if sid in self._sessions_cache:
                continue
            self._sessions_cache.add(sid)
            try:
                model_data = json.loads(s["model"]) if isinstance(s["model"], str) else {}
            except Exception:
                model_data = {}
            raw_messages = self._extract_messages(cur, sid)
            conversation = self._build_conversation(cur, raw_messages)
            decision_chain = compute_decision_chain(conversation)
            geometry = compute_semantic_geometry(conversation)
            session_entry = {
                "source": "opencode",
                "session_id": sid,
                "title": s["title"],
                "model_id": model_data.get("id", "?"),
                "provider": model_data.get("providerID", "?"),
                "agent": s["agent"],
                "tokens_input": s["tokens_input"],
                "tokens_output": s["tokens_output"],
                "tokens_reasoning": s["tokens_reasoning"],
                "cost": s["cost"],
                "time_created": s["time_created"],
                "time_updated": s["time_updated"],
                "message_count": len(raw_messages),
                "part_count": len(conversation),
                "conversation": conversation,
                "decision_chain": decision_chain,
                "semantic_geometry": geometry,
                "session_blake": blake3_digest(s["title"] or sid),
            }
            stats["sessions"] += 1
            stats["tokens_input"] += s["tokens_input"] or 0
            stats["tokens_output"] += s["tokens_output"] or 0
            self._write_entry(self._opencode_file, session_entry)
            for part in conversation:
                self._write_entry(self._parts_file, part)
            self._write_entry(self._geometry_file, geometry)
        conn.close()
        stats["cached_sessions"] = len(self._sessions_cache)
        return stats

    def _extract_messages(self, cur, session_id: str) -> list:
        cur.execute("""
            SELECT id, time_created, data
            FROM message
            WHERE session_id = ?
            ORDER BY time_created ASC
        """, (session_id,))
        return cur.fetchall()

    def _build_conversation(self, cur, raw_messages: list) -> list:
        conversation = []
        for m in raw_messages:
            try:
                msg_data = json.loads(m["data"]) if isinstance(m["data"], str) else {}
            except Exception:
                msg_data = {}
            role = msg_data.get("role", "?")
            agent = msg_data.get("agent", "?")
            tokens = msg_data.get("tokens", {})
            tokens_total = tokens.get("total", 0) if isinstance(tokens, dict) else 0
            finish = msg_data.get("finish", "")
            cur.execute("""
                SELECT id, data, time_created
                FROM part
                WHERE message_id = ?
                ORDER BY time_created ASC
            """, (m["id"],))
            parts = cur.fetchall()
            for p in parts:
                try:
                    pdata = json.loads(p["data"]) if isinstance(p["data"], str) else {}
                except Exception:
                    pdata = {}
                ptype = pdata.get("type", "?")
                text = pdata.get("text", "")
                part_type = classify_part_type(role, pdata)
                tool_name = pdata.get("tool", "")
                call_id = pdata.get("callID", "")
                finish_reason = pdata.get("reason", "") if ptype == "step-finish" else finish
                state = pdata.get("state", {})
                tool_input = state.get("input", {})
                tool_output_raw = state.get("output", "")
                tool_output = tool_output_raw[:200].replace("\n", " ") if tool_output_raw else ""
                entry = {
                    "session_id": m["id"].rsplit("_", 1)[0] if "_" in m["id"] else "",
                    "message_id": m["id"],
                    "part_id": p["id"],
                    "time_created": p["time_created"],
                    "role": role,
                    "agent": agent,
                    "part_type": part_type,
                    "type_detail": ptype,
                    "text": text[:500] if text else "",
                    "tool_name": tool_name,
                    "tool_call_id": call_id,
                    "tool_input": str(tool_input)[:200] if tool_input else "",
                    "tool_output": tool_output,
                    "finish_reason": finish_reason,
                    "tokens_total": tokens_total,
                    "blake": blake3_digest(text or tool_name or str(p["id"])),
                }
                sig = semantic_signature(entry, conversation)
                trace_label = classify_trace(sig)
                entry["blake_signature"] = sig
                entry["trace_classification"] = trace_label
                conversation.append(entry)
        return conversation

    def start_udp_server(self, port: int = UDP_PORT):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((UDP_HOST, port))
        print(f"[TOKEN COLLECTOR] UDP server listening :{port}")
        while self._running:
            try:
                data, addr = sock.recvfrom(1024)
                if len(data) >= 32:
                    floats = struct.unpack("<8f", data[:32])
                    entry = {
                        "source": "win_collector",
                        "addr": addr[0],
                        "cpu_usage": floats[0],
                        "mem_used": floats[1],
                        "swap_used": floats[2],
                        "net_in": floats[3],
                        "net_out": floats[4],
                        "cpu_cores": floats[5],
                        "entropy": floats[6],
                        "reserved": floats[7],
                        "timestamp": time.time(),
                        "blake": blake3_digest(str(floats)),
                    }
                    self._write_entry(self._telemetry_file, entry)
            except Exception as e:
                print(f"[TOKEN COLLECTOR] UDP error: {e}")

    def _write_entry(self, path: str, entry: dict):
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[TOKEN COLLECTOR] Write error {path}: {e}")

    def start(self):
        self._running = True
        udp_thread = threading.Thread(target=self.start_udp_server,
                                       daemon=True, name="TokenCollector-UDP")
        udp_thread.start()
        def polling_loop():
            while self._running:
                try:
                    self.collect_opencode_sessions(limit=3)
                except Exception as e:
                    print(f"[TOKEN COLLECTOR] Polling error: {e}")
                time.sleep(30)
            print("[TOKEN COLLECTOR] Polling loop ended.")
        poll_thread = threading.Thread(target=polling_loop,
                                        daemon=True, name="TokenCollector-Poll")
        poll_thread.start()
        print(f"[TOKEN COLLECTOR] Dataset dir: {self.dataset_dir}")
        print(f"[TOKEN COLLECTOR] Sessions: {self._sessions_file}")
        print(f"[TOKEN COLLECTOR] Parts: {self._parts_file}")
        print(f"[TOKEN COLLECTOR] Geometry: {self._geometry_file}")
        print(f"[TOKEN COLLECTOR] OpenCode: {self._opencode_file}")
        return self

    def stop(self):
        self._running = False

    def stats(self) -> dict:
        def count_lines(path):
            try:
                with open(path) as f:
                    return sum(1 for _ in f)
            except FileNotFoundError:
                return 0
        return {
            "sessions_lines": count_lines(self._sessions_file),
            "parts_lines": count_lines(self._parts_file),
            "geometry_lines": count_lines(self._geometry_file),
            "opencode_sessions_lines": count_lines(self._opencode_file),
            "telemetry_lines": count_lines(self._telemetry_file),
            "cached_opencode_sessions": len(self._sessions_cache),
        }


_COLLECTOR_INSTANCE = None

def get_collector() -> TokenDatasetCollector:
    global _COLLECTOR_INSTANCE
    if _COLLECTOR_INSTANCE is None:
        _COLLECTOR_INSTANCE = TokenDatasetCollector()
        _COLLECTOR_INSTANCE.start()
    return _COLLECTOR_INSTANCE


if __name__ == "__main__":
    import signal
    collector = get_collector()
    print(json.dumps(collector.stats(), indent=2))
    def signal_handler(sig, frame):
        print("\nStopping...")
        collector.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
