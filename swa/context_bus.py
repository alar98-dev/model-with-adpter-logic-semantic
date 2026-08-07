"""
Context Bus — Shared Context Window in Memory

Todos os agentes externos (OpenCode, Copilot, Cursor, etc.) escrevem
seus tokens aqui. Aeon intercepta e compartilha a mesma janela de
contexto diretamente na memória via mmap.

Arquivo: /dev/shm/aeon_context_bus.dat
Capacidade: 4096 entradas em ring buffer (~4MB)
Acesso: zero-copy via mmap para todos os processos locais
"""
import os
import mmap
import struct
import time
import json
import threading
from enum import IntEnum
from typing import Optional, Callable, List, Tuple

CONTEXT_BUS_PATH = "/dev/shm/aeon_context_bus.dat"
MAX_AGENTS = 32
MAX_ENTRIES = 8192
ENTRY_SIZE = 384
HEADER_SIZE = 4096
BUS_SIZE = HEADER_SIZE + (MAX_ENTRIES * ENTRY_SIZE)

HEADER_FMT = "<IIIIIIIIQ"
HEADER_LEN = struct.calcsize(HEADER_FMT)
assert HEADER_LEN <= HEADER_SIZE, "Header struct excede HEADER_SIZE"

ENTRY_FMT = "<IIIQHBB192s"
ENTRY_LEN = struct.calcsize(ENTRY_FMT)

class Role(IntEnum):
    SYSTEM = 0
    USER = 1
    ASSISTANT = 2
    INTERCEPT = 3
    INJECTED = 4

class Flag(IntEnum):
    NONE = 0
    INTERCEPTED = 1
    INJECTED = 2
    AEON_MODULATED = 4
    BROADCAST = 8

class ContextBusHeader:
    __slots__ = ("magic", "version", "num_agents", "write_seq", "read_seq",
                 "capacity", "entry_size", "flags", "created_at")
    
    def __init__(self):
        self.magic = 0xAE0CB0C0
        self.version = 1
        self.num_agents = 0
        self.write_seq = 0
        self.read_seq = 0
        self.capacity = MAX_ENTRIES
        self.entry_size = ENTRY_SIZE
        self.flags = 0
        self.created_at = int(time.time() * 1000)
    
    def pack(self) -> bytes:
        return struct.pack(HEADER_FMT, self.magic, self.version, self.num_agents,
                          self.write_seq, self.read_seq, self.capacity,
                          self.entry_size, self.flags, self.created_at)
    
    @classmethod
    def unpack(cls, data: bytes) -> "ContextBusHeader":
        h = cls()
        (h.magic, h.version, h.num_agents, h.write_seq, h.read_seq,
         h.capacity, h.entry_size, h.flags, h.created_at) = struct.unpack(HEADER_FMT, data)
        return h


class ContextEntry:
    __slots__ = ("seq", "agent_id", "token_id", "timestamp", "token_len",
                 "role", "flags", "token_text")
    
    def __init__(self):
        self.seq = 0
        self.agent_id = 0
        self.token_id = 0
        self.timestamp = 0
        self.token_len = 0
        self.role = 0
        self.flags = 0
        self.token_text = ""
    
    def pack(self) -> bytes:
        text_bytes = self.token_text.encode("utf-8")[:192].ljust(192, b"\x00")
        return struct.pack(ENTRY_FMT, self.seq, self.agent_id, self.token_id,
                          self.timestamp, self.token_len, self.role, self.flags, text_bytes)
    
    @classmethod
    def unpack(cls, data: bytes) -> "ContextEntry":
        e = cls()
        (e.seq, e.agent_id, e.token_id, e.timestamp, e.token_len,
         e.role, e.flags, text_bytes) = struct.unpack(ENTRY_FMT, data)
        e.token_text = text_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        return e


class ContextBus:
    """Shared context window via mmap — todos os agentes compartilham a mesma memória."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, path: str = CONTEXT_BUS_PATH, create: bool = True):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.path = path
        self._mm = None
        self._file = None
        self._callbacks: List[Callable] = []
        self._agent_name: Optional[str] = None
        self._agent_id: int = -1
        
        if create:
            self._ensure_bus()
    
    def _ensure_bus(self):
        if not os.path.exists(self.path):
            self._create_bus()
        self._open_bus()
    
    def _create_bus(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as f:
            header = ContextBusHeader()
            f.write(header.pack())
            f.write(b"\x00" * (HEADER_SIZE - HEADER_LEN))
            f.write(b"\x00" * (MAX_ENTRIES * ENTRY_SIZE))
        os.chmod(self.path, 0o666)
    
    def _open_bus(self):
        self._file = open(self.path, "r+b")
        self._mm = mmap.mmap(self._file.fileno(), BUS_SIZE, access=mmap.ACCESS_WRITE)
        self._validate_header()
    
    def _validate_header(self):
        data = self._mm[:HEADER_LEN]
        h = ContextBusHeader.unpack(data)
        assert h.magic == 0xAE0CB0C0, f"Magic inválido: {h.magic:#x}"
        assert h.version == 1, f"Versão inválida: {h.version}"
    
    def _read_header(self) -> ContextBusHeader:
        return ContextBusHeader.unpack(self._mm[:HEADER_LEN])
    
    def _write_header(self, h: ContextBusHeader):
        self._mm[:HEADER_LEN] = h.pack()
    
    def _entry_offset(self, seq: int) -> int:
        idx = seq % self._read_header().capacity
        return HEADER_SIZE + (idx * ENTRY_SIZE)
    
    def register_agent(self, name: str) -> int:
        """Registra um agente no barramento. Retorna agent_id."""
        h = self._read_header()
        for i in range(h.num_agents):
            offset = HEADER_LEN + (i * 48)
            existing = self._mm[offset:offset+48].rstrip(b"\x00").decode("utf-8", errors="replace")
            if existing == name:
                self._agent_id = i
                self._agent_name = name
                return i
        
        if h.num_agents >= MAX_AGENTS:
            raise RuntimeError(f"Máximo de {MAX_AGENTS} agentes atingido")
        
        aid = h.num_agents
        h.num_agents += 1
        self._write_header(h)
        
        offset = HEADER_LEN + (aid * 48)
        name_bytes = name.encode("utf-8").ljust(48, b"\x00")[:48]
        self._mm[offset:offset+48] = name_bytes
        
        self._agent_id = aid
        self._agent_name = name
        return aid
    
    def write_token(self, text: str, role: Role = Role.USER, flags: Flag = Flag.NONE) -> int:
        """Escreve um token no contexto compartilhado. Retorna sequence number."""
        if self._agent_id < 0:
            raise RuntimeError("Agent não registrado. Chame register_agent() primeiro.")
        
        h = self._read_header()
        seq = h.write_seq
        h.write_seq += 1
        self._write_header(h)
        
        entry = ContextEntry()
        entry.seq = seq
        entry.agent_id = self._agent_id
        entry.token_id = seq
        entry.timestamp = int(time.time() * 1000)
        entry.token_len = len(text.encode("utf-8"))
        entry.role = role.value
        entry.flags = flags.value
        entry.token_text = text
        
        offset = self._entry_offset(seq)
        self._mm[offset:offset+ENTRY_LEN] = entry.pack()
        
        self._notify_callbacks(entry)
        return seq
    
    def read_context(self, start_seq: int = 0, max_entries: int = 0) -> List[ContextEntry]:
        """Lê o contexto compartilhado a partir de start_seq."""
        h = self._read_header()
        if max_entries <= 0:
            max_entries = h.capacity
        
        entries = []
        for seq in range(start_seq, h.write_seq):
            if len(entries) >= max_entries:
                break
            offset = self._entry_offset(seq)
            raw = self._mm[offset:offset+ENTRY_LEN]
            if raw[0:4] == b"\x00\x00\x00\x00":
                continue
            entry = ContextEntry.unpack(raw)
            entries.append(entry)
        
        return entries
    
    def read_all(self, start_seq: int = 0, max_entries: int = 0) -> list[dict]:
        """Lê todo o contexto atual como lista de dicionários (compat com BlakeObserver)."""
        return [
            {
                "agent_id": e.agent_id,
                "role": e.role,
                "tokens": e.token_text,
                "blake_signature": None,
                "timestamp": e.timestamp,
                "seq": e.seq,
            }
            for e in self.read_context(start_seq, max_entries)
        ]

    def read_context_text(self, start_seq: int = 0, separator: str = "") -> str:
        """Retorna o contexto como texto contínuo."""
        entries = self.read_context(start_seq)
        texts = []
        for e in entries:
            if e.role == Role.USER:
                texts.append(f"[USER] {e.token_text}")
            elif e.role == Role.ASSISTANT:
                texts.append(f"[AEON] {e.token_text}")
            elif e.role == Role.SYSTEM:
                texts.append(f"[SYS] {e.token_text}")
            elif e.role == Role.INTERCEPT:
                texts.append(f"[INTERCEPT] {e.token_text}")
            elif e.role == Role.INJECTED:
                texts.append(f"[INJECT] {e.token_text}")
        return separator.join(texts)
    
    def read_full_context(self, separator: str = " ") -> str:
        """Lê todo o contexto do bus como texto."""
        return self.read_context_text(0, separator)
    
    def intercept(self, token: str, source_agent: str = "unknown") -> int:
        """Intercepta um token de qualquer agente externo."""
        if self._agent_id < 0:
            self.register_agent("aeon-intercept")
        return self.write_token(
            f"[INTERCEPT:{source_agent}] {token}",
            role=Role.INTERCEPT,
            flag=Flag.INTERCEPTED
        )
    
    def inject(self, token: str) -> int:
        """Aeon injeta seu próprio token no contexto compartilhado."""
        if self._agent_id < 0:
            self.register_agent("aeon")
        return self.write_token(
            token,
            role=Role.INJECTED,
            flag=Flag.INJECTED
        )
    
    def broadcast(self, event_type: str, payload: dict):
        """Transmite um evento no barramento (via flag BROADCAST)."""
        data = json.dumps({"type": event_type, "payload": payload})
        seq = self.write_token(
            data,
            role=Role.SYSTEM,
            flag=Flag.BROADCAST
        )
        return seq
    
    def on_token(self, callback: Callable):
        """Registra callback para novos tokens no barramento."""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self, entry: ContextEntry):
        for cb in self._callbacks:
            try:
                cb(entry)
            except Exception:
                pass
    
    @property
    def write_seq(self) -> int:
        return self._read_header().write_seq
    
    @property
    def agent_id(self) -> int:
        return self._agent_id
    
    @property
    def stats(self) -> dict:
        h = self._read_header()
        return {
            "magic": f"{h.magic:#x}",
            "version": h.version,
            "num_agents": h.num_agents,
            "write_seq": h.write_seq,
            "read_seq": h.read_seq,
            "capacity": h.capacity,
            "entry_size": h.entry_size,
            "created_at": h.created_at,
            "bus_size": BUS_SIZE,
            "usage_pct": (h.write_seq / h.capacity) * 100 if h.capacity > 0 else 0,
            "agents": self._list_agents(),
        }
    
    def _list_agents(self) -> list:
        agents = []
        h = self._read_header()
        for i in range(h.num_agents):
            offset = HEADER_LEN + (i * 48)
            name = self._mm[offset:offset+48].rstrip(b"\x00").decode("utf-8", errors="replace")
            agents.append(name)
        return agents
    
    def __del__(self):
        self.close()

    def close(self):
        if self._mm:
            try:
                self._mm.flush()
                self._mm.close()
            except Exception:
                pass
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    import sys
    
    with ContextBus() as bus:
        if len(sys.argv) > 1 and sys.argv[1] == "stats":
            print(json.dumps(bus.stats, indent=2))
        elif len(sys.argv) > 1 and sys.argv[1] == "read":
            text = bus.read_full_context("\n")
            print(text if text else "(vazio)")
        elif len(sys.argv) > 1 and sys.argv[1] == "watch":
            print("Observando contexto compartilhado (Ctrl+C para sair)...")
            last_seq = bus.write_seq
            try:
                while True:
                    if bus.write_seq > last_seq:
                        entries = bus.read_context(last_seq)
                        for e in entries:
                            _role_name = ["SYS", "USER", "ASST", "INT", "INJ"][e.role]
                            print(f"[{_role_name}] {e.token_text}")
                        last_seq = bus.write_seq
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nEncerrado.")
        else:
            print("Uso: context_bus.py [stats|read|watch]")
