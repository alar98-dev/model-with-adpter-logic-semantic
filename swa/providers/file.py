from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional

import torch

from swa.providers.abc import AbstractStateProvider
from swa.providers.ram import RAMStateProvider

_HEADER_FORMAT = "=I"  # 4-byte unsigned int for slot count
_SLOT_HEADER_FORMAT = "=32sII"  # name (32 bytes), offset, size


class FileStateProvider(AbstractStateProvider):
    """File-backed state provider for testing and development.

    Each named state is serialised as a contiguous tensor dump inside a
    simple binary container file.  Because the file is pre-allocated,
    random-access reads and writes are efficient.

    .. caution::
       This is **not** a concurrent-safe format.  It is designed for single-
       process testing only.
    """

    def __init__(self, path: str | Path, vector_dim: int = 2560, bias_dim: int = 2560) -> None:
        self._path = Path(path)
        self._vector_dim = vector_dim
        self._bias_dim = bias_dim
        self._ram = RAMStateProvider(vector_dim, bias_dim)

        # If the file already exists, replay its contents into RAM.
        if self._path.exists():
            self._load()
        # Reserve space on disk to allow seek-based writes.
        else:
            self._flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_state(self, state_name: str, data: torch.Tensor) -> bool:
        self._ram.write_state(state_name, data)
        self._flush()
        return True

    def read_state(self, state_name: str) -> Optional[torch.Tensor]:
        return self._ram.read_state(state_name)

    def read_states(self, state_names: list[str]) -> Dict[str, Optional[torch.Tensor]]:
        return {name: self.read_state(name) for name in state_names}

    def available_slots(self) -> frozenset[str]:
        return self._ram.available_slots()

    def close(self) -> None:
        self._ram.close()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _flush(self) -> None:
        slots = self._ram._slots  # noqa: SLF001  — intentional for serialised access
        self._path.parent.mkdir(parents=True, exist_ok=True)

        with self._path.open("wb") as f:
            # Header: number of slots
            f.write(struct.pack(_HEADER_FORMAT, len(slots)))

            # We need two passes: first compute offsets, then write data.
            entries: list[tuple[str, int, int]] = []
            data_offset = struct.calcsize(_HEADER_FORMAT) + len(slots) * struct.calcsize(
                _SLOT_HEADER_FORMAT
            )
            cursor = data_offset

            for name, tensor in slots.items():
                nbytes = tensor.numel() * tensor.element_size()
                entries.append((name, cursor, nbytes))
                cursor += nbytes

            # Write slot headers
            for name, offset, size in entries:
                name_bytes = name.encode("utf-8").ljust(32, b"\x00")[:32]
                f.write(struct.pack(_SLOT_HEADER_FORMAT, name_bytes, offset, size))

            # Write tensor data
            for name in slots:
                tensor = slots[name]
                f.write(tensor.detach().cpu().numpy().tobytes())

    def _load(self) -> None:
        if self._path.stat().st_size == 0:
            return
        with self._path.open("rb") as f:
            (count,) = struct.unpack(_HEADER_FORMAT, f.read(struct.calcsize(_HEADER_FORMAT)))

            for _ in range(count):
                raw = f.read(struct.calcsize(_SLOT_HEADER_FORMAT))
                name_bytes, offset, size = struct.unpack(_SLOT_HEADER_FORMAT, raw)
                name = name_bytes.rstrip(b"\x00").decode("utf-8")

                # Seek to data offset and read
                pos = f.tell()
                f.seek(offset)
                data_bytes = f.read(size)
                f.seek(pos)

                # Best-effort shape recovery
                arr = __import__("numpy").frombuffer(data_bytes, dtype="float32")
                # Infer reasonable shape
                if arr.size == self._vector_dim:
                    shape = (self._vector_dim,)
                elif arr.size == self._bias_dim:
                    shape = (self._bias_dim,)
                else:
                    shape = (arr.size,)
                tensor = torch.from_numpy(arr.reshape(shape).copy())
                self._ram.write_state(name, tensor)
