from __future__ import annotations

import math
import os
import struct
import time
from typing import Dict, Optional

import blake3
import numpy as np
import torch

from swa.config import get_config
from swa.providers.abc import AbstractStateProvider

SECTOR_SIZE = 4096

# Default geometric offsets for named states on the block device.
# These remain as defaults but are no longer hardcoded magic numbers
# buried in a monolithic class — they are configurable here and can
# be overridden via the provider constructor.
_DEFAULT_OFFSETS: dict[str, int] = {
    "reality": 0x0000,
    "prediction": 0x4000,
    "shadow_bias": 0x8000,
    "navigation_map": 0x100000,
    "recursion": 0x12000,
    "dreamwords": 0x20000,
}


class HardwareStateProvider(AbstractStateProvider):
    """Block-device backed state provider.

    Reads and writes named state slots at fixed offsets on a raw block
    device (e.g. ``/dev/sde``).  When the device is unavailable the
    constructor raises ``OSError`` so callers can fall back explicitly.
    """

    def __init__(
        self,
        device_path: str | None = None,
        offsets: dict[str, int] | None = None,
        vector_dim: int = 2560,
        bias_dim: int = 2560,
    ) -> None:
        cfg = get_config()
        self._device_path = device_path or cfg.hardware_state_device
        self._offsets = _DEFAULT_OFFSETS if offsets is None else offsets
        self._vector_dim = vector_dim
        self._bias_dim = bias_dim

        self._state_shapes: dict[str, tuple[int, ...]] = {
            "reality": (self._vector_dim,),
            "prediction": (self._vector_dim,),
            "shadow_bias": (self._bias_dim,),
            "navigation_map": (self._vector_dim,),
            "recursion": (self._vector_dim,),
            "dreamwords": (self._vector_dim,),
        }

        self._fd: int | None = None
        self._mem: memoryview | None = None
        self._metadata: dict[str, _StateMeta] = {}

        self._open_device()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_state(self, state_name: str, data: torch.Tensor) -> bool:
        self._require_slot(state_name)
        self._require_shape(state_name, data)

        offset = self._offsets[state_name]
        data_bytes = data.detach().cpu().numpy().tobytes()
        self._write_at(offset, data_bytes)

        self._metadata[state_name] = _StateMeta(
            offset=offset,
            size=len(data_bytes),
            hash=blake3.blake3(data_bytes).hexdigest(),
            timestamp=time.time(),
        )
        return True

    def read_state(self, state_name: str) -> Optional[torch.Tensor]:
        self._require_slot(state_name)

        offset = self._offsets[state_name]
        shape = self._state_shapes[state_name]
        expected_size = int(np.prod(shape)) * 4  # float32

        data_bytes = self._read_at(offset, expected_size)
        if data_bytes is None or len(data_bytes) != expected_size:
            return None

        # Integrity check when we have a stored hash
        meta = self._metadata.get(state_name)
        if meta is not None:
            actual_hash = blake3.blake3(data_bytes).hexdigest()
            if actual_hash != meta.hash:
                return None

        arr = np.frombuffer(data_bytes, dtype=np.float32).copy().reshape(shape)
        return torch.from_numpy(arr)

    def read_states(self, state_names: list[str]) -> Dict[str, Optional[torch.Tensor]]:
        return {name: self.read_state(name) for name in state_names}

    def available_slots(self) -> frozenset[str]:
        return frozenset(self._offsets)

    def close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_device(self) -> None:
        try:
            fd = os.open(self._device_path, os.O_RDWR | os.O_SYNC)
            # Determine device size
            try:
                size = os.fstat(fd).st_size
            except OSError:
                size = 0
            if size <= 0:
                size = os.lseek(fd, 0, os.SEEK_END)
                os.lseek(fd, 0, os.SEEK_SET)

            import mmap

            mem = mmap.mmap(fd, size, flags=mmap.MAP_SHARED, access=mmap.ACCESS_WRITE)
            self._fd = fd
            self._mem = mem
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        except (PermissionError, FileNotFoundError, OSError):
            self.close()
            raise

    def _write_at(self, offset: int, data: bytes) -> None:
        if self._mem is None:
            raise RuntimeError("Hardware provider not open")

        aligned_size = ((len(data) + SECTOR_SIZE - 1) // SECTOR_SIZE) * SECTOR_SIZE
        buf = data + b"\x00" * (aligned_size - len(data))

        self._mem.seek(offset)
        self._mem.write(buf)

    def _read_at(self, offset: int, size: int) -> Optional[bytes]:
        if self._mem is None:
            return None

        aligned_offset = (offset // SECTOR_SIZE) * SECTOR_SIZE
        read_size = size + (offset - aligned_offset)

        self._mem.seek(aligned_offset)
        chunk = self._mem.read(read_size)
        if len(chunk) < read_size:
            return None

        return chunk[offset - aligned_offset : offset - aligned_offset + size]

    def _require_slot(self, name: str) -> None:
        if name not in self._offsets:
            msg = f"Unknown state slot: {name!r}. Available: {sorted(self._offsets)}"
            raise ValueError(msg)

    def _require_shape(self, name: str, data: torch.Tensor) -> None:
        expected = self._state_shapes[name]
        if data.shape != expected:
            msg = f"Shape mismatch for {name!r}: expected {expected}, got {data.shape}"
            raise ValueError(msg)


class _StateMeta:
    __slots__ = ("offset", "size", "hash", "timestamp")

    def __init__(self, offset: int, size: int, hash: str, timestamp: float) -> None:
        self.offset = offset
        self.size = size
        self.hash = hash
        self.timestamp = timestamp
