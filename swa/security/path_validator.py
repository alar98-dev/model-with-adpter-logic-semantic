from __future__ import annotations

from pathlib import Path
from typing import Sequence

from swa.config import get_config


class PathValidator:
    """Validates file-system paths against configurable allow-lists.

    Prevents arbitrary path traversal outside of permitted zones.
    """

    def __init__(
        self,
        write_prefixes: Sequence[Path] | None = None,
        read_prefixes: Sequence[Path] | None = None,
        additional_blocked: Sequence[Path] | None = None,
    ) -> None:
        cfg = get_config()
        self._write_prefixes = _resolve_all(write_prefixes or cfg.allowed_write_prefixes)
        self._read_prefixes = _resolve_all(read_prefixes or cfg.allowed_read_prefixes)
        self._blocked = _resolve_all(additional_blocked or ())

        # Always block dangerous kernel pseudo-filesystems
        self._blocked |= frozenset(
            Path(p).resolve()
            for p in ("/dev", "/proc", "/sys", "/run")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_write(self, path: str | Path) -> bool:
        resolved = Path(path).resolve()
        if self._inside_blocked(resolved):
            return False
        return any(_is_subdir(resolved, prefix) for prefix in self._write_prefixes)

    def can_read(self, path: str | Path) -> bool:
        resolved = Path(path).resolve()
        if self._inside_blocked(resolved):
            return False
        return any(_is_subdir(resolved, prefix) for prefix in self._read_prefixes)

    def assert_can_write(self, path: str | Path) -> None:
        if not self.can_write(path):
            raise PermissionError(f"Write not allowed: {path}")

    def assert_can_read(self, path: str | Path) -> None:
        if not self.can_read(path):
            raise PermissionError(f"Read not allowed: {path}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _inside_blocked(self, resolved: Path) -> bool:
        return any(_is_subdir(resolved, b) for b in self._blocked)


def _resolve_all(paths: Sequence[Path]) -> frozenset[Path]:
    return frozenset(p.expanduser().resolve() for p in paths)


def _is_subdir(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
