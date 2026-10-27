"""Canonical filesystem paths for the WSL runtime."""

from __future__ import annotations

import os
from pathlib import Path


_SOURCE_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = Path(
    os.environ.get("SWA_PROJECT_ROOT", str(_SOURCE_ROOT))
).expanduser().resolve()


def project_path(*parts: str) -> Path:
    """Return a path anchored in the canonical WSL project root."""
    return PROJECT_ROOT.joinpath(*parts)
