"""Structural geometry async logger."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from swa.db.async_worker import DBAsyncWorker


class AsyncStructuralGeometryLogger:
    """Periodically logs system geometry metrics via ``DBAsyncWorker``."""

    def __init__(self, worker: Optional[DBAsyncWorker] = None) -> None:
        self._worker = worker or DBAsyncWorker(flush_interval=10.0)

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def log_geometry(self, geometry_data: Dict[str, Any]) -> None:
        self._worker.enqueue(geometry_data)
