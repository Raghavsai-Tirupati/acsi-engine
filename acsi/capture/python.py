from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AsyncJsonlWriter:
    """Fail-open JSONL writer for production capture wrappers."""

    def __init__(self, path: Path, max_queue: int = 10_000) -> None:
        self.path = path
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def write(self, payload: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait(payload)
        except Exception:
            logger.debug("ACSI capture dropped an event due to local backpressure.", exc_info=True)

    def close(self, timeout: float = 5.0) -> None:
        try:
            self._queue.put_nowait(None)
            self._thread.join(timeout=timeout)
        except Exception:
            logger.debug("ACSI capture close signal could not be queued.", exc_info=True)

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
                    handle.flush()
        except Exception:
            logger.debug("ACSI capture writer stopped after a local error.", exc_info=True)


def capture_event(writer: AsyncJsonlWriter, payload: dict[str, Any]) -> None:
    try:
        writer.write(payload)
    except Exception:
        logger.debug("ACSI capture failed open.", exc_info=True)
