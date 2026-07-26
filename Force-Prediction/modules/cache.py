"""Dataset-scoped content-addressed JSON cache with legacy read-through."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


class DiskCache:
    VERSION = "v2"

    def __init__(self, root: Path, *, legacy_root: Path | None = None) -> None:
        self.root = root
        self.legacy_root = legacy_root if legacy_root != root else None
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.read_errors = 0
        self.legacy_hits = 0

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(
            (DiskCache.VERSION, *parts), sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Any | None:
        value = self._read(self.root / f"{key}.json")
        if value is not None:
            self.hits += 1
            return value
        legacy = self.legacy_root / f"{key}.json" if self.legacy_root else None
        if legacy is not None:
            value = self._read(legacy)
            if value is not None:
                self.hits += 1
                self.legacy_hits += 1
                self._write(self.root / f"{key}.json", value)
                return value
        self.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._write(self.root / f"{key}.json", value)
        self.writes += 1

    def _read(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            self.read_errors += 1
            return None

    def _write(self, path: Path, value: Any) -> None:
        payload = json.dumps(value)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as fh:
            fh.write(payload)
            temporary = Path(fh.name)
        temporary.replace(path)

    def stats(self) -> dict[str, int]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "read_errors": self.read_errors,
            "legacy_hits": self.legacy_hits,
        }
