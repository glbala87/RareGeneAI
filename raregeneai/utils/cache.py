"""Annotation result caching for RareGeneAI.

Provides a disk-backed LRU cache for expensive annotation lookups
(VEP, gnomAD, CADD, ClinVar API calls). Repeated analyses of
the same variant skip the API call entirely.

Cache keys are variant_key (chr-pos-ref-alt). Values are dicts
of annotation fields. The cache is a simple JSON-lines file
with an in-memory dict overlay for fast lookups.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock

from loguru import logger


class AnnotationCache:
    """Thread-safe disk-backed annotation cache."""

    def __init__(self, cache_dir: str | Path, max_memory_entries: int = 100_000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "annotation_cache.jsonl"
        self._memory: dict[str, dict] = {}
        self._lock = Lock()
        self._max_memory = max_memory_entries
        self._hits = 0
        self._misses = 0
        self._load_cache()

    def _load_cache(self) -> None:
        """Load existing cache from disk into memory."""
        if not self._cache_file.exists():
            return

        count = 0
        try:
            with open(self._cache_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        key = entry.pop("_key", None)
                        if key:
                            self._memory[key] = entry
                            count += 1
                            if count >= self._max_memory:
                                break
                    except json.JSONDecodeError:
                        continue

            logger.debug(f"Loaded {count} entries from annotation cache")
        except Exception as e:
            logger.warning(f"Failed to load annotation cache: {e}")

    def get(self, variant_key: str) -> dict | None:
        """Look up cached annotation for a variant."""
        with self._lock:
            result = self._memory.get(variant_key)
            if result is not None:
                self._hits += 1
            else:
                self._misses += 1
            return result

    def put(self, variant_key: str, annotations: dict) -> None:
        """Store annotation result in cache."""
        with self._lock:
            self._memory[variant_key] = annotations

            # Append to disk
            entry = {"_key": variant_key, **annotations}
            try:
                with open(self._cache_file, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except Exception:
                pass  # Non-critical: cache write failure shouldn't break pipeline

    def get_or_compute(self, variant_key: str, compute_fn) -> dict:
        """Get from cache or compute and cache the result."""
        cached = self.get(variant_key)
        if cached is not None:
            return cached

        result = compute_fn()
        if result:
            self.put(variant_key, result)
        return result or {}

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._memory),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Clear both memory and disk cache."""
        with self._lock:
            self._memory.clear()
            self._hits = 0
            self._misses = 0
            if self._cache_file.exists():
                self._cache_file.unlink()
