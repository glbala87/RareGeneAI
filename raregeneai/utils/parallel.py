"""Parallel processing utilities for RareGeneAI.

Provides batch-parallel execution for independent pipeline steps
and variant-level annotation using ThreadPoolExecutor.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, TypeVar

from loguru import logger

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    n_threads: int = 4,
    desc: str = "Processing",
    batch_size: int | None = None,
) -> list[R]:
    """Apply fn to each item in parallel using a thread pool.

    Preserves input order. Falls back to sequential if n_threads <= 1.

    Args:
        fn: Function to apply to each item.
        items: Input items.
        n_threads: Number of worker threads.
        desc: Description for logging.
        batch_size: If set, process items in batches of this size.

    Returns:
        Results in same order as input items.
    """
    if not items:
        return []

    if n_threads <= 1:
        return [fn(item) for item in items]

    results: list[R | None] = [None] * len(items)

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        future_to_idx = {
            executor.submit(fn, item): idx
            for idx, item in enumerate(items)
        }

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"{desc} failed for item {idx}: {e}")
                results[idx] = items[idx]  # Return original on failure
            completed += 1

    return results


def batch_items(items: list[T], batch_size: int) -> list[list[T]]:
    """Split items into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
