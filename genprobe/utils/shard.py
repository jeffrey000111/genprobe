"""Deterministic sharding.

Items are sorted by a key, then shard k takes ``ordered[k::num_shards]``. Sorting makes
the assignment independent of input order, so a resubmitted array job always sees the
same samples in the same shard.
"""

from __future__ import annotations

from typing import Callable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def select_shard(
    items: Sequence[T], shard: int, num_shards: int, key: Callable[[T], str]
) -> list[T]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    keys = [key(x) for x in items]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})[:5]
        raise ValueError(f"duplicate keys, e.g. {dupes}")
    ordered = sorted(items, key=key)
    return list(ordered[shard::num_shards])


def chunks(items: Sequence[T], size: int) -> Iterator[list[T]]:
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    for i in range(0, len(items), size):
        yield list(items[i : i + size])
