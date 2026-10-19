"""Crash-safe file writes.

Every output is written to ``<path>.tmp`` first and then renamed with ``os.replace``,
which is atomic on POSIX filesystems. A job killed mid-write leaves only a ``.tmp``
file behind, never a truncated PNG at the real path.
"""

from __future__ import annotations

import errno
import os
import time
from pathlib import Path

TMP_SUFFIX = ".tmp"
# Transient NFS errors on HPC3's /home: "Stale file handle" killed one generation shard
# (job 91975_2) at its first image save; retrying the write is enough.
RETRY_ERRNOS = frozenset({errno.ESTALE})


def retry_transient(fn, attempts: int = 4, delay: float = 2.0, sleep=time.sleep):
    """Call fn(), retrying (with growing pauses) only on transient filesystem errors."""
    for i in range(attempts):
        try:
            return fn()
        except OSError as e:
            if e.errno not in RETRY_ERRNOS or i == attempts - 1:
                raise
            sleep(delay * (i + 1))


def atomic_save_image(img, path: str | Path, fmt: str = "PNG") -> None:
    """Save a PIL image atomically. Format is explicit because '.png.tmp' has no known extension."""
    path = Path(path)
    tmp = path.with_name(path.name + TMP_SUFFIX)

    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(tmp, format=fmt)
        os.replace(tmp, path)
    retry_transient(write)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def clean_tmp(root: str | Path) -> int:
    """Delete leftover *.tmp files under root. Returns how many were removed."""
    root = Path(root)
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob(f"*{TMP_SUFFIX}"):
        if p.is_file():
            p.unlink()
            n += 1
    return n
