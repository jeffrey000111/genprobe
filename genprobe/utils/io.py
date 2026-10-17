"""Crash-safe file writes.

Every output is written to ``<path>.tmp`` first and then renamed with ``os.replace``,
which is atomic on POSIX filesystems. A job killed mid-write leaves only a ``.tmp``
file behind, never a truncated PNG at the real path.
"""

from __future__ import annotations

import os
from pathlib import Path

TMP_SUFFIX = ".tmp"


def atomic_save_image(img, path: str | Path, fmt: str = "PNG") -> None:
    """Save a PIL image atomically. Format is explicit because '.png.tmp' has no known extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    img.save(tmp, format=fmt)
    os.replace(tmp, path)


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
