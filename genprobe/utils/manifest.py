"""Append-only JSONL manifests: one line per sample, per attempt.

Rules (guide §4.4):
- each shard writes its own file; never let two jobs append to the same file;
- every line is flushed as soon as it is written;
- readers tolerate a truncated last line (job killed mid-write);
- when a sample_id appears several times, the *last* terminal record wins.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable

# A sample in one of these states is finished and will not be retried on resume.
# "blank" is terminal on purpose: the same seed and settings reproduce the same blank
# image, so retrying would loop forever. It is reported as a failure rate instead.
TERMINAL_STATUSES = frozenset({"ok", "blank"})


class ManifestWriter:
    def __init__(self, path: str | Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._f = None

    def __enter__(self) -> "ManifestWriter":
        # If a previous job was killed mid-line, start on a fresh line so our first record
        # isn't glued onto the broken one (which readers then skip).
        needs_newline = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                needs_newline = f.read(1) != b"\n"
        self._f = open(self.path, "a", encoding="utf-8")
        if needs_newline:
            self._f.write("\n")
        return self

    def write(self, record: dict[str, Any]) -> None:
        assert self._f is not None, "use ManifestWriter as a context manager"
        self._f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._f.flush()
        if self.fsync:
            os.fsync(self._f.fileno())

    def __exit__(self, *exc) -> None:
        if self._f is not None:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()
            self._f = None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one JSONL file, skipping (with a warning) lines that fail to parse."""
    out = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        warnings.warn(f"{path}: skipped {bad} unparseable line(s) (likely a killed job)")
    return out


def read_many(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for p in sorted(str(x) for x in paths):  # sorted: deterministic "last wins"
        records.extend(read_jsonl(p))
    return records


def latest_by_id(records: Iterable[dict[str, Any]], key: str = "sample_id") -> dict[str, dict[str, Any]]:
    """Collapse attempts: last terminal record wins; otherwise keep the last record seen.

    "Last" means by ``created_at`` when present, else file order.
    """
    recs = list(records)
    recs.sort(key=lambda r: r.get("created_at", ""))  # stable: ties keep file order
    best: dict[str, dict[str, Any]] = {}
    for r in recs:
        k = r[key]
        prev = best.get(k)
        if r.get("status") in TERMINAL_STATUSES or prev is None or prev.get("status") not in TERMINAL_STATUSES:
            best[k] = r
    return best


def done_ids(
    records: Iterable[dict[str, Any]],
    *,
    match: dict[str, Any] | None = None,
    require_file: bool = False,
) -> set[str]:
    """IDs whose latest record is terminal and (optionally) matches fields like config_hash.

    ``require_file=True`` also checks that the record's ``path`` exists on disk, so a
    deleted or never-renamed image gets regenerated.
    """
    match = match or {}
    out = set()
    relevant = [r for r in records if all(r.get(k) == v for k, v in match.items())]
    for sid, r in latest_by_id(relevant).items():
        if r.get("status") not in TERMINAL_STATUSES:
            continue
        if require_file and r.get("status") == "ok" and not Path(r.get("path", "")).is_file():
            continue
        out.add(sid)
    return out
