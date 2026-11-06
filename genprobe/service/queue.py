"""The work queue, on Postgres (D-040).

A job is one image and one prompt; it has one *task* per requested metric. Workers claim tasks
of their metric with ``SELECT ... FOR UPDATE SKIP LOCKED``, so any number of workers can poll the
same table without handing a task to two of them, and without blocking each other.

Delivery is at-least-once, completion is exactly-once:

- A claim takes a *lease* (``lease_expires_at``) that the worker renews while it scores. If the
  worker dies, the lease runs out and the task can be claimed again, up to ``max_attempts``.
- Every claim increments ``attempts``; the pair (``lease_owner``, ``attempts``) is a fencing
  token. ``complete`` and ``fail`` only change a task whose token still matches, so a worker that
  lost its lease (stalled past it, then woke up) cannot overwrite the result of the worker that
  took over. Its work is wasted, never written twice.
- A failed attempt goes back to the queue after an exponential backoff, or to ``failed`` (with
  the error) once its attempts are used up.

Scores are also kept in ``score_cache`` under (image, prompt, metric, metric version), so a
repeated request is answered without scoring, and a model upgrade (new version) starts afresh.

Every function takes an open psycopg connection in autocommit mode (``db.connect``).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

PENDING = ("queued", "running")
CHANNEL = "genprobe_tasks"   # NOTIFY channel: payload is the metric of a newly queued task


class IdempotencyConflict(Exception):
    """The Idempotency-Key was already used for a different request."""


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def request_hash(image_sha256: str, prompt: str, metrics: list[str]) -> str:
    return sha256_text(json.dumps([image_sha256, prompt, sorted(set(metrics))], ensure_ascii=False))


def _jsonb(x: Any):
    from psycopg.types.json import Jsonb

    return Jsonb(x)


@dataclass(frozen=True)
class Claimed:
    task_id: int
    job_id: uuid.UUID
    metric: str
    attempt: int          # 1 on the first claim; with the owner, the fencing token
    max_attempts: int
    image_sha256: str
    prompt: str
    created_at: datetime  # when the job was submitted


# ---------------------------------------------------------------------------- submitting


def find_by_key(conn, idempotency_key: str) -> dict | None:
    return conn.execute("SELECT id, request_hash FROM jobs WHERE idempotency_key = %s",
                        (idempotency_key,)).fetchone()


def pending_count(conn, metric: str) -> int:
    return conn.execute("SELECT count(*) AS n FROM tasks WHERE metric = %s AND status IN ('queued', 'running')",
                        (metric,)).fetchone()["n"]


def cache_get(conn, image_sha256: str, prompt_sha256: str, metric: str, version: str) -> dict | None:
    return conn.execute(
        "SELECT value, extra FROM score_cache WHERE image_sha256 = %s AND prompt_sha256 = %s "
        "AND metric = %s AND metric_version = %s",
        (image_sha256, prompt_sha256, metric, version)).fetchone()


def create_job(conn, *, image_sha256: str, prompt: str, metrics: list[str], max_attempts: int,
               idempotency_key: str | None = None, versions: dict[str, str] | None = None
               ) -> tuple[uuid.UUID, bool]:
    """Create a job and its tasks in one transaction; return (job_id, created).

    With an idempotency key already used for the same request, nothing is created and the first
    job is returned (created=False); used for a different request, IdempotencyConflict.
    ``versions`` maps a metric to the version live workers serve; a cached score for that
    version completes the task at once.
    """
    metrics = sorted(set(metrics))
    if not metrics:
        raise ValueError("no metric requested")
    rh = request_hash(image_sha256, prompt, metrics)
    versions = versions or {}
    psha = sha256_text(prompt)
    for _ in range(3):  # a concurrent first request with the same key may roll back; then retry
        job_id = uuid.uuid4()
        with conn.transaction():
            row = conn.execute(
                "INSERT INTO jobs (id, idempotency_key, request_hash, image_sha256, prompt) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (idempotency_key) DO NOTHING RETURNING id",
                (job_id, idempotency_key, rh, image_sha256, prompt)).fetchone()
            if row is None:
                old = find_by_key(conn, idempotency_key)
                if old is None:
                    continue
                if old["request_hash"] != rh:
                    raise IdempotencyConflict(f"Idempotency-Key {idempotency_key!r} was used for another request")
                return old["id"], False
            for m in metrics:
                v = versions.get(m)
                hit = cache_get(conn, image_sha256, psha, m, v) if v else None
                if hit is not None:
                    conn.execute(
                        "INSERT INTO tasks (job_id, metric, status, max_attempts, value, extra, metric_version, "
                        "cached, started_at, finished_at) VALUES (%s, %s, 'done', %s, %s, %s, %s, true, now(), now())",
                        (job_id, m, max_attempts, hit["value"], _jsonb(hit["extra"]), v))
                else:
                    conn.execute("INSERT INTO tasks (job_id, metric, status, max_attempts) VALUES (%s, %s, 'queued', %s)",
                                 (job_id, m, max_attempts))
                    # delivered at commit: idle workers wake now instead of at their next poll
                    conn.execute("SELECT pg_notify(%s, %s)", (CHANNEL, m))
        return job_id, True
    raise RuntimeError("could not create or find the job for this Idempotency-Key")


# ---------------------------------------------------------------------------- reading


def job_status(task_statuses: list[str]) -> str:
    if any(s in PENDING for s in task_statuses):
        return "pending"
    return "done" if all(s == "done" for s in task_statuses) else "failed"


def get_job(conn, job_id: uuid.UUID) -> tuple[dict, list[dict]] | None:
    job = conn.execute("SELECT id, image_sha256, prompt, created_at FROM jobs WHERE id = %s", (job_id,)).fetchone()
    if job is None:
        return None
    tasks = conn.execute(
        "SELECT metric, status, attempts, max_attempts, value, extra, metric_version, cached, error, "
        "created_at, started_at, finished_at FROM tasks WHERE job_id = %s ORDER BY metric", (job_id,)).fetchall()
    return job, tasks


def queue_depths(conn) -> dict[tuple[str, str], int]:
    rows = conn.execute("SELECT metric, status, count(*) AS n FROM tasks WHERE status IN ('queued', 'running') "
                        "GROUP BY metric, status").fetchall()
    return {(r["metric"], r["status"]): r["n"] for r in rows}


# ---------------------------------------------------------------------------- scorers


def register_scorer(conn, worker_id: str, metric: str, version: str) -> None:
    conn.execute(
        "INSERT INTO scorers (worker_id, metric, metric_version) VALUES (%s, %s, %s) "
        "ON CONFLICT (worker_id) DO UPDATE SET metric = EXCLUDED.metric, "
        "metric_version = EXCLUDED.metric_version, last_seen = now()",
        (worker_id, metric, version))


def unregister_scorer(conn, worker_id: str) -> None:
    conn.execute("DELETE FROM scorers WHERE worker_id = %s", (worker_id,))


def live_scorers(conn, ttl_seconds: float) -> list[dict]:
    return conn.execute(
        "SELECT worker_id, metric, metric_version, started_at, last_seen FROM scorers "
        "WHERE last_seen > now() - make_interval(secs => %s) ORDER BY metric, worker_id",
        (float(ttl_seconds),)).fetchall()


def live_versions(conn, ttl_seconds: float) -> dict[str, str]:
    """The version each metric's most recently seen worker serves."""
    rows = conn.execute(
        "SELECT DISTINCT ON (metric) metric, metric_version FROM scorers "
        "WHERE last_seen > now() - make_interval(secs => %s) ORDER BY metric, last_seen DESC",
        (float(ttl_seconds),)).fetchall()
    return {r["metric"]: r["metric_version"] for r in rows}


# ---------------------------------------------------------------------------- working

_CLAIM = """
WITH picked AS (
    SELECT id FROM tasks
    WHERE metric = %(metric)s
      AND ((status = 'queued' AND available_at <= now())
           OR (status = 'running' AND lease_expires_at < now() AND attempts < max_attempts))
    ORDER BY id
    LIMIT %(n)s
    FOR UPDATE SKIP LOCKED
)
UPDATE tasks t
SET status = 'running',
    attempts = t.attempts + 1,
    lease_owner = %(owner)s,
    lease_expires_at = now() + make_interval(secs => %(lease)s),
    started_at = coalesce(t.started_at, now()),
    error = NULL
FROM picked, jobs j
WHERE t.id = picked.id AND j.id = t.job_id
RETURNING t.id AS task_id, t.job_id, t.metric, t.attempts AS attempt, t.max_attempts,
          j.image_sha256, j.prompt, j.created_at
"""


def claim(conn, metric: str, owner: str, n: int, lease_seconds: float) -> list[Claimed]:
    """Take up to ``n`` tasks of ``metric``, oldest first: queued ones whose backoff has passed,
    and running ones whose lease ran out with attempts left."""
    rows = conn.execute(_CLAIM, {"metric": metric, "owner": owner, "n": int(n),
                                 "lease": float(lease_seconds)}).fetchall()
    return sorted((Claimed(**r) for r in rows), key=lambda c: c.task_id)


def heartbeat(conn, task_ids: list[int], owner: str, lease_seconds: float) -> int:
    """Extend the leases this worker still holds; return how many."""
    if not task_ids:
        return 0
    cur = conn.execute(
        "UPDATE tasks SET lease_expires_at = now() + make_interval(secs => %s) "
        "WHERE id = ANY(%s) AND status = 'running' AND lease_owner = %s",
        (float(lease_seconds), list(task_ids), owner))
    return cur.rowcount


_FENCE = "id = %(id)s AND status = 'running' AND lease_owner = %(owner)s AND attempts = %(attempt)s"


def complete(conn, c: Claimed, owner: str, value: float, extra: dict, version: str, cached: bool = False) -> bool:
    """Record a score. False if this worker no longer holds the task (nothing is written)."""
    with conn.transaction():
        cur = conn.execute(
            "UPDATE tasks SET status = 'done', value = %(value)s, extra = %(extra)s, metric_version = %(version)s, "
            "cached = %(cached)s, error = NULL, lease_owner = NULL, lease_expires_at = NULL, finished_at = now() "
            "WHERE " + _FENCE,
            {"value": float(value), "extra": _jsonb(extra), "version": version, "cached": cached,
             "id": c.task_id, "owner": owner, "attempt": c.attempt})
        if cur.rowcount != 1:
            return False
        if not cached:
            conn.execute(
                "INSERT INTO score_cache (image_sha256, prompt_sha256, metric, metric_version, value, extra) "
                "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (c.image_sha256, sha256_text(c.prompt), c.metric, version, float(value), _jsonb(extra)))
    return True


def fail(conn, c: Claimed, owner: str, error: str, permanent: bool = False,
         backoff_seconds: float = 2.0) -> str | None:
    """Record a failed attempt. The task is queued again after ``backoff * 2**(attempt-1)``
    seconds, or marked failed when ``permanent`` or out of attempts. Returns the new status, or
    None if this worker no longer holds the task."""
    row = conn.execute(
        "UPDATE tasks SET "
        "status = CASE WHEN %(permanent)s OR attempts >= max_attempts THEN 'failed' ELSE 'queued' END, "
        "available_at = now() + make_interval(secs => %(backoff)s * power(2, attempts - 1)), "
        "finished_at = CASE WHEN %(permanent)s OR attempts >= max_attempts THEN now() END, "
        "error = %(error)s, lease_owner = NULL, lease_expires_at = NULL "
        "WHERE " + _FENCE + " RETURNING status",
        {"permanent": bool(permanent), "backoff": float(backoff_seconds), "error": error[:2000],
         "id": c.task_id, "owner": owner, "attempt": c.attempt}).fetchone()
    return None if row is None else row["status"]


def reap(conn) -> int:
    """Fail running tasks whose lease ran out on their last attempt (their worker died or hung);
    tasks with attempts left are simply claimed again."""
    cur = conn.execute(
        "UPDATE tasks SET status = 'failed', finished_at = now(), lease_owner = NULL, lease_expires_at = NULL, "
        "error = 'lease expired on the last attempt (worker died or stalled)' "
        "WHERE status = 'running' AND lease_expires_at < now() AND attempts >= max_attempts")
    return cur.rowcount
