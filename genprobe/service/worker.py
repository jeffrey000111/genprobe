"""A scoring worker: claims tasks of one metric in batches and scores them (D-040).

    python -m genprobe.service.worker --metric clipscore --batch-size 16 --device cpu

Loop: claim up to ``batch_size`` tasks (whatever is waiting: batches grow under load and
latency stays low when it is quiet), answer the ones already in the cache, score the rest
in one call while a background thread renews their leases, and write each result. When idle it
waits on Postgres ``LISTEN`` for the API's notice of a new task, so pickup is immediate, and also
polls with exponential backoff (50 ms up to 1 s) for what a notice does not announce (retries
after their backoff, expired leases, a notice missed while reconnecting). Every
``housekeeping_seconds`` it fails tasks
whose lease ran out on their last attempt and refreshes its row in ``scorers``.

If a batch raises, it is split in halves and retried down to single items, so one bad input
fails alone (and is retried later, up to its attempts) instead of failing its neighbours.
SIGTERM or SIGINT finishes the current batch and exits; a hard kill loses nothing either, since
the leases run out and other workers take the tasks over.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from . import db, queue
from .config import WorkerSettings
from .scorers import Item, Scorer, ScoreResult, load_scorer
from .store import ImageStore
from .telemetry import BATCH_SECONDS, BATCH_SIZE, TASK_SECONDS, TASKS, ctx, setup_logging

log = logging.getLogger("genprobe.service.worker")


class Worker:
    def __init__(self, cfg: WorkerSettings, scorer: Scorer):
        if scorer.name != cfg.metric:
            raise ValueError(f"scorer {scorer.name!r} does not serve metric {cfg.metric!r}")
        self.cfg = cfg
        self.scorer = scorer
        self.store = ImageStore(cfg.data_dir)
        self.stopping = threading.Event()
        self.conn = db.connect(cfg.database_url)
        self.hb_conn = db.connect(cfg.database_url)   # the lease-renewal thread's own connection
        self.listen_conn = db.connect(cfg.database_url)
        self.listen_conn.execute(f"LISTEN {queue.CHANNEL}")
        self.batches = 0

    # ------------------------------------------------------------------ lifecycle

    def stop(self, *_args) -> None:
        if not self.stopping.is_set():
            log.info("stopping after the current batch", extra=ctx(worker=self.cfg.worker_id))
        self.stopping.set()

    def close(self) -> None:
        try:
            queue.unregister_scorer(self.conn, self.cfg.worker_id)
        finally:
            for c in (self.conn, self.hb_conn, self.listen_conn):
                c.close()

    def housekeeping(self) -> None:
        queue.register_scorer(self.conn, self.cfg.worker_id, self.cfg.metric, self.scorer.version)
        n = queue.reap(self.conn)
        if n:
            log.warning("failed %d tasks whose lease expired on their last attempt", n,
                        extra=ctx(worker=self.cfg.worker_id))

    def run(self, max_batches: int | None = None, until_idle: bool = False) -> int:
        """Work until stopped. ``max_batches`` and ``until_idle`` (return when nothing is left
        to claim) are for tests and one-off runs. Returns the number of batches processed."""
        cfg = self.cfg
        log.info("worker started", extra=ctx(worker=cfg.worker_id, metric=cfg.metric,
                                             version=self.scorer.version, batch_size=cfg.batch_size))
        next_housekeeping = 0.0
        idle = cfg.poll_min_seconds
        done = 0
        while not self.stopping.is_set():
            if time.monotonic() >= next_housekeeping:
                self.housekeeping()
                next_housekeeping = time.monotonic() + cfg.housekeeping_seconds
            batch = queue.claim(self.conn, cfg.metric, cfg.worker_id, cfg.batch_size, cfg.lease_seconds)
            if not batch:
                if until_idle:
                    break
                if self._wait_for_notice(idle):
                    idle = cfg.poll_min_seconds
                else:
                    idle = min(idle * 2, cfg.poll_max_seconds)
                continue
            idle = cfg.poll_min_seconds
            self.process(batch)
            done += 1
            self.batches += 1
            if max_batches is not None and done >= max_batches:
                break
        return done

    def _wait_for_notice(self, timeout: float) -> bool:
        """Block until a task of this worker's metric is announced or ``timeout`` passes."""
        deadline = time.monotonic() + timeout
        while not self.stopping.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            for n in self.listen_conn.notifies(timeout=min(left, 0.5), stop_after=1):
                if n.payload == self.cfg.metric:
                    return True
        return False

    # ------------------------------------------------------------------ one batch

    def process(self, batch: list[queue.Claimed]) -> None:
        todo: list[tuple[queue.Claimed, Item]] = []
        for c in batch:
            hit = queue.cache_get(self.conn, c.image_sha256, queue.sha256_text(c.prompt), c.metric, self.scorer.version)
            if hit is not None:
                self._complete(c, ScoreResult(hit["value"], hit["extra"]), cached=True)
                continue
            path = self.store.path(c.image_sha256)
            if not path.exists():
                self._fail(c, "image file is missing from the store", permanent=True)
                continue
            todo.append((c, Item(str(c.task_id), str(path), c.prompt)))
        if not todo:
            return
        t0 = time.perf_counter()
        with self._renewing_leases([c.task_id for c, _ in todo]):
            outcomes = self._score_isolating(todo)
        BATCH_SIZE.labels(self.cfg.metric).observe(len(todo))
        BATCH_SECONDS.labels(self.cfg.metric).observe(time.perf_counter() - t0)
        for c, res in outcomes:
            if isinstance(res, Exception):
                self._fail(c, f"{type(res).__name__}: {res}")
            else:
                self._complete(c, res, cached=False)

    def _score_isolating(self, todo: list[tuple[queue.Claimed, Item]]) -> list[tuple[queue.Claimed, ScoreResult | Exception]]:
        try:
            results = self.scorer.score([it for _, it in todo])
            if len(results) != len(todo):
                raise RuntimeError(f"scorer returned {len(results)} results for {len(todo)} items")
            return [(c, r) for (c, _), r in zip(todo, results)]
        except Exception as e:  # noqa: BLE001 - any scorer error is recorded on the task
            if len(todo) == 1:
                return [(todo[0][0], e)]
            log.warning("batch of %d failed (%s); splitting it", len(todo), e, extra=ctx(worker=self.cfg.worker_id))
            mid = len(todo) // 2
            return self._score_isolating(todo[:mid]) + self._score_isolating(todo[mid:])

    @contextmanager
    def _renewing_leases(self, task_ids: list[int]):
        stop = threading.Event()
        every = self.cfg.lease_seconds / 3

        def renew():
            while not stop.wait(every):
                try:
                    queue.heartbeat(self.hb_conn, task_ids, self.cfg.worker_id, self.cfg.lease_seconds)
                except Exception:  # noqa: BLE001 - a missed renewal only risks a duplicate attempt
                    log.exception("lease renewal failed", extra=ctx(worker=self.cfg.worker_id))

        t = threading.Thread(target=renew, name="lease-renewal", daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join()

    def _complete(self, c: queue.Claimed, r: ScoreResult, cached: bool) -> None:
        ok = queue.complete(self.conn, c, self.cfg.worker_id, r.value, r.extra, self.scorer.version, cached=cached)
        if not ok:
            TASKS.labels(c.metric, "lost_lease").inc()
            log.warning("lease lost before completion; result discarded",
                        extra=ctx(worker=self.cfg.worker_id, task_id=c.task_id, attempt=c.attempt))
            return
        TASKS.labels(c.metric, "cached" if cached else "done").inc()
        TASK_SECONDS.labels(c.metric).observe((datetime.now(timezone.utc) - c.created_at).total_seconds())

    def _fail(self, c: queue.Claimed, error: str, permanent: bool = False) -> None:
        status = queue.fail(self.conn, c, self.cfg.worker_id, error, permanent=permanent,
                            backoff_seconds=self.cfg.retry_backoff_seconds)
        outcome = {"queued": "retry", "failed": "failed", None: "lost_lease"}[status]
        TASKS.labels(c.metric, outcome).inc()
        log.warning("task attempt failed: %s", error,
                    extra=ctx(worker=self.cfg.worker_id, task_id=c.task_id, attempt=c.attempt, outcome=outcome))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    ap.add_argument("--metrics-port", type=int, help="serve Prometheus metrics on this port")
    ap.add_argument("--until-idle", action="store_true", help="exit when the queue is empty")
    a = ap.parse_args(argv)
    setup_logging()
    cfg = WorkerSettings.from_env(metric=a.metric, batch_size=a.batch_size, device=a.device,
                                  metrics_port=a.metrics_port)
    t0 = time.perf_counter()
    scorer = load_scorer(cfg.metric, cfg.device)
    log.info("scorer loaded in %.1f s", time.perf_counter() - t0,
             extra=ctx(metric=cfg.metric, version=scorer.version, device=cfg.device))
    if cfg.metrics_port:
        from prometheus_client import start_http_server

        start_http_server(cfg.metrics_port)
    w = Worker(cfg, scorer)
    signal.signal(signal.SIGTERM, w.stop)
    signal.signal(signal.SIGINT, w.stop)
    try:
        w.run(until_idle=a.until_idle)
    finally:
        w.close()
    log.info("worker stopped", extra=ctx(worker=cfg.worker_id, batches=w.batches))
    return 0


if __name__ == "__main__":
    sys.exit(main())
