"""The scoring service (genprobe/service, D-040).

Pure-Python parts always run. The queue, worker and API tests need the service dependencies
(pip install -e ".[service]") and a Postgres to write to:

    GENPROBE_TEST_PG_DSN=postgresql://user@localhost:5432/postgres python -m unittest tests.test_service

Each test gets its own schema, dropped afterwards, so tests never see each other's rows.
"""

from __future__ import annotations

import importlib.util
import io
import os
import tempfile
import threading
import time
import unittest
import uuid

from PIL import Image

from genprobe.service import config as C
from genprobe.service import queue as Q
from genprobe.service.store import ImageRejected, ImageStore, inspect_image

HAVE_DEPS = all(importlib.util.find_spec(m) for m in ("psycopg", "psycopg_pool", "fastapi", "httpx", "prometheus_client"))
DSN = os.environ.get("GENPROBE_TEST_PG_DSN")
needs_db = unittest.skipUnless(HAVE_DEPS and DSN, "needs the service extras and GENPROBE_TEST_PG_DSN")


def png(color=(200, 30, 30), size=(64, 64), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------- no database


class TestPure(unittest.TestCase):
    def test_inspect_image(self):
        info = inspect_image(png(), max_pixels=10_000)
        self.assertEqual((info.format, info.width, info.height), ("PNG", 64, 64))
        self.assertEqual(len(info.sha256), 64)
        for data, status in [(b"", 400), (b"not an image", 400), (png(fmt="GIF"), 415),
                             (png(size=(200, 200)), 413), (png()[:60], 400)]:
            with self.assertRaises(ImageRejected) as e:
                inspect_image(data, max_pixels=10_000)
            self.assertEqual(e.exception.status, status)

    def test_store_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as d:
            s = ImageStore(d)
            a = s.put(png())
            self.assertEqual(a, s.put(png()))                 # same bytes, same file
            self.assertTrue(s.path(a).exists())
            self.assertEqual(s.path(a).read_bytes(), png())
            self.assertEqual(len(list((s.root / a[:2]).iterdir())), 1)   # no temp files left
            with self.assertRaises(ValueError):
                s.path("../etc/passwd")

    def test_job_status_and_hash(self):
        self.assertEqual(Q.job_status(["done", "queued"]), "pending")
        self.assertEqual(Q.job_status(["done", "done"]), "done")
        self.assertEqual(Q.job_status(["done", "failed"]), "failed")
        self.assertEqual(Q.request_hash("a", "p", ["y", "x", "x"]), Q.request_hash("a", "p", ["x", "y"]))
        self.assertNotEqual(Q.request_hash("a", "p", ["x"]), Q.request_hash("a", "p ", ["x"]))

    def test_settings_from_env(self):
        env = {"GENPROBE_SERVICE_METRICS": "clipscore, vqascore", "GENPROBE_MAX_QUEUE_DEPTH": "7",
               "GENPROBE_AUTO_MIGRATE": "yes", "GENPROBE_API_KEYS": ""}
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            s = C.Settings.from_env()
            self.assertEqual(s.metrics, ("clipscore", "vqascore"))
            self.assertEqual((s.max_queue_depth, s.auto_migrate, s.api_keys), (7, True, ()))
            w = C.WorkerSettings.from_env(metric="fake", batch_size=4)
            self.assertEqual((w.metric, w.batch_size), ("fake", 4))
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_fake_scorer(self):
        from genprobe.service.scorers import FakeScorer, Item

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "x.png")
            with open(p, "wb") as f:
                f.write(png())
            s = FakeScorer()
            a, b = s.score([Item("1", p, "a cat"), Item("2", p, "a dog")])
            self.assertTrue(0 <= a.value <= 1 and a.value != b.value)
            self.assertEqual(a.value, s.score([Item("9", p, "a cat")])[0].value)
            with self.assertRaises(RuntimeError):
                s.score([Item("1", p, "a cat"), Item("2", p, "[fail]")])

    def test_percentile(self):
        from genprobe.service.loadtest import pct

        self.assertEqual(pct([1, 2, 3, 4, 5], 0.5), 3)
        self.assertAlmostEqual(pct([1, 2, 3, 4], 0.5), 2.5)
        self.assertTrue(pct([], 0.5) != pct([], 0.5))        # NaN


# --------------------------------------------------------------------------- with Postgres


class DBTest(unittest.TestCase):
    """A fresh schema per test, migrated."""

    def setUp(self):
        from psycopg.conninfo import make_conninfo

        from genprobe.service import db

        self.schema = "t_" + uuid.uuid4().hex[:12]
        self.admin = db.connect(DSN)
        self.admin.execute(f"CREATE SCHEMA {self.schema}")
        self.dsn = make_conninfo(DSN, options=f"-c search_path={self.schema}")
        self.conn = db.connect(self.dsn)
        self.assertEqual(db.migrate(self.conn), ["0001_init"])
        self.tmp = tempfile.TemporaryDirectory()
        self.store = ImageStore(self.tmp.name)

    def tearDown(self):
        self.conn.close()
        self.admin.execute(f"DROP SCHEMA {self.schema} CASCADE")
        self.admin.close()
        self.tmp.cleanup()

    def job(self, prompt="a red square", metrics=("fake",), color=(200, 30, 30), key=None, max_attempts=3, versions=None):
        sha = self.store.put(png(color))
        return Q.create_job(self.conn, image_sha256=sha, prompt=prompt, metrics=list(metrics),
                            max_attempts=max_attempts, idempotency_key=key, versions=versions)[0]

    def worker(self, **kw):
        from genprobe.service.scorers import FakeScorer
        from genprobe.service.worker import Worker

        opts = {"poll_min_seconds": 0.01, "poll_max_seconds": 0.05, "retry_backoff_seconds": 0.0,
                "worker_id": "w-test", **kw}
        cfg = C.WorkerSettings(database_url=self.dsn, data_dir=self.tmp.name, metric="fake", **opts)
        return Worker(cfg, FakeScorer())

    def statuses(self, job_id):
        return [t["status"] for t in Q.get_job(self.conn, job_id)[1]]


@needs_db
class TestQueue(DBTest):
    def test_migrate_twice(self):
        from genprobe.service import db

        self.assertEqual(db.migrate(self.conn), [])
        self.assertEqual(db.pending(self.conn), [])

    def test_lifecycle(self):
        j = self.job(metrics=("fake", "other"))
        self.assertEqual(self.statuses(j), ["queued", "queued"])
        [c] = Q.claim(self.conn, "fake", "w1", 10, 30)
        self.assertEqual((c.job_id, c.attempt, c.prompt), (j, 1, "a red square"))
        self.assertEqual(Q.claim(self.conn, "fake", "w2", 10, 30), [])      # held by w1
        self.assertTrue(Q.complete(self.conn, c, "w1", 0.5, {"k": 1}, "v1"))
        job, tasks = Q.get_job(self.conn, j)
        self.assertEqual([(t["metric"], t["status"], t["value"]) for t in tasks],
                         [("fake", "done", 0.5), ("other", "queued", None)])
        self.assertEqual(Q.job_status([t["status"] for t in tasks]), "pending")

    def test_idempotency_key(self):
        a = self.job(key="k1")
        sha = self.store.put(png())
        b, created = Q.create_job(self.conn, image_sha256=sha, prompt="a red square", metrics=["fake"],
                                  max_attempts=3, idempotency_key="k1")
        self.assertEqual((a, created), (b, False))
        with self.assertRaises(Q.IdempotencyConflict):
            Q.create_job(self.conn, image_sha256=sha, prompt="another prompt", metrics=["fake"],
                         max_attempts=3, idempotency_key="k1")
        self.assertEqual(self.conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"], 1)

    def test_concurrent_claims_never_overlap(self):
        from genprobe.service import db

        n = 300
        for i in range(n):
            self.job(prompt=f"p{i}")
        got: list[list[int]] = [[] for _ in range(8)]

        def run(k):
            conn = db.connect(self.dsn)
            try:
                while batch := Q.claim(conn, "fake", f"w{k}", 7, 30):
                    got[k] += [c.task_id for c in batch]
            finally:
                conn.close()

        threads = [threading.Thread(target=run, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        ids = [i for g in got for i in g]
        self.assertEqual(len(ids), n)
        self.assertEqual(len(set(ids)), n)
        self.assertGreater(sum(1 for g in got if g), 1)                      # the work was shared

    def test_expired_lease_is_taken_over_and_fenced(self):
        j = self.job()
        [a] = Q.claim(self.conn, "fake", "slow", 1, 0.2)
        self.assertEqual(Q.claim(self.conn, "fake", "fast", 1, 30), [])
        time.sleep(0.3)
        [b] = Q.claim(self.conn, "fake", "fast", 1, 30)
        self.assertEqual((b.task_id, b.attempt), (a.task_id, 2))
        self.assertFalse(Q.complete(self.conn, a, "slow", 0.1, {}, "v"))   # stale worker: rejected
        self.assertIsNone(Q.fail(self.conn, a, "slow", "late"))
        self.assertEqual(Q.heartbeat(self.conn, [a.task_id], "slow", 30), 0)
        self.assertTrue(Q.complete(self.conn, b, "fast", 0.9, {}, "v"))
        t = Q.get_job(self.conn, j)[1][0]
        self.assertEqual((t["status"], t["value"], t["attempts"]), ("done", 0.9, 2))

    def test_heartbeat_keeps_the_lease(self):
        self.job()
        [a] = Q.claim(self.conn, "fake", "w1", 1, 0.3)
        for _ in range(3):
            time.sleep(0.15)
            self.assertEqual(Q.heartbeat(self.conn, [a.task_id], "w1", 0.3), 1)
        self.assertEqual(Q.claim(self.conn, "fake", "w2", 1, 30), [])

    def test_retry_backoff_then_dead_letter(self):
        j = self.job(max_attempts=2)
        [a] = Q.claim(self.conn, "fake", "w1", 1, 30)
        self.assertEqual(Q.fail(self.conn, a, "w1", "boom", backoff_seconds=60), "queued")
        self.assertEqual(Q.claim(self.conn, "fake", "w1", 1, 30), [])      # still backing off
        self.conn.execute("UPDATE tasks SET available_at = now()")
        [b] = Q.claim(self.conn, "fake", "w1", 1, 30)
        self.assertEqual(b.attempt, 2)
        self.assertEqual(Q.fail(self.conn, b, "w1", "boom again"), "failed")
        t = Q.get_job(self.conn, j)[1][0]
        self.assertEqual((t["status"], t["error"]), ("failed", "boom again"))
        self.assertEqual(Q.claim(self.conn, "fake", "w1", 1, 30), [])

    def test_permanent_failure_and_reaper(self):
        j1 = self.job(prompt="a")
        [a] = Q.claim(self.conn, "fake", "w1", 1, 30)
        self.assertEqual(Q.fail(self.conn, a, "w1", "bad input", permanent=True), "failed")
        j2 = self.job(prompt="b", max_attempts=1)
        Q.claim(self.conn, "fake", "dies", 1, 0.1)
        time.sleep(0.2)
        self.assertEqual(Q.claim(self.conn, "fake", "w2", 1, 30), [])       # no attempts left
        self.assertEqual(Q.reap(self.conn), 1)
        self.assertEqual((self.statuses(j1), self.statuses(j2)), (["failed"], ["failed"]))

    def test_cache_at_submit(self):
        j = self.job()
        [c] = Q.claim(self.conn, "fake", "w1", 1, 30)
        Q.complete(self.conn, c, "w1", 0.42, {"x": 1}, "v1")
        j2 = self.job(versions={"fake": "v1"})                          # same image and prompt
        t = Q.get_job(self.conn, j2)[1][0]
        self.assertEqual((t["status"], t["value"], t["cached"], t["extra"]), ("done", 0.42, True, {"x": 1}))
        j3 = self.job(versions={"fake": "v2"})                          # new model version: score again
        self.assertEqual(self.statuses(j3), ["queued"])
        self.assertNotEqual(j, j2)


@needs_db
class TestWorker(DBTest):
    def test_scores_everything(self):
        jobs = [self.job(prompt=f"prompt {i}", color=(i, 0, 0)) for i in range(25)]
        w = self.worker(batch_size=8)
        try:
            self.assertEqual(w.run(until_idle=True), 4)                 # 8 + 8 + 8 + 1
        finally:
            w.close()
        for j in jobs:
            t = Q.get_job(self.conn, j)[1][0]
            self.assertEqual((t["status"], t["metric_version"], t["cached"]), ("done", "fake-1", False))
        self.assertEqual(self.conn.execute("SELECT count(*) AS n FROM score_cache").fetchone()["n"], 25)
        self.assertEqual(self.conn.execute("SELECT count(*) AS n FROM scorers").fetchone()["n"], 0)  # unregistered

    def test_bad_item_fails_alone(self):
        good = [self.job(prompt=f"ok {i}", color=(0, i, 0)) for i in range(6)]
        bad = self.job(prompt="this one [fail]s", max_attempts=2)
        w = self.worker(batch_size=16)
        try:
            w.run(until_idle=True)   # first pass: the batch is split until the bad task fails alone
            w.run(until_idle=True)   # its second attempt (no backoff in tests) fails for good
        finally:
            w.close()
        self.assertTrue(all(self.statuses(j) == ["done"] for j in good))
        t = Q.get_job(self.conn, bad)[1][0]
        self.assertEqual((t["status"], t["attempts"]), ("failed", 2))
        self.assertIn("injected failure", t["error"])

    def test_missing_image_and_worker_cache(self):
        sha = "0" * 64                                                  # never stored
        lost = Q.create_job(self.conn, image_sha256=sha, prompt="x", metrics=["fake"], max_attempts=3)[0]
        first = self.job(prompt="same")
        w = self.worker()
        try:
            w.run(until_idle=True)
            again = self.job(prompt="same")                             # submitted with no live version known
            w.run(until_idle=True)
        finally:
            w.close()
        t = Q.get_job(self.conn, lost)[1][0]
        self.assertEqual((t["status"], t["attempts"]), ("failed", 1))
        a, b = Q.get_job(self.conn, first)[1][0], Q.get_job(self.conn, again)[1][0]
        self.assertEqual((b["status"], b["cached"], b["value"]), ("done", True, a["value"]))

    def test_notice_wakes_an_idle_worker(self):
        w = self.worker(poll_min_seconds=5.0, poll_max_seconds=5.0)  # polling alone would take 5 s
        t = threading.Thread(target=w.run)
        t.start()
        try:
            time.sleep(0.3)                                          # now waiting
            t0 = time.monotonic()
            j = self.job()
            while self.statuses(j) != ["done"] and time.monotonic() - t0 < 4:
                time.sleep(0.02)
            self.assertEqual(self.statuses(j), ["done"])
            self.assertLess(time.monotonic() - t0, 1.0)
        finally:
            w.stop()
            t.join(timeout=10)
            w.close()

    def test_stop_finishes_and_exits(self):
        w = self.worker(poll_max_seconds=0.02)
        t = threading.Thread(target=w.run)
        t.start()
        time.sleep(0.2)
        w.stop()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        w.close()


@needs_db
class TestAPI(DBTest):
    def client(self, **kw):
        from fastapi.testclient import TestClient

        from genprobe.service.api import create_app

        s = C.Settings(database_url=self.dsn, data_dir=self.tmp.name, metrics=("fake",), pool_max=4, **kw)
        return TestClient(create_app(s))

    def post(self, c, prompt="a red square", image=None, headers=None, **data):
        return c.post("/v1/jobs", files={"image": ("x.png", image if image is not None else png(), "image/png")},
                      data={"prompt": prompt, **data}, headers=headers or {})

    def test_submit_score_read(self):
        with self.client() as c:
            r = self.post(c)
            self.assertEqual(r.status_code, 202, r.text)
            jid = r.json()["job_id"]
            self.assertEqual(r.headers["location"], f"/v1/jobs/{jid}")
            g = c.get(f"/v1/jobs/{jid}")
            self.assertEqual((g.json()["status"], g.headers.get("retry-after")), ("pending", "1"))
            w = self.worker()
            try:
                w.run(until_idle=True)
                g = c.get(f"/v1/jobs/{jid}").json()
                self.assertEqual(g["status"], "done")
                res = g["results"]["fake"]
                self.assertEqual((res["cached"], res["attempts"], res["metric_version"]), (False, 1, "fake-1"))
                self.assertGreaterEqual(res["total_seconds"], res["queue_seconds"])
                # the worker registered its version, so the same request is answered at once
                w.housekeeping()
                r2 = self.post(c)
                self.assertEqual(r2.status_code, 200)
                self.assertEqual(r2.json()["results"]["fake"]["value"], res["value"])
                self.assertTrue(r2.json()["results"]["fake"]["cached"])
                self.assertEqual(c.get("/v1/scorers").json()["fake"]["workers"], 1)
            finally:
                w.close()
            self.assertEqual(c.get(f"/v1/jobs/{uuid.uuid4()}").status_code, 404)
            self.assertEqual(c.get("/v1/jobs/not-a-uuid").status_code, 422)

    def test_rejects_bad_requests(self):
        with self.client(max_image_pixels=100 * 100, max_prompt_chars=20) as c:
            self.assertEqual(self.post(c, image=b"nope").status_code, 400)
            self.assertEqual(self.post(c, image=png(fmt="GIF")).status_code, 415)
            self.assertEqual(self.post(c, image=png(size=(200, 200))).status_code, 413)
            self.assertEqual(self.post(c, prompt="   ").status_code, 422)
            self.assertEqual(self.post(c, prompt="x" * 21).status_code, 422)
            self.assertEqual(self.post(c, metrics="clipscore").status_code, 422)
            self.assertEqual(self.conn.execute("SELECT count(*) AS n FROM jobs").fetchone()["n"], 0)

    def test_idempotency_and_backpressure(self):
        with self.client(max_queue_depth=2) as c:
            a = self.post(c, headers={"Idempotency-Key": "abc"})
            b = self.post(c, headers={"Idempotency-Key": "abc"})
            self.assertEqual((a.status_code, b.status_code), (202, 200))
            self.assertEqual(a.json()["job_id"], b.json()["job_id"])
            self.assertEqual(self.post(c, prompt="other", headers={"Idempotency-Key": "abc"}).status_code, 409)
            self.assertEqual(self.post(c, prompt="second").status_code, 202)
            full = self.post(c, prompt="third")
            self.assertEqual((full.status_code, full.headers.get("retry-after")), (429, "5"))
            # a retry of an accepted request is still answered while the queue is full
            self.assertEqual(self.post(c, headers={"Idempotency-Key": "abc"}).status_code, 200)

    def test_auth_health_metrics(self):
        with self.client(api_keys=("s3cret",)) as c:
            self.assertEqual(self.post(c).status_code, 401)
            self.assertEqual(self.post(c, headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(self.post(c, headers={"Authorization": "Bearer s3cret"}).status_code, 202)
            self.assertEqual(c.get("/healthz").status_code, 200)            # no key needed
            self.assertEqual(c.get("/readyz").json(), {"status": "ready"})
            m = c.get("/metrics").text
            self.assertIn('genprobe_queue_tasks{metric="fake",status="queued"} 1.0', m)
            self.assertIn("genprobe_http_request_seconds_bucket", m)
            self.assertIn('route="/v1/jobs"', m)


if __name__ == "__main__":
    unittest.main()
