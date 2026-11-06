"""Kill workers while they score, and check that no task is lost or written twice (D-040).

    python -m genprobe.service.chaos --jobs 300 --workers 3 --kill-every 3 --out results/service/chaos.jsonl

Queues ``--jobs`` jobs for the fake metric (each image scored with a fixed cost per item, so
batches take a while), starts ``--workers`` worker processes, and every ``--kill-every`` seconds
sends SIGKILL to one of them (no chance to finish or unregister) and starts a replacement. When
every task has reached a final state it reports how many finished, how many needed a second or
third attempt, and checks each score against the fake scorer's deterministic value, so a result
written by the wrong attempt or for the wrong image would show up. Uses its own database schema.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from . import db, queue
from .scorers import FakeScorer, Item
from .store import ImageStore


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--database-url", default=os.environ.get("GENPROBE_DATABASE_URL"))
    ap.add_argument("--jobs", type=int, default=300)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--item-seconds", type=float, default=0.1, help="fake scoring cost per image")
    ap.add_argument("--lease", type=float, default=3.0)
    ap.add_argument("--kill-every", type=float, default=3.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from PIL import Image
    from psycopg.conninfo import make_conninfo

    rng = random.Random(a.seed)
    schema = "chaos_" + uuid.uuid4().hex[:8]
    admin = db.connect(a.database_url)
    admin.execute(f"CREATE SCHEMA {schema}")
    dsn = make_conninfo(a.database_url, options=f"-c search_path={schema}")
    conn = db.connect(dsn)
    db.migrate(conn)
    tmp = tempfile.TemporaryDirectory()
    store = ImageStore(tmp.name)
    for i in range(a.jobs):
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (i % 256, (i // 256) % 256, 7)).save(buf, format="PNG")
        sha = store.put(buf.getvalue())
        queue.create_job(conn, image_sha256=sha, prompt=f"prompt {i}", metrics=["fake"], max_attempts=3)

    env = {**os.environ, "GENPROBE_DATABASE_URL": dsn, "GENPROBE_SERVICE_DATA": tmp.name,
           "GENPROBE_FAKE_ITEM_SECONDS": str(a.item_seconds), "GENPROBE_LEASE_SECONDS": str(a.lease),
           "GENPROBE_RETRY_BACKOFF_SECONDS": "0.5", "GENPROBE_HOUSEKEEPING_SECONDS": "1"}
    n_started = 0

    def spawn():
        nonlocal n_started
        n_started += 1
        return subprocess.Popen([sys.executable, "-m", "genprobe.service.worker", "--metric", "fake",
                                 "--batch-size", str(a.batch_size)],
                                env={**env, "GENPROBE_WORKER_ID": f"chaos-{n_started}"},
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    procs = [spawn() for _ in range(a.workers)]
    kills = 0
    t0 = time.monotonic()
    next_kill = t0 + a.kill_every
    try:
        while time.monotonic() - t0 < a.timeout:
            left = conn.execute("SELECT count(*) AS n FROM tasks WHERE status IN ('queued', 'running')").fetchone()["n"]
            if left == 0:
                break
            if time.monotonic() >= next_kill:
                k = rng.randrange(len(procs))
                procs[k].send_signal(signal.SIGKILL)
                procs[k].wait()
                kills += 1
                procs[k] = spawn()
                next_kill += a.kill_every
            time.sleep(0.1)
        elapsed = time.monotonic() - t0
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    rows = conn.execute("SELECT t.status, t.attempts, t.value, j.image_sha256, j.prompt FROM tasks t "
                        "JOIN jobs j ON j.id = t.job_id").fetchall()
    fake = FakeScorer()
    wrong = sum(1 for r in rows if r["status"] == "done" and abs(
        r["value"] - fake.score([Item("x", str(store.path(r["image_sha256"])), r["prompt"])])[0].value) > 1e-12)
    res = {"jobs": a.jobs, "workers": a.workers, "batch_size": a.batch_size, "item_seconds": a.item_seconds,
           "lease_seconds": a.lease, "kill_every_seconds": a.kill_every, "workers_killed": kills,
           "seconds": round(elapsed, 1),
           "done": sum(r["status"] == "done" for r in rows), "failed": sum(r["status"] == "failed" for r in rows),
           "unfinished": sum(r["status"] in ("queued", "running") for r in rows),
           "attempts": {str(k): sum(r["attempts"] == k for r in rows) for k in sorted({r["attempts"] for r in rows})},
           "wrong_values": wrong, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(json.dumps(res))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
    conn.close()
    admin.execute(f"DROP SCHEMA {schema} CASCADE")
    admin.close()
    tmp.cleanup()
    return 0 if res["unfinished"] == 0 and res["wrong_values"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
