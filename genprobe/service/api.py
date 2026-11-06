"""HTTP API of the scoring service (D-040).

    uvicorn --factory genprobe.service.api:create_app --host 0.0.0.0 --port 8000

Endpoints:

- ``POST /v1/jobs`` (multipart: ``image``, ``prompt``, optional ``metrics`` comma-separated;
  optional ``Idempotency-Key`` header): validates the image, refuses with 429 when a requested
  metric's queue is full, stores the image and creates the job. 202 with the job when it is
  pending, 200 when it is already done (every score came from the cache) or the key was seen.
- ``GET /v1/jobs/{id}``: the job and one result per metric, with timings. ``Retry-After`` is set
  while it is pending.
- ``GET /v1/scorers``: the live workers and the metric version each serves.
- ``GET /healthz`` (the process is up), ``GET /readyz`` (database reachable, schema current),
  ``GET /metrics`` (Prometheus).

Scoring never happens in a request: requests only touch the database, so the API stays fast
while workers are busy, and overload shows up as a 429 instead of a growing wait.
"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import db, queue
from .config import Settings
from .store import ImageRejected, ImageStore, inspect_image
from .telemetry import CACHED_AT_SUBMIT, HTTP_SECONDS, QUEUE_TASKS, SUBMISSIONS, ctx

log = logging.getLogger("genprobe.service.api")


def _iso(t):
    return None if t is None else t.isoformat()


def job_view(conn, job_id: uuid.UUID) -> dict | None:
    got = queue.get_job(conn, job_id)
    if got is None:
        return None
    job, tasks = got
    results = {}
    for t in tasks:
        r = {"status": t["status"], "value": t["value"], "metric_version": t["metric_version"],
             "cached": t["cached"], "attempts": t["attempts"], "error": t["error"], "extra": t["extra"]}
        if t["started_at"] is not None and not t["cached"]:
            r["queue_seconds"] = round((t["started_at"] - job["created_at"]).total_seconds(), 4)
        if t["finished_at"] is not None:
            r["total_seconds"] = round((t["finished_at"] - job["created_at"]).total_seconds(), 4)
        results[t["metric"]] = r
    return {"job_id": str(job["id"]), "status": queue.job_status([t["status"] for t in tasks]),
            "prompt": job["prompt"], "image_sha256": job["image_sha256"],
            "created_at": _iso(job["created_at"]), "results": results}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = ImageStore(settings.data_dir)
    pool = db.pool(settings.database_url, settings.pool_min, settings.pool_max)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool.open(wait=True, timeout=30)
        if settings.auto_migrate:
            with pool.connection() as conn:
                applied = db.migrate(conn)
            if applied:
                log.info("applied migrations %s", ", ".join(applied))
        yield
        pool.close()

    app = FastAPI(title="GenProbe scoring service", version="1.0", lifespan=lifespan)
    app.state.settings, app.state.pool, app.state.store = settings, pool, store

    @app.middleware("http")
    async def observe(request: Request, call_next):
        t0 = time.perf_counter()
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        n = request.headers.get("content-length")
        if n is not None and n.isdigit() and int(n) > settings.max_image_bytes + 64 * 1024:
            response = JSONResponse({"detail": f"request body over {settings.max_image_bytes} bytes"}, status_code=413)
        else:
            response = await call_next(request)
        route = request.scope.get("route")
        HTTP_SECONDS.labels(request.method, getattr(route, "path", "unmatched"),
                            str(response.status_code)).observe(time.perf_counter() - t0)
        response.headers["X-Request-ID"] = rid
        return response

    def require_key(authorization: str | None = Header(default=None)) -> None:
        if not settings.api_keys:
            return
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not any(hmac.compare_digest(token.encode(), k.encode())
                                                 for k in settings.api_keys):
            raise HTTPException(401, "missing or wrong API key", headers={"WWW-Authenticate": "Bearer"})

    def reject(status: int, detail: str, outcome: str, headers: dict | None = None):
        SUBMISSIONS.labels(outcome).inc()
        raise HTTPException(status, detail, headers=headers)

    @app.post("/v1/jobs", dependencies=[Depends(require_key)])
    def submit(image: UploadFile = File(...), prompt: str = Form(...), metrics: str | None = Form(default=None),
               idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        names = sorted({m.strip() for m in (metrics or ",".join(settings.metrics)).split(",") if m.strip()})
        unknown = [m for m in names if m not in settings.metrics]
        if not names or unknown:
            reject(422, f"unknown metric(s) {unknown}; this service scores {list(settings.metrics)}", "invalid")
        prompt = prompt.strip()
        if not prompt or len(prompt) > settings.max_prompt_chars:
            reject(422, f"prompt must have 1 to {settings.max_prompt_chars} characters", "invalid")
        if idempotency_key is not None and not (0 < len(idempotency_key) <= 200 and idempotency_key.isprintable()):
            reject(422, "Idempotency-Key must be 1 to 200 printable characters", "invalid")
        data = image.file.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            reject(413, f"image over {settings.max_image_bytes} bytes", "invalid")
        try:
            info = inspect_image(data, settings.max_image_pixels)
        except ImageRejected as e:
            reject(e.status, str(e), "invalid")

        with pool.connection() as conn:
            if idempotency_key is not None:
                old = queue.find_by_key(conn, idempotency_key)
                if old is not None:   # a retry: answer it even when the queue is full
                    if old["request_hash"] != queue.request_hash(info.sha256, prompt, names):
                        reject(409, "Idempotency-Key was already used for a different request", "conflict")
                    SUBMISSIONS.labels("replayed").inc()
                    return JSONResponse(job_view(conn, old["id"]), status_code=200,
                                        headers={"Location": f"/v1/jobs/{old['id']}"})
            for m in names:
                if queue.pending_count(conn, m) >= settings.max_queue_depth:
                    reject(429, f"the {m} queue is full; retry later", "busy", headers={"Retry-After": "5"})
            store.put(data, info.sha256)       # before the job exists: a worker never misses its image
            versions = queue.live_versions(conn, settings.scorer_ttl_seconds)
            try:
                job_id, created = queue.create_job(conn, image_sha256=info.sha256, prompt=prompt, metrics=names,
                                                   max_attempts=settings.max_attempts,
                                                   idempotency_key=idempotency_key, versions=versions)
            except queue.IdempotencyConflict as e:
                reject(409, str(e), "conflict")
            view = job_view(conn, job_id)
        SUBMISSIONS.labels("created" if created else "replayed").inc()
        for m, r in view["results"].items():
            if created and r["cached"]:
                CACHED_AT_SUBMIT.labels(m).inc()
        log.info("job submitted", extra=ctx(job_id=view["job_id"], metrics=names, status=view["status"]))
        code = 202 if created and view["status"] == "pending" else 200
        return JSONResponse(view, status_code=code, headers={"Location": f"/v1/jobs/{job_id}"})

    @app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_key)])
    def get_job(job_id: uuid.UUID):
        with pool.connection() as conn:
            view = job_view(conn, job_id)
        if view is None:
            raise HTTPException(404, "no such job")
        headers = {"Retry-After": "1"} if view["status"] == "pending" else {}
        return JSONResponse(view, headers=headers)

    @app.get("/v1/scorers", dependencies=[Depends(require_key)])
    def scorers():
        with pool.connection() as conn:
            rows = queue.live_scorers(conn, settings.scorer_ttl_seconds)
        out: dict[str, dict] = {m: {"workers": 0, "versions": []} for m in settings.metrics}
        for r in rows:
            d = out.setdefault(r["metric"], {"workers": 0, "versions": []})
            d["workers"] += 1
            if r["metric_version"] not in d["versions"]:
                d["versions"].append(r["metric_version"])
        return out

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        try:
            with pool.connection(timeout=2) as conn:
                missing = db.pending(conn)
        except Exception as e:  # noqa: BLE001 - any database error means not ready
            return JSONResponse({"status": "unavailable", "detail": str(e)}, status_code=503)
        if missing:
            return JSONResponse({"status": "unavailable", "detail": f"migrations pending: {missing}"}, status_code=503)
        return {"status": "ready"}

    @app.get("/metrics")
    def metrics():
        try:
            with pool.connection(timeout=2) as conn:
                depths = queue.queue_depths(conn)
            for m in set(settings.metrics) | {k[0] for k in depths}:
                for st in queue.PENDING:
                    QUEUE_TASKS.labels(m, st).set(depths.get((m, st), 0))
        except Exception:  # noqa: BLE001 - still serve the other metrics
            log.exception("could not read queue depths")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
