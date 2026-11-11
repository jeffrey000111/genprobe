"""End-to-end check of a running scoring service, using only the standard library (D-041).

    python scripts/service_e2e.py --url http://localhost:8000 --metric fake --workers 2

CI starts the Docker Compose stack with the fake metric and runs this against it; it works the
same against any running instance. It waits for the API and the workers, then checks a job from
submission to score, the cache, idempotency keys, input validation and the metrics endpoint.
Exits 1 at the first failed check.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib


def png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """A valid PNG of one colour, built by hand so the script needs no imaging library."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def call(method: str, url: str, body: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def submit(url: str, image: bytes, prompt: str, key: str | None = None):
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\n{prompt}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"x.png\"\r\n"
            "Content-Type: image/png\r\n\r\n").encode() + image + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if key:
        headers["Idempotency-Key"] = key
    status, _, raw = call("POST", f"{url}/v1/jobs", body, headers)
    return status, (json.loads(raw) if raw.startswith(b"{") else {"raw": raw.decode(errors="replace")})


def check(ok: bool, what: str, detail: object = "") -> None:
    if not ok:
        print(f"FAIL  {what}  {detail}")
        sys.exit(1)
    print(f"ok    {what}")


def wait(what: str, cond, timeout: float) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if cond():
                print(f"ok    {what} ({time.monotonic() - t0:.1f} s)")
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(1)
    check(False, what, f"not within {timeout:.0f} s")


def poll(url: str, job_id: str, timeout: float = 60) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        status, _, raw = call("GET", f"{url}/v1/jobs/{job_id}")
        view = json.loads(raw)
        if status == 200 and view["status"] != "pending":
            return view
        time.sleep(0.2)
    check(False, f"job {job_id} finishes", f"still pending after {timeout:.0f} s")
    return {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--metric", default="fake")
    ap.add_argument("--workers", type=int, default=1, help="live workers to wait for")
    ap.add_argument("--timeout", type=float, default=120, help="seconds to wait for the stack")
    a = ap.parse_args(argv)
    url, m = a.url.rstrip("/"), a.metric
    salt = os.urandom(2)  # a fresh image per run, so a reused database cannot answer from its cache
    image = png(64, 64, (salt[0], salt[1], 90))

    wait("API ready", lambda: call("GET", f"{url}/readyz")[0] == 200, a.timeout)
    wait(f"{a.workers} {m} worker(s) registered",
         lambda: json.loads(call("GET", f"{url}/v1/scorers")[2])[m]["workers"] >= a.workers, a.timeout)

    status, view = submit(url, image, "a teal square")
    check(status == 202 and view["status"] == "pending", "new job is accepted as pending (202)", (status, view))
    done = poll(url, view["job_id"])
    r = done["results"][m]
    check(done["status"] == "done" and 0 <= r["value"] <= 1, "job is scored", done)
    check(r["cached"] is False and r["attempts"] == 1, "first score is computed once", r)

    status, again = submit(url, image, "a teal square")
    r2 = again["results"][m]
    check(status == 200 and r2["cached"] and r2["value"] == r["value"],
          "same image and prompt answered from the cache (200)", (status, again))

    key, image2 = uuid.uuid4().hex, png(64, 64, (salt[1], 30, salt[0]))
    s1, v1 = submit(url, image2, "another square", key)
    s2, v2 = submit(url, image2, "another square", key)
    check(s1 == 202 and s2 == 200 and v1["job_id"] == v2["job_id"], "retry with the same Idempotency-Key returns the first job",
          (s1, s2))
    s3, _ = submit(url, image2, "a different prompt", key)
    check(s3 == 409, "same key for a different request is a conflict (409)", s3)
    poll(url, v1["job_id"])

    s4, _ = submit(url, b"not an image", "a square")
    check(s4 == 400, "an unreadable upload is refused (400)", s4)

    _, _, metrics = call("GET", f"{url}/metrics")
    check(b'genprobe_submissions_total{outcome="created"}' in metrics, "Prometheus metrics are served")
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
