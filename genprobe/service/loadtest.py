"""Open-loop load test of the scoring service (D-040).

    python -m genprobe.service.loadtest --url http://localhost:8000 --rates 1,2,4 --duration 60 \
        --label clip-cpu-1worker --out results/service/loadtest.jsonl

Requests arrive as a Poisson process at each offered rate, whether or not earlier ones have
finished (an open loop: a closed loop, where each client waits for its last answer, hides
queueing delay by slowing down with the server). Each request uploads a fresh synthetic image
(512x512 JPEG, so nothing comes from the cache unless ``--repeat`` asks for it), then polls its
job until it is done.

Reported per rate: submit latency (the POST), end-to-end latency from the server's own
timestamps (submitted to scored, so polling granularity does not blur it), completed jobs per
second, and how many requests were refused (429), failed or timed out. Results are appended as
one JSON line per rate; ``--markdown`` prints a table.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

DEFAULT_PROMPTS = [
    "a red apple on a wooden table", "two dogs playing in the snow", "a street sign that says OPEN",
    "a cat sitting to the left of a blue vase", "three yellow balloons above a house",
    "an astronaut riding a horse on the moon", "a bowl of ramen with chopsticks", "a lighthouse at sunset",
]


def make_images(n: int, seed: int, side: int = 512) -> list[bytes]:
    """Smooth random colour fields with a little noise: realistic file sizes, all different."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side] / side
    out = []
    for _ in range(n):
        c = rng.uniform(0, 255, size=(4, 3))
        img = (c[0] * (1 - xx[..., None]) * (1 - yy[..., None]) + c[1] * xx[..., None] * (1 - yy[..., None])
               + c[2] * (1 - xx[..., None]) * yy[..., None] + c[3] * xx[..., None] * yy[..., None])
        img = np.clip(img + rng.normal(0, 8, size=img.shape), 0, 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=90)
        out.append(buf.getvalue())
    return out


@dataclass
class Outcome:
    submit_status: int = 0
    submit_seconds: float = math.nan
    job_status: str = ""               # done, failed, timeout, refused, error
    server_seconds: float = math.nan   # submitted -> scored, from the server's timestamps
    queue_seconds: float = math.nan
    client_seconds: float = math.nan   # POST sent -> done observed
    cached: bool = False
    error: str = ""


@dataclass
class RateResult:
    label: str
    offered_rate: float
    duration: float
    sent: int = 0
    done: int = 0
    failed: int = 0
    refused: int = 0
    timeouts: int = 0
    errors: int = 0
    cached: int = 0
    throughput: float = 0.0
    submit_p50: float = math.nan
    submit_p99: float = math.nan
    e2e_p50: float = math.nan
    e2e_p90: float = math.nan
    e2e_p99: float = math.nan
    e2e_max: float = math.nan
    queue_p50: float = math.nan
    queue_p99: float = math.nan
    env: dict = field(default_factory=dict)


def pct(xs: list[float], q: float) -> float:
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return math.nan
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def one_request(client, url: str, image: bytes, prompt: str, metrics: str | None, headers: dict,
                      poll: float, timeout: float) -> Outcome:
    o = Outcome()
    t0 = time.perf_counter()
    data = {"prompt": prompt}
    if metrics:
        data["metrics"] = metrics
    try:
        r = await client.post(f"{url}/v1/jobs", files={"image": ("img.jpg", image, "image/jpeg")},
                              data=data, headers=headers)
    except Exception as e:  # noqa: BLE001
        o.job_status, o.error = "error", f"submit: {type(e).__name__}: {e}"
        return o
    o.submit_status, o.submit_seconds = r.status_code, time.perf_counter() - t0
    if r.status_code == 429:
        o.job_status = "refused"
        return o
    if r.status_code not in (200, 202):
        o.job_status, o.error = "error", f"submit {r.status_code}: {r.text[:200]}"
        return o
    view = r.json()
    while view["status"] == "pending":
        if time.perf_counter() - t0 > timeout:
            o.job_status = "timeout"
            return o
        await asyncio.sleep(poll * random.uniform(0.8, 1.2))
        try:
            g = await client.get(f"{url}/v1/jobs/{view['job_id']}", headers=headers)
            g.raise_for_status()
            view = g.json()
        except Exception as e:  # noqa: BLE001 - keep polling; a lost GET is not a lost job
            o.error = f"poll: {type(e).__name__}: {e}"
    o.client_seconds = time.perf_counter() - t0
    o.job_status = view["status"]
    res = list(view["results"].values())
    o.cached = all(x.get("cached") for x in res)
    o.server_seconds = max((x.get("total_seconds") or math.nan) for x in res)
    qs = [x["queue_seconds"] for x in res if x.get("queue_seconds") is not None]
    o.queue_seconds = max(qs) if qs else math.nan
    return o


async def run_rate(args, rate: float, images: list[bytes], prompts: list[str]) -> RateResult:
    import httpx

    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    limits = httpx.Limits(max_connections=args.max_connections, max_keepalive_connections=args.max_connections)
    rng = random.Random(args.seed + int(rate * 1000))
    tasks: list[asyncio.Task] = []
    used = 0
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        start = time.perf_counter()
        t_next = start
        while True:
            t_next += rng.expovariate(rate)
            if t_next - start > args.duration:
                break
            await asyncio.sleep(max(0.0, t_next - time.perf_counter()))
            if args.repeat > 0 and used > 0 and rng.random() < args.repeat:
                k = rng.randrange(used)                      # an earlier (image, prompt): a cache hit
            else:
                k, used = used, used + 1
            img, prompt = images[k % len(images)], prompts[k % len(prompts)]
            tasks.append(asyncio.create_task(one_request(client, args.url, img, prompt, args.metrics, headers,
                                                         args.poll, args.timeout)))
        outcomes: list[Outcome] = await asyncio.gather(*tasks)
        end = time.perf_counter()
    res = RateResult(args.label, rate, args.duration, sent=len(outcomes))
    for o in outcomes:
        if o.job_status == "done":
            res.done += 1
            res.cached += o.cached
        elif o.job_status == "failed":
            res.failed += 1
        elif o.job_status == "refused":
            res.refused += 1
        elif o.job_status == "timeout":
            res.timeouts += 1
        else:
            res.errors += 1
    done = [o for o in outcomes if o.job_status == "done"]
    res.throughput = round(res.done / (end - start), 3)
    sub = [o.submit_seconds for o in outcomes]
    res.submit_p50, res.submit_p99 = pct(sub, .5), pct(sub, .99)
    e2e = [o.server_seconds for o in done if not o.cached]
    res.e2e_p50, res.e2e_p90, res.e2e_p99 = pct(e2e, .5), pct(e2e, .9), pct(e2e, .99)
    res.e2e_max = max(e2e) if e2e else math.nan
    q = [o.queue_seconds for o in done if not o.cached]
    res.queue_p50, res.queue_p99 = pct(q, .5), pct(q, .99)
    errs = [o.error for o in outcomes if o.error]
    res.env = {"client_host": platform.node(), "client_cpus": os.cpu_count(), "python": platform.python_version(),
               "metrics": args.metrics, "repeat": args.repeat, "seed": args.seed, "first_errors": errs[:3],
               "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    return res


def fmt(x: float, nd: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def markdown(rows: list[RateResult]) -> str:
    out = ["| label | offered/s | sent | done | 429 | failed | timeout | done/s | submit p50 / p99 s | "
           "end-to-end p50 / p90 / p99 s | queue p50 / p99 s |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        out.append(f"| {r.label} | {r.offered_rate:g} | {r.sent} | {r.done} | {r.refused} | {r.failed} | {r.timeouts} | "
                   f"{fmt(r.throughput)} | {fmt(r.submit_p50, 3)} / {fmt(r.submit_p99, 3)} | "
                   f"{fmt(r.e2e_p50)} / {fmt(r.e2e_p90)} / {fmt(r.e2e_p99)} | {fmt(r.queue_p50)} / {fmt(r.queue_p99)} |")
    return "\n".join(out)


async def amain(args) -> list[RateResult]:
    rates = [float(x) for x in args.rates.split(",")]
    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = [json.loads(line)["prompt"] for line in f if line.strip()]
    rows = []
    for i, rate in enumerate(rates):
        if i and args.pause:
            await asyncio.sleep(args.pause)
        # fresh images for every rate (different seed), more than the arrivals will need
        images = make_images(int(rate * args.duration * 1.5) + 20, args.seed * 1000 + i)
        r = await run_rate(args, rate, images, prompts)
        rows.append(r)
        print(markdown([r]).splitlines()[-1], flush=True)
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(r)) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rates", default="1", help="offered jobs per second, comma-separated; run in turn")
    ap.add_argument("--duration", type=float, default=60, help="seconds of arrivals per rate")
    ap.add_argument("--pause", type=float, default=5, help="seconds between rates")
    ap.add_argument("--metrics", default=None, help="comma-separated; default: the service's")
    ap.add_argument("--prompts", default=None, help="JSONL with a 'prompt' field (default: 8 built-in prompts)")
    ap.add_argument("--repeat", type=float, default=0.0, help="share of requests that repeat an earlier one")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between status polls")
    ap.add_argument("--timeout", type=float, default=300, help="give up on a job after this many seconds")
    ap.add_argument("--max-connections", type=int, default=100)
    ap.add_argument("--api-key", default=os.environ.get("GENPROBE_API_KEY"))
    ap.add_argument("--label", default="run")
    ap.add_argument("--seed", type=int, default=None, help="default: from the clock, so runs never share images")
    ap.add_argument("--out", default=None, help="append one JSON line per rate")
    ap.add_argument("--markdown", action="store_true", help="print the full table at the end")
    a = ap.parse_args(argv)
    if a.seed is None:
        a.seed = int(time.time()) % 1_000_000
    rows = asyncio.run(amain(a))
    if a.markdown:
        print(markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
