"""Scorer throughput by batch size, without the service around it (D-040).

    python -m genprobe.service.bench --metric clipscore --device cpu --batch-sizes 1,4,16 --images 64

Measures what one worker process can score per second, which bounds the service's throughput
per worker, and how much batching helps. Images are synthetic (as in the load test; a metric's
cost does not depend on the content), written to a temporary directory. Each batch size gets a
warm-up batch that is not timed, then ``--images`` images in batches, timed with the clock
around each ``score`` call. On CUDA the scorer's own synchronisation (results are copied to the
CPU) ends each timed call. One JSON line per batch size goes to ``--out``.

Runs on the laptop for ``fake``, in the worker container for CLIPScore on CPU, and in a Slurm
GPU job for VQAScore (slurm/service_bench.sbatch), never on a login node.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import tempfile
import time

from .loadtest import make_images, DEFAULT_PROMPTS
from .scorers import Item, load_scorer


def run(metric: str, device: str, batch_sizes: list[int], n_images: int, seed: int = 0) -> list[dict]:
    t0 = time.perf_counter()
    scorer = load_scorer(metric, device)
    load_seconds = time.perf_counter() - t0
    rows = []
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i, data in enumerate(make_images(n_images + max(batch_sizes), seed)):
            p = os.path.join(d, f"{i}.jpg")
            with open(p, "wb") as f:
                f.write(data)
            paths.append(p)
        items = [Item(str(i), p, DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)]) for i, p in enumerate(paths)]
        for bs in batch_sizes:
            scorer.score(items[:bs])                                   # warm-up, not timed
            times = []
            for k in range(0, n_images - n_images % bs, bs):
                t = time.perf_counter()
                scorer.score(items[k:k + bs])
                times.append(time.perf_counter() - t)
            total = sum(times)
            rows.append({"metric": metric, "metric_version": scorer.version, "device": device, "batch_size": bs,
                         "images": bs * len(times), "images_per_second": round(bs * len(times) / total, 3),
                         "batch_seconds_median": round(statistics.median(times), 4),
                         "batch_seconds_max": round(max(times), 4), "load_seconds": round(load_seconds, 1),
                         "host": platform.node(), "cpus": os.cpu_count(), "torch_threads": _torch_threads(),
                         "gpu": _gpu_name(device), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(json.dumps(rows[-1]), flush=True)
    return rows


def _torch_threads():
    try:
        import torch

        return torch.get_num_threads()
    except ImportError:
        return None


def _gpu_name(device: str):
    if not device.startswith("cuda"):
        return None
    import torch

    return torch.cuda.get_device_name(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="clipscore")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--images", type=int, default=64, help="timed images per batch size")
    ap.add_argument("--out", default=None, help="append JSON lines here")
    a = ap.parse_args(argv)
    rows = run(a.metric, a.device, [int(x) for x in a.batch_sizes.split(",")], a.images)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
