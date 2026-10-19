"""Pipeline check of one scoring run (no comparison with people): per metric, how many samples
were expected and scored, errors, the metric_version, the range of values and the measured
seconds per image, plus a cost projection for a larger set.

    python -m genprobe.score_check --samples data/manifests/ext_genai_bench.jsonl \
        --prompts data/prompts/genai_bench_smoke.jsonl --tag genai_bench_smoke \
        --project 9600 clipscore vqascore pickscore hpsv2

Reads data/scores/<metric>/<tag>_shard*.jsonl only. Exit status 1 if a metric has errors or
missing samples.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .metrics import REGISTRY
from .score import load_samples
from .utils.manifest import latest_by_id, read_many


def check(metric: str, samples, data: str, tag: str, project: int | None) -> tuple[list[str], bool]:
    m = REGISTRY[metric]()
    expected = {s.sample_id: s for s in samples if m.applies_to(s)}
    recs = read_many(sorted((Path(data) / "scores" / metric).glob(f"{tag}_shard*.jsonl")))
    latest = latest_by_id([r for r in recs if r.get("sample_id") in expected])
    status = Counter(r.get("status") for r in latest.values())
    ok = [r for r in latest.values() if r.get("status") == "ok"]
    vals = [float(r["value"]) for r in ok if r.get("value") is not None]
    bad = [v for v in vals if v != v or v in (float("inf"), float("-inf"))]
    secs = [float(r["seconds"]) for r in latest.values() if r.get("seconds") is not None]
    by_model: dict[str, list[float]] = defaultdict(list)
    for r in latest.values():
        if r.get("seconds") is not None:
            by_model[expected[r["sample_id"]].model].append(float(r["seconds"]))
    versions = Counter(r.get("metric_version") for r in latest.values())
    missing = len(expected) - len(latest)
    lines = [f"## {metric}",
             f"expected {len(expected)}, scored {len(latest)}, ok {status.get('ok', 0)}, "
             f"error {status.get('error', 0)}, missing {missing}",
             "metric_version: " + "; ".join(f"{v} ({n})" for v, n in versions.items())]
    if vals:
        q = statistics.quantiles(vals, n=20) if len(vals) > 1 else [vals[0]] * 19
        lines.append(f"value min {min(vals):.4g}, p5 {q[0]:.4g}, median {statistics.median(vals):.4g}, "
                     f"p95 {q[-1]:.4g}, max {max(vals):.4g}; non-finite {len(bad)}; distinct {len(set(vals))}")
    if secs:
        lines.append(f"seconds/image median {statistics.median(secs):.3f}, mean {statistics.fmean(secs):.3f}, "
                     f"total {sum(secs) / 60:.1f} min (scoring only, without model loading)")
        lines.append("by model (mean s/image): " + ", ".join(
            f"{k} {statistics.fmean(v):.3f}" for k, v in sorted(by_model.items())))
        if project:
            lines.append(f"projected for {project} images: {statistics.fmean(secs) * project / 3600:.2f} h of scoring")
    errs = Counter(str(r.get("error", ""))[:200] for r in latest.values() if r.get("status") == "error")
    for e, n in errs.most_common(3):
        lines.append(f"error ×{n}: {e}")
    healthy = missing == 0 and not errs and not bad and bool(vals)
    return lines, healthy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--samples", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--project", type=int, default=None, help="number of images to project the cost to")
    args = ap.parse_args(argv)
    samples = load_samples(args.samples, args.prompts)
    print(f"{len(samples)} samples in {args.prompts}")
    ok_all = True
    for m in args.metrics:
        lines, ok = check(m, samples, args.data, args.tag, args.project)
        ok_all &= ok
        print("\n".join(lines) + ("" if ok else "\n-> NOT OK"), end="\n\n")
    print("all ok" if ok_all else "PROBLEMS above")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
