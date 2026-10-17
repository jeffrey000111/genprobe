"""Run one metric over a set of samples with sharding and resume (guide §7.1).

Input is any JSONL whose rows have at least sample_id, path and prompt — generation
manifests work as-is, and later the GenAI-Bench human-rated images get converted to
the same shape for meta-evaluation.

    python -m genprobe.score --metric clipscore \
        --samples "data/manifests/gen_sdxl_*.jsonl" --tag sdxl \
        --shard $SLURM_ARRAY_TASK_ID --num-shards 4
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from .metrics import REGISTRY, Metric
from .samples import Sample, load_prompts
from .utils.env import fingerprint, now_iso, run_id
from .utils.io import atomic_write_text
from .utils.manifest import ManifestWriter, done_ids, latest_by_id, read_many
from .utils.shard import chunks, select_shard


def load_samples(pattern: str, prompts_path: str | None = None) -> list[Sample]:
    """Latest usable record per sample_id; joins tags/spec from a prompts file if given."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")
    latest = latest_by_id(read_many(paths))
    rows = [r for r in latest.values() if r.get("status", "ok") == "ok"]
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)} if prompts_path else {}
    out = []
    for r in rows:
        s = Sample.from_dict(r)
        p = prompts.get(s.prompt_id)
        if p is not None:
            s.tags, s.spec = list(p.tags), p.spec
        out.append(s)
    return out


def run_scoring(
    *,
    metric: Metric,
    samples: list[Sample],
    out: str | Path,
    tag: str,
    shard: int = 0,
    num_shards: int = 1,
    batch_size: int | None = None,
    device: str = "cuda",
    log=print,
) -> dict[str, int]:
    out = Path(out)
    bs = batch_size or metric.default_batch_size
    # batch size can change numerics, so it is part of the version string (guide §4.7)
    mversion = f"{metric.version}/bs{bs}"
    score_dir = out / "scores" / metric.name

    if "spec" in metric.needs and samples and not any(s.spec for s in samples):
        log(f"WARNING: {metric.name} needs prompt specs but no sample has one — "
            f"did you forget --prompts? Nothing will be scored.")
    applicable = [s for s in samples if metric.applies_to(s)]
    mine = select_shard(applicable, shard, num_shards, key=lambda s: s.sample_id)
    existing = read_many(score_dir.glob("*.jsonl")) if score_dir.exists() else []
    done = done_ids(existing, match={"metric_version": mversion})
    todo = [s for s in mine if s.sample_id not in done]
    log(f"[{metric.name}] {tag} shard {shard}/{num_shards}: {len(mine)} samples "
        f"({len(samples) - len(applicable)} not applicable), {len(todo)} to score")
    counts = {"ok": 0, "error": 0, "skipped": len(mine) - len(todo)}
    if not todo:
        return counts

    rid = run_id()
    fp = fingerprint()
    atomic_write_text(out / "runs" / f"score_{metric.name}_{tag}_{rid}.json",
                      json.dumps({"run_id": rid, "metric": metric.name, "metric_version": mversion,
                                  "env": fp}, indent=2, default=str))
    metric.setup(device)
    path = score_dir / f"{tag}_shard{shard:03d}of{num_shards:03d}.jsonl"
    with ManifestWriter(path) as w:
        for batch in chunks(todo, bs):
            t0 = time.perf_counter()
            try:
                results = metric.score_batch(batch)
                if len(results) != len(batch):
                    raise RuntimeError(f"score_batch returned {len(results)} results for {len(batch)} samples")
                err = None
            except Exception as e:  # noqa: BLE001
                results, err = [{"sample_id": s.sample_id, "value": None, "extra": {}} for s in batch], \
                    f"{type(e).__name__}: {e}"[:2000]
            per = (time.perf_counter() - t0) / len(batch)
            for s, r in zip(batch, results):
                if r.get("sample_id") != s.sample_id:
                    raise RuntimeError(f"{metric.name} returned results out of order")
                rec = {
                    "sample_id": s.sample_id, "metric": metric.name, "metric_version": mversion,
                    "value": r.get("value"), "extra": r.get("extra", {}), "seconds": round(per, 4),
                    "status": "error" if err else "ok", "created_at": now_iso(), "run_id": rid,
                    "git_commit": fp["git_commit"], "gpu": fp.get("gpu"),
                }
                if err:
                    rec["error"] = err
                counts[rec["status"]] += 1
                w.write(rec)
    log(f"[{metric.name}] {tag} shard {shard}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", required=True, choices=sorted(REGISTRY) or None)
    ap.add_argument("--samples", required=True, help="glob of JSONL files with sample_id/path/prompt")
    ap.add_argument("--tag", required=True, help="name for this input set, e.g. the model name")
    ap.add_argument("--prompts", default=None, help="prompts JSONL to join tags/spec from")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--out", default="data")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    metric = REGISTRY[args.metric]()
    samples = load_samples(args.samples, args.prompts)
    counts = run_scoring(metric=metric, samples=samples, out=args.out, tag=args.tag, shard=args.shard,
                         num_shards=args.num_shards, batch_size=args.batch_size, device=args.device)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
