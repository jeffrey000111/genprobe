"""Summarize one prompt set: metric scores by model and category, plus generation cost.

Reads the generation manifests and data/scores/<metric>/*.jsonl, keeps the samples of the
given prompts file, and writes to --out (default results/<prompts stem>/):

    summary.md       tables for people (scores with bootstrap CIs, cost, sanity checks)
    summary.json     the same numbers for scripts
    per_sample.csv   one row per image with every metric, for looking at examples

    python -m genprobe.summarize --prompts data/prompts/smoke.jsonl --metrics clipscore vqascore

Light (stdlib + numpy), but on HPC3 it still goes through Slurm: slurm/summarize.sbatch.
Confidence intervals resample prompts, not images (CLAUDE.md research rules).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .samples import load_prompts
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

N_BOOT = 2000


def bootstrap_mean_ci(per_prompt: dict[str, list[float]], n_boot: int = N_BOOT, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean over prompts (each prompt = mean over its seeds) and a percentile CI from
    resampling prompts with replacement."""
    vals = np.array([np.mean(v) for v in per_prompt.values() if v], dtype=float)
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    boots = vals[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi)


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def pick_scores(records: list[dict[str, Any]], ids: set[str]) -> tuple[dict[str, dict], str | None, Counter]:
    """Latest ok record per sample for ONE metric_version (the most recently written one),
    so runs with different weights/templates are never averaged together."""
    mine = [r for r in records if r.get("sample_id") in ids]
    versions = Counter(r.get("metric_version") for r in mine if r.get("status") == "ok")
    errors = Counter(r.get("metric_version") for r in mine if r.get("status") == "error")
    if not versions:
        return {}, None, errors
    newest = max((r for r in mine if r.get("status") == "ok"), key=lambda r: r.get("created_at", ""))
    version = newest["metric_version"]
    latest = latest_by_id([r for r in mine if r.get("metric_version") == version])
    ok = {sid: r for sid, r in latest.items() if r.get("status") == "ok"}
    return ok, version, Counter({"versions_seen": len(versions), "errors": errors.get(version, 0)})


def category(tags: list[str]) -> str:
    return tags[0] if tags else "untagged"


def fmt(x: float, nd: int = 3) -> str:
    return "–" if x != x else f"{x:.{nd}f}"


def summarize(*, prompts_path: str, manifests: str, data: str, metrics: list[str], out: str,
              full_run_images: int) -> dict[str, Any]:
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    gen = latest_by_id(read_many(sorted(glob.glob(manifests))))
    gen = {sid: r for sid, r in gen.items() if r.get("prompt_id") in prompts}
    ok_ids = {sid for sid, r in gen.items() if r.get("status") == "ok"}
    models = sorted({r["model"] for r in gen.values()})
    cats = sorted({category(p.tags) for p in prompts.values()})

    scores: dict[str, dict[str, dict]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for m in metrics:
        recs = read_many(sorted((Path(data) / "scores" / m).glob("*.jsonl")))
        ok, version, info = pick_scores(recs, ok_ids)
        scores[m] = ok
        vals = np.array([r["value"] for r in ok.values()], float)
        meta[m] = {"metric_version": version, "n_ok": len(ok), "n_expected": len(ok_ids),
                   "n_error": info.get("errors", 0), "versions_seen": info.get("versions_seen", 0),
                   "min": float(vals.min()) if len(vals) else float("nan"),
                   "max": float(vals.max()) if len(vals) else float("nan"),
                   "std": float(vals.std()) if len(vals) else float("nan"),
                   "n_truncated": sum(1 for r in ok.values() if r.get("extra", {}).get("truncated"))}

    # scores by model (CI over prompts) and by model x category
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    by_cat: dict[str, dict[str, dict]] = defaultdict(dict)
    for m in metrics:
        for model in models:
            per_prompt: dict[str, list[float]] = defaultdict(list)
            per_cat: dict[str, list[float]] = defaultdict(list)
            for sid, r in scores[m].items():
                g = gen[sid]
                if g["model"] != model:
                    continue
                per_prompt[g["prompt_id"]].append(r["value"])
                per_cat[category(prompts[g["prompt_id"]].tags)].append(r["value"])
            mean, lo, hi = bootstrap_mean_ci(per_prompt)
            by_model[m][model] = {"mean": mean, "ci95": [lo, hi], "n_prompts": len(per_prompt),
                                  "n_images": sum(len(v) for v in per_prompt.values())}
            by_cat[m][model] = {c: {"mean": float(np.mean(v)) if v else float("nan"), "n": len(v)}
                                for c, v in ((c, per_cat.get(c, [])) for c in cats)}

    # generation cost from the manifests (warmup batches excluded, D-004)
    cost = {}
    for model in models:
        rows = [r for r in gen.values() if r["model"] == model and r.get("status") == "ok"]
        timed = [r["gen_seconds"] for r in rows if not r.get("warmup") and r.get("gen_seconds") is not None]
        med = float(np.median(timed)) if timed else float("nan")
        cost[model] = {
            "gpu": Counter(r.get("gpu") for r in rows).most_common(1)[0][0] if rows else None,
            "batch_size": rows[0].get("batch_size") if rows else None,
            "steps": rows[0].get("steps") if rows else None,
            "n_images": len(rows), "n_timed": len(timed), "sec_per_image_median": med,
            "peak_mem_gb_max": max((r.get("peak_mem_gb") or 0 for r in rows), default=float("nan")),
            "n_blank": sum(1 for r in gen.values() if r["model"] == model and r.get("status") == "blank"),
            "n_error": sum(1 for r in gen.values() if r["model"] == model and r.get("status") == "error"),
            f"gpu_hours_for_{full_run_images}": med * full_run_images / 3600 if timed else float("nan"),
        }

    # metric-vs-metric rank agreement over images scored by both (not a validity measure)
    agreement = {}
    for i, a in enumerate(metrics):
        for b in metrics[i + 1:]:
            common = sorted(set(scores[a]) & set(scores[b]))
            agreement[f"{a}~{b}"] = {"spearman": spearman([scores[a][s]["value"] for s in common],
                                                          [scores[b][s]["value"] for s in common]),
                                     "n": len(common)}

    result = {"prompts": prompts_path, "created_at": now_iso(), "git_commit": git_commit(),
              "n_prompts": len(prompts), "models": models, "categories": cats, "metrics": meta,
              "by_model": by_model, "by_model_category": by_cat, "generation_cost": cost,
              "metric_agreement": agreement, "bootstrap": {"n_boot": N_BOOT, "seed": 0, "unit": "prompt"}}

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(outdir / "summary.json", json.dumps(result, indent=2, default=str))
    atomic_write_text(outdir / "summary.md", render_md(result, full_run_images))
    write_per_sample(outdir / "per_sample.csv", gen, prompts, scores, metrics)
    return result


def render_md(res: dict[str, Any], full_run_images: int) -> str:
    models, metrics, cats = res["models"], list(res["metrics"]), res["categories"]
    L = [f"# Summary: `{res['prompts']}`", "",
         f"Generated {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.summarize`. "
         f"{res['n_prompts']} prompts; means are over prompts, 95% CIs from {res['bootstrap']['n_boot']} "
         "bootstrap resamples of prompts.", ""]
    L += ["## Metric scores by model", "",
          "| Model | " + " | ".join(metrics) + " |", "| --- |" + " --- |" * len(metrics)]
    for model in models:
        cells = []
        for m in metrics:
            d = res["by_model"][m][model]
            cells.append(f"{fmt(d['mean'])} [{fmt(d['ci95'][0])}, {fmt(d['ci95'][1])}] (n={d['n_images']})")
        L.append(f"| {model} | " + " | ".join(cells) + " |")
    for m in metrics:
        L += ["", f"### {m} by category (mean, n images)", "",
              "| Model | " + " | ".join(cats) + " |", "| --- |" + " --- |" * len(cats)]
        for model in models:
            d = res["by_model_category"][m][model]
            L.append(f"| {model} | " + " | ".join(f"{fmt(d[c]['mean'])} ({d[c]['n']})" for c in cats) + " |")
    L += ["", "## Metric sanity checks", "",
          "| Metric | metric_version | scored / expected | errors | min | max | std | truncated prompts |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, d in res["metrics"].items():
        extra = " (other versions exist; only this one is used)" if d["versions_seen"] > 1 else ""
        L.append(f"| {m} | `{d['metric_version']}`{extra} | {d['n_ok']} / {d['n_expected']} | {d['n_error']} | "
                 f"{fmt(d['min'])} | {fmt(d['max'])} | {fmt(d['std'])} | {d['n_truncated']} |")
    if res["metric_agreement"]:
        L += ["", "Rank agreement between metrics over the same images (Spearman). This says how often "
              "the metrics agree with each other, not which one agrees with people.", ""]
        for k, d in res["metric_agreement"].items():
            L.append(f"- {k}: ρ = {fmt(d['spearman'])} (n = {d['n']})")
    L += ["", "## Generation cost", "",
          f"Median seconds per image from the manifests, warmup batches excluded (D-004). "
          f"GPU-hours = median × {full_run_images} images / 3600.", "",
          f"| Model | GPU | Batch | Steps | Images (timed) | s / image | Peak mem GB | Blank | Errors | GPU-h for {full_run_images} |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model in models:
        c = res["generation_cost"][model]
        L.append(f"| {model} | {c['gpu']} | {c['batch_size']} | {c['steps']} | {c['n_images']} ({c['n_timed']}) | "
                 f"{fmt(c['sec_per_image_median'], 2)} | {fmt(c['peak_mem_gb_max'], 1)} | {c['n_blank']} | "
                 f"{c['n_error']} | {fmt(c[f'gpu_hours_for_{full_run_images}'], 2)} |")
    return "\n".join(L) + "\n"


def write_per_sample(path: Path, gen: dict, prompts: dict, scores: dict, metrics: list[str]) -> None:
    cols = ["sample_id", "model", "prompt_id", "seed", "category", "prompt", "status", "gen_seconds",
            "warmup", *metrics, "clip_truncated"]
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sid in sorted(gen):
            g = gen[sid]
            row = [sid, g["model"], g["prompt_id"], g.get("seed"), category(prompts[g["prompt_id"]].tags),
                   g.get("prompt"), g.get("status"), g.get("gen_seconds"), g.get("warmup")]
            row += [scores[m][sid]["value"] if sid in scores[m] else "" for m in metrics]
            clip = scores.get("clipscore", {}).get(sid)
            row.append(clip["extra"].get("truncated") if clip else "")
            w.writerow(row)
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--metrics", nargs="+", default=["clipscore", "vqascore"])
    ap.add_argument("--manifests", default="data/manifests/gen_*.jsonl", help="glob of generation manifests")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=None, help="default: results/<prompts file stem>")
    ap.add_argument("--full-run-images", type=int, default=2400)
    args = ap.parse_args(argv)
    out = args.out or str(Path("results") / Path(args.prompts).stem)
    res = summarize(prompts_path=args.prompts, manifests=args.manifests, data=args.data,
                    metrics=args.metrics, out=out, full_run_images=args.full_run_images)
    missing = {m: d for m, d in res["metrics"].items() if d["n_ok"] < d["n_expected"]}
    print(f"wrote {out}/summary.md, summary.json, per_sample.csv")
    for m, d in res["metrics"].items():
        print(f"  {m}: {d['n_ok']}/{d['n_expected']} scored, version {d['metric_version']}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
