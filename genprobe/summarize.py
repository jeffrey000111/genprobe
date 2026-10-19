"""Summarize one prompt set: metric scores by model and category, plus generation cost.

Reads the generation manifests and data/scores/<metric>/*.jsonl, keeps the samples of the
given prompts file, and writes to --out (default results/<prompts stem>/):

    summary.md       tables for people (scores with bootstrap CIs, cost, sanity checks, and
                     agreement with human labels when data/labels/<stem>_labels.csv exists)
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

from .metrics import REGISTRY
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


LABEL_VALUES = ("y", "n", "?")


def load_labels(path: str | Path) -> dict[str, dict[str, str]]:
    """Human labels: CSV with sample_id, correct (y / n / ?), note. Full-width ？ is accepted."""
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            c = (row.get("correct") or "").strip().replace("？", "?").lower()
            if c not in LABEL_VALUES:
                raise ValueError(f"{path}: {row.get('sample_id')}: correct={row.get('correct')!r}, "
                                 f"expected one of {LABEL_VALUES}")
            out[row["sample_id"]] = {"correct": c, "note": row.get("note", "")}
    return out


def pairwise_agreement(values: dict[str, float], labels: dict[str, str], prompt_of: dict[str, str],
                       n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Within each prompt, every pair (image labelled y, image labelled n): does the metric score
    the y image higher? Ties count 1/2. Pairs never cross prompts, so prompt difficulty cancels
    out. CI: resample prompts with replacement, pool their pairs."""
    per_prompt: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    by_prompt: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"y": [], "n": []})
    for sid, lab in labels.items():
        if lab in ("y", "n") and sid in values:
            by_prompt[prompt_of[sid]][lab].append(sid)
    for pid, d in by_prompt.items():
        for a in d["y"]:
            for b in d["n"]:
                diff = values[a] - values[b]
                per_prompt[pid].append((1.0 if diff > 0 else 0.5 if diff == 0 else 0.0, a, b))
    pids = sorted(per_prompt)
    sums = np.array([sum(x for x, _, _ in per_prompt[p]) for p in pids], float)
    counts = np.array([len(per_prompt[p]) for p in pids], float)
    if not len(pids):
        return {"pairs": 0, "n_prompts": 0, "agreement": float("nan"), "ci95": [float("nan")] * 2, "worst": []}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
    boots = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    # strict mistakes only (the wrong image scored higher); ties are counted as 1/2 above
    wrong = sorted(((values[b] - values[a], a, b) for p in pids for x, a, b in per_prompt[p] if x == 0.0),
                   reverse=True)
    return {"pairs": int(counts.sum()), "n_prompts": len(pids), "agreement": float(sums.sum() / counts.sum()),
            "ci95": [float(lo), float(hi)], "ties": int(sum(x == 0.5 for p in pids for x, _, _ in per_prompt[p])),
            "worst": [{"labelled_correct": a, "labelled_wrong": b, "margin": round(m, 4)} for m, a, b in wrong[:5]]}


def decision_agreement(values: dict[str, float], labels: dict[str, str], prompt_of: dict[str, str],
                       threshold: float, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Checker verdict (value >= threshold) against the human y/n label, image by image.
    Accuracy CI: resample prompts with replacement, pool their images."""
    per_prompt: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for sid, lab in labels.items():
        if lab in ("y", "n") and sid in values:
            per_prompt[prompt_of[sid]].append((values[sid] >= threshold, lab == "y"))
    pairs = [x for v in per_prompt.values() for x in v]
    cm = Counter(("t" if pred == truth else "f") + ("p" if pred else "n") for pred, truth in pairs)
    out: dict[str, Any] = {"threshold": threshold, "images": len(pairs), "n_prompts": len(per_prompt),
                           "tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"]}
    if not pairs:
        return {**out, "accuracy": float("nan"), "ci95": [float("nan")] * 2}
    pids = sorted(per_prompt)
    hits = np.array([sum(p == t for p, t in per_prompt[k]) for k in pids], float)
    counts = np.array([len(per_prompt[k]) for k in pids], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pids), size=(n_boot, len(pids)))
    lo, hi = np.quantile(hits[idx].sum(axis=1) / counts[idx].sum(axis=1), [0.025, 0.975])
    return {**out, "accuracy": float(hits.sum() / counts.sum()), "ci95": [float(lo), float(hi)]}


def metric_applies(name: str):
    """Which prompts a metric scores (OCR only text prompts, ...); unknown metrics: all."""
    cls = REGISTRY.get(name)
    if cls is None:
        return lambda prompt: True
    inst = cls()
    return lambda prompt: inst.applies_to(prompt)


def category(tags: list[str]) -> str:
    return tags[0] if tags else "untagged"


def fmt(x: float, nd: int = 3) -> str:
    return "–" if x != x else f"{x:.{nd}f}"


def summarize(*, prompts_path: str, manifests: str, data: str, metrics: list[str], out: str,
              full_run_images: int, labels_path: str | None = None, split: str | None = None) -> dict[str, Any]:
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path, split=split)}
    gen = latest_by_id(read_many(sorted(glob.glob(manifests))))
    gen = {sid: r for sid, r in gen.items() if r.get("prompt_id") in prompts}
    ok_ids = {sid for sid, r in gen.items() if r.get("status") == "ok"}
    models = sorted({r["model"] for r in gen.values()})
    cats = sorted({category(p.tags) for p in prompts.values()})

    scores: dict[str, dict[str, dict]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for m in metrics:
        recs = read_many(sorted((Path(data) / "scores" / m).glob("*.jsonl")))
        applies = metric_applies(m)
        expected = {sid for sid in ok_ids if applies(prompts[gen[sid]["prompt_id"]])}
        ok, version, info = pick_scores(recs, expected)
        scores[m] = ok
        vals = np.array([r["value"] for r in ok.values()], float)
        meta[m] = {"metric_version": version, "n_ok": len(ok), "n_expected": len(expected),
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

    # agreement with human labels (y / n per image), within prompts
    human: dict[str, Any] = {}
    labels = {}
    if labels_path:
        labels = {sid: d for sid, d in load_labels(labels_path).items() if sid in gen}
        lab = {sid: d["correct"] for sid, d in labels.items()}
        prompt_of = {sid: gen[sid]["prompt_id"] for sid in gen}
        human = {"path": str(labels_path), "counts": dict(Counter(lab.values())), "by_metric": {}}
        for m in metrics:
            vals = {sid: r["value"] for sid, r in scores[m].items()}
            entry = {"all": pairwise_agreement(vals, lab, prompt_of)}
            for c in cats:
                entry[c] = pairwise_agreement(vals, {s: v for s, v in lab.items()
                                                     if category(prompts[gen[s]["prompt_id"]].tags) == c},
                                              prompt_of)
            ys = [vals[s] for s, v in lab.items() if v == "y" and s in vals]
            ns = [vals[s] for s, v in lab.items() if v == "n" and s in vals]
            entry["mean_if_y"] = float(np.mean(ys)) if ys else float("nan")
            entry["mean_if_n"] = float(np.mean(ns)) if ns else float("nan")
            thr = getattr(REGISTRY.get(m), "decision_threshold", None)
            if thr is not None:
                entry["verdict"] = decision_agreement(vals, lab, prompt_of, thr)
            human["by_metric"][m] = entry

    result = {"prompts": prompts_path, "split": split or "all", "created_at": now_iso(), "git_commit": git_commit(),
              "n_prompts": len(prompts), "models": models, "categories": cats, "metrics": meta,
              "by_model": by_model, "by_model_category": by_cat, "generation_cost": cost,
              "metric_agreement": agreement, "human_agreement": human,
              "bootstrap": {"n_boot": N_BOOT, "seed": 0, "unit": "prompt"}}

    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(outdir / "summary.json", json.dumps(result, indent=2, default=str))
    atomic_write_text(outdir / "summary.md", render_md(result, full_run_images))
    write_per_sample(outdir / "per_sample.csv", gen, prompts, scores, metrics, labels)
    return result


def render_md(res: dict[str, Any], full_run_images: int) -> str:
    models, metrics, cats = res["models"], list(res["metrics"]), res["categories"]
    L = [f"# Summary: `{res['prompts']}` (split: {res.get('split', 'all')})", "",
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
    h = res.get("human_agreement") or {}
    if h:
        L += ["", "## Agreement with human labels", "",
              f"Labels: `{h['path']}` ({', '.join(f'{k}={v}' for k, v in sorted(h['counts'].items()))}; "
              "`?` excluded). Within each prompt, every pair of one image labelled correct and one labelled "
              "wrong: how often the metric scores the correct one higher (ties = 1/2). 95% CI from "
              "resampling prompts. One annotator, small sample: a sanity check, not a result (D-010, D-015).", "",
              "| Metric | all | " + " | ".join(cats) + " | mean score if correct / wrong |",
              "| --- | --- |" + " --- |" * len(cats) + " --- |"]
        for m, e in h["by_metric"].items():
            def cell(d):
                return "–" if not d["pairs"] else (f"{fmt(d['agreement'], 2)} [{fmt(d['ci95'][0], 2)}, "
                                                   f"{fmt(d['ci95'][1], 2)}] ({d['pairs']} pairs)")
            L.append(f"| {m} | {cell(e['all'])} | " + " | ".join(cell(e[c]) for c in cats) +
                     f" | {fmt(e['mean_if_y'])} / {fmt(e['mean_if_n'])} |")
        verdicts = {m: e["verdict"] for m, e in h["by_metric"].items() if e.get("verdict")}
        if verdicts:
            L += ["", "Checker verdicts against the same labels, image by image (an image passes when the "
                  "checker's value ≥ its threshold). Accuracy CI from resampling prompts.", "",
                  "| Checker | threshold | images | accuracy | passes & correct | passes & wrong | "
                  "fails & correct | fails & wrong |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for m, v in verdicts.items():
                L.append(f"| {m} | {v['threshold']} | {v['images']} | {fmt(v['accuracy'], 2)} "
                         f"[{fmt(v['ci95'][0], 2)}, {fmt(v['ci95'][1], 2)}] | {v['tp']} | {v['fp']} | "
                         f"{v['fn']} | {v['tn']} |")
        for m, e in h["by_metric"].items():
            if e["all"]["worst"]:
                L += ["", f"Largest {m} mistakes (image labelled wrong scored above one labelled correct, same prompt):", ""]
                L += [f"- `{w['labelled_wrong']}` > `{w['labelled_correct']}` by {w['margin']}"
                      for w in e["all"]["worst"]]
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


def checker_detail(metric: str, extra: dict) -> str:
    """One readable cell per checker verdict, for looking at examples in per_sample.csv."""
    if metric == "ocr":
        return extra.get("ocr_norm", "")
    if metric == "detect":
        if extra.get("check") == "count":
            return f"count {extra.get('count_pred')}/{extra.get('count_target')}"
        if extra.get("missing"):
            return f"{extra.get('check')}: missing {','.join(extra['missing'])}"
        a, b = extra.get("centre_a"), extra.get("centre_b")
        if a and b:
            return f"{extra.get('check')}: a at ({a[0]:.2f},{a[1]:.2f}) b at ({b[0]:.2f},{b[1]:.2f})"
    return ""


DETAIL_COLUMNS = {"ocr": "ocr_read", "detect": "detect_detail"}


def write_per_sample(path: Path, gen: dict, prompts: dict, scores: dict, metrics: list[str],
                     labels: dict | None = None) -> None:
    labels = labels or {}
    details = [m for m in metrics if m in DETAIL_COLUMNS]
    cols = ["sample_id", "model", "prompt_id", "seed", "category", "prompt", "status", "gen_seconds",
            "warmup", *metrics, *(DETAIL_COLUMNS[m] for m in details), "clip_truncated", "human_correct",
            "human_note"]
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for sid in sorted(gen):
            g = gen[sid]
            row = [sid, g["model"], g["prompt_id"], g.get("seed"), category(prompts[g["prompt_id"]].tags),
                   g.get("prompt"), g.get("status"), g.get("gen_seconds"), g.get("warmup")]
            row += [scores[m][sid]["value"] if sid in scores[m] else "" for m in metrics]
            row += [checker_detail(m, scores[m][sid].get("extra", {})) if sid in scores[m] else "" for m in details]
            clip = scores.get("clipscore", {}).get(sid)
            row.append(clip["extra"].get("truncated") if clip else "")
            row += [labels.get(sid, {}).get("correct", ""), labels.get(sid, {}).get("note", "")]
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
    ap.add_argument("--split", default=None, choices=["dev", "test"],
                    help="only prompts of this split (reported numbers come from test)")
    ap.add_argument("--labels", default=None,
                    help="human labels CSV (sample_id, correct=y/n/?); default data/labels/<prompts stem>_labels.csv "
                         "if it exists")
    args = ap.parse_args(argv)
    stem = Path(args.prompts).stem
    out = args.out or str(Path("results") / (f"{stem}_{args.split}" if args.split else stem))
    labels = args.labels
    if labels is None and (Path("data/labels") / f"{stem}_labels.csv").is_file():
        labels = str(Path("data/labels") / f"{stem}_labels.csv")
    res = summarize(prompts_path=args.prompts, manifests=args.manifests, data=args.data,
                    metrics=args.metrics, out=out, full_run_images=args.full_run_images, labels_path=labels,
                    split=args.split)
    missing = {m: d for m, d in res["metrics"].items() if d["n_ok"] < d["n_expected"]}
    print(f"wrote {out}/summary.md, summary.json, per_sample.csv")
    for m, d in res["metrics"].items():
        print(f"  {m}: {d['n_ok']}/{d['n_expected']} scored, version {d['metric_version']}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
