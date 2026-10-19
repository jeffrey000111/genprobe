"""Sanity checks every metric must pass before its numbers are used (guide §7.9, D-023).

Run on a fixed random sample of dev-split images (never test), for one metric per job:

1. **Mismatch**: score each image against its own prompt and against another prompt of the
   same kind (any prompt for alignment metrics; a prompt with the same kind of spec for the
   checkers). A metric that measures alignment should prefer the own prompt almost always.
2. **Counterfactual**: for template prompts, change one thing (the count by 2, left <-> right /
   above <-> below, or the requested text) and score again. A score that doesn't drop is
   reading the prompt as a bag of words. On random images this is confounded: when the
   generator drew the scene wrong (often, for counts and positions), the changed prompt may fit
   the image better. So the clean version runs on the images a person labelled correct
   (data/labels/<prompt set>_labels.csv, ``--part labelled``): there the original prompt is
   right and the changed one is wrong, and a valid metric should prefer the original.
3. **Repeatability**: score the same images a second time in the same job, and compare with the
   scores stored by the scoring run (same ``metric_version``).
4. **Direction and range** are covered by the human-label tables of the summaries (mean score of
   images labelled correct vs wrong) and their min / max columns.

    python -m genprobe.sanity --metric vqascore --prompts data/prompts/prompts_v1.jsonl --device cuda
    python -m genprobe.sanity --metric vqascore --part labelled --device cuda   # adds the clean check
    python -m genprobe.sanity --report      # results/sanity/summary.md from all metrics run so far

Metrics without a prompt (the safety filters) only get the repeatability check.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .build_prompts import COUNT_OBJECTS, NUMBER_WORDS, TEXT_PHRASES
from .metrics import REGISTRY
from .samples import Sample, load_prompts
from .score import load_samples
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

OUT_DIR = Path("results/sanity")
PLURAL = dict(COUNT_OBJECTS)
FLIP = {"left_of": ("right_of", "to the left of", "to the right of"),
        "right_of": ("left_of", "to the right of", "to the left of"),
        "above": ("below", " above ", " below "), "below": ("above", " below ", " above ")}


# ---------- prompt edits ----------

def spec_kind(spec: dict | None) -> str | None:
    if not spec:
        return None
    if spec.get("objects"):
        return "count"
    if spec.get("relation"):
        return "relation"
    if isinstance(spec.get("text"), str):
        return "text"
    return None


def counterfactual(prompt: str, spec: dict | None) -> tuple[str, dict, str] | None:
    """(new prompt, new spec, what changed) with exactly one thing changed, or None."""
    kind = spec_kind(spec)
    if kind == "count":
        obj = spec["objects"][0]
        name, n = obj["name"], int(obj["count"])
        m = n + 2 if n <= 5 else n - 2
        noun = lambda k: name if k == 1 else PLURAL.get(name, name + "s")  # noqa: E731
        for old, new in ((f"{NUMBER_WORDS.get(n, n)} {noun(n)}", f"{NUMBER_WORDS.get(m, m)} {noun(m)}"),
                         (f"{n} {noun(n)}", f"{m} {noun(m)}")):
            pat = re.compile(rf"\b{re.escape(old)}\b")
            if pat.search(prompt):
                return pat.sub(new, prompt, count=1), {**spec, "objects": [{**obj, "count": m}]}, f"count {n}->{m}"
        return None
    if kind == "relation":
        rel = spec["relation"]
        new_type, old_phrase, new_phrase = FLIP[rel["type"]]
        padded = f" {prompt} "
        if old_phrase not in padded:
            return None
        new_prompt = padded.replace(old_phrase, new_phrase, 1).strip()
        return new_prompt, {**spec, "relation": {**rel, "type": new_type}}, f"{rel['type']}->{new_type}"
    if kind == "text":
        text = spec["text"]
        for phrases in TEXT_PHRASES.values():
            if text in phrases:
                other = phrases[(phrases.index(text) + 1) % len(phrases)]
                if f'"{text}"' in prompt:
                    return prompt.replace(f'"{text}"', f'"{other}"', 1), {**spec, "text": other}, "text"
        return None
    return None


# ---------- checks ----------

def own_higher(own: list[float], other: list[float]) -> dict[str, Any]:
    """Share of images where the own prompt scores higher (ties = 1/2), and the strict counts."""
    o, x = np.asarray(own, float), np.asarray(other, float)
    if not len(o):
        return {"n": 0, "rate": float("nan"), "higher": 0, "ties": 0, "lower": 0}
    hi, ti = int((o > x).sum()), int((o == x).sum())
    return {"n": len(o), "rate": float((hi + 0.5 * ti) / len(o)), "higher": hi, "ties": ti,
            "lower": len(o) - hi - ti, "mean_drop": float((o - x).mean())}


def repeat_stats(a: list[float], b: list[float]) -> dict[str, Any]:
    a_, b_ = np.asarray(a, float), np.asarray(b, float)
    if not len(a_):
        return {"n": 0}
    d = np.abs(a_ - b_)
    return {"n": len(a_), "identical": float((d == 0).mean()), "max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean())}


def score_values(metric, samples: list[Sample], bs: int) -> list[float]:
    vals: list[float] = []
    for i in range(0, len(samples), bs):
        batch = samples[i:i + bs]
        res = metric.score_batch(batch)
        vals += [float(r["value"]) for r in res]
    return vals


def pick(samples: list[Sample], n: int, rng: random.Random) -> list[Sample]:
    return sorted(rng.sample(samples, min(n, len(samples))), key=lambda s: s.sample_id)


def run(metric_name: str, prompts_path: str, manifests: str, data: str = "data", split: str = "dev",
        n: int = 200, n_repeat: int = 50, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    metric = REGISTRY[metric_name]()
    bs = metric.default_batch_size
    mversion = f"{metric.version}/bs{bs}"
    split_ids = {p.prompt_id for p in load_prompts(prompts_path, split=split)}
    samples = [s for s in load_samples(manifests, prompts_path) if s.prompt_id in split_ids]
    applicable = [s for s in samples if metric.applies_to(s)]
    rng = random.Random(seed)
    base = pick(applicable, n, rng)
    uses_prompt = "prompt" in metric.needs
    uses_spec = "spec" in metric.needs

    # mismatched prompt (or spec) for each image, from a different prompt of the same kind
    pool: dict[Any, list[Sample]] = {}
    for s in applicable:
        pool.setdefault(spec_kind(s.spec) if uses_spec else "any", []).append(s)
    mismatched = []
    for s in base:
        cands = [c for c in pool[spec_kind(s.spec) if uses_spec else "any"] if c.prompt_id != s.prompt_id]
        o = rng.choice(cands) if cands else None
        mismatched.append(None if o is None else replace(s, prompt=o.prompt, spec=o.spec, tags=o.tags))
    cf = []
    for s in base:
        c = counterfactual(s.prompt, s.spec) if (uses_prompt or uses_spec) else None
        cf.append(None if c is None else (replace(s, prompt=c[0], spec=c[1]), c[2]))

    metric.setup(device)
    own = score_values(metric, base, bs)
    res: dict[str, Any] = {"metric": metric_name, "metric_version": mversion, "split": split, "seed": seed,
                           "n_images": len(base), "created_at": now_iso(), "git_commit": git_commit(),
                           "device": device, "metric_info": getattr(metric, "info", {})}
    # every score, so any rate in the table can be traced back to the images behind it
    per_image = [{"sample_id": s.sample_id, "prompt": s.prompt, "own": v} for s, v in zip(base, own)]
    if uses_prompt or uses_spec:
        idx = [i for i, m in enumerate(mismatched) if m is not None]
        other = score_values(metric, [mismatched[i] for i in idx], bs)
        res["mismatch"] = own_higher([own[i] for i in idx], other)
        for i, v in zip(idx, other):
            per_image[i].update(mismatch_prompt=mismatched[i].prompt, mismatch=v)
        idx = [i for i, c in enumerate(cf) if c is not None]
        if idx:
            cf_vals = score_values(metric, [cf[i][0] for i in idx], bs)
            for i, v in zip(idx, cf_vals):
                per_image[i].update(cf_prompt=cf[i][0].prompt, cf_change=cf[i][1], cf=v)
            res["counterfactual"] = {"all": own_higher([own[i] for i in idx], cf_vals)}
            by: dict[str, tuple[list, list]] = {}
            for j, i in enumerate(idx):
                kind = spec_kind(base[i].spec)
                by.setdefault(kind, ([], []))
                by[kind][0].append(own[i])
                by[kind][1].append(cf_vals[j])
            for kind, (a, b) in sorted(by.items()):
                res["counterfactual"][kind] = own_higher(a, b)
    k = min(n_repeat, len(base))
    res["repeat_in_job"] = repeat_stats(own[:k], score_values(metric, base[:k], bs))
    score_dir = Path(data) / "scores" / metric_name
    stored = latest_by_id(r for r in read_many(score_dir.glob("*.jsonl"))
                          if r.get("metric_version") == mversion and r.get("status") == "ok") \
        if score_dir.exists() else {}
    pairs = [(v, stored[s.sample_id]["value"]) for s, v in zip(base, own) if s.sample_id in stored]
    res["repeat_vs_stored"] = repeat_stats([p[0] for p in pairs], [p[1] for p in pairs])
    res["per_image"] = per_image
    old = OUT_DIR / f"{metric_name}.json"
    if old.exists():                              # keep a labelled-images check run earlier
        prev = json.loads(old.read_text()).get("counterfactual_labelled")
        if prev:
            res["counterfactual_labelled"] = prev
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(OUT_DIR / f"{metric_name}.json", json.dumps(res, indent=2, default=str))
    return res


def fmt_check(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "–"
    return f"{d['rate']:.2f} ({d['higher']}↑ {d['ties']}= {d['lower']}↓ of {d['n']})"


def fmt_repeat(d: dict | None) -> str:
    if not d or not d.get("n"):
        return "–"
    return f"{d['identical']:.0%} identical, max |Δ| {d['max_abs_diff']:.2g} (n={d['n']})"


def run_labelled(metric_name: str, prompts_path: str, manifests: str, labels_path: str | None = None,
                 split: str = "dev", device: str = "cuda") -> dict[str, Any]:
    """Counterfactual check on images people labelled correct; merged into <metric>.json."""
    from .summarize import load_labels

    metric = REGISTRY[metric_name]()
    bs = metric.default_batch_size
    labels_path = labels_path or str(Path("data/labels") / f"{Path(prompts_path).stem}_labels.csv")
    labels = load_labels(labels_path)
    split_ids = {p.prompt_id for p in load_prompts(prompts_path, split=split)}
    good = [s for s in load_samples(manifests, prompts_path)
            if s.prompt_id in split_ids and labels.get(s.sample_id, {}).get("correct") == "y"
            and metric.applies_to(s)]
    pairs = [(s, counterfactual(s.prompt, s.spec)) for s in good]
    pairs = [(s, c) for s, c in pairs if c is not None]
    metric.setup(device)
    own = score_values(metric, [s for s, _ in pairs], bs)
    cfv = score_values(metric, [replace(s, prompt=c[0], spec=c[1]) for s, c in pairs], bs)
    out: dict[str, Any] = {"labels": labels_path, "all": own_higher(own, cfv)}
    by: dict[str, tuple[list, list]] = {}
    for (s, _), a, b in zip(pairs, own, cfv):
        by.setdefault(spec_kind(s.spec), ([], []))
        by[spec_kind(s.spec)][0].append(a)
        by[spec_kind(s.spec)][1].append(b)
    for kind, (a, b) in sorted(by.items()):
        out[kind] = own_higher(a, b)
    out["per_image"] = [{"sample_id": s.sample_id, "own": a, "cf_prompt": c[0], "cf": b}
                        for (s, c), a, b in zip(pairs, own, cfv)]
    path = OUT_DIR / f"{metric_name}.json"
    res = json.loads(path.read_text()) if path.exists() else {"metric": metric_name, "n_images": 0,
                                                               "metric_version": f"{metric.version}/bs{bs}",
                                                               "device": device}
    res["counterfactual_labelled"] = out
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(res, indent=2, default=str))
    return res


def report(out_dir: Path | None = None) -> str:
    out_dir = out_dir or OUT_DIR
    rows = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]
    L = ["# Metric sanity checks (guide §7.9)", "",
         f"Written {now_iso()} at commit `{git_commit()}` by `python -m genprobe.sanity --report`. Dev-split "
         "images only. Rates: share of images where the own prompt scores higher than the changed one "
         "(ties count 1/2); ↑ = / ↓ give the raw counts. Binary checkers tie often (both 0): read the ↓ "
         "column, which should be near 0.", "",
         "## Mismatch and repeatability (200 random dev images)", "",
         "| Metric | images | mismatched prompt | repeat in job | repeat vs scoring run |",
         "| --- | --- | --- | --- | --- |"]
    for r in rows:
        L.append(f"| {r['metric']} | {r['n_images']} | {fmt_check(r.get('mismatch'))} | "
                 f"{fmt_repeat(r.get('repeat_in_job'))} | {fmt_repeat(r.get('repeat_vs_stored'))} |")
    L += ["", "## One-word counterfactuals", "",
          "**On images labelled correct** (the clean test: the original prompt is right, the changed one wrong):", "",
          "| Metric | all | count ±2 | left/right, above/below | other text |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        c = r.get("counterfactual_labelled")
        if c:
            L.append(f"| {r['metric']} | {fmt_check(c.get('all'))} | {fmt_check(c.get('count'))} | "
                     f"{fmt_check(c.get('relation'))} | {fmt_check(c.get('text'))} |")
    L += ["", "On the random template images of the sample (confounded: where the generator got the count or "
          "position wrong, the changed prompt can fit the image better, so even a perfect metric scores below 1):", "",
          "| Metric | all | count ±2 | left/right, above/below | other text |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        c = r.get("counterfactual")
        if c:
            L.append(f"| {r['metric']} | {fmt_check(c.get('all'))} | {fmt_check(c.get('count'))} | "
                     f"{fmt_check(c.get('relation'))} | {fmt_check(c.get('text'))} |")
    L += ["", "Versions: " + "; ".join(f"{r['metric']} `{r['metric_version']}` on {r['device']}" for r in rows)]
    text = "\n".join(L) + "\n"
    atomic_write_text(out_dir / "summary.md", text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=sorted(REGISTRY))
    ap.add_argument("--prompts", default="data/prompts/prompts_v1.jsonl")
    ap.add_argument("--samples", default="data/manifests/gen_*.jsonl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--part", choices=["random", "labelled"], default="random",
                    help="random: mismatch, counterfactual and repeat checks on --n random dev images; "
                         "labelled: counterfactuals on the images people labelled correct")
    ap.add_argument("--labels", default=None, help="default data/labels/<prompts stem>_labels.csv")
    ap.add_argument("--report", action="store_true", help="only rebuild results/sanity/summary.md")
    args = ap.parse_args(argv)
    if not args.report:
        if not args.metric:
            ap.error("--metric is required unless --report")
        if args.part == "labelled":
            r = run_labelled(args.metric, args.prompts, args.samples, args.labels, device=args.device)
        else:
            r = run(args.metric, args.prompts, args.samples, args.data, n=args.n, device=args.device)
        print(json.dumps({k: v for k, v in r.items() if k not in ("metric_info", "per_image")}, indent=2, default=str))
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
