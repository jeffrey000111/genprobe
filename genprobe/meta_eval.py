"""Metric meta-evaluation on GenAI-Bench (module B, guide §9.2): which automatic metrics order
images the way people do, by skill, and the pre-registered tests H1 and H2.

Written and tested before any metric was scored on GenAI-Bench. The method is the one fixed in
docs/decisions.md ("Pre-registered hypotheses"):

- Pairs: within each prompt, every two models' images whose mean human ratings differ (pairs with
  equal means are excluded; their share is reported). Ratings outside 1-5 are dropped, and those
  images use the mean of their valid ratings.
- Agreement of a metric: share of pairs it orders like the mean rating; equal metric scores = 1/2.
  Comparisons between metrics use the pairs where both scored both images.
- CIs: 2,000 bootstrap resamples of prompts (seed 0); 95 %, or 97.5 % for the two H2 tests.
- Ceiling: on the same pairs, each rating position against the mean of the other two
  (leave-one-rater-out; pairs where those two tie are skipped; equal ratings = 1/2), averaged over
  the three positions; images without three valid ratings are left out of it. Plus
  Krippendorff's alpha (ordinal) over all valid ratings.
- Kendall's tau-b between metric and mean rating within each prompt, averaged (exploratory).

    python -m genprobe.meta_eval --metrics clipscore vqascore pickscore hpsv2

Writes results/meta_eval/: hypotheses.md, agreement_by_tag.csv, summary.json and
fig_meta_eval_heatmap.png. Light (numpy + matplotlib); on HPC3 run it through Slurm
(scripts/submit_meta_eval.sh).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .genai_bench import FOLDER, load_ratings
from .samples import load_prompts
from .summarize import N_BOOT, pick_scores
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

OUT = Path("results/meta_eval")
MIN_PROMPTS_FIG = 30          # tags with fewer prompts stay in the table but not in the figure (as in D-020)
HUMAN = "human (leave-one-out)"
H1 = {"name": "H1", "tag": "counting", "a": "vqascore", "b": "clipscore", "level": 0.95}
H2 = [{"name": "H2a", "tag": "all", "a": "vqascore", "b": "pickscore", "level": 0.975},
      {"name": "H2b", "tag": "all", "a": "vqascore", "b": "hpsv2", "level": 0.975}]

Pairs = dict[str, list[tuple[str, str, int]]]


class IncompleteScores(RuntimeError):
    pass


# ---------- pairs and agreement ----------

def make_pairs(means: dict[str, dict[str, float]]) -> tuple[Pairs, int, int]:
    """means: prompt_id -> {sample_id: mean human rating}. Returns prompt_id -> [(a, b, sign)],
    sign +1 when a is rated higher and -1 when lower, plus the numbers of tied and of all pairs."""
    pairs: Pairs = {}
    ties = total = 0
    for pid, imgs in means.items():
        ids = sorted(imgs)
        out = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                total += 1
                d = imgs[a] - imgs[b]
                if d == 0:
                    ties += 1
                    continue
                out.append((a, b, 1 if d > 0 else -1))
        if out:
            pairs[pid] = out
    return pairs, ties, total


def agree(x: float, y: float, sign: int) -> float:
    """1 if x vs y is ordered like sign, 0 if the other way, 1/2 if x == y."""
    if x == y:
        return 0.5
    return 1.0 if (x > y) == (sign > 0) else 0.0


def prompt_sums(pairs: Pairs, pids: list[str], score_sets: list[dict[str, float]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per prompt, over the pairs where every score set has both images: agreement sums (one row
    per score set) and pair counts. Prompts with no such pair are left out."""
    kept, sums, counts = [], [], []
    for pid in pids:
        ps = [(a, b, s) for a, b, s in pairs.get(pid, []) if all(a in sc and b in sc for sc in score_sets)]
        if not ps:
            continue
        kept.append(pid)
        sums.append([sum(agree(sc[a], sc[b], s) for a, b, s in ps) for sc in score_sets])
        counts.append(len(ps))
    return kept, np.array(sums, float).reshape(len(kept), len(score_sets)).T, np.array(counts, float)


def _boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci(b: np.ndarray, level: float) -> list[float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(b, [a, 1 - a])
    return [float(lo), float(hi)]


def agreement(pairs: Pairs, pids: list[str], scores: dict[str, float], n_boot: int = N_BOOT, seed: int = 0,
              level: float = 0.95) -> dict[str, Any]:
    kept, s, c = prompt_sums(pairs, pids, [scores])
    if not kept:
        return {"agreement": math.nan, "ci": [math.nan, math.nan], "pairs": 0, "n_prompts": 0}
    idx = _boot_idx(len(kept), n_boot, seed)
    b = s[0][idx].sum(1) / c[idx].sum(1)
    return {"agreement": float(s[0].sum() / c.sum()), "ci": _ci(b, level), "pairs": int(c.sum()),
            "n_prompts": len(kept)}


def difference(pairs: Pairs, pids: list[str], sa: dict[str, float], sb: dict[str, float], n_boot: int = N_BOOT,
               seed: int = 0, level: float = 0.95) -> dict[str, Any]:
    """agreement(a) - agreement(b) over the same pairs; CI from resampling prompts."""
    kept, s, c = prompt_sums(pairs, pids, [sa, sb])
    if not kept:
        return {"difference": math.nan, "ci": [math.nan, math.nan], "level": level, "pairs": 0, "n_prompts": 0}
    idx = _boot_idx(len(kept), n_boot, seed)
    d = (s[0][idx].sum(1) - s[1][idx].sum(1)) / c[idx].sum(1)
    return {"difference": float((s[0].sum() - s[1].sum()) / c.sum()), "ci": _ci(d, level), "level": level,
            "agreement_a": float(s[0].sum() / c.sum()), "agreement_b": float(s[1].sum() / c.sum()),
            "pairs": int(c.sum()), "n_prompts": len(kept)}


def verdict(ci: list[float]) -> str:
    """Prediction: difference > 0."""
    lo, hi = ci
    if math.isnan(lo):
        return "no data"
    if lo > 0:
        return "supported"
    if hi < 0:
        return "contradicted"
    return "inconclusive"


# ---------- human ceiling ----------

def ceiling(pairs: Pairs, pids: list[str], raw: dict[str, dict[str, list[float]]], n_boot: int = N_BOOT,
            seed: int = 0) -> dict[str, Any]:
    """Rating position r (0, 1, 2) against the mean of the other two, on the metric pairs where
    both images have three ratings and the other two's means differ; agreement per position,
    then the mean of the three. The CI resamples prompts (same draws for all positions)."""
    kept, sums, counts = [], [], []
    for pid in pids:
        imgs = raw.get(pid, {})
        s, c = [0.0, 0.0, 0.0], [0, 0, 0]
        for a, b, _ in pairs.get(pid, []):
            ra, rb = imgs.get(a, []), imgs.get(b, [])
            if len(ra) != 3 or len(rb) != 3:
                continue
            for r in range(3):
                oa, ob = sum(ra) - ra[r], sum(rb) - rb[r]
                if oa == ob:
                    continue
                s[r] += agree(ra[r], rb[r], 1 if oa > ob else -1)
                c[r] += 1
        if any(c):
            kept.append(pid)
            sums.append(s)
            counts.append(c)
    if not kept:
        return {"agreement": math.nan, "ci": [math.nan, math.nan], "pairs": 0, "n_prompts": 0, "by_position": []}
    s, c = np.array(sums), np.array(counts, float)
    idx = _boot_idx(len(kept), n_boot, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.nanmean(s[idx].sum(1) / c[idx].sum(1), axis=1)
        pos = s.sum(0) / c.sum(0)                    # a position with no usable pair is nan, left out
    return {"agreement": float(np.nanmean(pos)), "ci": _ci(b, 0.95), "pairs": int(c.sum(0).max()),
            "n_prompts": len(kept), "by_position": [float(x) for x in pos]}


def krippendorff_alpha_ordinal(units: list[list[float]]) -> float:
    """Krippendorff's alpha with the ordinal distance; units (images) with fewer than 2 ratings are
    not pairable and are ignored."""
    units = [u for u in units if len(u) >= 2]
    values = sorted({v for u in units for v in u})
    if len(values) < 2:
        return math.nan
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))                     # coincidence matrix
    for u in units:
        m = len(u)
        for i, a in enumerate(u):
            for j, b in enumerate(u):
                if i != j:
                    o[idx[a], idx[b]] += 1 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    delta = np.zeros((k, k))
    for c in range(k):
        for e in range(c, k):
            d = n_c[c:e + 1].sum() - (n_c[c] + n_c[e]) / 2
            delta[c, e] = delta[e, c] = d * d
    d_o = (o * delta).sum()
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float(1 - d_o / d_e) if d_e > 0 else math.nan


def kendall_tau_b(x: list[float], y: list[float]) -> float:
    conc = disc = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif (dx > 0) == (dy > 0):
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    return (conc - disc) / denom if denom > 0 else math.nan


# ---------- run ----------

def load_metric_scores(data: str, metric: str, ids: set[str]) -> tuple[dict[str, float], str | None, int]:
    recs = read_many(sorted((Path(data) / "scores" / metric).glob("*.jsonl")))
    ok, version, info = pick_scores(recs, ids)
    return {sid: float(r["value"]) for sid, r in ok.items()}, version, int(info.get("errors", 0))


def run(metrics: list[str], prompts_path: str = "data/prompts/genai_bench_1600.jsonl",
        manifest: str = "data/manifests/ext_genai_bench.jsonl", data: str = "data", folder: str = str(FOLDER),
        out: Path | None = None, n_boot: int = N_BOOT, allow_incomplete: bool = False) -> dict[str, Any]:
    """Refuses to compute anything unless every metric has scored every image (so results are
    looked at once, on the full set), unless allow_incomplete."""
    out = Path(out or OUT)
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    gen = {sid: r for sid, r in latest_by_id(read_many(sorted(glob.glob(manifest)))).items()
           if r.get("status") == "ok" and r.get("prompt_id") in prompts}
    ratings, dropped = load_ratings(folder)
    raw: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for sid, r in gen.items():
        v = ratings.get((r["prompt_id"], r["model"]), [])
        if v:
            raw[r["prompt_id"]][sid] = v
    means = {pid: {sid: float(np.mean(v)) for sid, v in imgs.items()} for pid, imgs in raw.items()}
    pairs, ties, total = make_pairs(means)

    groups: dict[str, list[str]] = defaultdict(list)
    for p in prompts.values():
        groups["all"].append(p.prompt_id)
        for t in p.tags:
            groups[t].append(p.prompt_id)
    fixed = [g for g in ("all", "basic", "advanced") if g in groups]
    order = fixed + sorted(g for g in groups if g not in fixed)

    scores, versions, coverage = {}, {}, {}
    for m in metrics:
        scores[m], versions[m], errors = load_metric_scores(data, m, set(gen))
        coverage[m] = {"scored": len(scores[m]), "images": len(gen), "errors": errors}
    short = {m: c for m, c in coverage.items() if c["scored"] < c["images"]}
    if short and not allow_incomplete:
        raise IncompleteScores("not every image is scored yet, nothing computed: " + "; ".join(
            f"{m} {c['scored']}/{c['images']} ({c['errors']} errors)" for m, c in short.items()))

    rows = []
    for g in order:
        pids = sorted(groups[g])
        c = ceiling(pairs, pids, raw, n_boot)
        rows.append({"metric": HUMAN, "tag": g, **_flat(c)})
        for m in metrics:
            sc = scores[m]
            taus = []
            for pid in pids:
                ids = [s for s in sorted(means.get(pid, {})) if s in sc]
                t = kendall_tau_b([sc[s] for s in ids], [means[pid][s] for s in ids]) if len(ids) > 1 else math.nan
                if not math.isnan(t):
                    taus.append(t)
            rows.append({"metric": m, "tag": g, **_flat(agreement(pairs, pids, sc, n_boot)),
                         "kendall_tau_mean": float(np.mean(taus)) if taus else math.nan})

    hyp = {}
    for h in [H1] + H2:
        if h["a"] in scores and h["b"] in scores:
            d = difference(pairs, sorted(groups.get(h["tag"], [])), scores[h["a"]], scores[h["b"]], n_boot,
                           level=h["level"])
            where = "all prompts" if h["tag"] == "all" else f"prompts tagged '{h['tag']}'"
            hyp[h["name"]] = {**d, "comparison": f"{h['a']} - {h['b']}, {where}", "verdict": verdict(d["ci"])}

    alpha = krippendorff_alpha_ordinal([v for imgs in raw.values() for v in imgs.values()])
    res = {"created_at": now_iso(), "git_commit": git_commit(), "prompts": len(prompts), "images": len(gen),
           "pairs": sum(len(v) for v in pairs.values()), "pairs_tied_excluded": ties, "pairs_total": total,
           "dropped_ratings": dropped, "krippendorff_alpha_ordinal": alpha, "metric_versions": versions,
           "coverage": coverage, "hypotheses": hyp, "n_boot": n_boot}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "agreement_by_tag.csv", rows)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2, default=str))
    atomic_write_text(out / "hypotheses.md", render(res, rows, order, metrics))
    if metrics:
        heatmap(rows, order, metrics, out / "fig_meta_eval_heatmap.png")
    return {**res, "rows": rows}


def _flat(d: dict) -> dict:
    return {"agreement": d["agreement"], "ci95_lo": d["ci"][0], "ci95_hi": d["ci"][1], "pairs": d["pairs"],
            "n_prompts": d["n_prompts"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["metric", "tag", "agreement", "ci95_lo", "ci95_hi", "pairs", "n_prompts", "kendall_tau_mean"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else
                            ("" if isinstance(v, float) else v)) for k, v in r.items() if k in fields})


def _f(x, nd: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict, rows: list[dict], order: list[str], metrics: list[str]) -> str:
    L = ["# Metric meta-evaluation on GenAI-Bench", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.meta_eval`. "
         f"{res['prompts']} prompts, {res['images']} images. {res['pairs']} within-prompt image pairs whose mean "
         f"human ratings differ; {res['pairs_tied_excluded']} of {res['pairs_total']} pairs "
         f"({100 * res['pairs_tied_excluded'] / max(res['pairs_total'], 1):.1f} %) have equal means and are "
         f"excluded. {len(res['dropped_ratings'])} out-of-range ratings dropped. Krippendorff's α (ordinal) over "
         f"the human ratings: {_f(res['krippendorff_alpha_ordinal'])}.", "",
         "Coverage: " + "; ".join(f"{m} {c['scored']}/{c['images']} images scored"
                                  + (f", {c['errors']} errors" if c["errors"] else "")
                                  for m, c in res["coverage"].items()) + ".", "",
         "## Pre-registered hypotheses (docs/decisions.md)", "",
         "A hypothesis is supported when its CI lies above 0, contradicted when below, inconclusive otherwise.", "",
         "| Test | Comparison | Agreement a / b | Difference | CI | Pairs | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, h in res["hypotheses"].items():
        L.append(f"| {k} | {h['comparison']} | {_f(h.get('agreement_a'))} / {_f(h.get('agreement_b'))} | "
                 f"{_f(h['difference'])} | {h['level'] * 100:g} %: [{_f(h['ci'][0])}, {_f(h['ci'][1])}] | "
                 f"{h['pairs']} | **{h['verdict']}** |")
    by = {(r["metric"], r["tag"]): r for r in rows}
    L += ["", "## Agreement with people by tag", "",
          "Exploratory, apart from the cells the tests above use. Share of within-prompt image pairs ordered like "
          "the mean human rating, 95 % CI from resampling prompts. Human ceiling: each rating against the mean of "
          "the other two. GenAI-Bench prompts carry several tags, so rows overlap.", "",
          "| Tag | Prompts | Human ceiling | " + " | ".join(metrics) + " |",
          "| --- | --- | --- |" + " --- |" * len(metrics)]
    for g in order:
        h = by[(HUMAN, g)]
        n = max([by[(m, g)]["n_prompts"] for m in metrics] + [h["n_prompts"]])
        cells = [f"{_f(by[(m, g)]['agreement'], 2)} [{_f(by[(m, g)]['ci95_lo'], 2)}, {_f(by[(m, g)]['ci95_hi'], 2)}]"
                 for m in metrics]
        L.append(f"| {g} | {n} | {_f(h['agreement'], 2)} | " + " | ".join(cells) + " |")
    if metrics:
        L += ["", "Kendall's τ-b between metric and mean rating within each prompt, averaged over prompts "
              "(exploratory): " + "; ".join(f"{m} {_f(by[(m, 'all')].get('kendall_tau_mean'))}" for m in metrics) + "."]
    L += ["", "Metric versions: " + "; ".join(f"{m} `{v}`" for m, v in res["metric_versions"].items()) + "."]
    return "\n".join(L) + "\n"


def heatmap(rows: list[dict], order: list[str], metrics: list[str], out: Path, what: str = "image") -> None:
    from matplotlib.colors import LinearSegmentedColormap

    from .figures import INK, INK2, SEQ_RAMP, SURFACE, _fig, _footnote

    names = [HUMAN] + metrics
    by = {(r["metric"], r["tag"]): r["agreement"] for r in rows}
    n_of = {r["tag"]: r["n_prompts"] for r in rows if r["metric"] == HUMAN}
    small = [g for g in order if n_of.get(g, 0) < MIN_PROMPTS_FIG]
    order = [g for g in order if g not in small]
    if not order:
        return
    grid = np.array([[by.get((m, g), math.nan) for g in order] for m in names], float)
    if np.all(np.isnan(grid)):
        return
    plt, fig, ax = _fig(max(7.0, 0.55 * len(order) + 2.8), 0.45 * len(names) + 2.6)
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    vmin = math.floor(np.nanmin(grid) * 20) / 20
    vmax = max(math.ceil(np.nanmax(grid) * 20) / 20, vmin + 0.05)
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(names)):
        for j in range(len(order)):
            if not np.isnan(grid[i, j]):
                frac = (grid[i, j] - vmin) / (vmax - vmin)
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if frac > 0.6 else INK)
    ax.axhline(0.5, color=SURFACE, linewidth=3)       # sets the human row apart
    n_fixed = sum(g in ("all", "basic", "advanced") for g in order)
    if 0 < n_fixed < len(order):
        ax.axvline(n_fixed - 0.5, color=SURFACE, linewidth=3)   # overall columns | skill columns
    ax.set_xticks(range(len(order)), [g.replace("_", " ") for g in order], rotation=40, ha="right", fontsize=8,
                  color=INK2)
    ax.set_yticks(range(len(names)), ["human ceiling" if n == HUMAN else n for n in names], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2, length=0)
    j = order.index("all") if "all" in order else 0
    overall = {m: by.get((m, order[j]), math.nan) for m in metrics}
    best = max(overall, key=lambda m: -1 if math.isnan(overall[m]) else overall[m])
    bench = "GenAI-Bench" if what == "image" else "GenAI-Bench-Video"
    ax.set_title(f"Agreement with people on {bench} pairs: {best} highest overall ({overall[best]:.2f}), "
                 f"human ceiling {grid[0, j]:.2f}", fontsize=10, color=INK, loc="left", pad=10)
    _footnote(fig, f"Share of within-prompt {what} pairs a metric orders like the mean of three human ratings "
                   "(pairs with equal means excluded; equal metric scores count 1/2). Human ceiling: each rating "
                   "against the mean of the other two. Prompts carry several tags, so columns overlap. "
                   "CIs in hypotheses.md."
                   + (f" Not drawn (fewer than {MIN_PROMPTS_FIG} prompts): {', '.join(small)}." if small else ""))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", nargs="+", default=["clipscore", "vqascore", "pickscore", "hpsv2"])
    ap.add_argument("--prompts", default="data/prompts/genai_bench_1600.jsonl")
    ap.add_argument("--manifest", default="data/manifests/ext_genai_bench.jsonl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-incomplete", action="store_true", help="compute even if some images are unscored")
    args = ap.parse_args(argv)
    try:
        res = run(args.metrics, args.prompts, args.manifest, args.data, out=Path(args.out),
                  allow_incomplete=args.allow_incomplete)
    except IncompleteScores as e:
        print(e, file=sys.stderr)
        return 1
    for m, c in res["coverage"].items():
        print(f"{m}: {c['scored']}/{c['images']} scored, {c['errors']} errors")
    for k, h in res["hypotheses"].items():
        print(f"{k}: {h['comparison']}: {h['difference']:+.3f} {[round(x, 3) for x in h['ci']]} -> {h['verdict']}")
    print(f"wrote {args.out}/hypotheses.md, agreement_by_tag.csv, summary.json, fig_meta_eval_heatmap.png")
    complete = all(c["scored"] == c["images"] for c in res["coverage"].values())
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
