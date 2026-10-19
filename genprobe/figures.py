"""Leaderboard, paired comparisons and the M2 figures for one summarized prompt set (guide §8).

Reads ``<results>/per_sample.csv`` (written by genprobe.summarize) and the prompts file, and
writes next to them:

    leaderboard.csv            model x metric: mean over prompts, 95% CI (prompt bootstrap),
                               category-weighted mean (each category counts once), coverage
    pairwise.csv               model A - model B per metric: paired mean difference over the
                               same prompts, 95% CI, share of prompts where A > B
    fig_category_heatmap.png   main metric by model x prompt category / GenAI-Bench skill
    fig_frontier.png           main metric vs median generation seconds per image, Pareto front
    fig_text_length.png        OCR exact-match rate by number of requested words (needs ocr)
    figures.md                 what each figure shows, with its numbers

    python -m genprobe.figures --results results/prompts_v1_test --prompts data/prompts/prompts_v1.jsonl

Titles state what the data shows and are computed from it, never written by hand.
Light: numpy + matplotlib (in the eval env); on HPC3 run it through Slurm like the summary.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .samples import Prompt, load_prompts
from .summarize import N_BOOT, bootstrap_mean_ci
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text

# Fixed model order and colours: the same model has the same colour in every figure
# (categorical slots 1-4 of the dataviz reference palette, validated for adjacent pairs; the
# scatter adds marker shapes and direct labels because 4 series exceed its all-pairs cap).
MODEL_ORDER = ["sd15", "sdxl", "sd35m", "flux_schnell"]
MODEL_LABEL = {"sd15": "SD 1.5", "sdxl": "SDXL", "sd35m": "SD3.5 Medium", "flux_schnell": "FLUX.1 schnell"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SMALL_N = 30          # cells with fewer prompts are flagged (guide §8.2)
TEXT_BINS = [(1, 2, "1–2 words"), (3, 4, "3–4 words"), (5, 99, "5–8 words")]


# ---------- data ----------

def read_per_sample(path: str | Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def metric_columns(rows: list[dict[str, str]]) -> list[str]:
    """Columns of per_sample.csv that hold a registered metric's value (not details or labels)."""
    from .metrics import REGISTRY
    return [c for c in (rows[0].keys() if rows else []) if c in REGISTRY]


def ordered_models(models) -> list[str]:
    models = set(models)
    return [m for m in MODEL_ORDER if m in models] + sorted(models - set(MODEL_ORDER))


def per_prompt_values(rows, metric: str) -> dict[str, dict[str, list[float]]]:
    """model -> prompt_id -> values over seeds (ok images with a score only)."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["status"] == "ok" and r.get(metric, "") != "":
            out[r["model"]][r["prompt_id"]].append(float(r[metric]))
    return out


def prompt_groups(p: Prompt) -> list[str]:
    """Heatmap columns a prompt belongs to: its template category, or each GenAI-Bench skill."""
    if p.source == "genai_bench":
        return [f"GenAI · {t.replace('_', ' ')}" for t in p.tags[1:]]
    return [f"template · {p.tags[0]}"] if p.tags else []


def text_bin(p: Prompt) -> str | None:
    text = (p.spec or {}).get("text")
    if not isinstance(text, str):
        return None
    n = len(text.split())
    return next(label for lo, hi, label in TEXT_BINS if lo <= n <= hi)


def paired(a: dict[str, list[float]], b: dict[str, list[float]], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Mean of per-prompt differences (A - B) over prompts both models have, bootstrap CI,
    and the share of prompts where A scores higher / ties."""
    common = sorted(set(a) & set(b))
    if not common:
        return {"n_prompts": 0, "mean_diff": math.nan, "ci95": [math.nan, math.nan], "win_rate": math.nan,
                "tie_rate": math.nan}
    d = np.array([np.mean(a[p]) - np.mean(b[p]) for p in common])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
    return {"n_prompts": len(common), "mean_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "win_rate": float((d > 0).mean()), "tie_rate": float((d == 0).mean())}


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Models no other model beats on both cost (lower) and score (higher), sorted by cost."""
    front = [m for m, (c, s) in points.items()
             if not any((c2 <= c and s2 >= s) and (c2 < c or s2 > s) for m2, (c2, s2) in points.items() if m2 != m)]
    return sorted(front, key=lambda m: points[m][0])


def build_tables(rows, prompts: dict[str, Prompt], metrics: list[str]) -> dict[str, Any]:
    models = ordered_models(r["model"] for r in rows)
    ok_by_model = defaultdict(int)
    for r in rows:
        if r["status"] == "ok":
            ok_by_model[r["model"]] += 1
    leaderboard, pairs = [], []
    per_metric = {}
    for m in metrics:
        vals = per_prompt_values(rows, m)
        per_metric[m] = vals
        for model in models:
            pp = vals.get(model, {})
            mean, lo, hi = bootstrap_mean_ci(pp)
            by_cat = defaultdict(list)
            for pid, v in pp.items():
                by_cat[prompts[pid].tags[0] if prompts[pid].tags else "untagged"].append(np.mean(v))
            n_img = sum(len(v) for v in pp.values())
            leaderboard.append({
                "model": model, "metric": m, "mean": mean, "ci95_lo": lo, "ci95_hi": hi,
                "mean_category_weighted": float(np.mean([np.mean(v) for v in by_cat.values()])) if by_cat else math.nan,
                "n_prompts": len(pp), "n_images": n_img, "n_ok_images": ok_by_model[model]})
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                r = paired(vals.get(a, {}), vals.get(b, {}))
                pairs.append({"metric": m, "model_a": a, "model_b": b, "n_prompts": r["n_prompts"],
                              "mean_diff": r["mean_diff"], "ci95_lo": r["ci95"][0], "ci95_hi": r["ci95"][1],
                              "a_wins": r["win_rate"], "ties": r["tie_rate"],
                              "ci_excludes_0": bool(r["ci95"][0] > 0 or r["ci95"][1] < 0)})
    return {"models": models, "leaderboard": leaderboard, "pairwise": pairs, "per_metric": per_metric}


def gen_cost(rows) -> dict[str, float]:
    """Median seconds per image from the manifests' timings, warmup batches excluded (D-004)."""
    t = defaultdict(list)
    for r in rows:
        if r["status"] == "ok" and r.get("warmup") != "True" and r.get("gen_seconds") not in ("", None):
            t[r["model"]].append(float(r["gen_seconds"]))
    return {m: float(np.median(v)) for m, v in t.items() if v}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in r.items()})
    tmp.replace(path)


# ---------- figures ----------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fig(w: float, h: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return plt, fig, ax


def _footnote(fig, text: str) -> None:
    lines = textwrap.wrap(text, width=int(fig.get_figwidth() * 22))
    fig.text(0.01, 0.01, "\n".join(lines), fontsize=6.5, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.025 + 0.022 * len(lines), 1, 1))


def colour(models: list[str], m: str) -> str:
    return SERIES[models.index(m) % len(SERIES)]


def heatmap(models, vals, prompts, metric, out: Path) -> str | None:
    from matplotlib.colors import LinearSegmentedColormap

    groups: dict[str, set[str]] = defaultdict(set)
    for pid, p in prompts.items():
        for g in prompt_groups(p):
            groups[g].add(pid)
    cols = sorted((g for g in groups if g.startswith("template")), key=str) + \
        sorted(g for g in groups if not g.startswith("template"))
    if not cols or not models:
        return None
    grid = np.full((len(models), len(cols)), np.nan)
    for i, model in enumerate(models):
        pp = vals.get(model, {})
        for j, g in enumerate(cols):
            v = [np.mean(pp[p]) for p in groups[g] if p in pp]
            if v:
                grid[i, j] = np.mean(v)
    if np.all(np.isnan(grid)):
        return None
    lo_i, lo_j = np.unravel_index(np.nanargmin(grid), grid.shape)
    title = (f"{metric} by prompt category: lowest is {MODEL_LABEL.get(models[lo_i], models[lo_i])} on "
             f"{cols[lo_j].split(' · ')[1]} ({grid[lo_i, lo_j]:.2f})")
    plt, fig, ax = _fig(max(7.0, 0.62 * len(cols) + 2.2), 0.5 * len(models) + 2.6)
    ax.grid(False)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_RAMP)
    vmin, vmax = math.floor(np.nanmin(grid) * 20) / 20, math.ceil(np.nanmax(grid) * 20) / 20
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=max(vmax, vmin + 0.05), aspect="auto")
    for i in range(len(models)):
        for j in range(len(cols)):
            if not np.isnan(grid[i, j]):
                frac = (grid[i, j] - vmin) / max(vmax - vmin, 1e-9)
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if frac > 0.55 else INK)
    labels = [f"{g.split(' · ')[1]}\n{g.split(' · ')[0]} · n={len(groups[g])}{' †' if len(groups[g]) < SMALL_N else ''}"
              for g in cols]
    rot = 0 if len(cols) <= 5 else 35
    ax.set_xticks(range(len(cols)), labels, rotation=rot, ha="center" if rot == 0 else "right", fontsize=7.5,
                  color=INK2)
    ax.set_yticks(range(len(models)), [MODEL_LABEL.get(m, m) for m in models], fontsize=9, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2, length=0)
    cb.set_label(f"mean {metric} (over prompts)", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=10)
    note = ("Cell = mean over the prompts in that column (each prompt averaged over its seeds). "
            "GenAI-Bench prompts carry several skills, so columns overlap.")
    if any(len(groups[g]) < SMALL_N for g in cols):
        note += f" † fewer than {SMALL_N} prompts: read with care."
    _footnote(fig, note)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return title


def frontier(models, board, cost, metric, gpu: str | None, out: Path) -> str | None:
    pts = {r["model"]: (cost[r["model"]], r["mean"], r["ci95_lo"], r["ci95_hi"]) for r in board
           if r["metric"] == metric and r["model"] in cost and not math.isnan(r["mean"])}
    if len(pts) < 2:
        return None
    front = pareto_front({m: (c, s) for m, (c, s, _, _) in pts.items()})
    best = max(pts, key=lambda m: pts[m][1])
    cheapest = min(pts, key=lambda m: pts[m][0])
    lab = lambda m: MODEL_LABEL.get(m, m)  # noqa: E731
    title = (f"Highest {metric}: {lab(best)} ({pts[best][1]:.2f}, {pts[best][0]:.2f} s/image); "
             f"cheapest: {lab(cheapest)} ({pts[cheapest][1]:.2f}, {pts[cheapest][0]:.2f} s/image)")
    plt, fig, ax = _fig(7.0, 4.6)
    fx = [pts[m][0] for m in front]
    fy = [pts[m][1] for m in front]
    ax.plot(fx, fy, color=INK2, linewidth=1, linestyle="-", zorder=1, label="Pareto front")
    for m in ordered_models(pts):
        c, s, lo, hi = pts[m]
        ax.errorbar(c, s, yerr=[[s - lo], [hi - s]], fmt="none", ecolor=colour(models, m), elinewidth=1.5,
                    capsize=0, zorder=2)
        ax.scatter(c, s, s=64, marker=MARKERS[models.index(m) % len(MARKERS)], color=colour(models, m),
                   edgecolor=SURFACE, linewidth=2, zorder=3, label=lab(m))
        ax.annotate(lab(m), (c, s), xytext=(8, -12), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    xs = [p[0] for p in pts.values()]
    lo_x, hi_x = min(xs) / 1.6, max(xs) * 1.6
    ax.set_xlim(lo_x, hi_x)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    nice = [v * 10 ** e for e in range(-2, 4) for v in (1, 2, 5)]
    ax.xaxis.set_major_locator(FixedLocator([v for v in nice if lo_x <= v <= hi_x]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"median generation time per image, seconds (log scale){' · ' + gpu if gpu else ''}",
                  fontsize=8.5, color=INK2)
    ax.set_ylabel(f"mean {metric} (95% CI over prompts)", fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right")
    _footnote(fig, "Time: median over images, warmup batches excluded; each model at its model-card settings "
              "(D-006, D-007). Pareto front: no other model is both faster and higher-scoring.")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return title


def text_length(models, rows, prompts, out: Path, n_boot: int = N_BOOT) -> tuple[str | None, list[dict]]:
    vals = per_prompt_values(rows, "ocr")
    bins = [b[2] for b in TEXT_BINS]
    table = []
    for model in models:
        for b in bins:
            pp = {pid: [float(v == 1.0) for v in vs] for pid, vs in vals.get(model, {}).items()
                  if text_bin(prompts[pid]) == b}
            mean, lo, hi = bootstrap_mean_ci(pp, n_boot=n_boot)
            table.append({"model": model, "words": b, "exact_rate": mean, "ci95_lo": lo, "ci95_hi": hi,
                          "n_prompts": len(pp)})
    if all(math.isnan(r["exact_rate"]) for r in table):
        return None, table
    avg = {}
    for b in bins:
        v = [r["exact_rate"] for r in table if r["words"] == b and not math.isnan(r["exact_rate"])]
        if v:
            avg[b] = float(np.mean(v))
    b0, b1 = min(avg, key=bins.index), max(avg, key=bins.index)
    first, last = avg[b0], avg[b1]
    if first == last:
        title = f"Exact text: {first:.0%} for both {b0} and {b1}, averaged over models"
    else:
        verb = "falls" if last < first else "rises"
        title = f"Exact text {verb} from {first:.0%} ({b0}) to {last:.0%} ({b1}), averaged over models"
    plt, fig, ax = _fig(7.0, 4.4)
    x = np.arange(len(bins))
    ends = []
    for k, model in enumerate(models):
        r = [t for t in table if t["model"] == model]
        y = np.array([t["exact_rate"] for t in r])
        if np.all(np.isnan(y)):
            continue
        off = (k - (len(models) - 1) / 2) * 0.06       # small dodge so error bars don't overlap
        ax.errorbar(x + off, y, yerr=[y - [t["ci95_lo"] for t in r], [t["ci95_hi"] for t in r] - y],
                    color=colour(models, model), linewidth=2, elinewidth=1, capsize=0,
                    marker=MARKERS[k % len(MARKERS)], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    label=MODEL_LABEL.get(model, model))
        last = np.where(~np.isnan(y))[0][-1]
        ends.append([float(y[last]), float(x[last]), MODEL_LABEL.get(model, model)])
    # end labels, nudged apart so equal values don't print on top of each other
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for yv, xv, name in ends:
        ax.text(xv + 0.25, yv, name, va="center", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{b}\n(n={max(t['n_prompts'] for t in table if t['words'] == b)} prompts)" for b in bins],
                  fontsize=8.5, color=INK2)
    ax.set_xlim(-0.4, len(bins) - 1 + 1.1)
    ax.set_ylim(-0.03, 1.12)
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("images with the text exactly right (OCR checker)", fontsize=8.5, color=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=26)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, ncol=len(models), loc="lower left",
              bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6)
    _footnote(fig, "Exact = OCR finds the requested text with case, punctuation and spaces ignored (D-017). "
              "95% CI over prompts. The OCR checker's own error rate is in the summary.")
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return title, table


def run(results: str | Path, prompts_path: str, metric: str = "vqascore", split: str | None = None) -> dict:
    results = Path(results)
    rows = read_per_sample(results / "per_sample.csv")
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path, split=split)}
    rows = [r for r in rows if r["prompt_id"] in prompts]
    metrics = metric_columns(rows)
    t = build_tables(rows, prompts, metrics)
    models = t["models"]
    write_csv(results / "leaderboard.csv", t["leaderboard"])
    write_csv(results / "pairwise.csv", t["pairwise"])
    cost = gen_cost(rows)
    gpu = None
    sj = results / "summary.json"
    if sj.is_file():
        import json
        gc = json.loads(sj.read_text()).get("generation_cost", {})
        gpus = {c.get("gpu") for c in gc.values() if c.get("gpu")}
        gpu = gpus.pop() if len(gpus) == 1 else None
    titles = {}
    if metric in metrics:
        titles["fig_category_heatmap.png"] = heatmap(models, t["per_metric"][metric], prompts, metric,
                                                     results / "fig_category_heatmap.png")
        titles["fig_frontier.png"] = frontier(models, t["leaderboard"], cost, metric, gpu,
                                              results / "fig_frontier.png")
    text_table: list[dict] = []
    if "ocr" in metrics:
        titles["fig_text_length.png"], text_table = text_length(models, rows, prompts,
                                                                results / "fig_text_length.png")
        write_csv(results / "text_length.csv", text_table)
    L = [f"# Figures: `{results}`", "",
         f"Made {now_iso()} at commit `{git_commit()}` by `python -m genprobe.figures` from `per_sample.csv` "
         f"and `{prompts_path}`. Means are over prompts; CIs resample prompts ({N_BOOT}×).", ""]
    for f, title in titles.items():
        L.append(f"- `{f}`: {title}" if title else f"- `{f}`: not drawn (no data)")
    L += ["", "## Leaderboard", "", "| Model | metric | mean [95% CI] | category-weighted | prompts | images |",
          "| --- | --- | --- | --- | --- | --- |"]
    for r in t["leaderboard"]:
        L.append(f"| {r['model']} | {r['metric']} | {r['mean']:.3f} [{r['ci95_lo']:.3f}, {r['ci95_hi']:.3f}] | "
                 f"{r['mean_category_weighted']:.3f} | {r['n_prompts']} | {r['n_images']} |")
    L += ["", "## Paired differences (A − B over the same prompts)", "",
          "| metric | A | B | mean diff [95% CI] | A higher on | prompts |", "| --- | --- | --- | --- | --- | --- |"]
    for r in t["pairwise"]:
        mark = "" if r["ci_excludes_0"] else " (CI includes 0)"
        L.append(f"| {r['metric']} | {r['model_a']} | {r['model_b']} | {r['mean_diff']:+.3f} "
                 f"[{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]{mark} | {r['a_wins']:.0%} | {r['n_prompts']} |")
    atomic_write_text(results / "figures.md", "\n".join(L) + "\n")
    return {"titles": titles, **t, "text_length": text_table}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="folder with per_sample.csv (and summary.json)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--metric", default="vqascore", help="metric for the heatmap and frontier")
    ap.add_argument("--split", default=None, choices=["dev", "test"])
    args = ap.parse_args(argv)
    res = run(args.results, args.prompts, args.metric, args.split)
    for f, title in res["titles"].items():
        print(f"{f}: {title or 'not drawn (no data)'}")
    print(f"wrote {args.results}/leaderboard.csv, pairwise.csv, figures.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
