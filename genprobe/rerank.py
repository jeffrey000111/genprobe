"""Re-ranking curve (guide §11.1, D-034): generate 9 seeds per test prompt, pick the image with the
highest VQAScore among the first N (N = 1, 2, 4, 9), and let the independent judge (Gemma 4 31B,
pass = score 5) grade the picks; plot pass rate against GPU-seconds per prompt with module C's
groups (single, best-of-4, routed best-of-k, the agents) on the same axes.

Two curves: SDXL (the agent's base model) and the routed model (the category's dev-best model,
module C's routing table: SD3.5 Medium for counting and spatial, FLUX.1 schnell for text and
GenAI-Bench). Seeds 0-3 are prompt set v1's images; seeds 4-8 are generated into data/rerank/ so
nothing of M2 changes.

    bash scripts/run_rerank.sh smoke|full     (on HPC3) runs, in Slurm jobs:
    python -m genprobe.rerank prompts [--per-category 5]  -> data/rerank/prompts_<model>.jsonl
    genprobe.generate (seeds 4-8, --out data/rerank) and genprobe.score (VQAScore, --out data/rerank)
    python -m genprobe.rerank picks --out data/rerank/picks_full.jsonl
    python -m genprobe.judge grade --images data/rerank/picks_full.jsonl --tag grade_rerank_full_gemma4_31b
    python -m genprobe.rerank analyze --picks data/rerank/picks_full.jsonl --grades grade_rerank_full_gemma4_31b
        -> results/rerank/summary.md, summary.json, rerank_pass.csv, fig_rerank.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .agent import core
from .samples import load_prompts, make_sample_id
from .summarize import pick_scores
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

NS = (1, 2, 4, 9)                    # D-034
NEW_SEEDS = (4, 5, 6, 7, 8)
CURVES = ("sdxl", "routed")
CATEGORIES = ("counting", "spatial", "text", "genai")
PASS_SCORE = 5
N_BOOT = 2000
OUT = Path("data/rerank")
PROMPTS = "data/prompts/prompts_v1.jsonl"
AGENT_RUNS = {"testrule": "grade_testrule_gemma4_31b", "testsdxl": "grade_testsdxl_gemma4_31b"}
MODULE_C = ("single", "best_of_4", "routed_best_of_k", "agent_rule", "agent_rule_sdxl")
LABEL = {"sdxl": "SDXL, best of N by VQAScore", "routed": "routed model, best of N by VQAScore",
         "single": "single SDXL image", "best_of_4": "SDXL best-of-4 (agent's checks)",
         "routed_best_of_k": "routed best-of-k (agent's checks)", "agent_rule": "agent",
         "agent_rule_sdxl": "agent, SDXL only"}


def model_for(curve: str, category: str) -> str:
    return core.BASE_MODEL if curve == "sdxl" else core.ROUTE[category]


def test_prompts(prompts_path: str = PROMPTS, per_category: int | None = None) -> list:
    """Test prompts sorted by id; with per_category, the first n of each category (the smoke run)."""
    ps = load_prompts(prompts_path, split="test")
    if per_category is None:
        return ps
    out, seen = [], defaultdict(int)
    for p in ps:
        c = core.category_of(p.tags)
        if seen[c] < per_category:
            out.append(p)
            seen[c] += 1
    return out


def write_prompt_files(prompts_path: str = PROMPTS, per_category: int | None = None, out: Path = OUT) -> dict[str, int]:
    """data/rerank/prompts_<model>.jsonl: the test prompts each model needs seeds 4-8 for."""
    rows = {json.loads(line)["prompt_id"]: line for line in Path(prompts_path).read_text(encoding="utf-8").splitlines()
            if line.strip()}
    need = defaultdict(list)
    for p in test_prompts(prompts_path, per_category):
        for curve in CURVES:
            m = model_for(curve, core.category_of(p.tags))
            if p.prompt_id not in need[m]:
                need[m].append(p.prompt_id)
    out.mkdir(parents=True, exist_ok=True)
    for m, pids in sorted(need.items()):
        atomic_write_text(out / f"prompts_{m}.jsonl", "".join(rows[p] + "\n" for p in pids))
    return {m: len(p) for m, p in sorted(need.items())}


# ---------- picks ----------

def pick(scores: list[float | None], n: int) -> int:
    """Index of the highest VQAScore among the first n (ties: the lower seed)."""
    best = 0
    for s in range(1, n):
        if scores[s] > scores[best]:
            best = s
    return best


def make_picks(prompts: list, paths: dict[str, str], vqa: dict[str, float]) -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for p in prompts:
        cat = core.category_of(p.tags)
        for curve in CURVES:
            m = model_for(curve, cat)
            sids = [make_sample_id(m, p.prompt_id, s) for s in range(max(NS))]
            gaps = [s for s in sids if s not in vqa or s not in paths]
            if gaps:
                missing += gaps
                continue
            sc = [vqa[s] for s in sids]
            for n in NS:
                k = pick(sc, n)
                rows.append({"group": f"{curve}@{n}", "curve": curve, "n": n, "prompt_id": p.prompt_id,
                             "category": cat, "prompt": p.prompt, "model": m, "seed": k, "sample_id": sids[k],
                             "path": paths[sids[k]], "vqascore": sc[k]})
    return rows, sorted(set(missing))


def picks(out: str, per_category: int | None = None, data: str = "data", prompts_path: str = PROMPTS,
          allow_missing: bool = False) -> dict[str, Any]:
    prompts = test_prompts(prompts_path, per_category)
    man = latest_by_id(read_many(sorted(Path(data, "manifests").glob("gen_*.jsonl")) +
                                 sorted(Path(data, "rerank", "manifests").glob("gen_*.jsonl"))))
    paths = {sid: r["path"] for sid, r in man.items() if r.get("status") == "ok" and r.get("path")}
    ids = {make_sample_id(model_for(c, core.category_of(p.tags)), p.prompt_id, s)
           for p in prompts for c in CURVES for s in range(max(NS))}
    recs = read_many(sorted(Path(data, "scores", "vqascore").glob("*.jsonl")) +
                     sorted(Path(data, "rerank", "scores", "vqascore").glob("*.jsonl")))
    ok, version, info = pick_scores(recs, ids)
    rows, missing = make_picks(prompts, paths, {s: float(r["value"]) for s, r in ok.items()})
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} of {len(ids)} images lack an image or a VQAScore ({version}), "
                         f"e.g. {missing[:5]}: generate and score them first")
    atomic_write_text(Path(out), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    meta = {"created_at": now_iso(), "git_commit": git_commit(), "prompts": len(prompts), "rows": len(rows),
            "distinct_images": len({r["sample_id"] for r in rows}), "vqascore_version": version,
            "vqascore_versions_seen": info.get("versions_seen"), "missing": len(missing)}
    atomic_write_text(Path(out).with_suffix(".meta.json"), json.dumps(meta, indent=2))
    return meta


# ---------- analysis ----------

def mean_ci(per_prompt: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    pids = sorted(per_prompt)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per_prompt[p] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def cost(n: int, model: str, costs: dict) -> float:
    """GPU-seconds of best-of-n by VQAScore: n generations, plus n VQAScore runs when there is a
    choice to make (median seconds per image, module C's measured costs)."""
    gen = float(costs["gen"][model])
    vqa = float(costs["score"]["vqascore"])
    return n * gen + (n * vqa if n > 1 else 0.0)


def judge_version(tag: str, data: str = "data") -> str | None:
    p = Path(data, "judge", f"{tag}.meta.json")
    return json.loads(p.read_text()).get("judge_version") if p.exists() else None


def load_module_c(data: str = "data", root: Path = core.ROOT) -> tuple[dict, dict, dict, set, dict]:
    """Per-prompt judge pass and GPU-seconds of module C's groups on the test split, exactly as
    genprobe.agent.compare computes them for results/agent/testrule+testsdxl/judged.md: the grades
    of both runs are merged, a later run's grade replacing an earlier one's for an image graded
    twice (the base image, when both agents kept it). Also returns the merged grades."""
    passed, spend, cats, versions = defaultdict(dict), defaultdict(dict), {}, set()
    recs: dict[str, dict] = {}
    rows = []
    for i, (run, tag) in enumerate(AGENT_RUNS.items()):
        recs.update(latest_by_id(read_many([Path(data, "judge", f"{tag}.jsonl")])))
        versions.add(judge_version(tag, data))
        for line in (Path(root) / run / "final.jsonl").read_text().splitlines():
            if line.strip():
                f = json.loads(line)
                if i == 0 or f["group"].startswith("agent"):
                    rows.append(f)
    for f in rows:
        g = recs.get(f["sample_id"])
        passed[f["group"]][f["prompt_id"]] = float(bool(g) and g.get("score") == PASS_SCORE)
        spend[f["group"]][f["prompt_id"]] = float(f["gpu_seconds"])
        cats[f["prompt_id"]] = f["category"]
    return passed, spend, cats, versions, recs


def analyze(picks_path: str, grades: str, data: str = "data", out: str = "results/rerank", costs_run: str = "testrule",
            n_boot: int = N_BOOT, root: Path = core.ROOT, figure: bool = True) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(picks_path).read_text().splitlines() if line.strip()]
    g = latest_by_id(read_many([Path(data, "judge", f"{grades}.jsonl")]))
    costs = json.loads((Path(root) / costs_run / "state.json").read_text())["costs"]
    passed, spend, cats, vqa = defaultdict(dict), defaultdict(dict), {}, defaultdict(dict)
    ungraded = 0
    for r in rows:
        rec = g.get(r["sample_id"])
        if rec is None or rec.get("score") is None:
            ungraded += 1
        passed[r["group"]][r["prompt_id"]] = float(bool(rec) and rec.get("score") == PASS_SCORE)
        spend[r["group"]][r["prompt_id"]] = cost(r["n"], r["model"], costs)
        vqa[r["group"]][r["prompt_id"]] = float(r["vqascore"])
        cats[r["prompt_id"]] = r["category"]
    mc_passed, mc_spend, mc_cats, versions, mc_recs = load_module_c(data, root)
    # images graded both here and in module C (e.g. SDXL seed 0 = sdxl@1 = single): how often the
    # judge, run again in another job, gives the same score and the same pass / fail
    both = [s for s in g if s in mc_recs and g[s].get("score") is not None and mc_recs[s].get("score") is not None]
    regrade = {"images": len(both),
               "same_score": float(np.mean([g[s]["score"] == mc_recs[s]["score"] for s in both])) if both else math.nan,
               "same_pass": float(np.mean([(g[s]["score"] == PASS_SCORE) == (mc_recs[s]["score"] == PASS_SCORE)
                                           for s in both])) if both else math.nan}
    version = judge_version(grades, data)
    versions.discard(None)
    if versions and versions != {version}:
        raise SystemExit(f"re-ranking picks graded by {version}, module C by {versions}: not comparable")
    prompts = set(cats)
    for grp in MODULE_C:                       # module C's groups on the same prompts
        passed[grp] = {p: v for p, v in mc_passed.get(grp, {}).items() if p in prompts}
        spend[grp] = {p: v for p, v in mc_spend.get(grp, {}).items() if p in prompts}
    groups = [f"{c}@{n}" for c in CURVES for n in NS] + [m for m in MODULE_C if passed[m]]
    cat_list = ["all"] + [c for c in CATEGORIES if c in set(cats.values())]
    rate = {grp: {c: mean_ci({p: v for p, v in passed[grp].items() if c == "all" or cats[p] == c}, n_boot)
                  for c in cat_list} for grp in groups}
    own = {grp: {"mean_vqascore": float(np.mean(list(vqa[grp].values()))),
                 "vqascore_ge_0.9": float(np.mean([v >= core.TAU_GENAI for v in vqa[grp].values()]))}
           for grp in groups if grp in vqa}
    gpu = {grp: float(np.mean(list(spend[grp].values()))) if spend[grp] else math.nan for grp in groups}
    per_success = {grp: (float(sum(spend[grp].values()) / sum(passed[grp].values()))
                         if sum(passed[grp].values()) else math.nan) for grp in groups}
    pairs = [(f"{c}@{b}", f"{c}@{a}") for c in CURVES for a, b in ((1, 2), (2, 4), (4, 9), (1, 9))]
    pairs += [(f"routed@{n}", f"sdxl@{n}") for n in NS]
    pairs += [("best_of_4", "sdxl@4"), ("routed_best_of_k", "routed@4"),
              ("agent_rule", "sdxl@4"), ("agent_rule", "sdxl@9"), ("agent_rule", "routed@1"),
              ("agent_rule", "routed@2"), ("agent_rule", "routed@4"), ("agent_rule", "routed@9")]
    diffs = {}
    for a, b in pairs:
        if not passed[a] or not passed[b]:
            continue
        for c in cat_list:
            common = [p for p in passed[a] if p in passed[b] and (c == "all" or cats[p] == c)]
            diffs[f"{a} - {b}|{c}"] = mean_ci({p: passed[a][p] - passed[b][p] for p in common}, n_boot)
    res = {"created_at": now_iso(), "git_commit": git_commit(), "picks": picks_path, "grades": grades,
           "judge_version": version, "pass_rule": f"judge score == {PASS_SCORE}", "prompts": len(prompts),
           "ns": list(NS), "costs": costs, "groups": groups, "pass_rate": rate, "gpu_seconds": gpu,
           "gpu_seconds_per_success": per_success, "own_scores": own, "paired_difference": diffs,
           "ungraded": ungraded, "distinct_images": len({r["sample_id"] for r in rows}), "regrade": regrade}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    atomic_write_text(o / "summary.json", json.dumps(res, indent=2))
    with open(o / "rerank_pass.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "category", "pass_rate", "ci95_lo", "ci95_hi", "prompts", "gpu_seconds_per_prompt"])
        for grp in groups:
            for c in cat_list:
                r = rate[grp][c]
                w.writerow([grp, c, round(r["value"], 4), round(r["ci"][0], 4), round(r["ci"][1], 4), r["n"],
                            round(gpu[grp], 3)])
    if figure:
        res["figure_title"] = plot(res, o / "fig_rerank.png")
        atomic_write_text(o / "summary.json", json.dumps(res, indent=2))
    atomic_write_text(o / "summary.md", render(res))
    return res


def plot(res: dict, path: Path) -> str:
    from .figures import INK, INK2, SURFACE, _fig, _footnote
    colours = {"sdxl": "#eb6834", "routed": "#6250d6"}       # validated pair (dataviz validator)
    plt, fig, ax = _fig(7.4, 4.8)
    r, gpu = res["pass_rate"], res["gpu_seconds"]
    xs = []
    for c in CURVES:
        pts = [(gpu[f"{c}@{n}"], r[f"{c}@{n}"]["all"]) for n in res["ns"]]
        x = [p[0] for p in pts]
        y = [p[1]["value"] for p in pts]
        lo = [p[1]["value"] - p[1]["ci"][0] for p in pts]
        hi = [p[1]["ci"][1] - p[1]["value"] for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], color=colours[c], linewidth=2, elinewidth=1, capsize=0, marker="o",
                    markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label=LABEL[c])
        above = c == "routed"                       # N labels above the upper curve, below the lower one
        for n, xv, yv in zip(res["ns"], x, y):
            ax.annotate(f"N={n}", (xv, yv), xytext=(-6, 8) if above else (6, -13), textcoords="offset points",
                        fontsize=7.5, color=INK2, ha="right" if above else "left")
        xs += x
    markers = {"single": "v", "best_of_4": "s", "routed_best_of_k": "D", "agent_rule": "*", "agent_rule_sdxl": "P"}
    for m in MODULE_C:
        if m not in r:
            continue
        v = r[m]["all"]
        ax.errorbar(gpu[m], v["value"], yerr=[[v["value"] - v["ci"][0]], [v["ci"][1] - v["value"]]], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=0, alpha=0.55, zorder=2)
        ax.scatter(gpu[m], v["value"], s=180 if m == "agent_rule" else 60, marker=markers[m], color=INK,
                   edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.annotate(LABEL[m], (gpu[m], v["value"]), xytext=(8, -3), textcoords="offset points", fontsize=8,
                    color=INK)
        xs.append(gpu[m])
    ax.set_xscale("log")
    lo_x, hi_x = min(xs) / 1.4, max(xs) * 1.5
    ax.set_xlim(lo_x, hi_x)
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    nice = [v * 10 ** e for e in range(0, 4) for v in (1, 2, 5)]
    ax.xaxis.set_major_locator(FixedLocator([v for v in nice if lo_x <= v <= hi_x]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("GPU-seconds per prompt (generation + checks; log scale)", fontsize=8.5, color=INK2)
    ax.set_ylabel("prompts the judge passes (score 5; 95% CI)", fontsize=8.5, color=INK2)
    parts = [f"SDXL N=9 {r['sdxl@9']['all']['value']:.2f}", f"routed N=9 {r['routed@9']['all']['value']:.2f}"]
    if "agent_rule" in r:
        parts.append(f"agent {r['agent_rule']['all']['value']:.2f}")
    title = "Judge pass rate of best-of-N by VQAScore: " + ", ".join(parts)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=10)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK2, loc="lower right")
    _footnote(fig, f"Test split, {res['prompts']} prompts; judge Gemma 4 31B, original prompt, pass = score 5. "
              "Curves: best of the first N seeds by VQAScore. Black: module C's groups at the GPU-seconds they "
              "spent (D-031, D-032). Costs exclude model loading.")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return title


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    cats = list(next(iter(res["pass_rate"].values())).keys())
    L = ["# Re-ranking curve (guide §11.1)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.rerank analyze` (D-034). "
         f"Judge `{res['judge_version']}` (grades `{res['grades']}`), original prompts, pass = {res['pass_rule']}. "
         f"{res['prompts']} test prompts, {res['distinct_images']} distinct picked images; 95 % CIs from resampling "
         "prompts." + (f" Ungraded (counted as fails): {res['ungraded']}." if res["ungraded"] else ""), "",
         "`sdxl@N` / `routed@N`: of the first N seeds, the image with the highest VQAScore. Module C's groups "
         "(below the line) are the same prompts' images from `results/agent/testrule+testsdxl/`.", "",
         "## Pass rate", "",
         "| Group | " + " | ".join(cats) + " | GPU-s / prompt | GPU-s / success | mean VQAScore | VQAScore ≥ 0.9 |",
         "| --- |" + " --- |" * (len(cats) + 4)]
    for grp in res["groups"]:
        cells = [f"{_f(res['pass_rate'][grp][c]['value'])} [{_f(res['pass_rate'][grp][c]['ci'][0], 2)}, "
                 f"{_f(res['pass_rate'][grp][c]['ci'][1], 2)}]" for c in cats]
        own = res["own_scores"].get(grp, {})
        L.append(f"| {grp} | " + " | ".join(cells) + f" | {_f(res['gpu_seconds'][grp], 1)} | "
                 f"{_f(res['gpu_seconds_per_success'][grp], 1)} | {_f(own.get('mean_vqascore'))} | "
                 f"{_f(own.get('vqascore_ge_0.9'))} |")
    L += ["", "## Paired differences (same prompts)", "", "| Comparison | " + " | ".join(cats) + " |",
          "| --- |" + " --- |" * len(cats)]
    for pr in dict.fromkeys(k.split("|")[0] for k in res["paired_difference"]):
        cells = []
        for c in cats:
            d = res["paired_difference"].get(f"{pr}|{c}")
            cells.append("–" if d is None else f"{_f(d['value'])} [{_f(d['ci'][0], 2)}, {_f(d['ci'][1], 2)}]")
        L.append(f"| {pr} | " + " | ".join(cells) + " |")
    rg = res.get("regrade", {})
    if rg.get("images"):
        L += ["", f"Grading again: {rg['images']} images were graded both here and in module C (separate jobs). "
              f"Same score for {_f(rg['same_score'])}, same pass / fail for {_f(rg['same_pass'])}. Module C's rows "
              "use module C's grades, as in `results/agent/testrule+testsdxl/judged.md`."]
    c = res["costs"]
    L += ["", f"Costs (median seconds per image, from module C's run): generation "
          f"{', '.join(f'{m} {v:.2f}' for m, v in c['gen'].items())}; VQAScore {c['score']['vqascore']:.2f}. "
          "Best of N costs N generations plus N VQAScore runs (none for N = 1). Model loading is not counted.",
          "", "![re-ranking curve](fig_rerank.png)" if res.get("figure_title") else ""]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prompts")
    pr.add_argument("--per-category", type=int, default=None, help="smoke run: first n test prompts per category")
    pk = sub.add_parser("picks")
    pk.add_argument("--out", default=str(OUT / "picks.jsonl"))
    pk.add_argument("--per-category", type=int, default=None)
    an = sub.add_parser("analyze")
    an.add_argument("--picks", default=str(OUT / "picks.jsonl"))
    an.add_argument("--grades", required=True, help="tag of data/judge/<tag>.jsonl")
    an.add_argument("--out", default="results/rerank")
    args = ap.parse_args(argv)
    if args.cmd == "prompts":
        print(json.dumps(write_prompt_files(per_category=args.per_category)))
    elif args.cmd == "picks":
        print(json.dumps(picks(args.out, args.per_category), indent=2))
    else:
        print(render(analyze(args.picks, args.grades, out=args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
