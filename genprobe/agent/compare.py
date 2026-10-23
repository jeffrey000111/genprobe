"""Module C's result (guide §10.4, D-032): pass rates of the groups as graded by the independent
judge (Gemma 4 31B, frozen rubric, original prompt; pass = score 5), with 95 % CIs over prompts,
paired differences between the agent and each baseline, GPU-seconds per prompt and per success,
and what the agent could not fix.

    python -m genprobe.agent.compare --run devrule1 --grades grade_devrule1
    -> results/agent/<run>/judged.md, judged.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.env import git_commit, now_iso
from ..utils.io import atomic_write_text
from ..utils.manifest import latest_by_id, read_many
from . import core
from .core import ROOT, PromptState, run_dir

N_BOOT = 2000
PASS_SCORE = 5            # D-032: only "every requirement is met"
BASELINES = ("single", "best_of_4", "routed_best_of_k")


def judged(rec: dict | None) -> bool:
    return bool(rec) and rec.get("score") == PASS_SCORE


def mean_ci(per_prompt: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    pids = sorted(per_prompt)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per_prompt[p] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def compare(runs: list[str] | str, grades: list[str] | str, data: str = "data", root: Path = ROOT,
            results: Path = Path("results/agent"), n_boot: int = N_BOOT, name: str | None = None) -> dict[str, Any]:
    """One or more agent runs on the same split, each with its grades tag. The baselines come from
    the first run (they are the same images in every run); each run adds its agent group."""
    runs = [runs] if isinstance(runs, str) else list(runs)
    grades = [grades] if isinstance(grades, str) else list(grades)
    if len(runs) != len(grades):
        raise ValueError("one grades tag per run")
    final, recs, states, versions = [], {}, {}, []
    for i, (run, tag) in enumerate(zip(runs, grades)):
        d = run_dir(run, root)
        rows = [json.loads(line) for line in (d / "final.jsonl").read_text().splitlines() if line.strip()]
        meta_path = Path(data) / "judge" / f"{tag}.meta.json"
        versions.append(json.loads(meta_path.read_text()).get("judge_version") if meta_path.exists() else None)
        for f in rows:
            if not f["group"].startswith("agent") and i > 0:
                continue
            if f["group"].startswith("agent") and len(runs) > 1:
                f = {**f, "group": f"{f['group']}@{run}"}
            final.append(f)
        recs.update(latest_by_id(read_many([Path(data) / "judge" / f"{tag}.jsonl"])))
        for st in json.loads((d / "state.json").read_text())["states"]:
            states[(run, st["prompt_id"])] = PromptState.from_dict(st)
    if len(set(versions)) > 1:
        raise ValueError(f"runs were graded by different judges: {versions}")
    groups = sorted({f["group"] for f in final},
                    key=lambda g: (g.startswith("agent"), BASELINES.index(g) if g in BASELINES else 0, g))
    passed, cost, cat, missing = defaultdict(dict), defaultdict(dict), {}, Counter()
    for f in final:
        r = recs.get(f["sample_id"])
        if r is None or r.get("score") is None:
            missing[f["group"]] += 1
        passed[f["group"]][f["prompt_id"]] = float(judged(r))
        cost[f["group"]][f["prompt_id"]] = float(f["gpu_seconds"])
        cat[f["prompt_id"]] = f["category"]
    cats = ["all"] + sorted(set(cat.values()))
    rates = {g: {c: mean_ci({p: v for p, v in passed[g].items() if c == "all" or cat[p] == c}, n_boot) for c in cats}
             for g in groups}
    agents = [g for g in groups if g.startswith("agent")]
    pairs = [(a, b) for a in agents for b in groups if not b.startswith("agent")]
    pairs += [(a, b) for i, a in enumerate(agents) for b in agents[i + 1:]]
    diffs = {}
    for a, b in pairs:
        for c in cats:
            common = [p for p in passed[a] if p in passed[b] and (c == "all" or cat[p] == c)]
            diffs[f"{a} - {b}|{c}"] = mean_ci({p: passed[a][p] - passed[b][p] for p in common}, n_boot)
    spend = {g: {"mean_gpu_seconds": float(np.mean(list(cost[g].values()))),
                 "gpu_seconds_per_success": (float(sum(cost[g].values()) / sum(passed[g].values()))
                                             if sum(passed[g].values()) else math.nan)} for g in groups}
    unfixed = {}
    for g in agents:
        run = g.split("@")[1] if "@" in g else runs[0]
        c = Counter()
        for p, ok in passed[g].items():
            st = states[(run, p)]
            if not ok:
                a0 = st.attempts[0].checks
                c[core.diagnose(st.category, a0) if not core.passed(st.category, a0) else "passed own checks"] += 1
        unfixed[g] = dict(c)
    res = {"runs": runs, "grades": grades, "judge_version": versions[0], "created_at": now_iso(),
           "git_commit": git_commit(), "pass_rule": f"judge score == {PASS_SCORE}", "groups": groups,
           "pass_rate": rates, "paired_difference": diffs, "spend": spend, "ungraded": dict(missing),
           "agent_unfixed_by_first_diagnosis": unfixed, "prompts": len(cat)}
    out = results / (name or "+".join(runs))
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "judged.json", json.dumps(res, indent=2))
    atomic_write_text(out / "judged.md", render(res))
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    cats = list(next(iter(res["pass_rate"].values())).keys())
    L = [f"# Agent run(s) {', '.join('`' + r + '`' for r in res['runs'])} graded by the independent judge", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.agent.compare`. Judge "
         f"`{res['judge_version']}` (grades {', '.join('`' + g + '`' for g in res['grades'])}), original prompts, "
         f"pass = {res['pass_rule']}. "
         f"{res['prompts']} prompts; 95 % CIs from resampling prompts." +
         (f" Ungraded images (counted as fails): {res['ungraded']}." if res["ungraded"] else ""), "",
         "## Pass rate", "", "| Group | " + " | ".join(cats) + " | GPU-s / prompt | GPU-s / success |",
         "| --- |" + " --- |" * (len(cats) + 2)]
    for g in res["groups"]:
        cells = [f"{_f(res['pass_rate'][g][c]['value'])} [{_f(res['pass_rate'][g][c]['ci'][0], 2)}, "
                 f"{_f(res['pass_rate'][g][c]['ci'][1], 2)}]" for c in cats]
        sp = res["spend"][g]
        L.append(f"| {g} | " + " | ".join(cells) + f" | {_f(sp['mean_gpu_seconds'], 1)} | "
                 f"{_f(sp['gpu_seconds_per_success'], 1)} |")
    L += ["", "## Agent minus baseline (same prompts)", "", "| Comparison | " + " | ".join(cats) + " |",
          "| --- |" + " --- |" * len(cats)]
    pairs = sorted({k.split("|")[0] for k in res["paired_difference"]})
    for pr in pairs:
        cells = []
        for c in cats:
            dd = res["paired_difference"][f"{pr}|{c}"]
            cells.append(f"{_f(dd['value'], 3)} [{_f(dd['ci'][0], 2)}, {_f(dd['ci'][1], 2)}]")
        L.append(f"| {pr} | " + " | ".join(cells) + " |")
    for g, c in res["agent_unfixed_by_first_diagnosis"].items():
        if c:
            L += ["", f"{g}: prompts whose final image still fails, by the diagnosis of the first image: " +
                  ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="repeat for several runs on the same split")
    ap.add_argument("--grades", action="append", required=True,
                    help="tag of data/judge/<tag>.jsonl from `genprobe.judge grade`, one per --run")
    ap.add_argument("--name", default=None, help="output folder under results/agent (default: the run names)")
    args = ap.parse_args(argv)
    print(render(compare(args.run, args.grades, name=args.name)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
