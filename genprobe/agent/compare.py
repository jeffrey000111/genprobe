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


def compare(run: str, grades: str, data: str = "data", root: Path = ROOT, results: Path = Path("results/agent"),
            n_boot: int = N_BOOT) -> dict[str, Any]:
    d = run_dir(run, root)
    final = [json.loads(line) for line in (d / "final.jsonl").read_text().splitlines() if line.strip()]
    recs = latest_by_id(read_many([Path(data) / "judge" / f"{grades}.jsonl"]))
    meta_path = Path(data) / "judge" / f"{grades}.meta.json"
    jmeta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    states = {s["prompt_id"]: PromptState.from_dict(s)
              for s in json.loads((d / "state.json").read_text())["states"]}
    groups = sorted({f["group"] for f in final}, key=lambda g: (g.startswith("agent"), BASELINES.index(g)
                                                                 if g in BASELINES else 0, g))
    passed = defaultdict(dict)            # group -> prompt -> 0/1
    cost = defaultdict(dict)
    cat = {}
    missing = Counter()
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
    diffs = {}
    for a in (g for g in groups if g.startswith("agent")):
        for b in (g for g in groups if not g.startswith("agent")):
            for c in cats:
                common = [p for p in passed[a] if p in passed[b] and (c == "all" or cat[p] == c)]
                diffs[f"{a} - {b}|{c}"] = mean_ci({p: passed[a][p] - passed[b][p] for p in common}, n_boot)
    spend = {g: {"mean_gpu_seconds": float(np.mean(list(cost[g].values()))),
                 "gpu_seconds_per_success": (float(sum(cost[g].values()) / sum(passed[g].values()))
                                             if sum(passed[g].values()) else math.nan)} for g in groups}
    agent = next((g for g in groups if g.startswith("agent")), None)
    unfixed = Counter()
    if agent:
        for p, ok in passed[agent].items():
            st = states[p]
            if not ok:
                unfixed[core.diagnose(st.category, st.attempts[0].checks) if not core.passed(st.category, st.attempts[0].checks)
                        else "passed own checks"] += 1
        attempts = float(np.mean([len(s.attempts) for s in states.values()]))
    res = {"run": run, "grades": grades, "judge_version": jmeta.get("judge_version"), "created_at": now_iso(),
           "git_commit": git_commit(), "pass_rule": f"judge score == {PASS_SCORE}", "groups": groups,
           "pass_rate": rates, "paired_difference": diffs, "spend": spend, "ungraded": dict(missing),
           "agent_unfixed_by_first_diagnosis": dict(unfixed), "agent_attempts_mean": attempts if agent else None,
           "prompts": len(cat)}
    out = results / run
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "judged.json", json.dumps(res, indent=2))
    atomic_write_text(out / "judged.md", render(res))
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    cats = list(next(iter(res["pass_rate"].values())).keys())
    L = [f"# Agent run `{res['run']}` graded by the independent judge", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.agent.compare`. Judge "
         f"`{res['judge_version']}` (grades `{res['grades']}`), original prompts, pass = {res['pass_rule']}. "
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
    if res["agent_unfixed_by_first_diagnosis"]:
        L += ["", "Prompts the agent's final image still fails, by the diagnosis of the first image: " +
              ", ".join(f"{k} {v}" for k, v in sorted(res["agent_unfixed_by_first_diagnosis"].items())) + "."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--grades", required=True, help="tag of data/judge/<tag>.jsonl from `genprobe.judge grade`")
    args = ap.parse_args(argv)
    print(render(compare(args.run, args.grades)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
