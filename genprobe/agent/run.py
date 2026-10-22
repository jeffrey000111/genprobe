"""Round driver for the generation-debugging agent (guide §10, D-031). One Slurm job
(slurm/agent.sbatch) loops: plan -> generate (genprobe.agent.gen, gen env) -> check (genprobe.score:
VQAScore on the GPU, detector and OCR on CPU as in prompt set v1) -> ingest, until no prompt has a
next step.

    python -m genprobe.agent.run init   --run devrule --split dev
    python -m genprobe.agent.run plan   --run devrule --round 1      -> data/agent/<run>/round1/requests.jsonl
    python -m genprobe.agent.run ingest --run devrule --round 1
    python -m genprobe.agent.run report --run devrule                -> data/agent/<run>/final.jsonl,
                                                                        results/agent/<run>/
State: data/agent/<run>/state.json; audit log: data/agent/<run>/audit.jsonl (one line per attempt).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..samples import image_path, load_prompts, make_sample_id
from ..summarize import pick_scores
from ..utils.env import git_commit, now_iso
from ..utils.io import atomic_write_text
from ..utils.manifest import latest_by_id, read_many
from . import core
from .core import Attempt, PromptState

ROOT = core.ROOT
run_dir = core.run_dir
GEN_MODELS = ("sdxl", "sd35m", "flux_schnell")
SCORE_METRICS = ("vqascore", "detect", "ocr")


def load_states(run: str, root: Path = ROOT) -> list[PromptState]:
    return [PromptState.from_dict(d) for d in json.loads((run_dir(run, root) / "state.json").read_text())["states"]]


def save_states(run: str, states: list[PromptState], meta: dict, root: Path = ROOT) -> None:
    atomic_write_text(run_dir(run, root) / "state.json",
                      json.dumps({**meta, "states": [s.to_dict() for s in states]}, indent=1))


def load_meta(run: str, root: Path = ROOT) -> dict:
    d = json.loads((run_dir(run, root) / "state.json").read_text())
    return {k: v for k, v in d.items() if k != "states"}


def append_audit(run: str, lines: list[dict], root: Path = ROOT) -> None:
    with open(run_dir(run, root) / "audit.jsonl", "a", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


# ---------- reading checks ----------

def read_checks(data: str | Path, ids: set[str]) -> tuple[dict[str, dict], dict[str, float]]:
    """sample_id -> checks dict (vqascore, detect + detect_extra, ocr + ocr_read) and the summed
    scoring seconds, from data/scores/<metric>/*.jsonl."""
    checks: dict[str, dict] = defaultdict(dict)
    secs: dict[str, float] = defaultdict(float)
    for m in SCORE_METRICS:
        recs = read_many(sorted((Path(data) / "scores" / m).glob("*.jsonl")))
        ok, _, _ = pick_scores(recs, ids)
        for sid, r in ok.items():
            checks[sid][m] = float(r["value"])
            if m == "detect":
                checks[sid]["detect_extra"] = r.get("extra", {})
            if m == "ocr":
                checks[sid]["ocr_read"] = (r.get("extra") or {}).get("ocr_norm", "")
            secs[sid] += float(r.get("seconds") or 0.0)
    return dict(checks), dict(secs)


def measure_costs(data: str | Path, ids_by_model: dict[str, set[str]], score_ids: set[str]) -> dict:
    """Median seconds per image: generation per model (prompt set v1, warm-up batches left out)
    and scoring per metric."""
    gen = {}
    recs = read_many(sorted((Path(data) / "manifests").glob("gen_*.jsonl")))
    for m in GEN_MODELS:
        t = [float(r["gen_seconds"]) for r in recs if r.get("model") == m and r.get("status") == "ok"
             and not r.get("warmup") and r.get("gen_seconds") is not None and r["sample_id"] in ids_by_model.get(m, set())]
        if t:
            gen[m] = round(statistics.median(t), 4)
    score = {}
    for m in SCORE_METRICS:
        rs = read_many(sorted((Path(data) / "scores" / m).glob("*.jsonl")))
        t = [float(r["seconds"]) for r in rs if r.get("status") == "ok" and r.get("sample_id") in score_ids
             and r.get("seconds") is not None]
        if t:
            score[m] = round(statistics.median(t), 4)
    return {"gen": gen, "score": score}


# ---------- commands ----------

def init(run: str, split: str, policy: str = "rule", prompts_path: str = "data/prompts/prompts_v1.jsonl",
         data: str = "data", root: Path = ROOT, limit: int | None = None) -> dict:
    prompts = load_prompts(prompts_path, split=split, limit=limit)
    v1 = {pid: {m: {s: make_sample_id(m, p.prompt_id, s) for s in range(4)} for m in GEN_MODELS}
          for p in prompts for pid in [p.prompt_id]}
    ids_by_model = {m: {v1[p.prompt_id][m][s] for p in prompts for s in range(4)} for m in GEN_MODELS}
    base_ids = {v1[p.prompt_id][core.BASE_MODEL][s] for p in prompts for s in range(4)}
    costs = measure_costs(data, ids_by_model, base_ids)
    missing = [m for m in GEN_MODELS if m not in costs["gen"]] + [m for m in SCORE_METRICS if m not in costs["score"]]
    if missing:
        raise SystemExit(f"no timing data for {missing}: prompt set v1 must be generated and scored first")
    gen = latest_by_id(read_many(sorted((Path(data) / "manifests").glob(f"gen_{core.BASE_MODEL}_*.jsonl"))))
    checks, secs = read_checks(data, base_ids)
    states, lines = [], []
    for p in prompts:
        cat = core.category_of(list(p.tags))
        sid = v1[p.prompt_id][core.BASE_MODEL][core.BASE_SEED]
        g = gen.get(sid, {})
        if g.get("status") != "ok" or sid not in checks:
            raise SystemExit(f"{sid}: no ok v1 image or no checks")
        a0 = Attempt(k=0, model=core.BASE_MODEL, gen_prompt=p.prompt, cfg=None, seed=core.BASE_SEED, sample_id=sid,
                     action="base", reason="the base model's seed-0 image (prompt set v1)", status="ok",
                     checks=checks[sid], gen_seconds=costs["gen"][core.BASE_MODEL] if g.get("warmup") else g.get("gen_seconds"),
                     score_seconds=secs.get(sid, 0.0), path=g.get("path"))
        st = PromptState(prompt_id=p.prompt_id, prompt=p.prompt, category=cat, spec=p.spec, tags=list(p.tags),
                         budget=round(core.budget_for(cat, costs), 3), policy=policy, attempts=[a0])
        if core.passed(cat, a0.checks):
            st.done, st.stop_reason = True, "passed"
        states.append(st)
        lines.append(core.audit(st, a0, "start" + (" (passed)" if st.done else "")))
    meta = {"run": run, "split": split, "policy": policy, "created_at": now_iso(), "git_commit": git_commit(),
            "costs": costs, "settings": {"tau_genai": core.TAU_GENAI, "delta": core.DELTA, "cfg_high": core.CFG_HIGH,
                                         "route": core.ROUTE, "rules": core.RULES, "max_attempts": core.MAX_ATTEMPTS}}
    d = run_dir(run, root)
    d.mkdir(parents=True, exist_ok=True)
    (d / "audit.jsonl").write_text("")
    save_states(run, states, meta, root)
    append_audit(run, lines, root)
    return {"prompts": len(states), "passed_at_start": sum(s.done for s in states), "costs": costs}


def plan(run: str, rnd: int, root: Path = ROOT, chooser=None) -> int:
    """Choose the next attempt for every prompt that is still open; write the round's requests.
    `chooser(state, costs)` returns (action, params, reason) or None; the rule policy by default."""
    states, meta = load_states(run, root), load_meta(run, root)
    open_ = [a.sample_id for st in states for a in st.attempts if a.status == "planned"]
    if open_:
        raise SystemExit(f"{len(open_)} attempts are planned but not ingested yet (e.g. {open_[0]}); "
                         "run ingest for the previous round first")
    costs = meta["costs"]
    chooser = chooser or core.next_rule_action
    out = run_dir(run, root) / f"round{rnd}"
    out.mkdir(parents=True, exist_ok=True)
    reqs, score_prompts = [], []
    for st in states:
        if st.done:
            continue
        if len(st.attempts) >= core.MAX_ATTEMPTS:
            st.done, st.stop_reason = True, "max attempts"
            continue
        nxt = chooser(st, costs)
        if nxt is None:
            st.done, st.stop_reason = True, "no affordable action left"
            continue
        action, params, reason = nxt
        k = len(st.attempts)
        apid = core.attempt_prompt_id(st.prompt_id, k, run)
        sid = make_sample_id(params["model"], apid, params["seed"])
        a = Attempt(k=k, model=params["model"], gen_prompt=params["gen_prompt"], cfg=params["cfg"], seed=params["seed"],
                    sample_id=sid, action=action, reason=reason,
                    path=str(image_path(run_dir(run, root), params["model"], apid, params["seed"])),
                    llm_seconds=float(params.get("llm_seconds", 0.0)))
        st.attempts.append(a)
        reqs.append({"sample_id": sid, "model": a.model, "prompt_id": apid, "prompt": st.prompt,
                     "gen_prompt": a.gen_prompt, "cfg": a.cfg, "seed": a.seed, "path": a.path})
        score_prompts.append({"prompt_id": apid, "prompt": st.prompt, "source": "agent", "tags": st.tags,
                              "spec": st.spec, "split": meta["split"]})
    atomic_write_text(out / "requests.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs))
    atomic_write_text(out / "score_prompts.jsonl",
                      "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in score_prompts))
    save_states(run, states, meta, root)
    return len(reqs)


def ingest(run: str, rnd: int, data: str = "data", root: Path = ROOT) -> dict:
    states, meta = load_states(run, root), load_meta(run, root)
    d = run_dir(run, root)
    gen = latest_by_id(read_many(sorted((d / "manifests").glob(f"gen_r{rnd}_*.jsonl"))))
    pending = {a.sample_id: (st, a) for st in states for a in st.attempts if a.status == "planned"}
    checks, secs = read_checks(data, set(pending))
    counts, lines = Counter(), []
    for sid, (st, a) in pending.items():
        g = gen.get(sid)
        if not g or g.get("status") != "ok" or sid not in checks:
            a.status = "error"
            a.gen_seconds = (g or {}).get("gen_seconds")
            counts["error"] += 1
            lines.append(core.audit(st, a, "failed: " + ("not generated" if not g or g.get("status") != "ok"
                                                         else "not checked")))
            continue
        a.status, a.checks = "ok", checks[sid]
        a.gen_seconds = core.expected_cost(a.model, meta["costs"]) if g.get("warmup") else g.get("gen_seconds")
        a.score_seconds = secs.get(sid, 0.0)
        if core.better(a.checks, st.best_attempt().checks, st.category):
            st.best = a.k
            decision = "new best"
        else:
            decision = "kept attempt %d" % st.best
        if core.passed(st.category, a.checks):
            st.done, st.stop_reason = True, "passed"
            decision += "; passed"
        counts[decision.split(";")[0]] += 1
        lines.append(core.audit(st, a, decision))
    save_states(run, states, meta, root)
    append_audit(run, lines, root)
    return dict(counts)


def best_of_n(st: PromptState, v1_checks: dict[str, dict], n: int = 4, model: str = core.BASE_MODEL) -> str | None:
    """The image best-of-n picks among `model`'s prompt-set-v1 seeds 0..n-1, with the agent's own
    comparison, in seed order."""
    best = None
    for s in range(n):
        sid = make_sample_id(model, st.prompt_id, s)
        c = v1_checks.get(sid)
        if c is None:
            continue
        if best is None or core.better(c, v1_checks[best], st.category):
            best = sid
    return best


def routed_k(st: PromptState, costs: dict, max_k: int = 4) -> int:
    """How many images of the category's routed model the same budget pays for (at most the four
    seeds prompt set v1 already has; FLUX could afford more, so this baseline is if anything
    a little weak)."""
    per = core.expected_cost(core.ROUTE[st.category], costs) + core.score_cost(st.category, costs)
    return max(1, min(max_k, int(st.budget // per)))


def report(run: str, data: str = "data", root: Path = ROOT, results: Path = Path("results/agent")) -> dict:
    """Groups (guide §10.4, D-032): single = the base image; best_of_4 = SDXL seeds 0-3; routed_best_of_k
    = the category's dev-best model, as many v1 seeds as the budget pays for; agent_<policy>. All
    picks use the agent's comparison; final.jsonl lists them for the independent judge."""
    states, meta = load_states(run, root), load_meta(run, root)
    costs = meta["costs"]
    ids = {make_sample_id(m, st.prompt_id, s) for st in states for s in range(4)
           for m in {core.BASE_MODEL, core.ROUTE[st.category]}}
    v1_checks, _ = read_checks(data, ids)
    gen = latest_by_id(read_many(sorted((Path(data) / "manifests").glob("gen_*.jsonl"))))
    final, rows = [], []
    for st in states:
        k = routed_k(st, costs)
        per_routed = core.expected_cost(core.ROUTE[st.category], costs) + core.score_cost(st.category, costs)
        picks = {"single": (st.attempts[0].sample_id, st.attempts[0].cost()),
                 "best_of_4": (best_of_n(st, v1_checks), st.budget),
                 "routed_best_of_k": (best_of_n(st, v1_checks, k, core.ROUTE[st.category]), k * per_routed),
                 f"agent_{st.policy}": (st.best_attempt().sample_id, st.spent())}
        for group, (sid, cost) in picks.items():
            if sid is None:
                continue
            agent = group.startswith("agent")
            path = st.best_attempt().path if agent else gen.get(sid, {}).get("path")
            c = st.best_attempt().checks if agent else v1_checks.get(sid, {})
            final.append({"prompt_id": st.prompt_id, "group": group, "sample_id": sid, "path": path,
                          "prompt": st.prompt, "category": st.category, "gpu_seconds": round(cost, 3),
                          **({"k": k, "model": core.ROUTE[st.category]} if group == "routed_best_of_k" else {})})
            rows.append({"group": group, "category": st.category, "passed_internal": core.passed(st.category, c)})
    by = defaultdict(list)
    for r in rows:
        by[(r["group"], r["category"])].append(r["passed_internal"])
        by[(r["group"], "all")].append(r["passed_internal"])
    internal = {f"{g}|{c}": {"n": len(v), "pass_rate": sum(v) / len(v)} for (g, c), v in sorted(by.items())}
    agent_stats = {"attempts_mean": statistics.fmean(len(st.attempts) for st in states),
                   "spent_mean": statistics.fmean(st.spent() for st in states),
                   "budget_mean": statistics.fmean(st.budget for st in states),
                   "over_budget": sum(st.spent() > st.budget + 1e-6 for st in states),
                   "stop_reasons": dict(Counter(st.stop_reason or "open" for st in states)),
                   "actions": dict(Counter(a.action for st in states for a in st.attempts[1:])),
                   "improved_by": dict(Counter(st.best_attempt().action for st in states if st.best > 0)),
                   "routed_k": dict(Counter(f"{st.category}:{routed_k(st, costs)}" for st in states))}
    res = {"run": run, "split": meta["split"], "policy": meta["policy"], "created_at": now_iso(),
           "git_commit": git_commit(), "costs": costs, "settings": meta["settings"],
           "internal_pass_rate": internal, "agent": agent_stats, "prompts": len(states)}
    d = run_dir(run, root)
    atomic_write_text(d / "final.jsonl", "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in final))
    out = results / run
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2))
    atomic_write_text(out / "summary.md", render(res))
    (out / "audit.jsonl").write_text((d / "audit.jsonl").read_text())
    return res


def render(res: dict) -> str:
    a = res["agent"]
    L = [f"# Agent run `{res['run']}` ({res['policy']} policy, {res['split']} split)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.agent.run report`. "
         f"{res['prompts']} prompts. **Pass rates here use the agent's own checkers** (detector, OCR, VQAScore), "
         "which the agent optimizes; the comparison that counts is the independent judge's (D-030).", "",
         "| Group | Category | Prompts | Passes own checks |", "| --- | --- | --- | --- |"]
    for key, v in res["internal_pass_rate"].items():
        g, c = key.split("|")
        L.append(f"| {g} | {c} | {v['n']} | {v['pass_rate']:.3f} |")
    L += ["", f"Agent: {a['attempts_mean']:.2f} attempts per prompt on average (the first is the base image); "
              f"spent {a['spent_mean']:.1f} of {a['budget_mean']:.1f} GPU-seconds on average; "
              f"{a['over_budget']} prompts over budget.",
          f"Stops: {a['stop_reasons']}. Actions taken: {a['actions']}. The final image came from: {a['improved_by']} "
          "(prompts where it is not the base image).", "",
          f"Routed best-of-k used k = {a['routed_k']} (category:k: prompts).",
          "Costs (median seconds per image): " + json.dumps(res["costs"]) + ".",
          "Settings: " + json.dumps(res["settings"]) + "."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--run", required=True)
    i.add_argument("--split", choices=["dev", "test"], required=True)
    i.add_argument("--policy", choices=["rule", "llm"], default="rule")
    i.add_argument("--limit", type=int, default=None)
    for name in ("plan", "ingest"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--round", type=int, required=True)
    r = sub.add_parser("report")
    r.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "init":
        print(json.dumps(init(args.run, args.split, args.policy, limit=args.limit), indent=1))
    elif args.cmd == "plan":
        n = plan(args.run, args.round)
        print(f"round {args.round}: {n} attempts planned")
        atomic_write_text(run_dir(args.run) / f"round{args.round}" / "n_requests", f"{n}\n")
    elif args.cmd == "ingest":
        print(f"round {args.round}: " + json.dumps(ingest(args.run, args.round)))
    else:
        print(render(report(args.run)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
