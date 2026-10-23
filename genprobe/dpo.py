"""Diffusion-DPO on SD 1.5 with VQAScore preferences (guide §11.4, D-037), and the checks for reward
hacking. The training script is diffusers' own research script (train_diffusion_dpo.py at the v0.41.0
tag, fetched by scripts/setup_envs.sh train); this module prepares its data and evaluates its LoRA.

    python -m genprobe.dpo prompts [--smoke]       -> data/prompts/dpo_train[_smoke].jsonl
        GenAI-Bench-1600 prompts outside prompt set v1 (1,200; smoke: the first 20)
    (SD 1.5 seeds 0-3 for them: genprobe.generate --out data/dpo/cand; VQAScore: genprobe.score --out data/dpo/cand)
    python -m genprobe.dpo pairs [--smoke]         -> data/dpo/pairs[_smoke]/train-00000-of-00001.parquet
        per prompt the highest- vs the lowest-VQAScore image, kept when they differ by >= 0.1
    python -m genprobe.dpo generate --arm <name> [--lora <dir>] --split dev|test --seeds 0 [1 2 3] [--limit n]
        SD 1.5 at prompt set v1's settings, with or without the LoRA, on prompt set v1's dev or test prompts
        -> data/dpo/eval/images/<arm>/, data/dpo/eval/manifests/gen_<arm>_*.jsonl
    python -m genprobe.dpo images --arms a b --split dev --seed 0 --out f.jsonl   (input for genprobe.judge grade)
    python -m genprobe.dpo select --grades <tag> --arms base s250 s500 ...         (dev: which checkpoint)
    python -m genprobe.dpo analyze --grades <tag> --base <arm> --dpo <arm>           (test: H6 and the checks)
        -> results/dpo/summary.md, summary.json
"""

from __future__ import annotations

import argparse
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

V1 = "data/prompts/prompts_v1.jsonl"
GENAI_1600 = "data/prompts/genai_bench_1600.jsonl"
SD15 = "configs/gen/sd15.yaml"
ROOT = Path("data/dpo")
MIN_GAP = 0.1                  # D-037: a pair is used only if the two VQAScores differ by at least this
SMOKE_N = 20
PASS_SCORE = 5
N_BOOT = 2000
METRICS = ("vqascore", "clipscore", "pickscore", "hpsv2")
CHECKERS = ("detect", "ocr")


def train_prompts_path(smoke: bool) -> str:
    return "data/prompts/dpo_train_smoke.jsonl" if smoke else "data/prompts/dpo_train.jsonl"


def pairs_dir(smoke: bool) -> Path:
    return ROOT / ("pairs_smoke" if smoke else "pairs")


def write_train_prompts(smoke: bool = False, source: str = GENAI_1600, v1: str = V1) -> dict:
    """Every GenAI-Bench-1600 prompt that is not in prompt set v1 (dev or test), so no evaluation
    prompt is ever trained on."""
    used = {p.prompt_id for p in load_prompts(v1)}
    rows = [json.loads(x) for x in Path(source).read_text(encoding="utf-8").splitlines() if x.strip()]
    keep = sorted((r for r in rows if r["prompt_id"] not in used), key=lambda r: r["prompt_id"])
    if smoke:
        keep = keep[:SMOKE_N]
    keep = [{**r, "split": "train"} for r in keep]
    out = train_prompts_path(smoke)
    atomic_write_text(Path(out), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    return {"prompts": len(keep), "excluded_v1": len(used & {r["prompt_id"] for r in rows}), "file": out}


def make_pairs(images: dict[str, list[tuple[str, float, str]]], min_gap: float = MIN_GAP) -> tuple[list[dict], dict]:
    """images: prompt_id -> [(sample_id, vqascore, path)]. Winner = highest VQAScore, loser = lowest
    (ties: the lower sample_id); a prompt yields a pair only if they differ by >= min_gap."""
    pairs, gaps, dropped = [], [], 0
    for pid in sorted(images):
        ims = sorted(images[pid], key=lambda t: t[0])
        if len(ims) < 2:
            dropped += 1
            continue
        hi = max(ims, key=lambda t: t[1])
        lo = min(ims, key=lambda t: t[1])
        gap = hi[1] - lo[1]
        if gap < min_gap:
            dropped += 1
            continue
        gaps.append(gap)
        pairs.append({"prompt_id": pid, "win": hi[0], "lose": lo[0], "win_path": hi[2], "lose_path": lo[2],
                      "win_vqa": hi[1], "lose_vqa": lo[1]})
    stats = {"prompts": len(images), "pairs": len(pairs), "dropped": dropped,
             "mean_gap": float(np.mean(gaps)) if gaps else math.nan}
    return pairs, stats


def build_pairs(smoke: bool = False, root: Path = ROOT) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    prompts = {p.prompt_id: p for p in load_prompts(train_prompts_path(smoke))}
    man = latest_by_id(read_many(sorted((root / "cand" / "manifests").glob("gen_sd15_*.jsonl"))))
    gen = {s: r for s, r in man.items() if r.get("status") == "ok" and r["prompt_id"] in prompts}
    ok, version, _ = pick_scores(read_many(sorted((root / "cand" / "scores" / "vqascore").glob("*.jsonl"))), set(gen))
    images: dict[str, list] = defaultdict(list)
    for s, r in gen.items():
        if s in ok:
            images[r["prompt_id"]].append((s, float(ok[s]["value"]), r["path"]))
    pairs, stats = make_pairs(images)
    table = pa.table({"jpg_0": [Path(p["win_path"]).read_bytes() for p in pairs],
                      "jpg_1": [Path(p["lose_path"]).read_bytes() for p in pairs],
                      "label_0": [1] * len(pairs),
                      "caption": [prompts[p["prompt_id"]].prompt for p in pairs]})
    out = pairs_dir(smoke)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "train-00000-of-00001.parquet.tmp"
    pq.write_table(table, tmp)
    tmp.replace(out / "train-00000-of-00001.parquet")
    meta = {"created_at": now_iso(), "git_commit": git_commit(), "vqascore_version": version, "min_gap": MIN_GAP,
            "candidates": len(gen), "scored": len(ok), **stats,
            "pairs_list": [{k: p[k] for k in ("prompt_id", "win", "lose", "win_vqa", "lose_vqa")} for p in pairs]}
    atomic_write_text(root / f"pairs{'_smoke' if smoke else ''}.meta.json", json.dumps(meta, indent=1))
    return {k: v for k, v in meta.items() if k != "pairs_list"}


# ---------- evaluation images ----------

def eval_prompts(split: str, limit: int | None = None) -> list:
    """Prompt set v1's dev or test prompts; with limit, the first n of each category (smoke)."""
    ps = load_prompts(V1, split=split)
    if limit is None:
        return ps
    out, seen = [], defaultdict(int)
    for p in ps:
        c = core.category_of(p.tags)
        if seen[c] < limit:
            out.append(p)
            seen[c] += 1
    return out


def generate(arms: list[tuple[str, str | None]], split: str, seeds: list[int], limit: int | None = None,
             root: Path = ROOT, log=print) -> dict:
    """arms: (name, LoRA dir or None). One pipeline, LoRA loaded and unloaded per arm, the generation
    path of prompt set v1 (genprobe.generate) with the arm's name as the model name."""
    from .backends import DiffusersBackend
    from .config import load_gen_config
    from .generate import run_generation

    base_cfg = load_gen_config(SD15)
    prompts = eval_prompts(split, limit)
    backend = None
    counts = {}
    for name, lora in arms:
        cfg = base_cfg.model_copy(update={"name": name, "notes": f"{base_cfg.notes} LoRA: {lora or 'none'} (D-037)"})
        if backend is None:
            backend = DiffusersBackend(cfg)
        backend.cfg = cfg
        if lora:
            backend.pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors", adapter_name=name)
        counts[name] = run_generation(cfg=cfg, prompts=prompts, out=root / "eval", seeds=seeds, backend=backend, log=log)
        if lora:
            backend.pipe.unload_lora_weights()
    return counts


def eval_images(arms: list[str], split: str, seed: int | None = None, limit: int | None = None, root: Path = ROOT) -> list[dict]:
    want = {p.prompt_id: p for p in eval_prompts(split, limit)}
    man = latest_by_id(read_many(sorted((root / "eval" / "manifests").glob("gen_*.jsonl"))))
    return [{"sample_id": s, "prompt_id": r["prompt_id"], "prompt": r["prompt"], "path": r["path"], "group": r["model"],
             "seed": r["seed"], "category": core.category_of(want[r["prompt_id"]].tags)}
            for s, r in sorted(man.items()) if r.get("status") == "ok" and r["model"] in arms
            and r["prompt_id"] in want and (seed is None or r["seed"] == seed)]


def _ci(per_prompt: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    pids = sorted(per_prompt)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per_prompt[p] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def judge_pass(grades: dict, images: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for im in images:
        g = grades.get(im["sample_id"])
        out[im["group"]][im["prompt_id"]] = float(bool(g) and g.get("score") == PASS_SCORE)
    return out


def select(grades_tag: str, arms: list[str], limit: int | None = None, data: str = "data", root: Path = ROOT,
           results: Path = Path("results/dpo")) -> dict:
    """Dev: the checkpoint arm with the highest judge pass rate on seed 0; ties go to fewer steps
    (arms are given in step order). 'base' is shown for reference and is not selectable."""
    grades = latest_by_id(read_many([Path(data, "judge", f"{grades_tag}.jsonl")]))
    images = eval_images(arms, "dev", seed=0, limit=limit, root=root)
    jp = judge_pass(grades, images)
    rates = {a: _ci(jp.get(a, {})) for a in arms}
    ckpts = [a for a in arms if a != "base" and not a.endswith("base")]
    best = max(ckpts, key=lambda a: (rates[a]["value"], -ckpts.index(a)))
    res = {"created_at": now_iso(), "git_commit": git_commit(), "grades": grades_tag, "dev_prompts": len(jp.get(best, {})),
           "pass_rate": rates, "selected": best}
    atomic_write_text(root / "selected.json", json.dumps(res, indent=2))
    results.mkdir(parents=True, exist_ok=True)
    L = ["# DPO checkpoint chosen on the dev prompts (D-037)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.dpo select`. Judge pass rate "
         f"(score 5) on seed 0 of {res['dev_prompts']} dev prompts, grades `{grades_tag}`; highest wins, ties go to fewer steps.",
         "", "| Arm | Pass rate [95 % CI] |", "| --- | --- |"]
    L += [f"| {a} | {rates[a]['value']:.3f} [{rates[a]['ci'][0]:.2f}, {rates[a]['ci'][1]:.2f}] |" for a in arms]
    L += ["", f"Selected: **{best}**."]
    atomic_write_text(results / f"dev_selection{'_smoke' if limit else ''}.md", "\n".join(L) + "\n")
    return res


def analyze(grades_tag: str, base: str, dpo: str, limit: int | None = None, data: str = "data", root: Path = ROOT,
            results: Path = Path("results/dpo"), n_boot: int = N_BOOT) -> dict:
    """Test: H6 (judge pass rate on seed 0, DPO - base > 0) and the reward-hacking checks: the metric
    DPO was trained on (VQAScore) against the judge, preference scores, the checkers."""
    grades = latest_by_id(read_many([Path(data, "judge", f"{grades_tag}.jsonl")]))
    seed0 = eval_images([base, dpo], "test", seed=0, limit=limit, root=root)
    allimg = eval_images([base, dpo], "test", limit=limit, root=root)
    cat = {im["prompt_id"]: im["category"] for im in allimg}
    jp = judge_pass(grades, seed0)
    expected = defaultdict(dict)
    for im in seed0:
        g = grades.get(im["sample_id"]) or {}
        if g.get("expected_score") is not None:
            expected[im["group"]][im["prompt_id"]] = float(g["expected_score"])
    cats = ["all"] + sorted(set(cat.values()))
    sub = lambda d, c: {p: v for p, v in d.items() if c == "all" or cat[p] == c}  # noqa: E731
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "grades": grades_tag, "base": base,
                           "dpo": dpo, "prompts": len(cat), "pass_rate": {}, "difference": {}, "metrics": {}}
    for arm in (base, dpo):
        res["pass_rate"][arm] = {c: _ci(sub(jp.get(arm, {}), c), n_boot) for c in cats}
    for c in cats:
        common = [p for p in sub(jp.get(dpo, {}), c) if p in jp.get(base, {})]
        res["difference"][c] = _ci({p: jp[dpo][p] - jp[base][p] for p in common}, n_boot)
    lo = res["difference"]["all"]["ci"][0]
    hi = res["difference"]["all"]["ci"][1]
    res["H6"] = {"comparison": f"judge pass rate, {dpo} - {base}, seed 0, all test prompts",
                 "verdict": "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive",
                 **res["difference"]["all"]}
    if expected:
        common = [p for p in expected.get(dpo, {}) if p in expected.get(base, {})]
        res["expected_score_difference"] = _ci({p: expected[dpo][p] - expected[base][p] for p in common}, n_boot)
    ids = {im["sample_id"]: im for im in allimg}
    for m in METRICS + CHECKERS:
        # GPU metrics are scored into data/dpo/eval/scores, the CPU checkers (scripts/submit_scoring.sh)
        # into data/scores; pick_scores keeps only these images and one metric version.
        recs = read_many(sorted(Path(data, "dpo", "eval", "scores", m).glob("*.jsonl")) +
                         sorted(Path(data, "scores", m).glob("*.jsonl")))
        ok, version, _ = pick_scores(recs, set(ids))
        per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for s, r in ok.items():
            per[ids[s]["group"]][ids[s]["prompt_id"]].append(float(r["value"]))
        means = {arm: {p: float(np.mean(v)) for p, v in per[arm].items()} for arm in (base, dpo)}
        common = [p for p in means[dpo] if p in means[base]]
        res["metrics"][m] = {"version": version, base: _ci(means[base], n_boot), dpo: _ci(means[dpo], n_boot),
                             "difference": _ci({p: means[dpo][p] - means[base][p] for p in common}, n_boot)}
    results.mkdir(parents=True, exist_ok=True)
    name = "summary_smoke" if limit else "summary"
    atomic_write_text(results / f"{name}.json", json.dumps(res, indent=2))
    atomic_write_text(results / f"{name}.md", render(res))
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _c(d: dict, nd=3) -> str:
    return f"{_f(d['value'], nd)} [{_f(d['ci'][0], 2)}, {_f(d['ci'][1], 2)}]"


def render(res: dict) -> str:
    b, d = res["base"], res["dpo"]
    cats = list(res["difference"])
    L = ["# Diffusion-DPO on SD 1.5 with VQAScore preferences (guide §11.4, D-037)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.dpo analyze`. Test prompts "
         f"of prompt set v1 ({res['prompts']}); judge Gemma 4 31B (grades `{res['grades']}`), pass = score 5, seed 0; "
         "automatic metrics averaged over seeds 0-3 per prompt; 95 % CIs from resampling prompts.", "",
         f"**H6 (pre-registered): DPO raises the judge pass rate.** {d} − {b} = {_c(res['H6'])}: "
         f"**{res['H6']['verdict']}**.", "",
         "| Judge pass rate | " + " | ".join(cats) + " |", "| --- |" + " --- |" * len(cats)]
    for arm in (b, d):
        L.append(f"| {arm} | " + " | ".join(_c(res["pass_rate"][arm][c]) for c in cats) + " |")
    L.append(f"| {d} − {b} | " + " | ".join(_c(res["difference"][c]) for c in cats) + " |")
    if res.get("expected_score_difference"):
        L += ["", f"Judge expected score, {d} − {b}: {_c(res['expected_score_difference'])}."]
    L += ["", "## Reward-hacking checks: the trained-on metric against the others", "",
          f"| Metric | {b} | {d} | {d} − {b} |", "| --- | --- | --- | --- |"]
    for m, r in res["metrics"].items():
        if r[b]["n"]:
            L.append(f"| {m}{' (trained on)' if m == 'vqascore' else ''} | {_c(r[b])} | {_c(r[d])} | {_c(r['difference'])} |")
    L += ["", "Detector and OCR apply to counting / spatial and text prompts only."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("pairs")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("generate")
    p.add_argument("--arm", action="append", required=True, help="name[=lora_dir]; repeat")
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--limit", type=int, default=None, help="first n prompts of each category (smoke)")
    p = sub.add_parser("images")
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--grades", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--limit", type=int, default=None)
    p = sub.add_parser("analyze")
    p.add_argument("--grades", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--dpo", required=True)
    p.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "prompts":
        out = write_train_prompts(a.smoke)
    elif a.cmd == "pairs":
        out = build_pairs(a.smoke)
    elif a.cmd == "generate":
        arms = [(x.split("=", 1)[0], x.split("=", 1)[1] if "=" in x else None) for x in a.arm]
        out = generate(arms, a.split, a.seeds, a.limit)
    elif a.cmd == "images":
        ims = eval_images(a.arms, a.split, a.seed, a.limit)
        atomic_write_text(Path(a.out), "".join(json.dumps(x) + "\n" for x in ims))
        out = {"images": len(ims), "file": a.out}
    elif a.cmd == "select":
        out = select(a.grades, a.arms, a.limit)
    else:
        out = analyze(a.grades, a.base, a.dpo, a.limit)
        out = {"H6": out["H6"]}
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
