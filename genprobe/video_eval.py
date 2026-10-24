"""Evaluation of the generated videos (guide §11.3, D-036): VBench's six custom-input dimensions
and VQAScore on the whole video, for Wan2.2 TI2V-5B and for GenAI-Bench-Video's four older models
on the same prompts (data/prompts/wan_video.jsonl).

    python -m genprobe.video_eval vbench --model wan22_f121     (vbench env, GPU job: slurm/vbench.sbatch)
    python -m genprobe.video_eval vbench --model Floor33         ... and Gen2, Pika_v1, Modelscope
        -> data/video/vbench/<model>.jsonl     one line per video: prompt_id and the six dimensions
    python -m genprobe.video_eval analyze --wan wan22_f121      (eval env, CPU)
        -> results/video/summary.md, summary.json

VBench runs from the pinned source (scripts/download_vbench.sh) in local mode, with its weights in
$VBENCH_CACHE_DIR; each model's videos are linked into one folder as <prompt_id>.mp4.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import genai_bench_video as gv
from .samples import make_sample_id
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

DIMENSIONS = ("subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree",
              "aesthetic_quality", "imaging_quality")
OUT = Path("data/video")
PROMPTS = "data/prompts/wan_video.jsonl"
N_BOOT = 2000


def wan_prompts(path: str = PROMPTS) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def video_paths(model: str, prompts: list[dict], out: Path = OUT) -> dict[str, str]:
    """prompt_id -> video file, for a Wan run (data/video/manifests) or an older model (GenAI-Bench-Video)."""
    if model.startswith("wan22"):
        recs = latest_by_id(read_many(sorted((out / "manifests").glob(f"gen_{model}_*.jsonl"))))
    else:
        recs = latest_by_id(read_many([gv.MANIFEST]))
    want = {p["prompt_id"] for p in prompts}
    return {r["prompt_id"]: r["path"] for r in recs.values()
            if r.get("model") == model and r.get("status") == "ok" and r["prompt_id"] in want}


def vbench(model: str, prompts_path: str = PROMPTS, out: Path = OUT, limit: int | None = None) -> dict:
    import torch
    from vbench import VBench

    prompts = sorted(wan_prompts(prompts_path), key=lambda p: p["prompt_id"])[:limit]
    paths = video_paths(model, prompts, out)
    if len(paths) < len(prompts):
        raise SystemExit(f"{model}: {len(prompts) - len(paths)} of {len(prompts)} videos missing")
    vin = out / "vbench_in" / model
    vin.mkdir(parents=True, exist_ok=True)
    for f in vin.glob("*.mp4"):
        f.unlink()
    for pid, p in paths.items():
        (vin / f"{pid}.mp4").symlink_to(Path(p).resolve())
    raw = out / "vbench" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    src = Path(os.environ.get("GENPROBE_ENVS", Path.home() / "envs")) / "vbench-src"
    bench = VBench(torch.device("cuda"), str(src / "vbench" / "VBench_full_info.json"), str(raw))
    name = f"{model}_{len(paths)}"
    bench.evaluate(videos_path=str(vin), name=name, dimension_list=list(DIMENSIONS), local=True,
                   read_frame=False, mode="custom_input")
    res = json.loads((raw / f"{name}_eval_results.json").read_text())
    per: dict[str, dict[str, Any]] = defaultdict(dict)
    for dim in DIMENSIONS:
        _, videos = res[dim]
        for v in videos:
            # VBench divides imaging quality (MUSIQ, 0-100) by 100 only in its mean; per video it is
            # left on 0-100. Put it on VBench's reported 0-1 scale here.
            scale = 100.0 if dim == "imaging_quality" else 1.0
            per[Path(v["video_path"]).stem][dim] = float(v["video_results"]) / scale
    rows = [{"model": model, "prompt_id": pid, "sample_id": make_sample_id(model, pid, 0), **per[pid]}
            for pid in sorted(per)]
    atomic_write_text(out / "vbench" / f"{model}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    return {"model": model, "videos": len(rows), "means": {d: sum(r.get(d, math.nan) for r in rows) / len(rows)
                                                         for d in DIMENSIONS}}


def _ci(per_prompt: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    import numpy as np
    pids = sorted(per_prompt)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per_prompt[p] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def analyze(wan: str, prompts_path: str = PROMPTS, out: Path = OUT, data: str = "data",
            results: Path = Path("results/video"), n_boot: int = N_BOOT) -> dict:
    from .meta_eval import load_metric_scores

    prompts = wan_prompts(prompts_path)
    models = [wan] + list(gv.MODELS)
    vq: dict[str, dict[str, float]] = {}
    versions = {}
    for m in models:
        ids = {make_sample_id(m, p["prompt_id"], 0): p["prompt_id"] for p in prompts}
        s, versions[m], _ = load_metric_scores(data, "vqascore_video", set(ids))
        vq[m] = {ids[k]: v for k, v in s.items()}
    vb: dict[str, dict[str, dict[str, float]]] = {}
    for m in models:
        f = out / "vbench" / f"{m}.jsonl"
        rows = [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
        vb[m] = {d: {r["prompt_id"]: r[d] for r in rows if d in r} for d in DIMENSIONS}
    measures = {"vqascore_video": vq, **{d: {m: vb[m][d] for m in models} for d in DIMENSIONS}}
    table = {k: {m: _ci(v[m], n_boot) for m in models} for k, v in measures.items()}
    diffs = {}
    for k, v in measures.items():
        for m in gv.MODELS:
            common = [p for p in v[wan] if p in v[m]]
            diffs[f"{k}|{wan} - {m}"] = _ci({p: v[wan][p] - v[m][p] for p in common}, n_boot)
    res = {"created_at": now_iso(), "git_commit": git_commit(), "wan": wan, "prompts": len(prompts),
           "models": models, "table": table, "paired_difference": diffs, "vqascore_versions": versions}
    results.mkdir(parents=True, exist_ok=True)
    atomic_write_text(results / "summary.json", json.dumps(res, indent=2))
    atomic_write_text(results / "summary.md", render(res))
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    models = res["models"]
    L = ["# Generated videos: Wan2.2 TI2V-5B and GenAI-Bench-Video's models on the same prompts", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.video_eval analyze` "
         f"(D-036). {res['prompts']} prompts; means over prompts with 95 % CIs from resampling prompts. Exploratory.", "",
         "| Measure | " + " | ".join(models) + " |", "| --- |" + " --- |" * len(models)]
    for k, row in res["table"].items():
        L.append(f"| {k} | " + " | ".join(f"{_f(row[m]['value'])} [{_f(row[m]['ci'][0], 2)}, {_f(row[m]['ci'][1], 2)}]"
                                         + ("" if row[m]["n"] == res["prompts"] else f" (n={row[m]['n']})")
                                         for m in models) + " |")
    L += ["", f"## {res['wan']} minus each older model (same prompts)", "",
          "| Measure | " + " | ".join(models[1:]) + " |", "| --- |" + " --- |" * (len(models) - 1)]
    for k in res["table"]:
        cells = [res["paired_difference"][f"{k}|{res['wan']} - {m}"] for m in models[1:]]
        L.append(f"| {k} | " + " | ".join(f"{_f(c['value'])} [{_f(c['ci'][0], 2)}, {_f(c['ci'][1], 2)}]" for c in cells) + " |")
    L += ["", "VQAScore on the whole video: D-035's metric (how far it agrees with people is in "
          "`results/video_meta_eval/`). The six VBench dimensions measure consistency, motion and image quality, "
          "not prompt following."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vbench")
    v.add_argument("--model", required=True)
    v.add_argument("--limit", type=int, default=None, help="first n prompts (smoke)")
    a = sub.add_parser("analyze")
    a.add_argument("--wan", required=True)
    args = ap.parse_args(argv)
    res = vbench(args.model, limit=args.limit) if args.cmd == "vbench" else analyze(args.wan)
    print(json.dumps(res if args.cmd == "vbench" else {"written": "results/video/summary.md"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
