"""Text-to-video with Wan2.2 TI2V-5B (guide §11.2, D-036), in the `video` env on an H100.

    python -m genprobe.video_gen select            -> data/prompts/wan_video.jsonl
        160 of GenAI-Bench-Video's 800 prompts, drawn at random (seed 0) within its basic and
        advanced tiers in proportion; the first 3 (by prompt_id) are the smoke prompts.
    python -m genprobe.video_gen generate --frames 121 [--limit 3] [--shard i --num-shards n]
        -> data/video/wan22_f<frames>/<prompt_id>_s0.mp4 and
           data/video/manifests/gen_wan22_f<frames>_shard<i>of<n>.jsonl
           (sample_id wan22_f<frames>__<prompt_id>__s0; resumable: finished videos are skipped)

Settings are the model card's: 1280 x 704, 50 steps, cfg 5.0, pipeline in bf16 with the VAE in
fp32, 24 fps; only the frame count varies (D-036 picks it from the smoke run). Timings are GPU
seconds per video measured around the pipeline call (model loading excluded).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from .samples import make_sample_id
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import ManifestWriter, latest_by_id, read_many

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"    # downloaded 2026-10-23 (D-036)
HEIGHT, WIDTH, STEPS, CFG, FPS, SEED = 704, 1280, 50, 5.0, 24, 0
N_PROMPTS = 160
SOURCE_PROMPTS = "data/prompts/genai_bench_video.jsonl"
PROMPTS = "data/prompts/wan_video.jsonl"
OUT = Path("data/video")


def model_name(frames: int) -> str:
    return f"wan22_f{frames}"


def select(source: str = SOURCE_PROMPTS, out: str = PROMPTS, n: int = N_PROMPTS, seed: int = 0) -> dict:
    rows = [json.loads(x) for x in Path(source).read_text(encoding="utf-8").splitlines() if x.strip()]
    tiers = defaultdict(list)
    for r in rows:
        tiers[r["tags"][0]].append(r)
    total = len(rows)
    alloc = {t: math.floor(n * len(v) / total) for t, v in tiers.items()}
    for t in sorted(tiers, key=lambda t: (-(n * len(tiers[t]) / total - alloc[t]), t))[: n - sum(alloc.values())]:
        alloc[t] += 1
    rng = random.Random(seed)
    chosen = []
    for t in sorted(tiers):
        chosen += rng.sample(sorted(tiers[t], key=lambda r: r["prompt_id"]), alloc[t])
    chosen.sort(key=lambda r: r["prompt_id"])
    atomic_write_text(Path(out), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen))
    return {"prompts": len(chosen), "by_tier": {t: alloc[t] for t in sorted(alloc)}, "smoke": [r["prompt_id"] for r in chosen[:3]]}


def generate(frames: int, prompts: str = PROMPTS, limit: int | None = None, shard: int = 0, num_shards: int = 1,
             out: Path = OUT, log=print) -> dict:
    import diffusers
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.utils import export_to_video

    if (frames - 1) % 4:
        raise ValueError(f"Wan needs 4k + 1 frames, got {frames}")
    rows = [json.loads(x) for x in Path(prompts).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = sorted(rows, key=lambda r: r["prompt_id"])[:limit]
    mine = rows[shard::num_shards]
    name = model_name(frames)
    mdir = out / "manifests"
    done = {s for s, r in latest_by_id(read_many(sorted(mdir.glob(f"gen_{name}_*.jsonl")))).items()
            if r.get("status") == "ok" and Path(r["path"]).is_file()} if mdir.exists() else set()
    todo = [r for r in mine if make_sample_id(name, r["prompt_id"], SEED) not in done]
    log(f"[{name}] shard {shard}/{num_shards}: {len(mine)} prompts, {len(mine) - len(todo)} done, {len(todo)} to generate")
    if not todo:
        return {"generated": 0, "skipped": len(mine)}
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible; video generation runs inside a GPU job")
    t0 = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32, revision=REVISION,
                                           local_files_only=True)
    pipe = WanPipeline.from_pretrained(REPO, vae=vae, torch_dtype=torch.bfloat16, revision=REVISION,
                                       local_files_only=True).to("cuda")
    load_s = time.perf_counter() - t0
    gpu = torch.cuda.get_device_name(0)
    commit = git_commit()
    vdir = out / name
    vdir.mkdir(parents=True, exist_ok=True)
    counts = {"ok": 0, "error": 0}
    with ManifestWriter(mdir / f"gen_{name}_shard{shard:03d}of{num_shards:03d}.jsonl") as w:
        for r in todo:
            sid = make_sample_id(name, r["prompt_id"], SEED)
            path = vdir / f"{r['prompt_id']}_s{SEED}.mp4"
            rec = {"sample_id": sid, "model": name, "prompt_id": r["prompt_id"], "prompt": r["prompt"], "seed": SEED,
                   "path": str(path), "frames": frames, "height": HEIGHT, "width": WIDTH, "steps": STEPS, "cfg": CFG,
                   "fps": FPS, "repo": REPO, "revision": REVISION, "diffusers": diffusers.__version__,
                   "torch": torch.__version__, "gpu": gpu, "git_commit": commit, "load_seconds": round(load_s, 1)}
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t = time.perf_counter()
                video = pipe(prompt=r["prompt"], height=HEIGHT, width=WIDTH, num_frames=frames,
                             num_inference_steps=STEPS, guidance_scale=CFG,
                             generator=torch.Generator("cuda").manual_seed(SEED)).frames[0]
                torch.cuda.synchronize()
                gen_s = time.perf_counter() - t
                tmp = path.with_suffix(".tmp.mp4")
                export_to_video(video, str(tmp), fps=FPS)
                tmp.replace(path)
                rec.update(status="ok", gen_seconds=round(gen_s, 2),
                           peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
                counts["ok"] += 1
                log(f"[{name}] {sid}: {gen_s:.1f} s")
            except Exception as e:  # noqa: BLE001
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:2000])
                counts["error"] += 1
            rec["created_at"] = now_iso()
            w.write(rec)
    log(f"[{name}] shard {shard}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


def timing(out: Path = OUT, results: Path = Path("results/video_gen_smoke")) -> dict:
    """Seconds per video by frame count, from every Wan manifest (the smoke run's D-036 choice)."""
    recs = [r for r in latest_by_id(read_many(sorted((out / "manifests").glob("gen_wan22_*.jsonl")))).values()
            if r.get("status") == "ok"]
    by = defaultdict(list)
    for r in recs:
        by[r["frames"]].append(r)
    res = {"created_at": now_iso(), "git_commit": git_commit(),
           "by_frames": {f: {"videos": len(v), "mean_gen_seconds": sum(x["gen_seconds"] for x in v) / len(v),
                             "max_gen_seconds": max(x["gen_seconds"] for x in v),
                             "peak_mem_gb": max(x.get("peak_mem_gb", 0) for x in v), "gpu": v[0]["gpu"],
                             "projected_gpu_hours_160": sum(x["gen_seconds"] for x in v) / len(v) * N_PROMPTS / 3600}
                         for f, v in sorted(by.items())}}
    results.mkdir(parents=True, exist_ok=True)
    atomic_write_text(results / "timing.json", json.dumps(res, indent=2))
    L = ["# Wan2.2 TI2V-5B smoke run: seconds per video (D-036)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.video_gen timing`. "
         f"{WIDTH} x {HEIGHT}, {STEPS} steps, cfg {CFG}; generation only, model loading excluded.", "",
         "| Frames | Videos | Mean s / video | Max s / video | Peak GPU memory (GB) | Projected GPU-hours for 160 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for f, v in res["by_frames"].items():
        L.append(f"| {f} | {v['videos']} | {v['mean_gen_seconds']:.1f} | {v['max_gen_seconds']:.1f} | "
                 f"{v['peak_mem_gb']} | {v['projected_gpu_hours_160']:.2f} |")
    atomic_write_text(results / "timing.md", "\n".join(L) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    g = sub.add_parser("generate")
    g.add_argument("--frames", type=int, required=True)
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--num-shards", type=int, default=1)
    sub.add_parser("timing")
    args = ap.parse_args(argv)
    if args.cmd == "select":
        print(json.dumps(select(), indent=2))
    elif args.cmd == "generate":
        res = generate(args.frames, limit=args.limit, shard=args.shard, num_shards=args.num_shards)
        print(json.dumps(res))
        return 1 if res.get("error") else 0
    else:
        print(json.dumps(timing(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
