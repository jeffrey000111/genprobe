"""Multi-shot videos from a short brief (guide §11.2's bonus, D-038). Gemma 4 31B splits each brief
into 3-4 shots once; four arms then make the shots with Wan2.2 TI2V-5B (49 frames, D-036's settings),
and Gemma 4 31B grades them on frames.

    A  text to video              Wan text-to-video from the shot description, seed 0
    B  keyframe, image to video   FLUX.1 schnell keyframe (seed 0), Wan image-to-video from it, seed 0
    C  agent                      B with checks: keyframe seeds 0, 1, 2 until VQAScore >= 0.9 (else the
                                  highest); video seeds 0, 1, 2 until VQAScore on the whole video >= tau_v
                                  (else the highest); only failing shots are redone
    D  re-ranking                 B's keyframe, video seeds 0 and 1, the higher VQAScore on video kept

Attempts that two arms share (same inputs and seed) are made once. NAME is smoke (the first dev
brief), dev (8 briefs, arm B only: it sets tau_v) or test (24 briefs). slurm/story.sbatch runs:

    python -m genprobe.story plan --name N                    (judge env)  -> data/story/shots.jsonl
    python -m genprobe.story keyframes --name N --round r     (gen env)    -> data/story/kf/
    python -m genprobe.story videos --name N --round r [--shard i --num-shards n]   (video env) -> data/story/video/
    (VQAScore on keyframes and on videos: genprobe.score, eval env)
    python -m genprobe.story judge --name N                   (judge env)  -> data/judge/story_<N>_<run>.jsonl
    python -m genprobe.story tune                             (after dev)  -> data/story/tau_v.json, results/story/dev_tau.md
    python -m genprobe.story analyze --name N                              -> results/story[_smoke]/summary.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .samples import Prompt, make_sample_id
from .summarize import pick_scores
from .utils.env import git_commit, now_iso, run_id
from .utils.io import atomic_write_text
from .utils.manifest import ManifestWriter, latest_by_id, read_many

BRIEFS = "data/prompts/story_briefs.jsonl"
SHOTS = "data/story/shots.jsonl"      # made on HPC3 by `plan`; a copy goes to results/story/ (never pushed over)
ROOT = Path("data/story")
JUDGE_DIR = Path("data/judge")
FLUX_CONFIG = "configs/gen/flux_schnell.yaml"
KF_MODEL, T2V = "fluxkf", "wanT2V"
FRAMES = 49                     # D-036
TAU_K = 0.9                     # VQAScore a keyframe needs: module C's threshold (D-031 rules r2)
TAU_GRID = (0.5, 0.6, 0.7, 0.8, 0.9)
SMOKE_TAU_V = 0.9               # smoke only (no dev result yet): high, so that the redo rounds run
TRIES = 3                       # keyframes and videos per shot for the agent
D_SEEDS = (0, 1)
PASS = 5
ARMS = {"smoke": "ABCD", "dev": "B", "test": "ABCD"}
N_BOOT, LEVEL = 2000, 0.975     # two pre-registered tests (H7, H8)
JUDGE = "gemma4_31b"


def i2v(k: int) -> str:
    return f"wanI2V_k{k}"


def frame_index(n: int = FRAMES) -> tuple[int, int, int]:
    return 0, (n - 1) // 2, n - 1


# ---------- briefs and the shot plan ----------

def load_rows(path: str | Path) -> list[dict]:
    p = Path(path)
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def briefs(name: str, path: str | None = None) -> list[dict]:
    rows = sorted(load_rows(path or BRIEFS), key=lambda r: r["prompt_id"])
    if name == "smoke":
        return [r for r in rows if r["split"] == "dev"][:1]
    if name in ("dev", "test"):
        return [r for r in rows if r["split"] == name]
    raise ValueError(f"name must be smoke, dev or test, not {name!r}")


def shot_rows(name: str, shots: str | None = None) -> list[dict]:
    want = {b["prompt_id"] for b in briefs(name)}
    return sorted((r for r in load_rows(shots or SHOTS) if r["brief_id"] in want), key=lambda r: (r["brief_id"], r["shot"]))


PLAN_TEMPLATE = (
    "You are planning a short video that is made of separate shots, each about two seconds long.\n\n"
    "Brief: {prompt}\n\n"
    "Split the brief into 3 or 4 shots that tell it in order. Each shot is generated on its own by a video "
    "model that never sees the other shots, so every shot description must stand alone: describe the main "
    "character with the same appearance words every time (what it is, its colours, clothing or markings), "
    "say where the shot takes place, and give the one action that happens in it. Keep to what the brief "
    "says: no new characters or events. Write each description as one or two plain sentences of 25 to 60 "
    "words, without camera terms.\n\n"
    'Reply with JSON only: {{"character": the main character\'s appearance in a short phrase, '
    '"shots": [{{"description": the first shot}}, ...]}}'
)
PLAN_SCHEMA = {"type": "object", "properties": {
    "character": {"type": "string"},
    "shots": {"type": "array", "minItems": 3, "maxItems": 4, "items": {
        "type": "object", "properties": {"description": {"type": "string"}},
        "required": ["description"], "additionalProperties": False}}},
    "required": ["character", "shots"], "additionalProperties": False}


def _judge_text():
    from . import judge as J
    rules = J.RULES.replace("see it in the image", "see it in the frames").replace(
        "unless the prompt asks", "unless the description asks")
    scale = ("5 = every requirement is met\n4 = all the main requirements are met; one minor detail is off\n"
             "3 = about half of the requirements are met\n2 = only a few requirements are met\n"
             "1 = nothing or almost nothing in the description is shown\n\n")
    reply = ('Reply with JSON only: {{"seen": what the frames show, "issues": [one short phrase per requirement '
             'that is not met; nothing that is fine], "score": 1-5}}')
    shot = ("You are checking whether an AI-generated video clip matches its text description. You see three "
            "frames of the clip in order: the first, the middle and the last.\n\nDescription: {prompt}\n\n"
            "First describe in one or two sentences what the clip actually shows, as if you had not read the "
            "description. Then check every requirement of the description against what you described: "
            + J.CHECKS + "; an action is met if the frames show it happening or done. " + rules
            + "\n\nThen score the match:\n" + scale + reply)
    story = ("You are checking whether a short AI-generated video tells its brief. The video is made of {n} "
             "shots; you see one frame from the middle of each shot, in order.\n\nBrief: {prompt}\n\n"
             "First describe in one or two sentences what the frames show, in order, as if you had not read "
             "the brief. Then check the brief against what you described: every event happens, in the brief's "
             "order, to the right main character in the right setting. A requirement is met only if you can "
             "actually see it in the frames; anything you cannot see counts as not met. Ignore image quality "
             "and artistic style.\n\nThen score:\n5 = every event of the brief is shown, in order\n"
             "4 = all the main events are shown; one minor detail is off\n3 = about half of the events are "
             "shown\n2 = only a few of the events are shown\n1 = nothing or almost nothing of the brief is "
             "shown\n\n" + reply)
    consist = ("You see one frame from each of the {n} shots of a short AI-generated video, in order. Its "
               "main character should be: {prompt}\n\nFirst describe in one or two sentences how the main "
               "character looks in each frame. Then judge whether it is the same individual in every frame: "
               "the same kind of character with the same colours, clothing, markings and other distinguishing "
               "features. Differences in pose, size in the frame, lighting or viewing angle are fine. A frame "
               "where the main character cannot be seen counts as a difference.\n\nThen score:\n"
               "5 = clearly the same individual in every frame\n4 = the same individual, with one small "
               "difference\n3 = similar, but clearly different in one frame\n2 = different in several frames\n"
               "1 = not the same individual in any two frames, or no main character\n\n"
               + reply.replace("per requirement that is not met; nothing that is fine", "per difference"))
    return shot, story, consist


def story_template_id() -> str:
    shot, story, consist = _judge_text()
    blob = json.dumps([PLAN_TEMPLATE, PLAN_SCHEMA, shot, story, consist, frame_index()], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def parse_plan(text: str) -> dict | None:
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    shots = [str(s.get("description", "")).strip() for s in d.get("shots", []) if isinstance(s, dict)]
    if not 3 <= len(shots) <= 4 or not all(shots) or not str(d.get("character", "")).strip():
        return None
    return {"character": str(d["character"]).strip(), "shots": shots}


def plan(name: str, shots: str | None = None, engine=None, log=print) -> dict:
    """Shot lists for the briefs of NAME that have none yet. A brief is planned once; every arm and
    every later stage reads the same plan."""
    shots = shots or SHOTS
    done = {r["brief_id"] for r in load_rows(shots)}
    todo = [b for b in briefs(name) if b["prompt_id"] not in done]
    log(f"[story] plan {name}: {len(todo)} brief(s) to plan, {len(done)} planned already")
    if not todo:
        return {"planned": 0}
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from .judge import VLLMJudge
    engine = engine or VLLMJudge(JUDGE, version_extra=f"/story-{story_template_id()}")
    params = SamplingParams(temperature=0.0, max_tokens=900, structured_outputs=StructuredOutputsParams(json=PLAN_SCHEMA))
    convs = [[{"role": "user", "content": [{"type": "text", "text": PLAN_TEMPLATE.format(prompt=json.dumps(
        b["prompt"], ensure_ascii=False))}]}] for b in todo]
    outs = engine.llm.chat(convs, params, **engine.chat_kwargs)
    rows, raw = [], []
    for b, o in zip(todo, outs):
        text = o.outputs[0].text
        p = parse_plan(text)
        raw.append({"brief_id": b["prompt_id"], "text": text, "parsed": p is not None, "judge_version": engine.version,
                    "created_at": now_iso()})
        if p is None:
            raise RuntimeError(f"{b['prompt_id']}: the plan did not parse: {text[:300]}")
        rows += [{"prompt_id": f"{b['prompt_id']}s{i}", "prompt": d, "brief_id": b["prompt_id"], "shot": i,
                  "split": b["split"], "character": p["character"], "source": "Gemma 4 31B shot plan (D-038)"}
                 for i, d in enumerate(p["shots"], 1)]
    Path(shots).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(shots).parent / "plans.jsonl", "a", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw))
    old = load_rows(shots)
    atomic_write_text(Path(shots), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in
                                           sorted(old + rows, key=lambda r: (r["brief_id"], r["shot"]))))
    return {"planned": len(todo), "shots": len(rows)}


# ---------- what each arm has, and what it needs next ----------

def load_scores(metric_dir: Path) -> dict[str, float]:
    recs = read_many(sorted(metric_dir.glob("*.jsonl"))) if metric_dir.exists() else []
    ok, _, _ = pick_scores(recs, {r.get("sample_id") for r in recs})
    return {s: float(r["value"]) for s, r in ok.items() if r.get("value") is not None}


def kf_scores(root: Path = ROOT) -> dict[str, float]:
    return load_scores(root / "kf" / "scores" / "vqascore")


def video_scores(root: Path = ROOT) -> dict[str, float]:
    return load_scores(root / "video" / "scores" / "vqascore_video")


def kf_state(shot: str, kf: dict[str, float], tau_k: float = TAU_K) -> dict:
    """The agent's keyframe search: seeds 0, 1, 2 until one reaches tau_k, else the highest.
    -> {"chosen": seed or None, "tried": [seeds], "next": seed to make or None}."""
    tried = []
    for k in range(TRIES):
        sid = make_sample_id(KF_MODEL, shot, k)
        if sid not in kf:
            return {"chosen": None, "tried": tried, "next": k}
        tried.append(k)
        if kf[sid] >= tau_k:
            return {"chosen": k, "tried": tried, "next": None}
    best = max(tried, key=lambda k: (kf[make_sample_id(KF_MODEL, shot, k)], -k))
    return {"chosen": best, "tried": tried, "next": None}


def c_video_state(shot: str, k: int, vid: dict[str, float], tau_v: float) -> dict:
    """The agent's video attempts from keyframe k: seeds 0, 1, 2 until one reaches tau_v, else the
    highest. -> {"final": sample_id or None, "tried": [sample_ids], "next": sample_id to make or None}."""
    tried = []
    for w in range(TRIES):
        sid = make_sample_id(i2v(k), shot, w)
        if sid not in vid:
            return {"final": None, "tried": tried, "next": sid}
        tried.append(sid)
        if vid[sid] >= tau_v:
            return {"final": sid, "tried": tried, "next": None}
    return {"final": max(tried, key=lambda s: (vid[s], -tried.index(s))), "tried": tried, "next": None}


def arm_state(arm: str, shot: str, kf: dict[str, float], vid: dict[str, float], tau_v: float | None) -> dict:
    """-> {"final": sample_id or None, "kf": keyframe sample_ids made, "videos": video sample_ids made,
    "next": video sample_ids still to make, "checks_kf": bool, "checks_video": bool}."""
    k0 = make_sample_id(KF_MODEL, shot, 0)
    if arm == "A":
        s = make_sample_id(T2V, shot, 0)
        return {"final": s if s in vid else None, "kf": [], "videos": [s], "next": [] if s in vid else [s],
                "checks_kf": False, "checks_video": False}
    if arm == "B":
        s = make_sample_id(i2v(0), shot, 0)
        return {"final": s if s in vid else None, "kf": [k0], "videos": [s], "next": [] if s in vid else [s],
                "checks_kf": False, "checks_video": False}
    if arm == "D":
        ss = [make_sample_id(i2v(0), shot, w) for w in D_SEEDS]
        miss = [s for s in ss if s not in vid]
        final = None if miss else max(ss, key=lambda s: (vid[s], -ss.index(s)))
        return {"final": final, "kf": [k0], "videos": ss, "next": miss, "checks_kf": False, "checks_video": True}
    if arm == "C":
        ks = kf_state(shot, kf)
        kfs = [make_sample_id(KF_MODEL, shot, k) for k in ks["tried"]]
        if ks["chosen"] is None:
            return {"final": None, "kf": kfs, "videos": [], "next": [], "keyframe_next": ks["next"],
                    "checks_kf": True, "checks_video": True}
        if tau_v is None:
            raise ValueError("the agent's video checks need tau_v")
        vs = c_video_state(shot, ks["chosen"], vid, tau_v)
        return {"final": vs["final"], "kf": kfs, "videos": vs["tried"], "next": [vs["next"]] if vs["next"] else [],
                "keyframe": ks["chosen"], "checks_kf": True, "checks_video": True}
    raise ValueError(arm)


def tau_v_for(name: str, root: Path = ROOT) -> float | None:
    if name == "smoke":
        return SMOKE_TAU_V
    if name == "dev":
        return None
    f = root / "tau_v.json"
    if not f.exists():
        raise SystemExit(f"{f} is missing: run the dev stage and `python -m genprobe.story tune` first")
    return float(json.loads(f.read_text())["tau_v"])


# ---------- keyframes (gen env) ----------

def keyframes(name: str, rnd: int, root: Path = ROOT, log=print) -> dict:
    """Round 0: keyframe seed 0 for every shot (B, C, D). Round r > 0: seed r for the shots whose agent
    keyframe search still needs it (needs the VQAScores of the earlier seeds)."""
    from .config import load_gen_config
    from .generate import run_generation

    rows = shot_rows(name)
    if rnd == 0:
        todo = rows
    elif "C" not in ARMS[name]:
        todo = []
    else:
        kf = kf_scores(root)
        todo = []
        for r in rows:
            st = kf_state(r["prompt_id"], kf)
            if st["chosen"] is None and st["next"] != rnd:
                raise SystemExit(f"{r['prompt_id']}: keyframe seed {st['next']} has no VQAScore yet; score round {rnd - 1} first")
            if st["next"] == rnd:
                todo.append(r)
    log(f"[story] keyframes {name} round {rnd}: {len(todo)} shot(s)")
    if not todo:
        return {"ok": 0}
    base = load_gen_config(FLUX_CONFIG)
    cfg = base.model_copy(update={"name": KF_MODEL, "height": 704, "width": 1280,
                                  "notes": f"{base.notes} Keyframes for Wan2.2 image-to-video at 1280 x 704 (D-038)."})
    prompts = [Prompt(prompt_id=r["prompt_id"], prompt=r["prompt"], source="story") for r in todo]
    return run_generation(cfg=cfg, prompts=prompts, out=root / "kf", seeds=[rnd], log=log)


# ---------- videos (video env) ----------

def video_todo(name: str, rnd: int, root: Path = ROOT) -> list[dict]:
    """Videos to make in round rnd. Round 0: A's text-to-video, B's and D's image-to-video from
    keyframe 0, the agent's first video from its keyframe. Round r > 0: the agent's next attempt for
    shots whose videos so far all missed tau_v (needs their VQAScores)."""
    rows = {r["prompt_id"]: r for r in shot_rows(name)}
    arms = ARMS[name]
    kf = kf_scores(root)
    vid = video_scores(root)
    made = set(video_manifest(root))
    tau_v = tau_v_for(name, root) if rnd > 0 else None
    want: dict[str, dict] = {}

    def add(sid: str, shot: str) -> None:
        model, _, seed = sid.split("__")
        k = int(model.rsplit("_k", 1)[1]) if model.startswith("wanI2V") else None
        want[sid] = {"sample_id": sid, "model": model, "prompt_id": shot, "prompt": rows[shot]["prompt"],
                     "seed": int(seed[1:]), "keyframe_seed": k}

    for shot in rows:
        if rnd == 0:
            if "A" in arms:
                add(make_sample_id(T2V, shot, 0), shot)
            if "B" in arms or "D" in arms:
                add(make_sample_id(i2v(0), shot, 0), shot)
            if "D" in arms:
                add(make_sample_id(i2v(0), shot, 1), shot)
            if "C" in arms:
                ks = kf_state(shot, kf)
                if ks["chosen"] is None:
                    raise SystemExit(f"{shot}: the agent's keyframe search is unfinished (seed {ks['next']} missing)")
                add(make_sample_id(i2v(ks["chosen"]), shot, 0), shot)
        elif "C" in arms:
            # The next attempt may already exist (D's second video is the agent's second attempt when
            # it kept keyframe 0); then it only needs its VQAScore.
            k = kf_state(shot, kf)["chosen"]
            vs = c_video_state(shot, k, vid, tau_v)
            if vs["next"] is not None and vs["next"] in made:
                raise SystemExit(f"{vs['next']} exists but has no VQAScore yet; score round {rnd - 1} first")
            if vs["next"] is not None:
                add(vs["next"], shot)
    return [want[s] for s in sorted(want)]


def _save_frame(arr, path: Path) -> None:
    from PIL import Image
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(a).save(tmp)
    tmp.replace(path)


def videos(name: str, rnd: int, shard: int = 0, num_shards: int = 1, root: Path = ROOT, log=print) -> dict:
    import diffusers
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    from . import video_gen as vg

    mdir = root / "video" / "manifests"
    done = {s for s, r in latest_by_id(read_many(sorted(mdir.glob("gen_*.jsonl")))).items()
            if r.get("status") == "ok" and Path(r["path"]).is_file()} if mdir.exists() else set()
    todo = [t for t in video_todo(name, rnd, root) if t["sample_id"] not in done][shard::num_shards]
    log(f"[story] videos {name} round {rnd} shard {shard}/{num_shards}: {len(todo)} to make")
    if not todo:
        return {"ok": 0, "error": 0}
    kman = latest_by_id(read_many(sorted((root / "kf" / "manifests").glob("gen_*.jsonl"))))
    t0 = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(vg.REPO, subfolder="vae", torch_dtype=torch.float32, revision=vg.REVISION,
                                           local_files_only=True)
    pipe = WanPipeline.from_pretrained(vg.REPO, vae=vae, torch_dtype=torch.bfloat16, revision=vg.REVISION,
                                       local_files_only=True).to("cuda")
    # Image to video with the same weights: Wan2.2 TI2V-5B conditions on the first frame's latent and
    # masks its timestep (expand_timesteps); it has no CLIP image encoder.
    ipipe = WanImageToVideoPipeline(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, vae=pipe.vae,
                                    scheduler=pipe.scheduler, transformer=pipe.transformer, expand_timesteps=True)
    load_s = time.perf_counter() - t0
    log(f"[story] loaded in {load_s:.0f} s; text-to-video expand_timesteps={pipe.config.get('expand_timesteps')}")
    gpu, commit = torch.cuda.get_device_name(0), git_commit()
    counts = {"ok": 0, "error": 0}
    fidx = frame_index()
    with ManifestWriter(mdir / f"gen_{name}_r{rnd}_shard{shard:03d}of{num_shards:03d}.jsonl") as w:
        for t in todo:
            sid = t["sample_id"]
            path = root / "video" / t["model"] / f"{t['prompt_id']}_s{t['seed']}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            frames = [str(root / "video" / "frames" / sid / f"f{i:02d}.png") for i in fidx]
            rec = {**t, "path": str(path), "frames_png": frames, "frames": FRAMES, "height": vg.HEIGHT, "width": vg.WIDTH,
                   "steps": vg.STEPS, "cfg": vg.CFG, "fps": vg.FPS, "repo": vg.REPO, "revision": vg.REVISION,
                   "diffusers": diffusers.__version__, "torch": torch.__version__, "gpu": gpu, "git_commit": commit,
                   "load_seconds": round(load_s, 1), "round": rnd}
            try:
                kw = dict(prompt=t["prompt"], height=vg.HEIGHT, width=vg.WIDTH, num_frames=FRAMES,
                          num_inference_steps=vg.STEPS, guidance_scale=vg.CFG,
                          generator=torch.Generator("cuda").manual_seed(t["seed"]))
                if t["keyframe_seed"] is not None:
                    kr = kman.get(make_sample_id(KF_MODEL, t["prompt_id"], t["keyframe_seed"]))
                    if not kr or kr.get("status") != "ok":
                        raise RuntimeError(f"keyframe seed {t['keyframe_seed']} of {t['prompt_id']} is missing")
                    rec["keyframe"] = kr["path"]
                    with Image.open(kr["path"]) as im:
                        image = im.convert("RGB")
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                out = (ipipe(image=image, **kw) if t["keyframe_seed"] is not None else pipe(**kw)).frames[0]
                torch.cuda.synchronize()
                gen_s = time.perf_counter() - t1
                tmp = path.with_suffix(".tmp.mp4")
                export_to_video(out, str(tmp), fps=vg.FPS)
                tmp.replace(path)
                for i, f in zip(fidx, frames):
                    _save_frame(out[i], Path(f))
                rec.update(status="ok", gen_seconds=round(gen_s, 2),
                           peak_mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
                counts["ok"] += 1
                log(f"[story] {sid}: {gen_s:.1f} s")
            except Exception as e:  # noqa: BLE001
                rec.update(status="error", error=f"{type(e).__name__}: {e}"[:2000])
                counts["error"] += 1
            rec["created_at"] = now_iso()
            w.write(rec)
    log(f"[story] videos shard {shard}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


# ---------- what the judge sees (judge env) ----------

def video_manifest(root: Path = ROOT) -> dict[str, dict]:
    return {s: r for s, r in latest_by_id(read_many(sorted((root / "video" / "manifests").glob("gen_*.jsonl")))).items()
            if r.get("status") == "ok"}


def finals(name: str, root: Path = ROOT, tau_v: float | None = None) -> dict[str, dict[str, str | None]]:
    """arm -> shot -> the video the arm ends with (None if not made yet)."""
    kf, vid = kf_scores(root), video_scores(root)
    tau_v = tau_v_for(name, root) if tau_v is None and "C" in ARMS[name] else tau_v
    rows = shot_rows(name)
    return {a: {r["prompt_id"]: arm_state(a, r["prompt_id"], kf, vid, tau_v)["final"] for r in rows} for a in ARMS[name]}


def judge_requests(name: str, root: Path = ROOT, tau_v: float | None = None) -> list:
    from .judge import Request
    shot_t, story_t, consist_t = _judge_text()
    rows = shot_rows(name)
    by_brief = defaultdict(list)
    for r in rows:
        by_brief[r["brief_id"]].append(r)
    brief_text = {b["prompt_id"]: b["prompt"] for b in briefs(name)}
    man = video_manifest(root)
    fin = finals(name, root, tau_v)
    missing = sorted({f"{a}:{s}" for a, d in fin.items() for s, v in d.items() if v is None})
    if missing:
        raise SystemExit(f"{len(missing)} arm/shot pair(s) have no final video yet, e.g. {missing[:5]}")
    reqs, seen = [], set()
    shot_of = {r["prompt_id"]: r for r in rows}
    for a in sorted(fin):
        for shot, sid in sorted(fin[a].items()):
            key = f"abs:shot:greedy:{sid}"
            if key not in seen:
                seen.add(key)
                reqs.append(Request(key, "abs", "greedy", shot, shot_of[shot]["prompt"], man[sid]["frames_png"],
                                    {"sample_id": sid, "variant": "shot"}, template=shot_t))
        for bid, shots in sorted(by_brief.items()):
            seq = [fin[a][s["prompt_id"]] for s in shots]
            mids = [man[s]["frames_png"][1] for s in seq]
            n = str(len(seq))
            for variant, t, text in (("story", story_t, brief_text[bid]), ("consist", consist_t, shots[0]["character"])):
                key = f"abs:{variant}:greedy:{'+'.join(seq)}"
                if key not in seen:
                    seen.add(key)
                    reqs.append(Request(key, "abs", "greedy", bid, text, mids,
                                        {"brief_id": bid, "shots": seq, "variant": variant},
                                        template=t.replace("{n}", n)))
    return reqs


def judge_records(name: str, judge_dir: Path = JUDGE_DIR) -> dict[str, dict]:
    return latest_by_id(read_many(sorted(judge_dir.glob(f"story_{name}_*.jsonl"))), key="key")


def judge(name: str, root: Path = ROOT, judge_dir: Path = JUDGE_DIR, engine=None, log=print) -> dict:
    from .judge import VLLMJudge, to_records
    have = judge_records(name, judge_dir)
    reqs = [r for r in judge_requests(name, root) if r.key not in have]
    log(f"[story] judge {name}: {len(reqs)} request(s), {len(have)} graded already")
    if not reqs:
        return {"graded": 0}
    engine = engine or VLLMJudge(JUDGE, max_images=4, version_extra=f"/story-{story_template_id()}")
    recs = to_records(reqs, engine.generate(reqs, log=log), engine.version)
    judge_dir.mkdir(parents=True, exist_ok=True)
    out = judge_dir / f"story_{name}_{run_id()}.jsonl"
    atomic_write_text(out, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    return {"graded": len(recs), "file": str(out)}


# ---------- tau_v on dev ----------

def balanced_accuracy(pred: list[bool], truth: list[bool]) -> float:
    pos = [p for p, t in zip(pred, truth) if t]
    neg = [not p for p, t in zip(pred, truth) if not t]
    parts = [float(np.mean(x)) for x in (pos, neg) if x]
    return float(np.mean(parts)) if parts else math.nan


def tune(root: Path = ROOT, judge_dir: Path = JUDGE_DIR, results: Path = Path("results/story")) -> dict:
    """tau_v: the value in TAU_GRID whose "VQAScore on video >= tau_v" best matches the judge's
    pass / fail on B's dev shots (balanced accuracy), ties to the lower (D-038)."""
    fin = finals("dev", root)["B"]
    vid = video_scores(root)
    g = judge_records("dev", judge_dir)
    xs, ys = [], []
    for shot, sid in sorted(fin.items()):
        rec = g.get(f"abs:shot:greedy:{sid}")
        if sid is None or rec is None or sid not in vid or rec.get("score") is None:
            raise SystemExit(f"{shot}: dev video, VQAScore or grade missing")
        xs.append(vid[sid])
        ys.append(rec["score"] >= PASS)
    table = []
    for t in TAU_GRID:
        pred = [x >= t for x in xs]
        table.append({"tau_v": t, "balanced_accuracy": balanced_accuracy(pred, ys),
                      "accuracy": float(np.mean([p == y for p, y in zip(pred, ys)])), "share_passing_check": float(np.mean(pred))})
    best = max(table, key=lambda r: (-math.inf if math.isnan(r["balanced_accuracy"]) else r["balanced_accuracy"], -r["tau_v"]))
    res = {"created_at": now_iso(), "git_commit": git_commit(), "shots": len(xs), "judge_pass": float(np.mean(ys)),
           "tau_v": best["tau_v"], "table": table}
    atomic_write_text(root / "tau_v.json", json.dumps(res, indent=2))
    results.mkdir(parents=True, exist_ok=True)
    L = ["# The agent's video threshold, chosen on dev (D-038)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.story tune`. Arm B's "
         f"{res['shots']} dev shots; the judge (Gemma 4 31B, 3 frames, pass = score 5) passes {_f(res['judge_pass'])} "
         "of them. A shot passes the agent's check when its VQAScore on the whole video is at least tau_v.", "",
         "| tau_v | Balanced accuracy against the judge | Accuracy | Shots passing the check |", "| --- | --- | --- | --- |"]
    L += [f"| {r['tau_v']} | {_f(r['balanced_accuracy'])} | {_f(r['accuracy'])} | {_f(r['share_passing_check'])} |" for r in table]
    L += ["", f"Chosen: **tau_v = {res['tau_v']}** (highest balanced accuracy, ties to the lower)."]
    atomic_write_text(results / "dev_tau.md", "\n".join(L) + "\n")
    return res


# ---------- analysis ----------

def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def boot_mean(by_brief: dict[str, list[float]], n_boot: int = N_BOOT, level: float = 0.95, seed: int = 0) -> dict:
    """Mean over all shots, CI from resampling briefs (shots of a brief move together)."""
    bids = sorted(b for b, v in by_brief.items() if v)
    if not bids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    s = np.array([sum(by_brief[b]) for b in bids], float)
    c = np.array([len(by_brief[b]) for b in bids], float)
    idx = np.random.default_rng(seed).integers(0, len(bids), size=(n_boot, len(bids)))
    stats = s[idx].sum(1) / c[idx].sum(1)
    a = (1 - level) / 2
    lo, hi = np.quantile(stats, [a, 1 - a])
    return {"value": float(s.sum() / c.sum()), "ci": [float(lo), float(hi)], "n": int(c.sum()), "briefs": len(bids)}


def verdict(ci: list[float]) -> str:
    return "supported" if ci[0] > 0 else "contradicted" if ci[1] < 0 else "inconclusive"


def costs(state: dict, kman: dict, vman: dict, kscore_s: dict, vscore_s: dict) -> float:
    """GPU-seconds an arm spends on one shot: its keyframes and videos, and the VQAScore runs of the
    checks it makes (model loading excluded)."""
    s = sum(kman[k].get("gen_seconds", 0.0) for k in state["kf"])
    s += sum(vman[v].get("gen_seconds", 0.0) for v in state["videos"])
    if state["checks_kf"]:
        s += sum(kscore_s.get(k, 0.0) for k in state["kf"])
    if state["checks_video"]:
        s += sum(vscore_s.get(v, 0.0) for v in state["videos"])
    return s


def score_seconds(metric_dir: Path) -> dict[str, float]:
    recs = read_many(sorted(metric_dir.glob("*.jsonl"))) if metric_dir.exists() else []
    return {s: float(r.get("seconds") or 0.0) for s, r in latest_by_id([r for r in recs if r.get("status") == "ok"]).items()}


def analyze(name: str, root: Path = ROOT, judge_dir: Path = JUDGE_DIR, results: Path | None = None,
            n_boot: int = N_BOOT) -> dict:
    results = results or Path("results/story" if name != "smoke" else "results/story_smoke")
    arms = ARMS[name]
    tau_v = tau_v_for(name, root) if "C" in arms else None
    rows = shot_rows(name)
    kf, vid = kf_scores(root), video_scores(root)
    kman = latest_by_id(read_many(sorted((root / "kf" / "manifests").glob("gen_*.jsonl"))))
    vman = video_manifest(root)
    ks_s, vs_s = score_seconds(root / "kf" / "scores" / "vqascore"), score_seconds(root / "video" / "scores" / "vqascore_video")
    g = judge_records(name, judge_dir)
    st = {a: {r["prompt_id"]: arm_state(a, r["prompt_id"], kf, vid, tau_v) for r in rows} for a in arms}
    brief_of = {r["prompt_id"]: r["brief_id"] for r in rows}

    def grade(key: str) -> dict:
        rec = g.get(key)
        if rec is None or rec.get("score") is None:
            raise SystemExit(f"no grade for {key}")
        return rec

    per = {a: {"pass": defaultdict(list), "expected": defaultdict(list), "vqa": defaultdict(list), "cost": defaultdict(list)}
           for a in arms}
    story = {a: {} for a in arms}
    for a in arms:
        for shot, s in st[a].items():
            b = brief_of[shot]
            rec = grade(f"abs:shot:greedy:{s['final']}")
            per[a]["pass"][b].append(float(rec["score"] >= PASS))
            per[a]["expected"][b].append(float(rec.get("expected_score") or rec["score"]))
            per[a]["vqa"][b].append(vid[s["final"]])
            per[a]["cost"][b].append(costs(s, kman, vman, ks_s, vs_s))
        for b in sorted({brief_of[x] for x in st[a]}):
            seq = "+".join(st[a][r["prompt_id"]]["final"] for r in rows if r["brief_id"] == b)
            story[a][b] = {"story": grade(f"abs:story:greedy:{seq}")["score"],
                           "consist": grade(f"abs:consist:greedy:{seq}")["score"]}
    out: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "name": name, "arms": arms,
                           "briefs": len({brief_of[x] for x in brief_of}), "shots": len(rows), "tau_k": TAU_K, "tau_v": tau_v,
                           "by_arm": {}, "tests": {}, "agent": {}}
    for a in arms:
        cost_brief = {b: [sum(v)] for b, v in per[a]["cost"].items()}
        out["by_arm"][a] = {
            "shot_pass": boot_mean(per[a]["pass"], n_boot),
            "shot_expected_score": boot_mean(per[a]["expected"], n_boot),
            "vqascore_video": boot_mean(per[a]["vqa"], n_boot),
            "story_pass": boot_mean({b: [float(v["story"] >= PASS)] for b, v in story[a].items()}, n_boot),
            "story_score": boot_mean({b: [float(v["story"])] for b, v in story[a].items()}, n_boot),
            "consistency_pass": boot_mean({b: [float(v["consist"] >= PASS)] for b, v in story[a].items()}, n_boot),
            "consistency_score": boot_mean({b: [float(v["consist"])] for b, v in story[a].items()}, n_boot),
            "gpu_seconds_per_brief": boot_mean(cost_brief, n_boot),
        }

    def diff(x: str, y: str, what: str = "pass", level: float = 0.95) -> dict:
        d = {b: [p - q for p, q in zip(per[x][what][b], per[y][what][b])] for b in per[x][what]}
        return boot_mean(d, n_boot, level)

    pre = {"H7": ("B", "A"), "H8": ("C", "B")}
    for h, (x, y) in pre.items():
        if x in arms and y in arms:
            d = diff(x, y, level=LEVEL)
            out["tests"][h] = {"comparison": f"shot pass rate, {x} - {y}", **d, "level": LEVEL, "verdict": verdict(d["ci"])}
    if "C" in arms and "D" in arms:
        out["tests"]["C - D (exploratory)"] = {"comparison": "shot pass rate, C - D", **diff("C", "D")}
    if "C" in arms and "A" in arms:
        out["tests"]["C - A (exploratory)"] = {"comparison": "shot pass rate, C - A", **diff("C", "A")}
    # Exploratory, added after the test results were seen (D-038 note of 2026-10-25): paired
    # differences of the brief-level story and consistency scores.
    out["brief_diffs"] = {}
    for x, y in (("B", "A"), ("C", "B"), ("C", "A"), ("C", "D"), ("D", "B")):
        if x in arms and y in arms:
            for what in ("story", "consist"):
                d = {b: [float(story[x][b][what] - story[y][b][what])] for b in story[x]}
                out["brief_diffs"][f"{what} score, {x} - {y}"] = boot_mean(d, n_boot)
    if "C" in arms:
        cs = list(st["C"].values())
        out["agent"] = {"shots": len(cs),
                        "keyframe_redone": float(np.mean([len(s["kf"]) > 1 for s in cs])),
                        "video_redone": float(np.mean([len(s["videos"]) > 1 for s in cs])),
                        "videos_per_shot": float(np.mean([len(s["videos"]) for s in cs])),
                        "keyframes_per_shot": float(np.mean([len(s["kf"]) for s in cs])),
                        "ended_below_tau_v": float(np.mean([vid[s["final"]] < tau_v for s in cs]))}
    results.mkdir(parents=True, exist_ok=True)
    atomic_write_text(results / "shots.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    atomic_write_text(results / "summary.json", json.dumps(out, indent=2))
    atomic_write_text(results / "summary.md", render(out))
    return out


ARM_NAMES = {"A": "A text to video", "B": "B keyframe → image to video", "C": "C agent (checks + redo)",
             "D": "D re-ranking (2 videos per shot)"}


def render(res: dict) -> str:
    def cell(d: dict, nd=3) -> str:
        return f"{_f(d['value'], nd)} [{_f(d['ci'][0], 2)}, {_f(d['ci'][1], 2)}]"
    L = [f"# Multi-shot videos from a brief ({res['name']}, D-038)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.story analyze`. "
         f"{res['briefs']} brief(s), {res['shots']} shots; Wan2.2 TI2V-5B, 49 frames per shot. Judge: Gemma 4 31B on "
         f"the first, middle and last frame of a shot (shot) or the middle frame of every shot (story, consistency); "
         f"pass = score 5. 95 % CIs from resampling briefs. Agent thresholds: keyframe VQAScore {res['tau_k']}, video "
         f"VQAScore {res['tau_v']}.", ""]
    if res["name"] == "smoke":
        L += ["**Smoke run: a pipeline check only; no number here is a result.**", ""]
    L += ["| Arm | Shot pass rate | Shot expected score | VQAScore (video) | Story pass | Story score | "
          "Consistency pass | Consistency score | GPU-s per brief |", "| --- |" + " --- |" * 8]
    for a in res["arms"]:
        r = res["by_arm"][a]
        L.append(f"| {ARM_NAMES[a]} | {cell(r['shot_pass'])} | {cell(r['shot_expected_score'], 2)} | "
                 f"{cell(r['vqascore_video'])} | {cell(r['story_pass'])} | {cell(r['story_score'], 2)} | "
                 f"{cell(r['consistency_pass'])} | {cell(r['consistency_score'], 2)} | "
                 f"{_f(r['gpu_seconds_per_brief']['value'], 0)} |")
    if res["tests"]:
        L += ["", "| Test | Difference [CI] | CI level | Verdict |", "| --- | --- | --- | --- |"]
        for k, t in res["tests"].items():
            L.append(f"| {k}: {t['comparison']} | {cell(t)} | {t.get('level', 0.95)} | {t.get('verdict', 'exploratory')} |")
    if res.get("brief_diffs"):
        L += ["", "Exploratory, added after the test results were seen: paired differences per brief.", "",
              "| Difference | Value [95 % CI] |", "| --- | --- |"]
        L += [f"| {k} | {cell(v, 2)} |" for k, v in res["brief_diffs"].items()]
    if res["agent"]:
        g = res["agent"]
        L += ["", f"Agent: keyframe redone on {_f(g['keyframe_redone'])} of shots, video redone on {_f(g['video_redone'])}; "
              f"{_f(g['keyframes_per_shot'], 2)} keyframes and {_f(g['videos_per_shot'], 2)} videos per shot; "
              f"{_f(g['ended_below_tau_v'])} of shots ended below tau_v."]
    L += ["", "GPU-seconds: generation plus the checks' VQAScore runs, measured per item; planning (the same for every "
          "arm) and model loading are not counted."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("plan", "keyframes", "videos", "judge", "analyze", "todo"):
        p = sub.add_parser(c)
        p.add_argument("--name", required=True, choices=sorted(ARMS))
        if c in ("keyframes", "videos", "todo"):
            p.add_argument("--round", type=int, required=True)
        if c == "videos":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--num-shards", type=int, default=1)
    sub.add_parser("tune")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        out = plan(a.name)
    elif a.cmd == "keyframes":
        out = keyframes(a.name, a.round)
    elif a.cmd == "videos":
        out = videos(a.name, a.round, a.shard, a.num_shards)
    elif a.cmd == "todo":
        out = {"videos": len(video_todo(a.name, a.round))}
    elif a.cmd == "judge":
        out = judge(a.name)
    elif a.cmd == "tune":
        out = tune()
    else:
        res = analyze(a.name)
        out = {"tests": res["tests"]}
    print(json.dumps(out, indent=2, default=str))
    return 1 if isinstance(out, dict) and out.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
