"""GenAI-Bench-Video (GenAI-Bench-800) for the video meta-evaluation (guide §11.3, D-035): 800
prompts, each rendered by four text-to-video models (Floor33, Gen-2, Pika v1, ModelScope) and each
video rated 1-5 by three people for how well it matches the prompt. `scripts/fetch_genai_bench_video.sh`
(coe-hpc1) downloads them to data/external/genai_bench_video/<model>/<index>.mp4.

    python -m genprobe.genai_bench_video prepare     (light: reads the JSON files and checks the videos exist)
        -> data/prompts/genai_bench_video.jsonl        800 prompts, prompt_id gv<index:04d>, skill tags
           data/prompts/genai_bench_video_smoke.jsonl  20 of them, for pipeline checks
           data/manifests/ext_genai_video.jsonl        one record per video (sample_id <model>__gv<index>__s0)
    python -m genprobe.genai_bench_video frames      (CPU node: scripts/prepare_genai_bench_video.sh)
        -> data/external/genai_bench_video/frames/<model>/<index>_f<k>.png   9 frames per video
           data/manifests/ext_genai_video_frames.jsonl  all 9 frames; frame k is stored as seed 100 + k
           data/manifests/ext_genai_video_mid.jsonl     the middle frame only (k = 4, seed 104)

Frames are taken at round(linspace(0, n - 1, 9)), so frame 0 is the first, 8 the last and 4 the
middle one. Image metrics score frames like images; genprobe.video_meta_eval averages them per video.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .build_prompts import genai_tags
from .samples import make_sample_id
from .utils.env import now_iso
from .utils.io import atomic_write_text

FOLDER = Path("data/external/genai_bench_video")
MODELS = ("Floor33", "Gen2", "Pika_v1", "Modelscope")
N_FRAMES = 9
MID = 4
FRAME_SEED = 100           # frame k of a video is stored as seed 100 + k
PROMPTS = "data/prompts/genai_bench_video.jsonl"
SMOKE = "data/prompts/genai_bench_video_smoke.jsonl"
MANIFEST = "data/manifests/ext_genai_video.jsonl"
FRAMES_MANIFEST = "data/manifests/ext_genai_video_frames.jsonl"
MID_MANIFEST = "data/manifests/ext_genai_video_mid.jsonl"


def prompt_id(index: int) -> str:
    return f"gv{index:04d}"


def source(folder: str | Path = FOLDER) -> str:
    p = Path(folder) / "SOURCE"
    return p.read_text().strip() if p.exists() else "zhiqiulin/GenAI-Bench-800"


def load(folder: str | Path = FOLDER) -> tuple[dict, dict[str, list[int]]]:
    folder = Path(folder)
    data = json.loads((folder / "genai_video.json").read_text())
    skills = json.loads((folder / "genai_skills.json").read_text())
    return data, {k: sorted({int(i) for i in v}) for k, v in skills.items()}


def load_ratings(folder: str | Path = FOLDER) -> tuple[dict[tuple[str, str], list[float]], list[dict]]:
    """{(prompt_id, model): valid ratings} and the dropped (out-of-range) ratings."""
    data, _ = load(folder)
    out, dropped = {}, []
    for key, item in data.items():
        pid = prompt_id(int(key))
        for model, ratings in item["models"].items():
            valid = [float(r) for r in ratings if 1 <= float(r) <= 5]
            dropped += [{"prompt_id": pid, "model": model, "rating": r} for r in ratings if not 1 <= float(r) <= 5]
            out[(pid, model)] = valid
    return out, dropped


def video_path(folder: str | Path, model: str, key: str) -> Path:
    return Path(folder) / model / f"{key}.mp4"


def frame_path(folder: str | Path, model: str, key: str, k: int) -> Path:
    return Path(folder) / "frames" / model / f"{key}_f{k}.png"


def frame_indices(n: int, k: int = N_FRAMES) -> list[int]:
    if n <= 0:
        return []
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def prepare(folder: str | Path = FOLDER, prompts_out: str = PROMPTS, smoke_out: str = SMOKE,
            manifest_out: str = MANIFEST, seed: int = 0) -> dict:
    folder = Path(folder)
    data, skills = load(folder)
    tags = genai_tags(skills)
    rows = []
    for key in sorted(data, key=int):
        i = int(key)
        t = tags.get(i, ["untagged"])
        rows.append({"prompt_id": prompt_id(i), "source": "genai_bench_video", "prompt": data[key]["prompt"].strip(),
                     "tags": t, "spec": None, "split": "test", "stratum": f"genai_{t[0]}", "key": key})
    atomic_write_text(Path(prompts_out), "".join(json.dumps(r) + "\n" for r in rows))
    smoke = sorted(random.Random(seed).sample(rows, min(20, len(rows))), key=lambda r: r["prompt_id"])
    atomic_write_text(Path(smoke_out), "".join(json.dumps(r) + "\n" for r in smoke))
    ratings, dropped = load_ratings(folder)
    recs, missing = [], 0
    for r in rows:
        for model in MODELS:
            if model not in data[r["key"]]["models"]:
                continue
            path = video_path(folder, model, r["key"])
            ok = path.is_file() and path.stat().st_size > 0
            missing += not ok
            recs.append({"sample_id": make_sample_id(model, r["prompt_id"], 0), "model": model,
                         "prompt_id": r["prompt_id"], "prompt": r["prompt"], "seed": 0, "path": str(path),
                         "status": "ok" if ok else "missing", "source": source(folder), "kind": "video",
                         "human_ratings": ratings.get((r["prompt_id"], model), []), "created_at": now_iso()})
    atomic_write_text(Path(manifest_out), "".join(json.dumps(x) + "\n" for x in recs))
    return {"prompts": len(rows), "videos": len(recs), "missing": missing, "dropped_ratings": dropped,
            "untagged": sum(r["tags"] == ["untagged"] for r in rows),
            "models": {m: sum(x["model"] == m for x in recs) for m in MODELS}}


def extract(path: Path, out: list[Path]) -> dict:
    """Write N_FRAMES frames of a video as PNGs; returns the video's frame count, fps and size."""
    import av
    from PIL import Image

    with av.open(str(path)) as c:
        st = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(st)]
        fps = float(st.average_rate) if st.average_rate else None
    idx = frame_indices(len(frames))
    for k, i in enumerate(idx):
        out[k].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[i]).save(out[k])
    h, w = frames[0].shape[:2] if frames else (None, None)
    return {"n_frames": len(frames), "fps": fps, "width": w, "height": h, "frame_indices": idx}


def frames(folder: str | Path = FOLDER, manifest: str = MANIFEST, frames_out: str = FRAMES_MANIFEST,
           mid_out: str = MID_MANIFEST) -> dict:
    folder = Path(folder)
    videos = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    key_of = {json.loads(line)["prompt_id"]: json.loads(line)["key"]
              for line in Path(PROMPTS).read_text().splitlines() if line.strip()}
    fr, mid, errors, info = [], [], 0, []
    for v in videos:
        if v["status"] != "ok":
            continue
        key = key_of[v["prompt_id"]]
        outs = [frame_path(folder, v["model"], key, k) for k in range(N_FRAMES)]
        try:
            meta = extract(Path(v["path"]), outs) if not all(p.is_file() for p in outs) else {}
            status = "ok"
        except Exception as e:  # noqa: BLE001
            meta, status, errors = {"error": f"{type(e).__name__}: {e}"[:500]}, "error", errors + 1
        info.append({"sample_id": v["sample_id"], **meta})
        for k, p in enumerate(outs):
            rec = {"sample_id": make_sample_id(v["model"], v["prompt_id"], FRAME_SEED + k), "model": v["model"],
                   "prompt_id": v["prompt_id"], "prompt": v["prompt"], "seed": FRAME_SEED + k, "path": str(p),
                   "status": status if p.is_file() else "error", "kind": "frame", "video": v["sample_id"],
                   "frame": k, "created_at": now_iso()}
            fr.append(rec)
            if k == MID:
                mid.append(rec)
    atomic_write_text(Path(frames_out), "".join(json.dumps(x) + "\n" for x in fr))
    atomic_write_text(Path(mid_out), "".join(json.dumps(x) + "\n" for x in mid))
    atomic_write_text(folder / "frames" / "video_info.jsonl", "".join(json.dumps(x) + "\n" for x in info))
    return {"videos": len(videos), "frames": len(fr), "errors": errors}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("frames")
    args = ap.parse_args(argv)
    res = prepare() if args.cmd == "prepare" else frames()
    print(json.dumps(res, indent=2))
    return 0 if not res.get("missing") and not res.get("errors") else 1


if __name__ == "__main__":
    sys.exit(main())
