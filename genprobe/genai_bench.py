"""GenAI-Bench-1600 images and human ratings, prepared for the metric meta-evaluation (module B,
guide §9.2; hypotheses in docs/decisions.md).

GenAI-Bench has 1,600 prompts, each rendered by 6 models (DALL-E 3, DeepFloyd IF XL, Midjourney 6,
SD 2.1, SDXL base, SDXL Turbo) and rated 1-5 by 3 people for how well the image matches the
prompt. `scripts/fetch_genai_bench.sh images` (coe-hpc1) downloads and unzips them to
data/external/genai_bench/images/<model>/<index:05d>.jpeg.

    python -m genprobe.genai_bench prepare
        -> data/prompts/genai_bench_1600.jsonl   all prompts; prompt_id gb<index:04d> as in prompt
                                                  set v1; split "dev" for the 80 prompts in v1's dev
                                                  split (used only to iterate on the VLM judge), else "test"
           data/prompts/genai_bench_smoke.jsonl  20 of them (10 dev, 10 test) for pipeline checks
           data/manifests/ext_genai_bench.jsonl  one record per image (sample_id <model>__gb<index>__s0),
                                                  the input to genprobe.score like generation manifests

The scoring runner reads these like our own images; `load_ratings` gives the human ratings,
with out-of-range ratings dropped (two in the dataset, D-016).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .build_prompts import genai_tags, load_genai_bench
from .samples import load_prompts, make_sample_id
from .utils.env import now_iso
from .utils.io import atomic_write_text

FOLDER = Path("data/external/genai_bench")
MODELS = ("DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo")
SOURCE = "BaiqiL/GenAI-Bench-1600@e51e8a8"


def prompt_id(index: int) -> str:
    return f"gb{index:04d}"


def image_path(folder: str | Path, model: str, index: int) -> Path:
    return Path(folder) / "images" / model / f"{index:05d}.jpeg"


def load_ratings(folder: str | Path = FOLDER) -> tuple[dict[tuple[str, str], list[float]], list[dict]]:
    """{(prompt_id, model): valid ratings} and the list of dropped (out-of-range) ratings."""
    data = json.loads((Path(folder) / "genai_image.json").read_text())
    out, dropped = {}, []
    for key, item in data.items():
        pid = prompt_id(int(key))
        for model, ratings in item["models"].items():
            valid = [float(r) for r in ratings if 1 <= float(r) <= 5]
            dropped += [{"prompt_id": pid, "model": model, "rating": r} for r in ratings if not 1 <= float(r) <= 5]
            out[(pid, model)] = valid
    return out, dropped


def prepare(folder: str | Path = FOLDER, v1_prompts: str = "data/prompts/prompts_v1.jsonl",
            prompts_out: str = "data/prompts/genai_bench_1600.jsonl",
            smoke_out: str = "data/prompts/genai_bench_smoke.jsonl",
            manifest_out: str = "data/manifests/ext_genai_bench.jsonl", verify: bool = False,
            seed: int = 0) -> dict:
    folder = Path(folder)
    texts, skills = load_genai_bench(folder)
    tags = genai_tags(skills)
    v1_dev = {p.prompt_id for p in load_prompts(v1_prompts, split="dev") if p.source == "genai_bench"}
    rows = []
    for i in sorted(texts):
        t = tags.get(i, ["untagged"])
        rows.append({"prompt_id": prompt_id(i), "source": "genai_bench", "prompt": texts[i], "tags": t,
                     "spec": None, "split": "dev" if prompt_id(i) in v1_dev else "test",
                     "stratum": f"genai_{t[0]}"})
    atomic_write_text(Path(prompts_out), "".join(json.dumps(r) + "\n" for r in rows))
    rng = random.Random(seed)
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    smoke = sorted(rng.sample(dev, min(10, len(dev))) + rng.sample(test, min(10, len(test))),
                   key=lambda r: r["prompt_id"])
    atomic_write_text(Path(smoke_out), "".join(json.dumps(r) + "\n" for r in smoke))

    ratings, dropped = load_ratings(folder)
    recs, missing, unreadable = [], 0, 0
    for r in rows:
        i = int(r["prompt_id"][2:])
        for model in MODELS:
            path = image_path(folder, model, i)
            status = "ok"
            size = None
            if not path.is_file():
                status, missing = "missing", missing + 1
            elif verify:
                try:
                    from PIL import Image
                    with Image.open(path) as im:
                        size = list(im.size)
                except Exception:  # noqa: BLE001
                    status, unreadable = "error", unreadable + 1
            rec = {"sample_id": make_sample_id(model, r["prompt_id"], 0), "model": model,
                   "prompt_id": r["prompt_id"], "prompt": r["prompt"], "seed": 0, "path": str(path),
                   "status": status, "source": SOURCE, "human_ratings": ratings.get((r["prompt_id"], model), []),
                   "created_at": now_iso()}
            if size:
                rec["image_size"] = size
            recs.append(rec)
    atomic_write_text(Path(manifest_out), "".join(json.dumps(x) + "\n" for x in recs))
    return {"prompts": len(rows), "dev": sum(r["split"] == "dev" for r in rows), "smoke": len(smoke),
            "images": len(recs), "missing": missing, "unreadable": unreadable, "dropped_ratings": dropped}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--folder", default=str(FOLDER))
    pr.add_argument("--verify", action="store_true", help="open every image (slow on NFS for DALL-E 3's)")
    args = ap.parse_args(argv)
    res = prepare(args.folder, verify=args.verify)
    print(json.dumps(res, indent=2))
    return 0 if res["missing"] == 0 and res["unreadable"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
