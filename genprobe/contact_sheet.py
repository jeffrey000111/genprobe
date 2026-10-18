"""Contact sheets for looking at every generated image of a prompt set (M1: all smoke images
reviewed by a person), plus a CSV template for writing down what you see.

One JPEG per prompt category: a row per prompt (with its text), a column per model and seed,
each tile captioned with the metric scores that exist so far. Thumbnails only; the sheets
and the label file go under data/ (not in git).

    python -m genprobe.contact_sheet --prompts data/prompts/smoke.jsonl --out data/review

On HPC3 run it through scripts/make_contact_sheets.sh (CPU node). The label template is
never overwritten, so labels typed into it survive re-running the sheets with new scores.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .samples import load_prompts
from .summarize import category, pick_scores
from .utils.manifest import latest_by_id, read_many

TILE = 256
PAD = 8
LABEL_W = 280
HEADER_H = 34
CAPTION_H = 26
ABBR = {"clipscore": "clip", "vqascore": "vqa"}
LABEL_COLUMNS = ["sample_id", "model", "prompt_id", "seed", "category", "prompt", "correct", "note"]


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)   # scalable font, Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def load_inputs(prompts_path: str, manifests: str, data: str, metrics: list[str]):
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    gen = latest_by_id(read_many(sorted(glob.glob(manifests))))
    gen = {sid: r for sid, r in gen.items()
           if r.get("prompt_id") in prompts and r.get("status") in ("ok", "blank")}
    scores: dict[str, dict[str, dict[str, Any]]] = {}
    for m in metrics:
        recs = read_many(sorted((Path(data) / "scores" / m).glob("*.jsonl")))
        scores[m], _, _ = pick_scores(recs, set(gen))
    return prompts, gen, scores


def make_sheets(*, prompts_path: str, manifests: str, data: str, metrics: list[str], out: str) -> list[Path]:
    prompts, gen, scores = load_inputs(prompts_path, manifests, data, metrics)
    if not gen:
        raise SystemExit(f"no generated images for {prompts_path} in {manifests}")
    by_key = {(r["model"], r["prompt_id"], int(r["seed"])): (sid, r) for sid, r in gen.items()}
    cols = sorted({(r["model"], int(r["seed"])) for r in gen.values()})
    big, small = _font(18), _font(15)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for cat in sorted({category(p.tags) for p in prompts.values()}):
        pids = [pid for pid in sorted(prompts) if category(prompts[pid].tags) == cat]
        row_h = TILE + CAPTION_H + PAD
        width = LABEL_W + len(cols) * (TILE + PAD) + PAD
        height = HEADER_H + len(pids) * row_h + PAD
        sheet = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((PAD, PAD), f"{Path(prompts_path).stem} / {cat}", fill="black", font=big)
        for j, (model, seed) in enumerate(cols):
            draw.text((LABEL_W + j * (TILE + PAD), PAD), f"{model} (seed {seed})", fill="black", font=big)
        for i, pid in enumerate(pids):
            y = HEADER_H + i * row_h
            label = pid + "\n" + "\n".join(textwrap.wrap(prompts[pid].prompt, 30))
            draw.multiline_text((PAD, y), label, fill="black", font=small, spacing=3)
            for j, (model, seed) in enumerate(cols):
                x = LABEL_W + j * (TILE + PAD)
                hit = by_key.get((model, pid, seed))
                if hit is None:
                    draw.rectangle((x, y, x + TILE, y + TILE), outline="red")
                    draw.text((x + PAD, y + PAD), "not generated", fill="red", font=small)
                    continue
                sid, rec = hit
                with Image.open(rec["path"]) as im:
                    thumb = im.convert("RGB")
                thumb.thumbnail((TILE, TILE))
                sheet.paste(thumb, (x, y))
                parts = ["BLANK"] if rec.get("status") == "blank" else []
                parts += [f"{ABBR.get(m, m)} {scores[m][sid]['value']:.2f}" for m in metrics if sid in scores[m]]
                draw.text((x, y + TILE + 3), "  ".join(parts) or "no scores yet", fill="black", font=small)
        path = outdir / f"{Path(prompts_path).stem}_{cat}.jpg"
        sheet.save(path, quality=85)
        written.append(path)
    labels = outdir / f"{Path(prompts_path).stem}_labels.csv"
    if not labels.exists():
        with open(labels, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(LABEL_COLUMNS)
            for sid in sorted(gen, key=lambda s: (gen[s]["prompt_id"], gen[s]["model"], gen[s]["seed"])):
                r = gen[sid]
                w.writerow([sid, r["model"], r["prompt_id"], r["seed"], category(prompts[r["prompt_id"]].tags),
                            prompts[r["prompt_id"]].prompt, "", ""])
        written.append(labels)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--manifests", default="data/manifests/gen_*.jsonl")
    ap.add_argument("--data", default="data")
    ap.add_argument("--metrics", nargs="+", default=["clipscore", "vqascore"])
    ap.add_argument("--out", default="data/review")
    args = ap.parse_args(argv)
    for p in make_sheets(prompts_path=args.prompts, manifests=args.manifests, data=args.data,
                         metrics=args.metrics, out=args.out):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
