"""Example panels for the report (guide §8.3, D-025): a few typical successes and failures per
prompt category, and "pretty but wrong" images (high PickScore, low VQAScore, guide §7.7).

Picked at random (fixed seed) from the test split among images that clearly pass or clearly
fail by the metrics, spread over the four models, so they are illustrations chosen by rule, not
by hand. Thumbnails only (256 px, JPEG): these are the only generated images in git (D-025).

    python -m genprobe.examples --results results/prompts_v1_test --prompts data/prompts/prompts_v1.jsonl

Writes results/examples/<panel>.jpg and results/examples/index.md (every image with its
sample_id and scores). On HPC3 run it through scripts/make_report_assets.sh (CPU node).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from .contact_sheet import _font
from .figures import MODEL_LABEL, ordered_models
from .score import load_samples
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text

TILE, PAD, CAP, HEAD = 256, 10, 88, 30
# The panels are public: prompts about children are left out of them (D-025).
NO_PUBLIC = re.compile(r"\b(child|children|kids?|bab(y|ies)|toddlers?|boys?|girls?|teen\w*|infants?|students?|pupils?|"
                       r"school\w*|classrooms?)\b", re.I)
PER_ROW = 4


def num(r: dict, k: str) -> float | None:
    v = r.get(k, "")
    return float(v) if v not in ("", None) else None


def ok(v: float | None, test: Callable[[float], bool]) -> bool:
    return v is not None and test(v)


# (file, title, [(row title, condition)]) — conditions are on per_sample.csv rows
def panels(pick_hi: float) -> list[tuple[str, str, list[tuple[str, Callable[[dict], bool]]]]]:
    vqa = lambda r: num(r, "vqascore")  # noqa: E731
    return [
        ("counting", "Counting prompts", [
            ("right (detector count matches, VQAScore >= 0.9)",
             lambda r: r["category"] == "counting" and ok(num(r, "detect"), lambda v: v == 1) and ok(vqa(r), lambda v: v >= 0.9)),
            ("wrong (detector count differs, VQAScore <= 0.1)",
             lambda r: r["category"] == "counting" and ok(num(r, "detect"), lambda v: v == 0) and ok(vqa(r), lambda v: v <= 0.1))]),
        ("spatial", "Spatial prompts", [
            ("right (detector agrees, VQAScore >= 0.9)",
             lambda r: r["category"] == "spatial" and ok(num(r, "detect"), lambda v: v == 1) and ok(vqa(r), lambda v: v >= 0.9)),
            ("wrong (detector disagrees, VQAScore <= 0.1)",
             lambda r: r["category"] == "spatial" and ok(num(r, "detect"), lambda v: v == 0) and ok(vqa(r), lambda v: v <= 0.1))]),
        ("text", "Text prompts", [
            ("right (OCR >= 0.85, VQAScore >= 0.9)",
             lambda r: r["category"] == "text" and ok(num(r, "ocr"), lambda v: v >= 0.85) and ok(vqa(r), lambda v: v >= 0.9)),
            ("wrong (OCR < 0.5, VQAScore <= 0.1)",
             lambda r: r["category"] == "text" and ok(num(r, "ocr"), lambda v: v < 0.5) and ok(vqa(r), lambda v: v <= 0.1))]),
        ("genai_bench", "GenAI-Bench prompts", [
            ("VQAScore >= 0.95", lambda r: r["category"] in ("basic", "advanced") and ok(vqa(r), lambda v: v >= 0.95)),
            ("VQAScore <= 0.05", lambda r: r["category"] in ("basic", "advanced") and ok(vqa(r), lambda v: v <= 0.05))]),
        ("pretty_but_wrong", "High PickScore (top 10%) but VQAScore <= 0.05", [
            ("", lambda r: ok(num(r, "pickscore"), lambda v: v >= pick_hi) and ok(vqa(r), lambda v: v <= 0.05))]),
    ]


def choose(rows: list[dict], cond, k: int, rng: random.Random) -> list[dict]:
    """k rows meeting cond, one model at a time in turn (so the models are spread)."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        if cond(r):
            by_model.setdefault(r["model"], []).append(r)
    for v in by_model.values():
        v.sort(key=lambda r: r["sample_id"])
        rng.shuffle(v)
    out: list[dict] = []
    models = ordered_models(by_model)
    rng.shuffle(models)
    while len(out) < k and any(by_model.values()):
        for m in models:
            if by_model[m] and len(out) < k:
                out.append(by_model[m].pop())
    return out


def caption(r: dict) -> str:
    scores = [f"VQA {float(r['vqascore']):.2f}"] if r.get("vqascore") else []
    for k, lab in (("ocr", "OCR"), ("pickscore", "Pick")):
        if r.get(k):
            scores.append(f"{lab} {float(r[k]):.2f}")
    lines = [MODEL_LABEL.get(r["model"], r["model"]), "  ".join(scores)] + textwrap.wrap(r["prompt"], 34)[:3]
    return "\n".join(lines)


def draw_panel(title: str, rows_of_tiles: list[tuple[str, list[dict]]], paths: dict[str, str], out: Path) -> None:
    font, bold = _font(12), _font(16)
    width = PAD + PER_ROW * (TILE + PAD)
    height = PAD + HEAD + sum(HEAD + TILE + CAP + PAD for _ in rows_of_tiles)
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD), title, fill="black", font=bold)
    y = PAD + HEAD
    for row_title, tiles in rows_of_tiles:
        if row_title:
            d.text((PAD, y + 6), row_title, fill="#52514e", font=font)
        y += HEAD
        for j, r in enumerate(tiles):
            x = PAD + j * (TILE + PAD)
            with Image.open(paths[r["sample_id"]]) as im:
                t = im.convert("RGB")
            t.thumbnail((TILE, TILE))
            img.paste(t, (x, y))
            d.multiline_text((x, y + TILE + 4), caption(r), fill="black", font=font, spacing=2)
        y += TILE + CAP + PAD
    img.save(out, quality=85)


def run(results: str, prompts: str, manifests: str, out: str = "results/examples", seed: int = 0) -> list[Path]:
    with open(Path(results) / "per_sample.csv", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    paths = {s.sample_id: s.path for s in load_samples(manifests, prompts)}
    rows = [r for r in rows if r["sample_id"] in paths and not NO_PUBLIC.search(r["prompt"])]
    picks = sorted(float(r["pickscore"]) for r in rows if r.get("pickscore"))
    pick_hi = picks[int(0.9 * len(picks))] if picks else float("inf")
    rng = random.Random(seed)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    written, index = [], ["# Example images", "",
                          f"Made {now_iso()} at commit `{git_commit()}` by `python -m genprobe.examples` from "
                          f"`{results}/per_sample.csv` (test split). Chosen at random (seed {seed}) among images "
                          "meeting each row's rule, spread over models; 256 px thumbnails (D-025).", ""]
    for name, title, row_rules in panels(pick_hi):
        tiles = [(rt, choose(rows, cond, PER_ROW, rng)) for rt, cond in row_rules]
        if not any(t for _, t in tiles):
            continue
        p = outdir / f"{name}.jpg"
        draw_panel(title, tiles, paths, p)
        written.append(p)
        index += [f"## {title}", "", f"![{title}]({p.name})", ""]
        for rt, ts in tiles:
            index += [f"- {rt or 'images'}: " + ", ".join(f"`{r['sample_id']}`" for r in ts)]
        index.append("")
    atomic_write_text(outdir / "index.md", "\n".join(index) + "\n")
    return written + [outdir / "index.md"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/prompts_v1_test")
    ap.add_argument("--prompts", default="data/prompts/prompts_v1.jsonl")
    ap.add_argument("--samples", default="data/manifests/gen_*.jsonl")
    ap.add_argument("--out", default="results/examples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    for p in run(args.results, args.prompts, args.samples, args.out, args.seed):
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
