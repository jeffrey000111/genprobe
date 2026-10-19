"""Blind labelling sets: a random sample of images for a person to judge, with no metric scores
and no model names in sight (D-024).

Used to measure the checkers on dev before any threshold is tuned (guide §7.5, §7.6: "label
~50 dev images, then report the checker's own accuracy"). Images are numbered and shuffled
across models; the item -> sample_id key is written to a separate file the labeller doesn't
need to open.

    python -m genprobe.label_set make --prompts data/prompts/prompts_v1.jsonl --out data/review/v1_dev
        -> sheet_*.jpg (25 numbered images each), labels.csv (to fill in), key.csv, README.txt
    python -m genprobe.label_set finalize --out data/review/v1_dev --labels data/labels/prompts_v1_labels.csv
        -> joins labels.csv with key.csv; appends to the tracked label file (sample_id, correct, note,
           true_count), refusing to overwrite labels that already exist

On HPC3 `make` runs through scripts/make_label_set.sh (CPU node); `finalize` runs anywhere.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from .contact_sheet import _font
from .samples import Sample, load_prompts
from .score import load_samples

KINDS = {"count": "objects", "spatial": "relation", "text": "text"}
TILE, PAD, CAP = 300, 10, 64
PER_ROW, PER_SHEET = 5, 25
LABEL_FIELDS = ["item", "kind", "prompt", "correct", "true_count", "note"]
README = """\
How to label (rules D-015). Fill labels.csv: one row per numbered image.

correct: y / n / ?   (? = can't tell from the picture)
  count   - y only if exactly the requested number can be counted; partly hidden but
            recognisable objects count. Also write the number you see in true_count
            (leave it empty if you can't count them).
  spatial - both objects must be there (otherwise n); left/right as seen by the viewer;
            "above" = higher in the image, resting on top included.
  text    - every requested word spelled right, nothing missing or repeated; case and
            punctuation don't matter.
note: anything worth remembering (optional).

Judge only the picture and its prompt. No scores or model names are shown on purpose.

标注说明（规则见 D-015）。在 labels.csv 里每张编号图片填一行：

correct：y / n / ?（? = 从图上判断不了）
  count   - 能数出的数量正好等于要求的才算 y；被挡住一部分但认得出的也算。另外请在
            true_count 里写你数到的数量（数不清就空着）。
  spatial - 两个物体都要出现（缺一个就是 n）；左右按看图的人来定；
            "above" = 在图里更高的位置，放在上面也算。
  text    - 要求的每个词都拼对，不缺不重复；大小写和标点不管。
note：想记下的任何情况（可不填）。

只看图片和它的提示词。故意不显示任何分数和模型名。
"""


def kind_of(s: Sample) -> str | None:
    spec = s.spec or {}
    for k, key in KINDS.items():
        if spec.get(key):
            return k
    return None


def choose(samples: list[Sample], n_per_kind: dict[str, int], seed: int) -> list[tuple[str, Sample]]:
    """n images per kind, as evenly spread over models as possible, in shuffled order."""
    rng = random.Random(seed)
    out: list[tuple[str, Sample]] = []
    for kind, n in n_per_kind.items():
        pool = sorted((s for s in samples if kind_of(s) == kind), key=lambda s: s.sample_id)
        by_model: dict[str, list[Sample]] = {}
        for s in pool:
            by_model.setdefault(s.model, []).append(s)
        for v in by_model.values():
            rng.shuffle(v)
        picked: list[Sample] = []
        models = sorted(by_model)
        while len(picked) < n and any(by_model.values()):
            for m in models:                      # round robin over models
                if by_model[m] and len(picked) < n:
                    picked.append(by_model[m].pop())
        rng.shuffle(picked)
        out += [(kind, s) for s in picked]
    return out


def draw_sheets(items: list[tuple[int, str, Sample]], outdir: Path) -> list[Path]:
    font = _font(15)
    paths = []
    for k in range(0, len(items), PER_SHEET):
        chunk = items[k:k + PER_SHEET]
        rows = (len(chunk) + PER_ROW - 1) // PER_ROW
        sheet = Image.new("RGB", (PER_ROW * (TILE + PAD) + PAD, rows * (TILE + CAP + PAD) + PAD), "white")
        draw = ImageDraw.Draw(sheet)
        for j, (item, _kind, s) in enumerate(chunk):
            x = PAD + (j % PER_ROW) * (TILE + PAD)
            y = PAD + (j // PER_ROW) * (TILE + CAP + PAD)
            with Image.open(s.path) as im:
                thumb = im.convert("RGB")
            thumb.thumbnail((TILE, TILE))
            sheet.paste(thumb, (x, y))
            caption = f"#{item}  " + s.prompt
            draw.multiline_text((x, y + TILE + 4), "\n".join(textwrap.wrap(caption, 38)[:3]), fill="black",
                                font=font, spacing=2)
        p = outdir / f"sheet_{k // PER_SHEET + 1:02d}.jpg"
        sheet.save(p, quality=90)
        paths.append(p)
    return paths


def make(prompts: str, manifests: str, out: str, split: str, n_per_kind: dict[str, int], seed: int) -> list[Path]:
    outdir = Path(out)
    if (outdir / "labels.csv").exists():
        raise SystemExit(f"{outdir}/labels.csv exists; refusing to replace a label file")
    ids = {p.prompt_id for p in load_prompts(prompts, split=split)}
    samples = [s for s in load_samples(manifests, prompts) if s.prompt_id in ids]
    chosen = choose(samples, n_per_kind, seed)
    items = [(i + 1, kind, s) for i, (kind, s) in enumerate(chosen)]
    outdir.mkdir(parents=True, exist_ok=True)
    written = draw_sheets(items, outdir)
    with open(outdir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(LABEL_FIELDS)
        for item, kind, s in items:
            w.writerow([item, kind, s.prompt, "", "", ""])
    with open(outdir / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "sample_id"])
        for item, _kind, s in items:
            w.writerow([item, s.sample_id])
    (outdir / "README.txt").write_text(README, encoding="utf-8")
    return written + [outdir / "labels.csv", outdir / "key.csv", outdir / "README.txt"]


def finalize(out: str, labels_path: str) -> int:
    outdir = Path(out)
    with open(outdir / "key.csv", newline="", encoding="utf-8") as f:
        key = {r["item"]: r["sample_id"] for r in csv.DictReader(f)}
    rows = []
    with open(outdir / "labels.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            c = (r.get("correct") or "").strip().replace("？", "?").lower()
            if c not in ("y", "n", "?"):
                raise SystemExit(f"item {r['item']}: correct={r.get('correct')!r}, expected y / n / ?")
            tc = (r.get("true_count") or "").strip()
            if tc and not tc.isdigit():
                raise SystemExit(f"item {r['item']}: true_count={tc!r} is not a whole number")
            rows.append({"sample_id": key[r["item"]], "correct": c, "note": (r.get("note") or "").strip(),
                         "true_count": tc})
    if len(rows) != len(key):
        raise SystemExit(f"labels.csv has {len(rows)} rows, key.csv {len(key)}: every item needs a label")
    dest = Path(labels_path)
    existing: list[dict] = []
    if dest.exists():
        with open(dest, newline="", encoding="utf-8-sig") as f:
            existing = list(csv.DictReader(f))
    clash = {r["sample_id"] for r in existing} & {r["sample_id"] for r in rows}
    if clash:
        raise SystemExit(f"{len(clash)} images already have labels in {dest} (e.g. {sorted(clash)[0]}); "
                         "labels are never overwritten (D-015)")
    fields = ["sample_id", "correct", "note", "true_count"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in existing + rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"added {len(rows)} labels to {dest} ({len(existing) + len(rows)} in total)")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--prompts", default="data/prompts/prompts_v1.jsonl")
    mk.add_argument("--samples", default="data/manifests/gen_*.jsonl")
    mk.add_argument("--split", default="dev")
    mk.add_argument("--out", default="data/review/v1_dev")
    mk.add_argument("--count", type=int, default=25)
    mk.add_argument("--spatial", type=int, default=25)
    mk.add_argument("--text", type=int, default=50)
    mk.add_argument("--seed", type=int, default=0)
    fin = sub.add_parser("finalize")
    fin.add_argument("--out", default="data/review/v1_dev")
    fin.add_argument("--labels", default="data/labels/prompts_v1_labels.csv")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        for p in make(args.prompts, args.samples, args.out, args.split,
                      {"count": args.count, "spatial": args.spatial, "text": args.text}, args.seed):
            print(f"wrote {p}")
    else:
        finalize(args.out, args.labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
