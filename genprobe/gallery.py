"""An HTML gallery of the agent's attempts (guide §10.5): for a few test prompts, every attempt's
image with its action, checks, diagnosis and decision, and the judge's grade of the image the agent
kept. Run where the images are (HPC3, CPU node):

    python -m genprobe.gallery --run testrule --grades grade_testrule_gemma4_31b
        -> results/examples/agent_gallery.html

The page is self-contained: 224-pixel JPEG thumbnails are inlined, like the example panels of D-025.
Prompts are drawn at random (seed 0), 3 per category, among those where the agent tried at least one
repair, so the page shows typical cases rather than chosen ones.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from .agent import core
from .samples import load_prompts
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

PER_CATEGORY, SEED, SIDE = 3, 0, 224
CATEGORIES = ("counting", "spatial", "text", "genai")
V1 = "data/prompts/prompts_v1.jsonl"
PASS_SCORE = 5


def load_audit(run: str, results: Path = Path("results/agent")) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = defaultdict(list)
    for line in (results / run / "audit.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by[r["prompt_id"]].append(r)
    return {p: sorted(v, key=lambda r: r["step"]) for p, v in by.items()}


def kept_step(steps: list[dict]) -> int:
    """The attempt the agent ends with: the last "start" or "new best" (a "kept attempt k" keeps it)."""
    best = 0
    for r in steps:
        res = r.get("result") or ""
        if res.startswith("start") or res.startswith("new best"):
            best = r["step"]
    return best


def choose(by: dict[str, list[dict]], category: dict[str, str], per: int = PER_CATEGORY, seed: int = SEED) -> list[str]:
    rng = random.Random(seed)
    out = []
    for c in CATEGORIES:
        pool = sorted(p for p, v in by.items() if category.get(p) == c and len(v) >= 2)
        out += rng.sample(pool, min(per, len(pool)))
    return out


def image_paths(run: str, data: str = "data") -> dict[str, str]:
    files = sorted((Path(data) / "manifests").glob("gen_*.jsonl")) + \
        sorted((Path(data) / "agent" / run / "manifests").glob("gen_*.jsonl"))
    return {s: r["path"] for s, r in latest_by_id(read_many(files)).items() if r.get("status") == "ok"}


def detector_extra(run: str, data: str = "data") -> dict[str, dict]:
    """The detector's details per image (count found, objects missing): the agent breaks ties between
    failed attempts with them (agent.core.closeness), and the audit log keeps only the pass / fail."""
    files = sorted((Path(data) / "scores" / "detect").glob("*.jsonl")) + \
        sorted((Path(data) / "agent" / run / "scores" / "detect").glob("*.jsonl"))
    return {s: r.get("extra") or {} for s, r in latest_by_id(read_many(files)).items() if r.get("status") == "ok"}


def _detector(x: dict) -> str:
    if "count_pred" in x:
        return f"detector counted {x['count_pred']} of {x['count_target']}"
    if x.get("missing"):
        return "detector: " + " and ".join(f"object {m}" for m in x["missing"]) + " not found"
    return ""


def thumb(path: str, side: int = SIDE) -> str:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _checks(c: dict) -> str:
    names = {"vqascore": "VQAScore", "detect": "detector", "ocr": "OCR"}
    return " · ".join(f"{names.get(k, k)} {v:.2f}" if isinstance(v, (int, float)) else f"{names.get(k, k)} {v}"
                      for k, v in sorted(c.items()))


def build(run: str, grades: str, data: str = "data", results: Path = Path("results/agent"),
          out: Path = Path("results/examples/agent_gallery.html"), prompts_path: str = V1) -> dict:
    by = load_audit(run, results)
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    category = {p: core.category_of(prompts[p].tags) for p in by if p in prompts}
    chosen = choose(by, category)
    paths = image_paths(run, data)
    det = detector_extra(run, data)
    g = latest_by_id(read_many([Path(data) / "judge" / f"{grades}.jsonl"]))
    cards = []
    for pid in chosen:
        steps = by[pid]
        k = kept_step(steps)
        final = next(r for r in steps if r["step"] == k)
        grade = (g.get(final["sample_id"]) or {}).get("score")
        tiles = []
        for r in steps:
            p = r.get("params") or {}
            how = [p.get("model", "")]
            if p.get("cfg"):
                how.append(f"cfg {p['cfg']}")
            if r["action"] in ("reseed", "base"):
                how.append(f"seed {p.get('seed')}")
            rewritten = p.get("gen_prompt") if r["action"] == "rewrite" else None
            img = thumb(paths[r["sample_id"]]) if r["sample_id"] in paths else ""
            tiles.append(
                f'<figure class="tile{" kept" if r["step"] == k else ""}">'
                + (f'<img src="{img}" alt="attempt {r["step"]}">' if img else '<div class="missing">image not found</div>')
                + f'<figcaption><b>{r["step"]}. {html.escape(r["action"].replace("_", " "))}</b> '
                f'<span class="dim">{html.escape(", ".join(x for x in how if x))}</span>'
                + (f'<br><span class="dim">rewritten: “{html.escape(rewritten)}”</span>' if rewritten else "")
                + f'<br>{html.escape(_checks(r.get("checks") or {}))} '
                + (f'<span class="dim">({html.escape(_detector(det[r["sample_id"]]))})</span> '
                   if _detector(det.get(r["sample_id"], {})) else "")
                + f'<span class="{"ok" if r.get("passed") else "no"}">{"passes its checks" if r.get("passed") else "fails its checks"}</span>'
                f'<br><span class="dim">{html.escape(r.get("decision_reason") or "")}</span>'
                f'<br><span class="dim">{r.get("gpu_seconds", 0):.1f} GPU-s, {r.get("budget_left", 0):.1f} left</span>'
                + ('<br><b>kept</b>' if r["step"] == k else "") + "</figcaption></figure>")
        verdict = ("–" if grade is None else f'{grade} of 5 — {"pass" if grade == PASS_SCORE else "fail"}')
        cards.append(f'<section class="card"><h2><span class="cat">{category[pid]}</span> {html.escape(prompts[pid].prompt)}</h2>'
                     f'<p class="dim"><code>{pid}</code> · {len(steps)} attempts · judge (Gemma 4 31B) on the kept image: '
                     f'<b>{verdict}</b></p><div class="row">{"".join(tiles)}</div></section>')
    page = PAGE.replace("{{CARDS}}", "\n".join(cards)).replace(
        "{{META}}", html.escape(f"Run {run}, test split; made {now_iso()} at commit {git_commit()} by python -m "
                                f"genprobe.gallery. {PER_CATEGORY} prompts per category drawn at random (seed {SEED}) "
                                "among those where the agent tried at least one repair."))
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out, page)
    return {"prompts": chosen, "file": str(out), "bytes": len(page)}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent attempts</title>
<style>
  :root { --ink:#1b1b1a; --ink2:#5c5b57; --line:#d9d8d3; --bg:#fafaf8; --card:#ffffff; --ok:#1a7f55; --no:#b3261e; --accent:#2a78d6; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --ink:#ecebe7; --ink2:#a7a6a1; --line:#3a3a37; --bg:#161615; --card:#20201e; --ok:#3cc795; --no:#ff8a80; --accent:#6aa8ef; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 -apple-system, "Segoe UI", sans-serif; }
  main { max-width:1280px; margin:0 auto; padding:16px; }
  h1 { font-size:22px; margin:8px 0 4px; }
  .dim { color:var(--ink2); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:14px 0; }
  .card h2 { font-size:16px; margin:0 0 4px; }
  .cat { font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--accent); margin-right:6px; }
  .row { display:flex; gap:10px; overflow-x:auto; padding-bottom:4px; }
  .tile { margin:0; flex:0 0 232px; border:1px solid var(--line); border-radius:6px; padding:4px; }
  .tile.kept { border:2px solid var(--accent); }
  .tile img { width:224px; height:224px; object-fit:contain; display:block; background:var(--bg); }
  .missing { width:224px; height:224px; display:flex; align-items:center; justify-content:center; color:var(--ink2); }
  figcaption { font-size:12px; padding-top:4px; }
  .ok { color:var(--ok); } .no { color:var(--no); }
</style>
</head>
<body>
<main>
<h1>The generation-debugging agent, attempt by attempt</h1>
<p class="dim">{{META}} Each row is one prompt: the base SDXL image, then the agent's repairs in order. The agent judges
its attempts with its own checks (detector, OCR, VQAScore); the judge's grade is independent of them (REPORT §4).
Between two attempts that both fail, the agent keeps the one that came nearer: for counting, the detected count nearer
the target; for spatial prompts, fewer objects missing; for text, the higher OCR match; then a VQAScore gain of at
least 0.02.</p>
{{CARDS}}
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="testrule")
    ap.add_argument("--grades", default="grade_testrule_gemma4_31b")
    a = ap.parse_args(argv)
    print(json.dumps(build(a.run, a.grades), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
