"""Blind human check of the multi-shot videos (D-038): does a person, blind to the arm, find the
agent's videos (C) tell the brief better, and keep the main character more consistent, than plain
text-to-video (A)? One annotator (the author). The protocol is docs/story_check_protocol.md.

    python -m genprobe.story_check make       (HPC3 CPU node, video env, after the test stage)
        -> data/story_check/story_check_package.zip, data/story_check/key.json (closed until labelling is done)
    python -m genprobe.story_check analyze --labels data/labels/story_check_labels.csv
        -> results/story/human_check.md, human_check.json

Items: the 24 test briefs, each with A's and C's shots joined into one video (sides random, seed 0).
Part 1 asks which video tells the brief better (A, B, same, neither) and adds 4 attention checks (a
video the judge gave story score 5 against another brief's video) and 3 hidden repeats; part 2 shows
the same 24 pairs with sides drawn again and asks in which video the main character stays the same
from shot to shot (A, B, same), with 3 hidden repeats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from . import story as S
from .dpo_check import choice, read_labels
from .human_eval import cohen_kappa
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text

N_GOLD, GOLD_PASS, N_REPEAT = 4, 3, 3          # fixed in docs/story_check_protocol.md
OUT = Path("data/story_check")
NEW, BASE = "C", "A"
N_BOOT = 2000
WIDTH = 960                                    # every video re-encoded alike, so files don't tell arms apart


def build(seqs: dict[tuple[str, str], list[str]], brief_text: dict[str, str], story_score: dict[tuple[str, str], int],
          seed: int = 0) -> tuple[list[dict], list[dict]]:
    """seqs: (brief, arm) -> the arm's shot videos in order; story_score: (brief, arm) -> the judge's
    story score. -> (public items, key)."""
    rng = random.Random(seed)
    briefs = sorted({b for b, a in seqs if a in (NEW, BASE) and (b, NEW) in seqs and (b, BASE) in seqs})

    def item(kind: str, q: str, b: str, left: tuple, right: tuple, **extra) -> dict:
        return {"kind": kind, "q": q, "brief_id": b, "prompt": brief_text[b], "left": list(left), "right": list(right), **extra}

    def sides(b: str, q: str) -> dict:
        x, y = (b, NEW), (b, BASE)
        return item("main", q, b, *((x, y) if rng.random() < 0.5 else (y, x)))

    part1 = [sides(b, "story") for b in briefs]
    # attention checks: a video the judge passed (story score 5; else the highest) against another
    # brief's agent video, the right answer being the real one
    cands = sorted(seqs, key=lambda k: (-story_score.get(k, 0), k))
    real_pool = [k for k in cands if story_score.get(k, 0) == 5] or cands
    picked, used = [], set()
    for k in rng.sample(real_pool, len(real_pool)):
        if k[0] not in used:
            picked.append(k)
            used.add(k[0])
        if len(picked) == N_GOLD:
            break
    for real in picked:
        other = rng.choice([b for b in briefs if b != real[0]])
        decoy = (other, NEW)
        l, r = (real, decoy) if rng.random() < 0.5 else (decoy, real)
        part1.append(item("gold", "story", real[0], l, r, real=list(real)))
    rng.shuffle(part1)
    part2 = [sides(b, "consist") for b in briefs]
    rng.shuffle(part2)
    order = []
    for part in (part1, part2):
        mains = [i for i, it in enumerate(part) if it["kind"] == "main"]
        originals = rng.sample([i for i in mains if i < len(part) // 2] or mains, N_REPEAT)
        seq = list(part)
        for i in originals:
            rep = {**part[i], "kind": "repeat", "of": id(part[i]), "left": part[i]["right"], "right": part[i]["left"]}
            lo = min(max(i + 12, len(seq) - 8), len(seq))
            seq.insert(rng.randint(lo, len(seq)), rep)
        order += seq
    public, key, id_of = [], [], {}
    for n, it in enumerate(order, 1):
        iid = f"q{n:03d}"
        if it["kind"] != "repeat":
            id_of[id(it)] = iid
        public.append({"item_id": iid, "part": "main", "q": it["q"], "prompt": it["prompt"],
                       "left": f"v/{iid}_A.mp4", "right": f"v/{iid}_B.mp4"})
        k = {"item_id": iid, "kind": it["kind"], "q": it["q"], "brief_id": it["brief_id"],
             "left": it["left"], "right": it["right"],
             "left_shots": seqs[tuple(it["left"])], "right_shots": seqs[tuple(it["right"])]}
        if "real" in it:
            k["real"] = it["real"]
        if it["kind"] == "repeat":
            k["repeat_of"] = id_of[it["of"]]
        key.append(k)
    return public, key


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def join_shots(paths: list[str], out: Path) -> None:
    """The shots one after another, re-encoded alike (H.264, WIDTH wide, no sound)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("".join(f"file '{Path(p).resolve()}'\n" for p in paths))
        lst = f.name
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", f"scale={WIDTH}:-2", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True)
    Path(lst).unlink()


def write_package(public: list[dict], key: list[dict], man: dict[str, dict], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "joined"
    cache.mkdir(exist_ok=True)
    tool = (Path(__file__).parent / "story_check_page.html").read_text(encoding="utf-8")
    zpath = out / "story_check_package.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    by_id = {k["item_id"]: k for k in key}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("story_check/index.html", tool)
        z.writestr("story_check/items.js", "window.ITEMS = " + json.dumps(public, ensure_ascii=False, indent=0) + ";\n")
        for it in public:
            k = by_id[it["item_id"]]
            for side, shots in ((it["left"], k["left_shots"]), (it["right"], k["right_shots"])):
                f = cache / (hashlib.sha1("+".join(shots).encode()).hexdigest()[:16] + ".mp4")
                if not f.exists():
                    join_shots([man[s]["path"] for s in shots], f)
                z.write(f, f"story_check/{side}")
    tmp.replace(zpath)
    atomic_write_text(out / "key.json", json.dumps({"created_at": now_iso(), "git_commit": git_commit(), "new": NEW,
                                                    "base": BASE, "items": key}, indent=1))
    return zpath


def make(name: str = "test") -> Path:
    if (OUT / "key.json").exists():
        raise SystemExit(f"{OUT / 'key.json'} exists: the items were drawn already")
    fin = S.finals(name)
    rows = S.shot_rows(name)
    seqs = {(b["prompt_id"], a): [fin[a][r["prompt_id"]] for r in rows if r["brief_id"] == b["prompt_id"]]
            for b in S.briefs(name) for a in fin}
    g = S.judge_records(name)
    score = {k: (g.get(f"abs:story:greedy:{'+'.join(v)}") or {}).get("score") or 0 for k, v in seqs.items()}
    public, key = build(seqs, {b["prompt_id"]: b["prompt"] for b in S.briefs(name)}, score)
    z = write_package(public, key, S.video_manifest())
    print(f"wrote {z} ({dict(Counter((k['q'], k['kind']) for k in key))}) and {OUT / 'key.json'} (keep it closed)")
    return z


def _ci(per: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    ids = sorted(per)
    if not ids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per[i] for i in ids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def analyze(labels: dict[str, str], key: list[dict], judge_story: dict[str, dict[str, int]] | None = None,
            n_boot: int = N_BOOT) -> dict[str, Any]:
    """judge_story: brief -> {"A": score, "C": score}, for the exploratory agreement with the judge."""
    by = {k["item_id"]: k for k in key}
    gold = [k for k in key if k["kind"] == "gold" and k["item_id"] in labels]
    right = sum(choice(k, labels[k["item_id"]]) == k["real"] for k in gold)
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "new": NEW, "base": BASE,
                           "labelled": sum(1 for k in key if k["item_id"] in labels), "items": len(key),
                           "gold": {"answered": len(gold), "right": right, "pass_mark": GOLD_PASS,
                                    "passed": len(gold) == N_GOLD and right >= GOLD_PASS}}

    def canon(k: dict, lab: str) -> str:
        c = choice(k, lab)
        return c if c in ("same", "neither") else ("new" if c[1] == NEW else "base")

    for q in ("story", "consist"):
        reps = [(by[k["repeat_of"]], k) for k in key if k["kind"] == "repeat" and k["q"] == q
                and k["item_id"] in labels and k["repeat_of"] in labels]
        a = [canon(o, labels[o["item_id"]]) for o, _ in reps]
        b = [canon(r, labels[r["item_id"]]) for _, r in reps]
        mains = [k for k in key if k["kind"] == "main" and k["q"] == q and k["item_id"] in labels]
        per = {k["brief_id"]: canon(k, labels[k["item_id"]]) for k in mains}
        decided = {p: float(v == "new") for p, v in per.items() if v in ("new", "base")}
        out = {"answers": dict(Counter(per.values())),
               "repeat": {"pairs": len(reps), "same_answer": float(np.mean([x == y for x, y in zip(a, b)])) if reps else math.nan,
                          "cohen_kappa": cohen_kappa(a, b)},
               "prefers_agent": _ci(decided, n_boot)}
        if q == "story" and judge_story:
            agree = [float((v == "new") == (judge_story[p][NEW] > judge_story[p][BASE]))
                     for p, v in per.items() if v in ("new", "base") and judge_story[p][NEW] != judge_story[p][BASE]]
            out["agrees_with_judge"] = {"pairs": len(agree), "share": float(np.mean(agree)) if agree else math.nan}
        res[q] = out
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    g = res["gold"]
    L = ["# Blind human check: the agent (C) against text to video (A), multi-shot videos (one annotator, D-038)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.story_check analyze`; protocol "
         f"in `docs/story_check_protocol.md`. Labelled {res['labelled']} of {res['items']} items. 95 % CIs from "
         "resampling briefs.", "",
         f"- Attention checks (part 1): {g['right']} of {g['answered']} right (pass mark {g['pass_mark']} of {N_GOLD}): "
         f"**{'passed' if g['passed'] else 'NOT passed'}**."
         + ("" if g["passed"] else " By the protocol, the answers below are not used for any conclusion."), ""]
    for q, title in (("story", "Which video tells the brief better?"),
                     ("consist", "In which video does the main character stay the same from shot to shot?")):
        r = res[q]
        p = r["prefers_agent"]
        L += [f"## {title}", "",
              "Answers: " + ", ".join(f"{k} {v}" for k, v in sorted(r["answers"].items())) +
              f". Hidden repeats: same answer on {_f(r['repeat']['same_answer'])} of {r['repeat']['pairs']} (κ "
              f"{_f(r['repeat']['cohen_kappa'])}).", "",
              f"Share of decided pairs where the agent's video was picked: **{_f(p['value'])}** [{_f(p['ci'][0], 2)}, "
              f"{_f(p['ci'][1], 2)}] over {p['n']} briefs (0.5 = no difference).", ""]
        if "agrees_with_judge" in r:
            a = r["agrees_with_judge"]
            L += [f"Exploratory: where both the person and the judge's story scores separate the two, they agree on "
                  f"{_f(a['share'])} of {a['pairs']} briefs.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make")
    an = sub.add_parser("analyze")
    an.add_argument("--labels", required=True)
    an.add_argument("--key", default=str(OUT / "key.json"))
    an.add_argument("--out", default="results/story")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        make()
        return 0
    kj = json.loads(Path(args.key).read_text())
    g = S.judge_records("test")
    js: dict[str, dict[str, int]] = {}
    for k in kj["items"]:
        if k["kind"] == "main":
            for side, arm in (("left_shots", k["left"][1]), ("right_shots", k["right"][1])):
                rec = g.get(f"abs:story:greedy:{'+'.join(k[side])}") or {}
                js.setdefault(k["brief_id"], {})[arm] = rec.get("score") or 0
    res = analyze(read_labels(args.labels), kj["items"], js)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "human_check.json", json.dumps(res, indent=2))
    atomic_write_text(out / "human_check.md", render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
