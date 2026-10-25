"""Blind human check of the DPO model against the base model (D-037): does a person also find the
DPO images closer to the text, and do they look as good? One annotator (the author), blind to which
model made which image. The protocol is docs/dpo_check_protocol.md.

    python -m genprobe.dpo_check make        (HPC3 CPU node, after the DPO test images exist)
        -> data/dpo_check/dpo_check_package.zip, data/dpo_check/key.json (closed until labelling is done)
    python -m genprobe.dpo_check analyze --labels data/labels/dpo_check_labels.csv
        -> results/dpo/human_check.md, human_check.json

Items (seed 0): 60 test prompts, 15 per stratum (counting, spatial, text, GenAI-Bench), drawn at
random; for each, the base and the DPO image of seed 0, sides random. Part 1 asks which image matches
the text better (A, B, same, neither) and adds 6 attention checks (one image belongs to a prompt of
another stratum) and 6 hidden repeats; part 2 shows the same 60 pairs again with sides re-drawn and
asks which image looks better (A, B, same), with 6 hidden repeats.

Round 2 (D-037 note of 2026-10-24) repeats the check on 60 new prompts, because round 1's attention
checks could be answered "neither" by the rules: their real image usually missed its own prompt too.
In round 2 the real image of an attention check is one the judge passed (score 5); everything else is
as in round 1.

    python -m genprobe.dpo_check make --round 2
        -> data/dpo_check_r2/dpo_check_r2_package.zip, data/dpo_check_r2/key.json
    python -m genprobe.dpo_check analyze --round 2 --labels data/labels/dpo_check_r2_labels.csv
        -> results/dpo/human_check_r2.md, human_check_r2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import dpo
from .human_eval import cohen_kappa, thumbnail
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text

STRATA = ("counting", "spatial", "text", "genai")
PER_STRATUM, N_GOLD, GOLD_PASS, N_REPEAT = 15, 6, 5, 6       # fixed in D-037 / docs/dpo_check_protocol.md
OUT = Path("data/dpo_check")
ROUND2_SEED = 1                                               # round 2 (D-037 note of 2026-10-24)
GRADES = "grade_dpo_test_full_gemma4_31b"                     # the judge's grades of the test images
LABELS = {"text": ("A", "B", "same", "neither"), "quality": ("A", "B", "same")}
N_BOOT = 2000


def out_dir(rnd: int = 1) -> Path:
    return OUT if rnd == 1 else Path(f"{OUT}_r{rnd}")


def round_name(rnd: int = 1) -> str:
    return "dpo_check" if rnd == 1 else f"dpo_check_r{rnd}"


def build(images: list[dict], base: str, new: str, seed: int = 0, exclude: frozenset[str] | set[str] = frozenset(),
          gold_ok: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """images: dpo.eval_images rows (sample_id, prompt_id, prompt, path, group, seed, category) of the
    two arms on the test split, seed 0. -> (public items, key).

    exclude: prompt ids never drawn (round 2: every prompt round 1 showed). gold_ok: if given (round 2),
    the real image of each attention check is one of these sample ids (the images the judge passed),
    from a prompt outside the 60 pairs, one check per prompt; otherwise (round 1) it is the base or DPO
    image of one of the 60 pairs."""
    rng = random.Random(seed)
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for im in images:
        if im["seed"] == 0 and im["group"] in (base, new) and im["prompt_id"] not in exclude:
            by[im["prompt_id"]][im["group"]] = im
    full = {p: v for p, v in by.items() if base in v and new in v}
    chosen = []
    for st in STRATA:
        pids = sorted(p for p, v in full.items() if v[base]["category"] == st)
        chosen += rng.sample(pids, min(PER_STRATUM, len(pids)))
    pairs = [{"prompt_id": p, "prompt": full[p][base]["prompt"], "stratum": full[p][base]["category"],
              "base": full[p][base], "new": full[p][new]} for p in chosen]
    others = sorted(p for p in full if p not in set(chosen))

    def sides(pr: dict, kind: str, q: str) -> dict:
        a, b = (pr["base"], pr["new"]) if rng.random() < 0.5 else (pr["new"], pr["base"])
        return {"kind": kind, "q": q, "prompt_id": pr["prompt_id"], "prompt": pr["prompt"], "stratum": pr["stratum"],
                "left": a["sample_id"], "right": b["sample_id"], "left_path": a["path"], "right_path": b["path"]}

    part1 = [sides(pr, "main", "text") for pr in pairs]
    gp: list[str] | None = None
    if gold_ok is not None:                       # round 2: images the judge passed, from outside the pairs
        ok = {p: [full[p][g] for g in (base, new) if full[p][g]["sample_id"] in gold_ok] for p in others}
        gp = rng.sample(sorted(p for p, v in ok.items() if v), N_GOLD)
        others = [o for o in others if o not in set(gp)]
    for n in range(N_GOLD):                       # the prompt's real image against another stratum's image
        if gp is None:                            # round 1 (random draws in the order round 1 was made)
            pr = rng.choice(pairs)
            q = rng.choice([o for o in others if full[o][base]["category"] != pr["stratum"]])
            real = pr[rng.choice(["base", "new"])]
        else:
            p = gp[n]
            pr = {"prompt_id": p, "prompt": full[p][base]["prompt"], "stratum": full[p][base]["category"]}
            q = rng.choice([o for o in others if full[o][base]["category"] != pr["stratum"]])
            real = rng.choice(ok[p])
        decoy = full[q][rng.choice([base, new])]
        a, b = (real, decoy) if rng.random() < 0.5 else (decoy, real)
        part1.append({"kind": "gold", "q": "text", "prompt_id": pr["prompt_id"], "prompt": pr["prompt"],
                      "stratum": pr["stratum"], "left": a["sample_id"], "right": b["sample_id"], "left_path": a["path"],
                      "right_path": b["path"], "real": real["sample_id"]})
    rng.shuffle(part1)
    part2 = [sides(pr, "main", "quality") for pr in pairs]
    rng.shuffle(part2)
    order = []
    for part in (part1, part2):
        mains = [i for i, it in enumerate(part) if it["kind"] == "main"]
        originals = rng.sample([i for i in mains if i < len(part) // 2] or mains, N_REPEAT)
        seq = list(part)
        for i in originals:
            rep = {**part[i], "kind": "repeat", "of": id(part[i])}
            rep["left"], rep["right"], rep["left_path"], rep["right_path"] = (part[i]["right"], part[i]["left"],
                                                                            part[i]["right_path"], part[i]["left_path"])
            lo = min(max(i + 20, len(seq) - 15), len(seq))
            seq.insert(rng.randint(lo, len(seq)), rep)
        order += seq
    public, key, id_of = [], [], {}
    for n, it in enumerate(order, 1):
        iid = f"q{n:03d}"
        if it["kind"] != "repeat":
            id_of[id(it)] = iid
        public.append({"item_id": iid, "part": "main", "q": it["q"], "prompt": it["prompt"],
                       "left": f"img/{iid}_A.jpg", "right": f"img/{iid}_B.jpg"})
        k = {"item_id": iid, "kind": it["kind"], "q": it["q"], "prompt_id": it["prompt_id"], "stratum": it["stratum"],
             "left": it["left"], "right": it["right"], "left_path": it["left_path"], "right_path": it["right_path"]}
        if "real" in it:
            k["real"] = it["real"]
        if it["kind"] == "repeat":
            k["repeat_of"] = id_of[it["of"]]
        key.append(k)
    return public, key


def write_package(public: list[dict], key: list[dict], base: str, new: str, out: Path = OUT, rnd: int = 1) -> Path:
    """Round 2 gets its own folder, zip name, export name and browser storage (window.ROUND), so its
    page never shows round 1's saved answers."""
    out.mkdir(parents=True, exist_ok=True)
    tool = (Path(__file__).parent / "dpo_check_page.html").read_text(encoding="utf-8")
    name = round_name(rnd)
    zpath = out / f"{name}_package.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    by_id = {k["item_id"]: k for k in key}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{name}/index.html", tool)
        z.writestr(f"{name}/items.js", f"window.ROUND = {rnd};\nwindow.ITEMS = "
                   + json.dumps(public, ensure_ascii=False, indent=0) + ";\n")
        for it in public:
            k = by_id[it["item_id"]]
            z.writestr(f"{name}/{it['left']}", thumbnail(k["left_path"]))
            z.writestr(f"{name}/{it['right']}", thumbnail(k["right_path"]))
    tmp.replace(zpath)
    atomic_write_text(out / "key.json", json.dumps({"created_at": now_iso(), "git_commit": git_commit(), "round": rnd,
                                                    "base": base, "new": new, "items": key}, indent=1))
    return zpath


def round1_prompts(key_path: Path = OUT / "key.json") -> set[str]:
    """Every prompt round 1 showed: its pairs, its attention checks' prompts and their decoys."""
    items = json.loads(Path(key_path).read_text())["items"]
    return {k["prompt_id"] for k in items} | {sid.split("__")[1] for k in items for sid in (k["left"], k["right"])}


def passed_ids(tag: str = GRADES, data: str = "data") -> set[str]:
    from .utils.manifest import latest_by_id, read_many
    grades = latest_by_id(read_many([Path(data, "judge", f"{tag}.jsonl")]))
    return {sid for sid, g in grades.items() if g.get("score") == 5}


def read_labels(path: str | Path) -> dict[str, str]:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip()
            if lab:
                if lab not in LABELS["text"]:
                    raise ValueError(f"{path}: {r['item_id']}: label {lab!r}")
                out[r["item_id"]] = lab
    return out


def choice(k: dict, label: str) -> str:
    return k["left"] if label == "A" else k["right"] if label == "B" else label


def analyze(labels: dict[str, str], key: list[dict], new: str, n_boot: int = N_BOOT, rnd: int = 1) -> dict[str, Any]:
    by = {k["item_id"]: k for k in key}
    gold = [k for k in key if k["kind"] == "gold" and k["item_id"] in labels]
    right = sum(choice(k, labels[k["item_id"]]) == k["real"] for k in gold)
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "round": rnd, "new": new,
                           "labelled": sum(1 for k in key if k["item_id"] in labels), "items": len(key),
                           "gold": {"answered": len(gold), "right": right, "pass_mark": GOLD_PASS,
                                    "passed": len(gold) == N_GOLD and right >= GOLD_PASS}}
    canon = lambda k, lab: (lambda c: c if c in ("same", "neither") else  # noqa: E731
                            ("new" if c.startswith(new + "__") else "base"))(choice(k, lab))
    for q in ("text", "quality"):
        reps = [(by[k["repeat_of"]], k) for k in key if k["kind"] == "repeat" and k["q"] == q
                and k["item_id"] in labels and k["repeat_of"] in labels]
        a = [canon(o, labels[o["item_id"]]) for o, _ in reps]
        b = [canon(r, labels[r["item_id"]]) for _, r in reps]
        mains = [k for k in key if k["kind"] == "main" and k["q"] == q and k["item_id"] in labels]
        answers = [canon(k, labels[k["item_id"]]) for k in mains]
        per = {k["prompt_id"]: canon(k, labels[k["item_id"]]) for k in mains}
        decided = {p: float(v == "new") for p, v in per.items() if v in ("new", "base")}
        out = {"answers": dict(Counter(answers)),
               "repeat": {"pairs": len(reps), "same_answer": float(np.mean([x == y for x, y in zip(a, b)])) if reps else math.nan,
                          "cohen_kappa": cohen_kappa(a, b)},
               "prefers_new": _ci(decided, n_boot), "by_stratum": {}}
        for st in STRATA:
            out["by_stratum"][st] = _ci({p: v for p, v in decided.items()
                                         if any(k["prompt_id"] == p and k["stratum"] == st for k in mains)}, n_boot)
        res[q] = out
    return res


def _ci(per: dict[str, float], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    pids = sorted(per)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0}
    x = np.array([per[p] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.quantile(x[idx].mean(1), [0.025, 0.975])
    return {"value": float(x.mean()), "ci": [float(lo), float(hi)], "n": len(x)}


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    g = res["gold"]
    rnd = res.get("round", 1)
    L = [f"# Blind human check{f', round {rnd}' if rnd > 1 else ''}: DPO vs base SD 1.5 (one annotator, D-037)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.dpo_check analyze`; protocol in "
         f"`docs/dpo_check_protocol.md`. Labelled {res['labelled']} of {res['items']} items. 95 % CIs from resampling prompts.",
         "", f"- Attention checks (part 1): {g['right']} of {g['answered']} right (pass mark {g['pass_mark']} of {N_GOLD}): "
         f"**{'passed' if g['passed'] else 'NOT passed'}**."
         + ("" if g["passed"] else " By the protocol, the answers below are not used for any conclusion."), ""]
    for q, title in (("text", "Which image matches the text better?"), ("quality", "Which image looks better?")):
        r = res[q]
        p = r["prefers_new"]
        L += [f"## {title}", "",
              "Answers: " + ", ".join(f"{k} {v}" for k, v in sorted(r["answers"].items())) +
              f". Hidden repeats: same answer on {_f(r['repeat']['same_answer'])} of {r['repeat']['pairs']} (κ "
              f"{_f(r['repeat']['cohen_kappa'])}).", "",
              f"Share of decided pairs where the DPO image was picked: **{_f(p['value'])}** [{_f(p['ci'][0], 2)}, "
              f"{_f(p['ci'][1], 2)}] over {p['n']} pairs (0.5 = no difference).", "",
              "By stratum: " + "; ".join(f"{st} {_f(v['value'], 2)} (n={v['n']})" for st, v in r["by_stratum"].items()) + ".", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--base", default="dpo_base")
    mk.add_argument("--round", type=int, default=1, choices=(1, 2))
    mk.add_argument("--grades", default=GRADES, help="round 2: the judge's grades of the test images")
    an = sub.add_parser("analyze")
    an.add_argument("--labels", required=True)
    an.add_argument("--round", type=int, default=1, choices=(1, 2))
    an.add_argument("--key", default=None, help="default: the round's key.json")
    an.add_argument("--out", default="results/dpo")
    args = ap.parse_args(argv)
    rnd = args.round
    if args.cmd == "make":
        out = out_dir(rnd)
        if (out / "key.json").exists():
            raise SystemExit(f"{out / 'key.json'} exists: round {rnd} was already drawn")
        new = json.loads((dpo.ROOT / "selected.json").read_text())["selected"]
        images = dpo.eval_images([args.base, new], "test", seed=0)
        if rnd == 1:
            public, key = build(images, args.base, new)
        else:
            public, key = build(images, args.base, new, seed=ROUND2_SEED, exclude=round1_prompts(),
                                gold_ok=passed_ids(args.grades))
        z = write_package(public, key, args.base, new, out=out, rnd=rnd)
        print(f"wrote {z} ({dict(Counter((k['q'], k['kind']) for k in key))}) and {out / 'key.json'} (keep it closed)")
        return 0
    kj = json.loads(Path(args.key or out_dir(rnd) / "key.json").read_text())
    res = analyze(read_labels(args.labels), kj["items"], kj["new"], rnd=rnd)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "human_check" if rnd == 1 else f"human_check_r{rnd}"
    atomic_write_text(out / f"{stem}.json", json.dumps(res, indent=2))
    atomic_write_text(out / f"{stem}.md", render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
