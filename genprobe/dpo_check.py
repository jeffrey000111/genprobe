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
LABELS = {"text": ("A", "B", "same", "neither"), "quality": ("A", "B", "same")}
N_BOOT = 2000


def build(images: list[dict], base: str, new: str, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """images: dpo.eval_images rows (sample_id, prompt_id, prompt, path, group, seed, category) of the
    two arms on the test split, seed 0. -> (public items, key)."""
    rng = random.Random(seed)
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for im in images:
        if im["seed"] == 0 and im["group"] in (base, new):
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
    for _ in range(N_GOLD):                       # the prompt's real image against another stratum's image
        pr = rng.choice(pairs)
        q = rng.choice([o for o in others if full[o][base]["category"] != pr["stratum"]])
        real, decoy = pr[rng.choice(["base", "new"])], full[q][rng.choice([base, new])]
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


def write_package(public: list[dict], key: list[dict], base: str, new: str, out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tool = (Path(__file__).parent / "dpo_check_page.html").read_text(encoding="utf-8")
    zpath = out / "dpo_check_package.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    by_id = {k["item_id"]: k for k in key}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("dpo_check/index.html", tool)
        z.writestr("dpo_check/items.js", "window.ITEMS = " + json.dumps(public, ensure_ascii=False, indent=0) + ";\n")
        for it in public:
            k = by_id[it["item_id"]]
            z.writestr(f"dpo_check/{it['left']}", thumbnail(k["left_path"]))
            z.writestr(f"dpo_check/{it['right']}", thumbnail(k["right_path"]))
    tmp.replace(zpath)
    atomic_write_text(out / "key.json", json.dumps({"created_at": now_iso(), "git_commit": git_commit(), "base": base,
                                                    "new": new, "items": key}, indent=1))
    return zpath


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


def analyze(labels: dict[str, str], key: list[dict], new: str, n_boot: int = N_BOOT) -> dict[str, Any]:
    by = {k["item_id"]: k for k in key}
    gold = [k for k in key if k["kind"] == "gold" and k["item_id"] in labels]
    right = sum(choice(k, labels[k["item_id"]]) == k["real"] for k in gold)
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "new": new,
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
    L = ["# Blind human check: DPO vs base SD 1.5 (one annotator, D-037)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.dpo_check analyze`; protocol in "
         f"`docs/dpo_check_protocol.md`. Labelled {res['labelled']} of {res['items']} items. 95 % CIs from resampling prompts.",
         "", f"- Attention checks (part 1): {g['right']} of {g['answered']} right (pass mark {g['pass_mark']} of {N_GOLD}): "
         f"**{'passed' if g['passed'] else 'NOT passed'}**.", ""]
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
    an = sub.add_parser("analyze")
    an.add_argument("--labels", required=True)
    an.add_argument("--key", default=str(OUT / "key.json"))
    an.add_argument("--out", default="results/dpo")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        new = json.loads((dpo.ROOT / "selected.json").read_text())["selected"]
        images = dpo.eval_images([args.base, new], "test", seed=0)
        public, key = build(images, args.base, new)
        z = write_package(public, key, args.base, new)
        print(f"wrote {z} ({dict(Counter((k['q'], k['kind']) for k in key))}) and {OUT / 'key.json'} (keep it closed)")
        return 0
    kj = json.loads(Path(args.key).read_text())
    res = analyze(read_labels(args.labels), kj["items"], kj["new"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "human_check.json", json.dumps(res, indent=2))
    atomic_write_text(out / "human_check.md", render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
