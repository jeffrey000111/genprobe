"""Our own small human evaluation (guide §9.5, D-029): does what §3 found on GenAI-Bench hold on
our four models and our template prompts? One annotator (the author), blind, with attention
checks and hidden repeats in place of a second annotator. The protocol is
docs/human_eval_protocol.md.

    python -m genprobe.human_eval make       (on HPC3, CPU node: scripts/make_human_eval.sh)
        -> data/human_eval/human_eval_package.zip   offline labelling page + 768 px images, no model
                                                     names anywhere in it
           data/human_eval/key.json                  which images each item shows (keep it closed
                                                     until labelling is done)
    python -m genprobe.human_eval analyze --labels data/labels/human_eval_labels.csv --key data/human_eval/key.json
        -> results/human_eval/summary.md, human_eval_agreement.csv, summary.json

What is asked, per item: which of two images (same prompt, same seed, two different models) matches
the prompt better: A, B, both equally (same), or neither. Items (seed 0):
    practice  20 pairs from the dev split (5 per stratum), labelled first, not analysed
    main     200 pairs from the test split, 50 per stratum (counting, spatial, text, GenAI-Bench)
    gold      20 attention checks: one image is the same model's image for a prompt of another stratum
    repeat    40 main pairs shown a second time, at least ~50 items later, sides re-drawn
Images flagged by either safety filter are never used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .summarize import N_BOOT
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text

STRATA = {"counting": ("counting",), "spatial": ("spatial",), "text": ("text",), "genai": ("basic", "advanced")}
N_MAIN, N_GOLD, N_REPEAT, N_PRACTICE = 50, 20, 40, 5      # main and practice are per stratum
GOLD_PASS = 18                                             # of 20, fixed before labelling (D-029)
SIDE = 768
LABELS = ("A", "B", "same", "neither")
METRICS = ("vqascore", "clipscore", "pickscore", "hpsv2", "detect", "ocr")
OUT = Path("data/human_eval")


def stratum_of(category: str) -> str | None:
    return next((s for s, cats in STRATA.items() if category in cats), None)


def flagged(r: dict) -> bool:
    return any(r.get(k) not in (None, "") and float(r[k]) >= 0.5 for k in ("nsfw_sd", "nsfw_vit"))


def load_rows(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["status"] == "ok" and not flagged(r) and stratum_of(r["category"])]


class Index:
    def __init__(self, rows: list[dict]):
        self.rows = {r["sample_id"]: r for r in rows}
        self.by = defaultdict(dict)                  # (prompt_id, seed) -> model -> sample_id
        self.prompts = defaultdict(set)              # stratum -> prompt ids
        for r in rows:
            self.by[(r["prompt_id"], r["seed"])][r["model"]] = r["sample_id"]
            self.prompts[stratum_of(r["category"])].add(r["prompt_id"])

    def seeds(self, pid: str, min_models: int = 2) -> list[str]:
        return sorted(s for (p, s), m in self.by.items() if p == pid and len(m) >= min_models)


def draw_pairs(ix: Index, per_stratum: int, rng: random.Random) -> list[dict]:
    """per_stratum pairs per stratum: prompts in random order (cycled if there are fewer prompts
    than pairs), a random seed, two random models; never the same pair twice."""
    out, seen = [], set()
    for stratum in STRATA:
        pids = sorted(ix.prompts[stratum])
        rng.shuffle(pids)
        k = 0
        while sum(p["stratum"] == stratum for p in out) < per_stratum and k < 10 * per_stratum:
            pid = pids[k % len(pids)]
            k += 1
            seeds = ix.seeds(pid)
            if not seeds:
                continue
            seed = rng.choice(seeds)
            models = sorted(ix.by[(pid, seed)])
            a, b = sorted(rng.sample(models, 2))
            key = (pid, seed, a, b)
            if key in seen:
                continue
            seen.add(key)
            r = ix.rows[ix.by[(pid, seed)][a]]
            out.append({"prompt_id": pid, "prompt": r["prompt"], "category": r["category"], "stratum": stratum,
                        "seed": seed, "sid_1": ix.by[(pid, seed)][a], "sid_2": ix.by[(pid, seed)][b]})
    return out


def draw_gold(ix: Index, n: int, rng: random.Random) -> list[dict]:
    """The real image for a prompt vs the same model's image (same seed) for a prompt of another
    stratum, so the right answer is obvious."""
    out = []
    keys = sorted(ix.by)
    while len(out) < n:
        pid, seed = rng.choice(keys)
        real_r = ix.rows[next(iter(ix.by[(pid, seed)].values()))]
        model = rng.choice(sorted(ix.by[(pid, seed)]))
        st = stratum_of(real_r["category"])
        others = sorted(q for (q, s), m in ix.by.items() if s == seed and model in m
                        and stratum_of(ix.rows[m[model]]["category"]) != st)
        if not others:
            continue
        q = rng.choice(others)
        out.append({"prompt_id": pid, "prompt": real_r["prompt"], "category": real_r["category"], "stratum": st,
                    "seed": seed, "real": ix.by[(pid, seed)][model], "decoy": ix.by[(q, seed)][model]})
    return out


def build(test_rows: list[dict], dev_rows: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    """-> (public items, key). Public items carry only id, part, prompt and image file names."""
    rng = random.Random(seed)
    test, dev = Index(test_rows), Index(dev_rows)
    main = [{"kind": "main", **p} for p in draw_pairs(test, N_MAIN, rng)]
    gold = [{"kind": "gold", **g} for g in draw_gold(test, N_GOLD, rng)]
    base = main + gold
    rng.shuffle(base)
    half = len(base) // 2
    originals = rng.sample([i for i, it in enumerate(base[:half]) if it["kind"] == "main"], N_REPEAT)
    for i, it in enumerate(base):
        it["_pos"] = i
    repeats = [{**base[i], "kind": "repeat", "of_pos": i} for i in originals]
    rng.shuffle(repeats)
    order = base[:]
    tail = len(base) - 60
    for rep in repeats:
        order.insert(rng.randint(tail, len(order)), rep)
    practice = [{"kind": "practice", **p} for p in draw_pairs(dev, N_PRACTICE, rng)]
    rng.shuffle(practice)

    public, key = [], []
    pos_to_id = {}
    for part, items, prefix in (("practice", practice, "p"), ("main", order, "q")):
        for n, it in enumerate(items, 1):
            iid = f"{prefix}{n:03d}"
            if it["kind"] != "repeat" and "_pos" in it:
                pos_to_id[it["_pos"]] = iid
            if it["kind"] == "gold":
                left, right = (it["real"], it["decoy"]) if rng.random() < 0.5 else (it["decoy"], it["real"])
            else:
                left, right = (it["sid_1"], it["sid_2"]) if rng.random() < 0.5 else (it["sid_2"], it["sid_1"])
            public.append({"item_id": iid, "part": part, "prompt": it["prompt"],
                           "left": f"img/{iid}_A.jpg", "right": f"img/{iid}_B.jpg"})
            k = {"item_id": iid, "part": part, "kind": it["kind"], "prompt_id": it["prompt_id"],
                 "category": it["category"], "stratum": it["stratum"], "seed": it["seed"], "left": left, "right": right}
            if it["kind"] == "gold":
                k["real"] = it["real"]
            if it["kind"] == "repeat":
                k["of_pos"] = it["of_pos"]
            key.append(k)
    for k in key:
        if "of_pos" in k:
            k["repeat_of"] = pos_to_id[k.pop("of_pos")]
    return public, key


def thumbnail(path: str, side: int = SIDE) -> bytes:
    """Every image becomes side x side JPEG, so file sizes and dimensions don't tell models apart."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB").resize((side, side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def write_package(public: list[dict], key: list[dict], paths: dict[str, str], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tool = (Path(__file__).parent / "human_eval_page.html").read_text(encoding="utf-8")
    zpath = out / "human_eval_package.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    by_id = {k["item_id"]: k for k in key}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("human_eval/index.html", tool)
        z.writestr("human_eval/items.js", "window.ITEMS = " + json.dumps(public, ensure_ascii=False, indent=0) + ";\n")
        for it in public:
            k = by_id[it["item_id"]]
            z.writestr(f"human_eval/{it['left']}", thumbnail(paths[k["left"]]))
            z.writestr(f"human_eval/{it['right']}", thumbnail(paths[k["right"]]))
    tmp.replace(zpath)
    atomic_write_text(out / "key.json", json.dumps({"created_at": now_iso(), "git_commit": git_commit(),
                                                    "items": key}, indent=1))
    return zpath


# ---------- analysis ----------

def read_labels(path: str | Path) -> dict[str, str]:
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lab = (r.get("label") or "").strip()
            if lab:
                if lab not in LABELS:
                    raise ValueError(f"{path}: {r['item_id']}: label {lab!r} not one of {LABELS}")
                out[r["item_id"]] = lab
    return out


def choice(k: dict, label: str) -> str:
    """'A'/'B' -> the chosen sample_id; 'same'/'neither' unchanged."""
    return k["left"] if label == "A" else k["right"] if label == "B" else label


def ratio_ci(per: dict[str, tuple[float, float]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    pids = sorted(p for p, (_, d) in per.items() if d > 0)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0, "n_prompts": 0}
    num = np.array([per[p][0] for p in pids], float)
    den = np.array([per[p][1] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(pids), size=(n_boot, len(pids)))
    b = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return {"value": float(num.sum() / den.sum()), "ci": [float(lo), float(hi)], "n": int(den.sum()),
            "n_prompts": len(pids)}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return math.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else math.nan


def analyze(labels: dict[str, str], key: list[dict], scores: dict[str, dict], n_boot: int = N_BOOT) -> dict[str, Any]:
    by = {k["item_id"]: k for k in key}
    main = [k for k in key if k["part"] == "main"]
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(),
                           "labelled": {part: sum(1 for k in key if k["part"] == part and k["item_id"] in labels)
                                        for part in ("practice", "main")},
                           "items": {part: sum(1 for k in key if k["part"] == part) for part in ("practice", "main")}}
    gold = [k for k in main if k["kind"] == "gold" and k["item_id"] in labels]
    right = sum(choice(k, labels[k["item_id"]]) == k["real"] for k in gold)
    res["gold"] = {"answered": len(gold), "right": right, "pass_mark": GOLD_PASS,
                   "passed": len(gold) == N_GOLD and right >= GOLD_PASS}
    canon = lambda k, lab: (lambda c: c if c in ("same", "neither") else  # noqa: E731
                            ("first" if c == min(k["left"], k["right"]) else "second"))(choice(k, lab))
    reps = [(by[k["repeat_of"]], k) for k in main if k["kind"] == "repeat"
            and k["item_id"] in labels and k["repeat_of"] in labels]
    a = [canon(o, labels[o["item_id"]]) for o, _ in reps]
    b = [canon(r, labels[r["item_id"]]) for _, r in reps]
    res["repeat"] = {"pairs": len(reps), "same_answer": float(np.mean([x == y for x, y in zip(a, b)])) if reps else math.nan,
                     "cohen_kappa": cohen_kappa(a, b)}
    pairs = [k for k in main if k["kind"] == "main" and k["item_id"] in labels]
    res["label_counts"] = dict(Counter(labels[k["item_id"]] for k in pairs))
    decided = [k for k in pairs if labels[k["item_id"]] in ("A", "B")]
    rows = []
    for m in METRICS:
        for stratum in ["all"] + list(STRATA):
            per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            for k in decided:
                if stratum != "all" and k["stratum"] != stratum:
                    continue
                win = choice(k, labels[k["item_id"]])
                lose = k["right"] if win == k["left"] else k["left"]
                x, y = scores.get(win, {}).get(m), scores.get(lose, {}).get(m)
                if x is None or y is None:
                    continue
                per[k["prompt_id"]][0] += 0.5 if x == y else float(x > y)
                per[k["prompt_id"]][1] += 1
            c = ratio_ci({p: (v[0], v[1]) for p, v in per.items()}, n_boot)
            if c["n"]:
                rows.append({"metric": m, "stratum": stratum, "agreement": c["value"], "ci95_lo": c["ci"][0],
                             "ci95_hi": c["ci"][1], "pairs": c["n"], "n_prompts": c["n_prompts"]})
    res["agreement"] = rows
    res["decided_pairs"] = len(decided)
    return res


def load_scores(paths: list[str]) -> dict[str, dict[str, float]]:
    out = {}
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["sample_id"]] = {m: float(r[m]) for m in METRICS if r.get(m) not in (None, "")}
    return out


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    g, rp = res["gold"], res["repeat"]
    L = ["# Human evaluation on our own images (one annotator)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.human_eval analyze`; "
         f"protocol in `docs/human_eval_protocol.md` (D-029). Labelled: {res['labelled']['main']} of "
         f"{res['items']['main']} main-part items.", "",
         f"- Attention checks: {g['right']} of {g['answered']} right (pass mark {g['pass_mark']} of {N_GOLD}): "
         f"**{'passed' if g['passed'] else 'NOT passed'}**.",
         f"- Hidden repeats: same answer on {_f(rp['same_answer'])} of {rp['pairs']} pairs; Cohen's κ "
         f"{_f(rp['cohen_kappa'])} (stands in for agreement between annotators).",
         "- Answers on the 200 main pairs: " + ", ".join(f"{k} {v}" for k, v in sorted(res["label_counts"].items())) +
         f"; {res['decided_pairs']} pairs where one image was picked are used below.", "",
         "## Does each metric prefer the image the annotator picked?", "",
         "Share of decided pairs where the metric scores the picked image higher (ties 1/2); 95 % CI over prompts. "
         "Detector and OCR only apply to their own prompt types.", "",
         "| Metric | Stratum | Agreement | CI | Pairs |", "| --- | --- | --- | --- | --- |"]
    for r in res["agreement"]:
        L.append(f"| {r['metric']} | {r['stratum']} | {_f(r['agreement'])} | [{_f(r['ci95_lo'])}, {_f(r['ci95_hi'])}] | "
                 f"{r['pairs']} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--test", default="results/prompts_v1_test/per_sample.csv")
    mk.add_argument("--dev", default="results/prompts_v1_dev/per_sample.csv")
    mk.add_argument("--samples", default="data/manifests/gen_*.jsonl")
    mk.add_argument("--prompts", default="data/prompts/prompts_v1.jsonl")
    an = sub.add_parser("analyze")
    an.add_argument("--labels", required=True)
    an.add_argument("--key", default=str(OUT / "key.json"))
    an.add_argument("--scores", nargs="+", default=["results/prompts_v1_test/per_sample.csv"])
    an.add_argument("--out", default="results/human_eval")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        from .score import load_samples
        public, key = build(load_rows(args.test), load_rows(args.dev))
        paths = {s.sample_id: s.path for s in load_samples(args.samples, args.prompts)}
        z = write_package(public, key, paths)
        print(f"wrote {z} ({sum(1 for i in public if i['part'] == 'practice')} practice + "
              f"{sum(1 for i in public if i['part'] == 'main')} items) and {OUT / 'key.json'} (don't open it before "
              "labelling is done)")
        return 0
    key = json.loads(Path(args.key).read_text())["items"]
    res = analyze(read_labels(args.labels), key, load_scores(args.scores))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2))
    with open(out / "human_eval_agreement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "stratum", "agreement", "ci95_lo", "ci95_hi", "pairs", "n_prompts"])
        w.writeheader()
        for r in res["agreement"]:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
    atomic_write_text(out / "summary.md", render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
