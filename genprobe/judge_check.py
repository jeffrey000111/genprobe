"""Blind human check of the final judge's pass / fail calls in module C (D-032, D-033): does a person
pass the same agent and baseline images that Gemma 4 31B passes, and do the two tell the same story
about agent vs routed best-of-k vs best-of-4? One annotator (the author), blind to group, model,
score and judge answer. The protocol is docs/judge_check_protocol.md.

    python -m genprobe.judge_check make       (on HPC3, CPU node: scripts/make_judge_check.sh)
        -> data/judge_check/judge_check_package.zip   offline labelling page + 768 px images
           data/judge_check/key.json                  which image each item shows (closed until done)
    python -m genprobe.judge_check analyze --labels data/labels/judge_check_labels.csv
        -> results/judge_check/summary.md, summary.json

Per item: one original prompt and one image. Question: does the image meet every requirement of
the text? yes / no / unsure. Items (seed 0):
    practice  10 images from the dev run devrule2, labelled first, not analysed
    main      the final images of best_of_4, routed_best_of_k and the agent for 50 test prompts
              (stratified in proportion to the test split); an image two groups share is shown once
    gold      10 attention checks: a test image shown with the prompt of another category (answer: no)
    repeat    15 main images shown a second time, later in the order
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

from .human_eval import cohen_kappa, thumbnail
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

GROUPS = ("best_of_4", "routed_best_of_k", "agent_rule")
CATEGORIES = ("genai", "counting", "spatial", "text")
N_PROMPTS, N_GOLD, GOLD_PASS, N_REPEAT, N_PRACTICE = 50, 10, 9, 15, 10     # fixed in D-033
PASS_SCORE = 5                                                             # D-032
LABELS = ("yes", "no", "unsure")
N_BOOT = 2000
OUT = Path("data/judge_check")
RUN, DEV_RUN = "testrule", "devrule2"
GRADES = "data/judge/grade_testrule_gemma4_31b.jsonl"


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def allocate(counts: dict[str, int], n: int) -> dict[str, int]:
    """n split over categories in proportion to counts (largest remainder, ties by name)."""
    total = sum(counts.values())
    raw = {c: n * counts[c] / total for c in counts}
    out = {c: int(math.floor(v)) for c, v in raw.items()}
    for c in sorted(counts, key=lambda c: (-(raw[c] - out[c]), c))[: n - sum(out.values())]:
        out[c] += 1
    return out


def build(final: list[dict], dev_final: list[dict], seed: int = 0) -> tuple[list[dict], list[dict]]:
    """final: rows of data/agent/<run>/final.jsonl (prompt_id, group, sample_id, path, prompt,
    category). -> (public items, key); public items carry only id, part, prompt and image name."""
    rng = random.Random(seed)
    rows = [r for r in final if r["group"] in GROUPS]
    by_prompt: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_prompt[r["prompt_id"]][r["group"]] = r
    cat = {p: next(iter(g.values()))["category"] for p, g in by_prompt.items()}
    counts = Counter(cat.values())
    alloc = allocate({c: counts[c] for c in CATEGORIES if counts[c]}, N_PROMPTS)
    chosen = []
    for c in sorted(alloc):
        pids = sorted(p for p in by_prompt if cat[p] == c)
        chosen += rng.sample(pids, alloc[c])
    images: dict[str, dict] = {}                     # sample_id -> image item (each image once)
    for p in sorted(chosen):
        for g in GROUPS:
            r = by_prompt[p].get(g)
            if r is None:
                continue
            it = images.setdefault(r["sample_id"], {"kind": "main", "prompt_id": p, "prompt": r["prompt"],
                                                    "category": r["category"], "sample_id": r["sample_id"],
                                                    "path": r["path"], "groups": []})
            it["groups"].append(g)
    main = [images[s] for s in sorted(images)]
    gold = []
    pool = sorted({(r["sample_id"], r["prompt_id"]) for r in rows})
    prompts_by_cat = defaultdict(list)
    for p in sorted(by_prompt):
        prompts_by_cat[cat[p]].append(p)
    used = set()
    while len(gold) < N_GOLD:
        sid, p = rng.choice(pool)
        if sid in images or sid in used:
            continue
        other = rng.choice(sorted(c for c in prompts_by_cat if c != cat[p]))
        q = rng.choice(prompts_by_cat[other])
        rq = next(iter(by_prompt[q].values()))
        src = next(x for x in rows if x["sample_id"] == sid)
        used.add(sid)
        gold.append({"kind": "gold", "prompt_id": q, "prompt": rq["prompt"], "category": cat[q], "sample_id": sid,
                     "path": src["path"], "image_prompt_id": p, "groups": []})
    base = main + gold
    rng.shuffle(base)
    half = len(base) // 2
    for i, it in enumerate(base):
        it["_pos"] = i
    originals = rng.sample([i for i, it in enumerate(base[:half]) if it["kind"] == "main"], N_REPEAT)
    repeats = [{**base[i], "kind": "repeat", "of_pos": i} for i in originals]
    rng.shuffle(repeats)
    order = base[:]
    tail = len(base) - 40
    for rep in repeats:
        order.insert(rng.randint(tail, len(order)), rep)
    dev = sorted({r["sample_id"]: r for r in dev_final if r["group"] in GROUPS}.values(), key=lambda r: r["sample_id"])
    practice = [{"kind": "practice", "prompt_id": r["prompt_id"], "prompt": r["prompt"], "category": r["category"],
                 "sample_id": r["sample_id"], "path": r["path"], "groups": [r["group"]]}
                for r in rng.sample(dev, N_PRACTICE)]

    public, key, pos_to_id = [], [], {}
    for part, items, prefix in (("practice", practice, "p"), ("main", order, "q")):
        for n, it in enumerate(items, 1):
            iid = f"{prefix}{n:03d}"
            if it["kind"] != "repeat" and "_pos" in it:
                pos_to_id[it["_pos"]] = iid
            public.append({"item_id": iid, "part": part, "prompt": it["prompt"], "image": f"img/{iid}.jpg"})
            k = {"item_id": iid, "part": part, "kind": it["kind"], "prompt_id": it["prompt_id"],
                 "category": it["category"], "sample_id": it["sample_id"], "path": it["path"], "groups": it["groups"]}
            if it["kind"] == "gold":
                k["image_prompt_id"] = it["image_prompt_id"]
            if it["kind"] == "repeat":
                k["of_pos"] = it["of_pos"]
            key.append(k)
    for k in key:
        if "of_pos" in k:
            k["repeat_of"] = pos_to_id[k.pop("of_pos")]
    return public, key


def write_package(public: list[dict], key: list[dict], out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tool = (Path(__file__).parent / "judge_check_page.html").read_text(encoding="utf-8")
    zpath = out / "judge_check_package.zip"
    tmp = zpath.with_suffix(".zip.tmp")
    by_id = {k["item_id"]: k for k in key}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        z.writestr("judge_check/index.html", tool)
        z.writestr("judge_check/items.js", "window.ITEMS = " + json.dumps(public, ensure_ascii=False, indent=0) + ";\n")
        for it in public:
            z.writestr(f"judge_check/{it['image']}", thumbnail(by_id[it["item_id"]]["path"]))
    tmp.replace(zpath)
    atomic_write_text(out / "key.json", json.dumps({"created_at": now_iso(), "git_commit": git_commit(),
                                                    "run": RUN, "items": key}, indent=1))
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


def boot(per_prompt: dict[str, list[tuple[float, float]]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Ratio sum(num) / sum(den) with a 95 % CI from resampling prompts."""
    pids = sorted(p for p, v in per_prompt.items() if sum(d for _, d in v) > 0)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0, "n_prompts": 0}
    num = np.array([sum(x for x, _ in per_prompt[p]) for p in pids], float)
    den = np.array([sum(d for _, d in per_prompt[p]) for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(pids), size=(n_boot, len(pids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    return {"value": float(num.sum() / den.sum()), "ci": [float(lo), float(hi)], "n": int(den.sum()),
            "n_prompts": len(pids)}


def kappa_ci(pairs: dict[str, list[tuple[int, int]]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    pids = sorted(p for p, v in pairs.items() if v)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0, "n_prompts": 0}
    flat = [x for p in pids for x in pairs[p]]
    val = cohen_kappa([str(a) for a, _ in flat], [str(b) for _, b in flat])
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(n_boot):
        s = [x for i in rng.integers(0, len(pids), len(pids)) for x in pairs[pids[i]]]
        ks.append(cohen_kappa([str(a) for a, _ in s], [str(b) for _, b in s]))
    lo, hi = np.nanquantile(np.array(ks, float), [0.025, 0.975])
    return {"value": val, "ci": [float(lo), float(hi)], "n": len(flat), "n_prompts": len(pids)}


def analyze(labels: dict[str, str], key: list[dict], grades: dict[str, dict], n_boot: int = N_BOOT) -> dict[str, Any]:
    by = {k["item_id"]: k for k in key}
    main = [k for k in key if k["part"] == "main"]
    res: dict[str, Any] = {"created_at": now_iso(), "git_commit": git_commit(), "pass_rule": f"judge score == {PASS_SCORE}",
                           "labelled": {p: sum(1 for k in key if k["part"] == p and k["item_id"] in labels)
                                        for p in ("practice", "main")},
                           "items": {p: sum(1 for k in key if k["part"] == p) for p in ("practice", "main")}}
    gold = [k for k in main if k["kind"] == "gold" and k["item_id"] in labels]
    right = sum(labels[k["item_id"]] == "no" for k in gold)
    res["gold"] = {"answered": len(gold), "right": right, "pass_mark": GOLD_PASS,
                   "passed": len(gold) == N_GOLD and right >= GOLD_PASS}
    reps = [(by[k["repeat_of"]], k) for k in main if k["kind"] == "repeat"
            and k["item_id"] in labels and k["repeat_of"] in labels]
    a = [labels[o["item_id"]] for o, _ in reps]
    b = [labels[r["item_id"]] for _, r in reps]
    res["repeat"] = {"pairs": len(reps), "same_answer": float(np.mean([x == y for x, y in zip(a, b)])) if reps else math.nan,
                     "cohen_kappa": cohen_kappa(a, b)}
    items = [k for k in main if k["kind"] == "main" and k["item_id"] in labels]
    res["label_counts"] = dict(Counter(labels[k["item_id"]] for k in items))

    def judge_pass(sid: str) -> int:
        g = grades.get(sid)
        return int(bool(g) and g.get("score") == PASS_SCORE)

    res["ungraded"] = sum(1 for k in items if grades.get(k["sample_id"], {}).get("score") is None)
    # (1) judge vs person on images the person answered yes or no
    decided = [k for k in items if labels[k["item_id"]] in ("yes", "no")]
    agree, ppv, npv, sens, pairs = (defaultdict(list) for _ in range(5))
    conf = Counter()
    for k in decided:
        h, j, p = int(labels[k["item_id"]] == "yes"), judge_pass(k["sample_id"]), k["prompt_id"]
        conf[(f"judge {'pass' if j else 'fail'}", f"person {'yes' if h else 'no'}")] += 1
        agree[p].append((float(h == j), 1.0))
        pairs[p].append((h, j))
        if j:
            ppv[p].append((float(h), 1.0))
        else:
            npv[p].append((float(not h), 1.0))
        if h:
            sens[p].append((float(j), 1.0))
    res["agreement"] = {"accuracy": boot(agree, n_boot), "cohen_kappa": kappa_ci(pairs, n_boot),
                        "judge_pass_person_yes": boot(ppv, n_boot), "judge_fail_person_no": boot(npv, n_boot),
                        "person_yes_judge_pass": boot(sens, n_boot),
                        "confusion": {f"{a} / {b}": v for (a, b), v in sorted(conf.items())}}
    res["by_category"] = {}
    for c in CATEGORIES:
        sub = {p: v for p, v in agree.items() if any(k["prompt_id"] == p and k["category"] == c for k in decided)}
        sub_pairs = {p: pairs[p] for p in sub}
        res["by_category"][c] = {"accuracy": boot(sub, n_boot), "cohen_kappa": kappa_ci(sub_pairs, n_boot)}
    # (2) pass rates per group on the sampled prompts: person (yes = pass) and judge, same images
    per_group: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)    # group -> prompt -> {person, judge}
    for k in items:
        for g in k["groups"]:
            per_group[g][k["prompt_id"]] = {"person": int(labels[k["item_id"]] == "yes"),
                                            "judge": judge_pass(k["sample_id"]),
                                            "unsure": int(labels[k["item_id"]] == "unsure")}
    rates = {}
    for g in GROUPS:
        rates[g] = {who: boot({p: [(float(v[who]), 1.0)] for p, v in per_group[g].items()}, n_boot)
                    for who in ("person", "judge")}
        rates[g]["unsure"] = sum(v["unsure"] for v in per_group[g].values())
        jp = {p: [(float(not v["person"]), 1.0)] for p, v in per_group[g].items() if v["judge"]}
        rates[g]["judge_pass_person_not"] = boot(jp, n_boot)
    res["pass_rate"] = rates
    diffs = {}
    for a_, b_ in (("agent_rule", "routed_best_of_k"), ("agent_rule", "best_of_4"), ("routed_best_of_k", "best_of_4")):
        common = sorted(set(per_group[a_]) & set(per_group[b_]))
        diffs[f"{a_} - {b_}"] = {who: boot({p: [(per_group[a_][p][who] - per_group[b_][p][who], 1.0)] for p in common}, n_boot)
                                 for who in ("person", "judge")}
    res["paired_difference"] = diffs
    res["prompts"] = len({k["prompt_id"] for k in items})
    return res


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _c(d: dict, nd=3) -> str:
    return f"{_f(d['value'], nd)} [{_f(d['ci'][0], 2)}, {_f(d['ci'][1], 2)}]"


def render(res: dict) -> str:
    g, rp, ag = res["gold"], res["repeat"], res["agreement"]
    L = ["# Blind human check of the judge's pass / fail calls (module C, one annotator)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.judge_check analyze`; "
         f"protocol in `docs/judge_check_protocol.md` (D-033). Judge pass = {res['pass_rule']}. Labelled: "
         f"{res['labelled']['main']} of {res['items']['main']} main-part items, {res['prompts']} test prompts. "
         "95 % CIs from resampling prompts.", "",
         f"- Attention checks: {g['right']} of {g['answered']} right (pass mark {g['pass_mark']} of {N_GOLD}): "
         f"**{'passed' if g['passed'] else 'NOT passed'}**.",
         f"- Hidden repeats: same answer on {_f(rp['same_answer'])} of {rp['pairs']} images; Cohen's κ "
         f"{_f(rp['cohen_kappa'])}.",
         "- Answers on the main images: " + ", ".join(f"{k} {v}" for k, v in sorted(res["label_counts"].items())) +
         (f"; {res['ungraded']} images have no judge score (counted as fails)." if res["ungraded"] else "."), "",
         "## Judge vs person (images answered yes or no)", "",
         "| Measure | Value [95 % CI] | Images |", "| --- | --- | --- |",
         f"| Same call (accuracy) | {_c(ag['accuracy'])} | {ag['accuracy']['n']} |",
         f"| Cohen's κ | {_c(ag['cohen_kappa'])} | {ag['cohen_kappa']['n']} |",
         f"| Judge passes the person also passes | {_c(ag['judge_pass_person_yes'])} | {ag['judge_pass_person_yes']['n']} |",
         f"| Judge fails the person also fails | {_c(ag['judge_fail_person_no'])} | {ag['judge_fail_person_no']['n']} |",
         f"| Person's passes the judge also passes | {_c(ag['person_yes_judge_pass'])} | {ag['person_yes_judge_pass']['n']} |",
         "", "Counts: " + ", ".join(f"{k} {v}" for k, v in ag["confusion"].items()) + ".", "",
         "By category (exploratory): " + "; ".join(
             f"{c} {_f(v['accuracy']['value'])} (κ {_f(v['cohen_kappa']['value'], 2)}, {v['accuracy']['n']} images)"
             for c, v in res["by_category"].items() if v["accuracy"]["n"]) + ".", "",
         "## Pass rates on the same images", "",
         "Person: yes = pass (no and unsure = not). Judge: score 5.", "",
         "| Group | Person | Judge | Unsure | Judge passes the person does not pass |", "| --- | --- | --- | --- | --- |"]
    for gname, r in res["pass_rate"].items():
        L.append(f"| {gname} | {_c(r['person'])} | {_c(r['judge'])} | {r['unsure']} | "
                 f"{_c(r['judge_pass_person_not'])} (of {r['judge_pass_person_not']['n']}) |")
    L += ["", "| Paired difference (same prompts) | Person | Judge |", "| --- | --- | --- |"]
    for name, d in res["paired_difference"].items():
        L.append(f"| {name} | {_c(d['person'])} | {_c(d['judge'])} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make")
    mk.add_argument("--final", default=f"data/agent/{RUN}/final.jsonl")
    mk.add_argument("--dev-final", default=f"data/agent/{DEV_RUN}/final.jsonl")
    an = sub.add_parser("analyze")
    an.add_argument("--labels", required=True)
    an.add_argument("--key", default=str(OUT / "key.json"))
    an.add_argument("--grades", default=GRADES)
    an.add_argument("--out", default="results/judge_check")
    args = ap.parse_args(argv)
    if args.cmd == "make":
        public, key = build(read_jsonl(args.final), read_jsonl(args.dev_final))
        z = write_package(public, key)
        kinds = Counter(k["kind"] for k in key)
        print(f"wrote {z} ({dict(kinds)}) and {OUT / 'key.json'} (don't open it before labelling is done)")
        return 0
    key = json.loads(Path(args.key).read_text())["items"]
    res = analyze(read_labels(args.labels), key, latest_by_id(read_many([args.grades])))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2))
    atomic_write_text(out / "summary.md", render(res))
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
