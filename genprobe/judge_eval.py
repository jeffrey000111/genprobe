"""Analysis of one VLM-judge round (genprobe/judge.py): agreement with the GenAI-Bench human
ratings, and the bias checks of guide §9.4, including the pre-registered H3 on the test round.

    python -m genprobe.judge_eval --tag dev_r1 --round dev

Reads data/judge/<tag>.jsonl; writes results/judge/<tag>/summary.md, judge_bias.csv and
summary.json. Checks:

- parsing: share of answers that are valid JSON (structured output should make this ~100 %);
- score distribution of the absolute scores (score collapse: everything a 4?);
- agreement with people, absolute: within-prompt pairs with different mean ratings (as in
  genprobe/meta_eval.py), ordered by the judge's integer score (ties 1/2) and by its expected
  score; VQAScore on the same pairs for reference;
- agreement with people, pairwise: each pair asked in both orders; a consistent pick counts 1 or 0
  against the human order, anything else (inconsistent, "same", unparsed) 1/2;
- position bias (H3): among greedy pairwise verdicts that pick an image, the share picking the
  image shown first, with a 95 % CI from resampling prompts; plus the share of pairs whose two
  orders agree;
- consistency: 3 samples at temperature 0.7 vs the greedy answer;
- blind test: scores with the real image vs a grey image vs another prompt's image.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .judge import Bench, h3_pairs, load_bench, round_prompts
from .meta_eval import agreement, load_metric_scores
from .summarize import N_BOOT
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import read_many


def _ci(per_prompt: dict[str, tuple[float, float]], n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Ratio sum(num) / sum(den) over prompts, with a prompt-bootstrap 95 % CI."""
    pids = sorted(p for p, (_, d) in per_prompt.items() if d > 0)
    if not pids:
        return {"value": math.nan, "ci": [math.nan, math.nan], "n": 0, "n_prompts": 0}
    num = np.array([per_prompt[p][0] for p in pids], float)
    den = np.array([per_prompt[p][1] for p in pids], float)
    idx = np.random.default_rng(seed).integers(0, len(pids), size=(n_boot, len(pids)))
    b = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.quantile(b, [0.025, 0.975])
    return {"value": float(num.sum() / den.sum()), "ci": [float(lo), float(hi)], "n": int(den.sum()),
            "n_prompts": len(pids)}


def position_bias(pair_recs: list[dict], n_boot: int = N_BOOT) -> dict[str, Any]:
    """H3 statistic: greedy verdicts that pick an image (not "same"/unparsed) -> share "first"."""
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in pair_recs:
        if r["mode"] == "greedy" and r.get("verdict") in ("first", "second"):
            per[r["prompt_id"]][0] += r["verdict"] == "first"
            per[r["prompt_id"]][1] += 1
    res = _ci({p: (v[0], v[1]) for p, v in per.items()}, n_boot)
    lo, hi = res["ci"]
    res["verdict"] = "no data" if math.isnan(lo) else ("supported" if lo > 0.5 or hi < 0.5 else "inconclusive")
    return res


def pair_decisions(pair_recs: list[dict]) -> dict[str, dict]:
    """pair_id -> {prompt_id, human (+1 a better, -1 b better), pick (image id or None), consistent}."""
    by: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in pair_recs:
        if r["mode"] == "greedy":
            by[r["pair_id"]][r["order"]] = r
    out = {}
    for pid_, orders in by.items():
        if set(orders) != {1, 2}:
            continue
        a, b = pid_.split("|")
        picks = []
        for o in (1, 2):
            r = orders[o]
            v = r.get("verdict")
            picks.append(r["first"] if v == "first" else r["second"] if v == "second" else "same" if v == "same" else None)
        consistent = picks[0] == picks[1] and picks[0] is not None
        pick = picks[0] if consistent and picks[0] in (a, b) else None
        out[pid_] = {"prompt_id": orders[1]["prompt_id"], "human": orders[1]["human"], "a": a, "b": b,
                     "picks": picks, "consistent": consistent, "pick": pick}
    return out


def pairwise_agreement(dec: dict[str, dict], n_boot: int = N_BOOT) -> dict[str, Any]:
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for d in dec.values():
        better = d["a"] if d["human"] > 0 else d["b"]
        per[d["prompt_id"]][0] += 0.5 if d["pick"] is None else float(d["pick"] == better)
        per[d["prompt_id"]][1] += 1
    return _ci({p: (v[0], v[1]) for p, v in per.items()}, n_boot)


def order_agreement(dec: dict[str, dict], n_boot: int = N_BOOT) -> dict[str, Any]:
    per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for d in dec.values():
        per[d["prompt_id"]][0] += d["consistent"]
        per[d["prompt_id"]][1] += 1
    return _ci({p: (v[0], v[1]) for p, v in per.items()}, n_boot)


def consistency(recs: list[dict]) -> dict[str, Any]:
    """Temperature-0.7 samples against the greedy answer, for absolute scores and pair verdicts."""
    greedy_abs = {r["sample_id"]: r.get("score") for r in recs
                  if r["kind"] == "abs" and r["mode"] == "greedy" and r["variant"] == "real"}
    greedy_pair = {r["pair_id"]: r.get("verdict") for r in recs if r["kind"] == "pair" and r["mode"] == "greedy"
                   and r["order"] == 1}
    samples: dict[tuple[str, str], list] = defaultdict(list)
    for r in recs:
        if r["mode"] == "sample":
            key = ("abs", r["sample_id"]) if r["kind"] == "abs" else ("pair", r["pair_id"])
            samples[key].append(r.get("score") if r["kind"] == "abs" else r.get("verdict"))
    out = {}
    for kind, greedy in (("abs", greedy_abs), ("pair", greedy_pair)):
        items = [(k[1], v) for k, v in samples.items() if k[0] == kind and k[1] in greedy]
        if not items:
            continue
        all_same = [len(set(v)) == 1 for _, v in items]
        majority = [Counter(v).most_common(1)[0][0] == greedy[i] for i, v in items]
        out[kind] = {"items": len(items), "samples_all_equal": float(np.mean(all_same)),
                     "sample_majority_equals_greedy": float(np.mean(majority))}
        if kind == "abs":
            sds = [float(np.std([x for x in v if x is not None])) for _, v in items if any(x is not None for x in v)]
            out[kind]["mean_sd_of_samples"] = float(np.mean(sds)) if sds else math.nan
    return out


def blind_test(abs_recs: list[dict]) -> dict[str, Any]:
    by: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in abs_recs:
        if r["mode"] == "greedy":
            by[r["sample_id"]][r["variant"]] = r
    out: dict[str, Any] = {}
    for other in ("grey", "mismatch"):
        rows = [(v["real"], v[other]) for v in by.values() if "real" in v and other in v
                and v["real"].get("score") is not None and v[other].get("score") is not None]
        if not rows:
            continue
        real = np.array([a["score"] for a, _ in rows], float)
        alt = np.array([b["score"] for _, b in rows], float)
        e_real = np.array([a.get("expected_score") or a["score"] for a, _ in rows], float)
        e_alt = np.array([b.get("expected_score") or b["score"] for _, b in rows], float)
        out[other] = {"images": len(rows), "mean_score_real": float(real.mean()), f"mean_score_{other}": float(alt.mean()),
                      "share_real_higher": float(np.mean((real > alt) + 0.5 * (real == alt))),
                      "share_real_higher_expected": float(np.mean((e_real > e_alt) + 0.5 * (e_real == e_alt)))}
    return out


def dev_examples(bench: Bench, pids: list[str], real: list[dict], dec: dict[str, dict], k: int = 8,
                 min_gap: float = 2.0) -> dict[str, list[dict]]:
    """Clear disagreements with people (mean ratings at least min_gap apart): absolute scores in the
    wrong order, and pairwise picks of the worse image in both orders."""
    by = {r["sample_id"]: r for r in real}
    model = lambda s: bench.images[s]["model"]  # noqa: E731
    absolute = []
    for pid in pids:
        m = bench.means.get(pid, {})
        for a, b, _ in bench.pairs.get(pid, []):
            hi, lo = (a, b) if m[a] > m[b] else (b, a)
            if m[hi] - m[lo] >= min_gap and hi in by and lo in by and (by[hi].get("score") or 0) < (by[lo].get("score") or 0):
                absolute.append({"prompt": bench.prompts[pid].prompt, "gap": m[hi] - m[lo],
                                 "people_prefer": {"model": model(hi), "mean": m[hi], "judge": by[hi].get("score"),
                                                   "issues": by[hi].get("issues")},
                                 "other": {"model": model(lo), "mean": m[lo], "judge": by[lo].get("score"),
                                           "issues": by[lo].get("issues")}})
    pairwise = []
    for d in dec.values():
        m = bench.means.get(d["prompt_id"], {})
        better, worse = (d["a"], d["b"]) if d["human"] > 0 else (d["b"], d["a"])
        if d["pick"] == worse and m[better] - m[worse] >= min_gap:
            pairwise.append({"prompt": bench.prompts[d["prompt_id"]].prompt, "gap": m[better] - m[worse],
                             "people_prefer": f"{model(better)} ({m[better]:.2f})",
                             "judge_picks": f"{model(worse)} ({m[worse]:.2f})"})
    key = lambda e: (-e["gap"], e["prompt"])  # noqa: E731
    return {"absolute": sorted(absolute, key=key)[:k], "pairwise": sorted(pairwise, key=key)[:k],
            "counts": {"absolute": len(absolute), "pairwise": len(pairwise)}}


def run(tag: str, rnd: str, data: str = "data", out_root: str = "results/judge", bench: Bench | None = None,
        n_boot: int = N_BOOT) -> dict[str, Any]:
    recs = read_many([Path(data) / "judge" / f"{tag}.jsonl"])
    meta_path = Path(data) / "judge" / f"{tag}.meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    bench = bench or load_bench()
    pids = round_prompts(bench, rnd)
    abs_recs = [r for r in recs if r["kind"] == "abs"]
    pair_recs = [r for r in recs if r["kind"] == "pair"]
    real = [r for r in abs_recs if r["mode"] == "greedy" and r["variant"] == "real"]

    res: dict[str, Any] = {"tag": tag, "round": rnd, "created_at": now_iso(), "git_commit": git_commit(),
                           "judge_version": meta.get("judge_version"), "records": len(recs),
                           "requests": meta.get("requests"), "generate_seconds": meta.get("generate_seconds")}
    res["parsing"] = {k: {"answers": len(v), "json_ok": float(np.mean([r["parse_ok"] for r in v])) if v else math.nan,
                          "unreadable": int(sum((r.get("score") if r["kind"] == "abs" else r.get("verdict")) is None
                                                for r in v)),
                          "truncated": int(sum(r.get("finish_reason") == "length" for r in v))}
                      for k, v in (("abs", abs_recs), ("pair", pair_recs))}
    res["score_distribution"] = {str(k): int(n) for k, n in sorted(Counter(r.get("score") for r in real).items(),
                                                                    key=lambda kv: (kv[0] is None, kv[0] or 0))}
    # absolute scores against people (same pairs and method as genprobe.meta_eval)
    ids = {r["sample_id"] for r in real}
    judge_int = {r["sample_id"]: float(r["score"]) for r in real if r.get("score") is not None}
    judge_exp = {r["sample_id"]: float(r["expected_score"]) for r in real if r.get("expected_score") is not None}
    vqa, _, _ = load_metric_scores(data, "vqascore", ids)
    res["absolute_vs_people"] = {name: _flat(agreement(bench.pairs, pids, sc, n_boot))
                                 for name, sc in (("judge score (1-5)", judge_int), ("judge expected score", judge_exp),
                                                  ("vqascore (reference)", vqa)) if sc}
    dec = pair_decisions(pair_recs)
    res["pairwise_vs_people"] = _flat2(pairwise_agreement(dec, n_boot))
    res["pairwise_vs_people"]["pick_rate"] = float(np.mean([d["pick"] is not None for d in dec.values()])) if dec else math.nan
    res["order_agreement"] = _flat2(order_agreement(dec, n_boot))
    res["position_bias"] = position_bias(pair_recs, n_boot)
    if rnd == "test":
        want = {f"{a}|{b}" for _, a, b, _ in h3_pairs(bench)}
        res["position_bias"]["h3_pairs_complete"] = want <= set(dec)
    res["consistency"] = consistency(recs)
    res["blind_test"] = blind_test(abs_recs)
    if rnd == "dev":                      # failure cases to iterate the rubric on; never shown for test
        res["examples"] = dev_examples(bench, pids, real, dec)

    out = Path(out_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2, default=str))
    write_bias_csv(out / "judge_bias.csv", res)
    atomic_write_text(out / "summary.md", render(res))
    return res


def _flat(d: dict) -> dict:
    return {"agreement": d["agreement"], "ci": d["ci"], "pairs": d["pairs"], "n_prompts": d["n_prompts"]}


def _flat2(d: dict) -> dict:
    return {"value": d["value"], "ci": d["ci"], "n": d["n"], "n_prompts": d["n_prompts"]}


def write_bias_csv(path: Path, res: dict) -> None:
    rows = []

    def add(check, stat, value, ci=(None, None), n=None):
        rows.append({"check": check, "statistic": stat, "value": value, "ci95_lo": ci[0], "ci95_hi": ci[1], "n": n})

    pb = res["position_bias"]
    add("position bias", "share of image-picking verdicts that pick the first image", pb["value"], pb["ci"], pb["n"])
    oa = res["order_agreement"]
    add("position bias", "share of pairs with the same verdict in both orders", oa["value"], oa["ci"], oa["n"])
    for kind, c in res["consistency"].items():
        add("consistency", f"{kind}: share of items whose 3 samples (T=0.7) agree", c["samples_all_equal"], n=c["items"])
        add("consistency", f"{kind}: share where the sample majority equals the greedy answer",
            c["sample_majority_equals_greedy"], n=c["items"])
    for other, b in res["blind_test"].items():
        add("blind test", f"mean score, real image (vs {other})", b["mean_score_real"], n=b["images"])
        add("blind test", f"mean score, {other} image", b[f"mean_score_{other}"], n=b["images"])
        add("blind test", f"share real image scored higher than {other} (ties 1/2)", b["share_real_higher"], n=b["images"])
    total = sum(res["score_distribution"].values()) or 1
    for k, n in res["score_distribution"].items():
        add("score distribution", f"share of real images scored {k}", n / total, n=n)
    for k, p in res["parsing"].items():
        add("parsing", f"{k}: share of answers that are valid JSON", p["json_ok"], n=p["answers"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "statistic", "value", "ci95_lo", "ci95_hi", "n"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in r.items()})


def _f(x, nd=3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render(res: dict) -> str:
    L = [f"# VLM judge, round `{res['round']}` (`{res['tag']}`)", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.judge_eval`. Judge "
         f"`{res['judge_version']}`; {res['records']} answers; generation took {res['generate_seconds']} s.", "",
         "## Parsing", ""]
    for k, p in res["parsing"].items():
        L.append(f"- {k}: {p['answers']} answers, {100 * p['json_ok']:.1f} % valid JSON, {p['unreadable']} unreadable, "
                 f"{p['truncated']} cut off at the token limit")
    total = sum(res["score_distribution"].values()) or 1
    L += ["", "## Absolute scores", "", "Distribution on real images: " + ", ".join(
        f"{k}: {n} ({100 * n / total:.0f} %)" for k, n in res["score_distribution"].items()) + ".", "",
        "Agreement with people (within-prompt pairs with different mean ratings, ties 1/2, 95 % CI over prompts):", "",
        "| Scorer | Agreement | CI | Pairs |", "| --- | --- | --- | --- |"]
    for name, a in res["absolute_vs_people"].items():
        L.append(f"| {name} | {_f(a['agreement'])} | [{_f(a['ci'][0])}, {_f(a['ci'][1])}] | {a['pairs']} |")
    pw, oa, pb = res["pairwise_vs_people"], res["order_agreement"], res["position_bias"]
    L += ["", "## Pairwise verdicts", "",
          f"- Agreement with people: {_f(pw['value'])} [{_f(pw['ci'][0])}, {_f(pw['ci'][1])}] over {pw['n']} pairs "
          f"(a pick counts only if both orders agree; otherwise 1/2). The judge picked an image consistently in "
          f"{100 * pw['pick_rate']:.0f} % of pairs.",
          f"- Same verdict in both orders: {_f(oa['value'])} [{_f(oa['ci'][0])}, {_f(oa['ci'][1])}].",
          f"- Position bias: {_f(pb['value'])} [{_f(pb['ci'][0])}, {_f(pb['ci'][1])}] of {pb['n']} image-picking "
          f"verdicts pick the image shown first (0.5 = no bias)"
          + (f"; **H3 {pb['verdict']}** (pre-registered: holds if the 95 % CI excludes 0.5)." if res["round"] == "test"
             else " (dev round: descriptive only; H3 is tested on the test round)."), "",
          "## Consistency (3 samples at temperature 0.7)", ""]
    for kind, c in res["consistency"].items():
        L.append(f"- {kind}: {c['items']} items; all 3 samples agree on {100 * c['samples_all_equal']:.0f} %; the "
                 f"samples' majority equals the greedy answer on {100 * c['sample_majority_equals_greedy']:.0f} %"
                 + (f"; mean SD of the sampled scores {_f(c['mean_sd_of_samples'], 2)}" if kind == "abs" else "") + ".")
    L += ["", "## Blind test", ""]
    for other, b in res["blind_test"].items():
        L.append(f"- Real image vs {other} image ({b['images']} images): mean score {_f(b['mean_score_real'], 2)} vs "
                 f"{_f(b[f'mean_score_{other}'], 2)}; the real image scores higher in "
                 f"{100 * b['share_real_higher']:.0f} % (ties 1/2; {100 * b['share_real_higher_expected']:.0f} % by "
                 f"expected score).")
    ex = res.get("examples")
    if ex:
        L += ["", "## Clear disagreements with people (dev only, for the rubric)", "",
              f"Mean human ratings at least 2 points apart. Absolute scores in the wrong order: "
              f"{ex['counts']['absolute']} pairs; pairwise picks of the worse image in both orders: "
              f"{ex['counts']['pairwise']}. The largest gaps:", ""]
        for e in ex["absolute"]:
            p, o = e["people_prefer"], e["other"]
            L.append(f"- {json.dumps(e['prompt'], ensure_ascii=False)}: people {p['model']} {p['mean']:.2f} > "
                     f"{o['model']} {o['mean']:.2f}; judge {p['judge']} vs {o['judge']}; issues it saw: "
                     f"{p['issues']} vs {o['issues']}")
        for e in ex["pairwise"]:
            L.append(f"- pairwise, {json.dumps(e['prompt'], ensure_ascii=False)}: people prefer {e['people_prefer']}, "
                     f"the judge picks {e['judge_picks']} in both orders")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--round", choices=["dev", "test"], required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results/judge")
    args = ap.parse_args(argv)
    res = run(args.tag, args.round, args.data, args.out)
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
