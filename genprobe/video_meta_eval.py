"""Metric meta-evaluation on GenAI-Bench-Video (guide §11.3, D-035): which automatic metrics order
videos the way people do, and the pre-registered tests H4 and H5. The method is §3's
(genprobe/meta_eval.py), applied to the four videos of each prompt:

- Pairs: within each prompt, every two videos whose mean human ratings differ; ratings outside 1-5
  are dropped. Agreement: share of pairs a metric orders like the mean rating (equal scores 1/2).
- CIs: 2,000 bootstrap resamples of prompts (seed 0); 97.5 % for H4 and H5 (two tests).
- Ceiling: each rating against the mean of the other two; Krippendorff's alpha (ordinal).

Scores per video:
    vqascore_video    VQAScore with the whole video as input (genprobe/metrics/vqascore_video.py)
    vqascore_mid      image VQAScore on the middle frame (frame 4 of 9)
    clipscore_frames, pickscore_frames, hpsv2_frames   the image metric averaged over 9 frames
    (frames: genprobe/genai_bench_video.py frames)

    python -m genprobe.video_meta_eval
        -> results/video_meta_eval/hypotheses.md, agreement_by_tag.csv, summary.json, fig_video_meta_eval_heatmap.png

Refuses to compute anything until every video has every score (results are looked at once).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import genai_bench_video as gv
from .meta_eval import (HUMAN, IncompleteScores, agreement, ceiling, difference, heatmap, kendall_tau_b,
                        krippendorff_alpha_ordinal, load_metric_scores, make_pairs, verdict, write_csv, _flat, _f)
from .samples import load_prompts, make_sample_id
from .summarize import N_BOOT
from .utils.env import git_commit, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

OUT = Path("results/video_meta_eval")
FRAME_METRICS = ("clipscore", "pickscore", "hpsv2")
METRICS = ("vqascore_video", "vqascore_mid") + tuple(f"{m}_frames" for m in FRAME_METRICS)
H4 = {"name": "H4", "tag": "all", "a": "vqascore_video", "b": "clipscore_frames", "level": 0.975}
H5 = {"name": "H5", "tag": "all", "a": "vqascore_video", "b": "vqascore_mid", "level": 0.975}


def video_scores(videos: dict[str, dict], data: str = "data") -> tuple[dict, dict, dict]:
    """Per-video score sets keyed by the video's sample_id, their metric versions and coverage."""
    vids = set(videos)
    frame_of = {make_sample_id(v["model"], v["prompt_id"], gv.FRAME_SEED + k): (sid, k)
                for sid, v in videos.items() for k in range(gv.N_FRAMES)}
    scores, versions, coverage = {}, {}, {}
    s, versions["vqascore_video"], err = load_metric_scores(data, "vqascore_video", vids)
    scores["vqascore_video"] = s
    coverage["vqascore_video"] = {"scored": len(s), "videos": len(vids), "errors": err}
    mids = {f: sv for f, (sv, k) in frame_of.items() if k == gv.MID}
    s, versions["vqascore_mid"], err = load_metric_scores(data, "vqascore", set(mids))
    scores["vqascore_mid"] = {mids[f]: v for f, v in s.items()}
    coverage["vqascore_mid"] = {"scored": len(scores["vqascore_mid"]), "videos": len(vids), "errors": err}
    for m in FRAME_METRICS:
        s, versions[f"{m}_frames"], err = load_metric_scores(data, m, set(frame_of))
        per = defaultdict(list)
        for f, v in s.items():
            per[frame_of[f][0]].append(v)
        full = {sv: float(np.mean(vs)) for sv, vs in per.items() if len(vs) == gv.N_FRAMES}
        scores[f"{m}_frames"] = full
        coverage[f"{m}_frames"] = {"scored": len(full), "videos": len(vids), "errors": err}
    return scores, versions, coverage


def run(prompts_path: str = gv.PROMPTS, manifest: str = gv.MANIFEST, data: str = "data", folder: str = str(gv.FOLDER),
        out: Path | None = None, n_boot: int = N_BOOT, allow_incomplete: bool = False) -> dict[str, Any]:
    out = Path(out or OUT)
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    videos = {sid: r for sid, r in latest_by_id(read_many([manifest])).items()
              if r.get("status") == "ok" and r.get("prompt_id") in prompts}
    ratings, dropped = gv.load_ratings(folder)
    raw: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for sid, r in videos.items():
        v = ratings.get((r["prompt_id"], r["model"]), [])
        if v:
            raw[r["prompt_id"]][sid] = v
    means = {pid: {sid: float(np.mean(v)) for sid, v in vs.items()} for pid, vs in raw.items()}
    pairs, ties, total = make_pairs(means)
    groups: dict[str, list[str]] = defaultdict(list)
    for p in prompts.values():
        groups["all"].append(p.prompt_id)
        for t in p.tags:
            groups[t].append(p.prompt_id)
    fixed = [g for g in ("all", "basic", "advanced") if g in groups]
    order = fixed + sorted(g for g in groups if g not in fixed)

    scores, versions, coverage = video_scores(videos, data)
    short = {m: c for m, c in coverage.items() if c["scored"] < c["videos"]}
    if short and not allow_incomplete:
        raise IncompleteScores("not every video is scored yet, nothing computed: " + "; ".join(
            f"{m} {c['scored']}/{c['videos']} ({c['errors']} errors)" for m, c in short.items()))
    rows = []
    for g in order:
        pids = sorted(groups[g])
        rows.append({"metric": HUMAN, "tag": g, **_flat(ceiling(pairs, pids, raw, n_boot))})
        for m in METRICS:
            sc = scores[m]
            taus = []
            for pid in pids:
                ids = [s for s in sorted(means.get(pid, {})) if s in sc]
                t = kendall_tau_b([sc[s] for s in ids], [means[pid][s] for s in ids]) if len(ids) > 1 else math.nan
                if not math.isnan(t):
                    taus.append(t)
            rows.append({"metric": m, "tag": g, **_flat(agreement(pairs, pids, sc, n_boot)),
                         "kendall_tau_mean": float(np.mean(taus)) if taus else math.nan})
    hyp = {}
    for h in (H4, H5):
        d = difference(pairs, sorted(groups[h["tag"]]), scores[h["a"]], scores[h["b"]], n_boot, level=h["level"])
        hyp[h["name"]] = {**d, "comparison": f"{h['a']} - {h['b']}, all prompts", "verdict": verdict(d["ci"])}
    by_model = {}
    for m in gv.MODELS:
        ids = [s for s, r in videos.items() if r["model"] == m]
        by_model[m] = {"videos": len(ids),
                       "mean_human": float(np.mean([np.mean(ratings[(videos[s]["prompt_id"], m)]) for s in ids
                                                    if ratings.get((videos[s]["prompt_id"], m))])) if ids else math.nan,
                       **{f"mean_{k}": float(np.mean([scores[k][s] for s in ids if s in scores[k]])) if ids else math.nan
                          for k in METRICS}}
    alpha = krippendorff_alpha_ordinal([v for vs in raw.values() for v in vs.values()])
    res = {"created_at": now_iso(), "git_commit": git_commit(), "source": gv.source(folder), "prompts": len(prompts),
           "videos": len(videos), "pairs": sum(len(v) for v in pairs.values()), "pairs_tied_excluded": ties,
           "pairs_total": total, "dropped_ratings": dropped, "krippendorff_alpha_ordinal": alpha,
           "metric_versions": versions, "coverage": coverage, "hypotheses": hyp, "by_model": by_model, "n_boot": n_boot}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "agreement_by_tag.csv", rows)
    atomic_write_text(out / "summary.json", json.dumps(res, indent=2, default=str))
    atomic_write_text(out / "hypotheses.md", render(res, rows, order))
    heatmap(rows, order, list(METRICS), out / "fig_video_meta_eval_heatmap.png", what="video")
    return {**res, "rows": rows}


def render(res: dict, rows: list[dict], order: list[str]) -> str:
    L = ["# Metric meta-evaluation on GenAI-Bench-Video", "",
         f"Made {res['created_at']} at commit `{res['git_commit']}` by `python -m genprobe.video_meta_eval` (D-035). "
         f"Data `{res['source']}`: {res['prompts']} prompts, {res['videos']} videos. {res['pairs']} within-prompt video "
         f"pairs whose mean human ratings differ; {res['pairs_tied_excluded']} of {res['pairs_total']} pairs "
         f"({100 * res['pairs_tied_excluded'] / max(res['pairs_total'], 1):.1f} %) have equal means and are excluded. "
         f"{len(res['dropped_ratings'])} out-of-range ratings dropped. Krippendorff's α (ordinal): "
         f"{_f(res['krippendorff_alpha_ordinal'])}.", "",
         "## Pre-registered hypotheses (docs/decisions.md, D-035)", "",
         "Supported when the CI lies above 0, contradicted when below, inconclusive otherwise.", "",
         "| Test | Comparison | Agreement a / b | Difference | CI | Pairs | Verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, h in res["hypotheses"].items():
        L.append(f"| {k} | {h['comparison']} | {_f(h.get('agreement_a'))} / {_f(h.get('agreement_b'))} | "
                 f"{_f(h['difference'])} | {h['level'] * 100:g} %: [{_f(h['ci'][0])}, {_f(h['ci'][1])}] | "
                 f"{h['pairs']} | **{h['verdict']}** |")
    by = {(r["metric"], r["tag"]): r for r in rows}
    L += ["", "## Agreement with people by tag (exploratory apart from the tests above)", "",
          "| Tag | Prompts | Human ceiling | " + " | ".join(METRICS) + " |", "| --- | --- | --- |" + " --- |" * len(METRICS)]
    for g in order:
        h = by[(HUMAN, g)]
        n = max([by[(m, g)]["n_prompts"] for m in METRICS] + [h["n_prompts"]])
        cells = [f"{_f(by[(m, g)]['agreement'], 2)} [{_f(by[(m, g)]['ci95_lo'], 2)}, {_f(by[(m, g)]['ci95_hi'], 2)}]"
                 for m in METRICS]
        L.append(f"| {g} | {n} | {_f(h['agreement'], 2)} | " + " | ".join(cells) + " |")
    L += ["", "Kendall's τ-b within prompts, averaged (exploratory): " +
          "; ".join(f"{m} {_f(by[(m, 'all')].get('kendall_tau_mean'))}" for m in METRICS) + ".", "",
          "## By model (means over its videos)", "", "| Model | Videos | Mean human rating | " +
          " | ".join(METRICS) + " |", "| --- | --- | --- |" + " --- |" * len(METRICS)]
    for m, r in res["by_model"].items():
        L.append(f"| {m} | {r['videos']} | {_f(r['mean_human'], 2)} | " +
                 " | ".join(_f(r[f'mean_{k}']) for k in METRICS) + " |")
    L += ["", "Coverage: " + "; ".join(f"{m} {c['scored']}/{c['videos']}" + (f" ({c['errors']} errors)" if c["errors"] else "")
                                       for m, c in res["coverage"].items()) + ".",
          "", "Metric versions: " + "; ".join(f"{m} `{v}`" for m, v in res["metric_versions"].items()) + "."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    try:
        res = run(out=Path(args.out))
    except IncompleteScores as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print((Path(args.out) / "hypotheses.md").read_text())
    return 0 if res else 1


if __name__ == "__main__":
    sys.exit(main())
