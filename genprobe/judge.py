"""VLM judge (guide §9.4, D-027): Qwen3-VL-8B-Instruct through vLLM grades GenAI-Bench images
against their prompts, in two ways, and is checked for the biases of guide §9.4 and H3.

- absolute: one image + prompt -> JSON {"issues": [...], "score": 1-5}; the score digit's token
  probabilities also give an expected score (1-5, continuous), which breaks ties.
- pairwise: two images of the same prompt -> JSON {"reason": ..., "better": "1" | "2" | "same"};
  every pair is asked in both orders (position bias, H3).
- blind test: the same absolute question with the image replaced by a grey image or by another
  prompt's image (same model); the score should drop.
- consistency: the greedy (temperature 0) answers are the results; 3 samples at temperature 0.7
  show how much the verdicts wobble.

Rounds:
    dev   the 80 GenAI-Bench prompts in prompt set v1's dev split (D-026): all 480 images, all their
          pairs with different mean human ratings; the rubric is iterated here only.
    test  the other 1,520 prompts: all their images; the 1,000 pairs pre-registered for H3 (drawn
          with seed 0 from pairs with different mean ratings); blind test and consistency on 500
          images and on those pairs.

    python -m genprobe.judge run --round dev --tag dev_r1      (GPU, judge env; slurm/judge.sbatch)
    python -m genprobe.judge_eval --tag dev_r1 --round dev     (analysis, CPU)

Raw answers go to data/judge/<tag>.jsonl (one record per request and sample) with
data/judge/<tag>.meta.json; nothing there is tuned on test.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .genai_bench import FOLDER, load_ratings
from .meta_eval import make_pairs
from .samples import load_prompts
from .utils.env import fingerprint, now_iso
from .utils.io import atomic_write_text
from .utils.manifest import latest_by_id, read_many

# Judge models (D-027, D-030). Both run the same frozen rubric; the Gemma judge is not tuned at all.
JUDGES: dict[str, dict[str, Any]] = {
    "qwen3vl8b": {"repo": "Qwen/Qwen3-VL-8B-Instruct", "short": "qwen3-vl-8b", "download": "judge",
                  "files": ("config.json", "model.safetensors.index.json"),
                  "gpu_memory_utilization": 0.85, "llm": {}, "chat": {}},
    # Gemma 4 31B (Google, Apache-2.0): a different model family from VQAScore's Qwen2.5-VL, so it
    # can grade an agent that optimizes VQAScore (guide §10.4). 560 vision tokens per image (its
    # default is 280); thinking off, as for the Qwen judge.
    "gemma4_31b": {"repo": "google/gemma-4-31B-it", "short": "gemma-4-31b", "download": "gemma4_31b",
                   "files": ("config.json", "model.safetensors.index.json", "chat_template.jinja"),
                   "gpu_memory_utilization": 0.92, "llm": {"mm_processor_kwargs": {"max_soft_tokens": 560}},
                   "chat": {"chat_template_kwargs": {"enable_thinking": False}}, "tag": "vis560"},
}
DEFAULT_JUDGE = "qwen3vl8b"
REPO = JUDGES[DEFAULT_JUDGE]["repo"]


def download_hint(name: str) -> str:
    return f'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh {JUDGES[name]["download"]}"'
MAX_SIDE = 1024          # images are shrunk (never enlarged) to fit 1024 x 1024 before the judge sees them
N_H3_PAIRS = 1000        # pre-registered (docs/decisions.md, H3)
N_TEST_BLIND = 500
N_SAMPLES, SAMPLE_TEMPERATURE = 3, 0.7
GREY = (128, 128, 128)

# ---------- prompts (the rubric is iterated on dev only; its hash goes into the version) ----------

# Rubric history (D-028 revisions; template hash in judge_version):
#   r1 (dev_r1): checks + 1-5 scale; pairs {"reason", "better"}.
#   r2 (dev_r2): "what can't be seen is unmet", scale anchored on how much is met, pairs list each
#       image's unmet requirements first. Absolute unchanged (0.659 vs 0.655), pairs much worse
#       (0.559 vs 0.660; 3 % cut off), and grey images still scored 5 with no issues.
#   r3 (dev_r3): absolute = r2 + a description of what the image shows before any check, so the
#       verdict rests on what was seen; pairs = r1 again, unchanged (also a check that reruns repeat).
CHECKS = ("every object it mentions is present; counts are exact; colours, shapes, materials and other "
          "attributes belong to the right objects; spatial relations (left/right, above/below, in front/"
          "behind, inside) and actions are as described; negations (\"no\", \"without\") are respected; "
          "comparisons (bigger, more, same) hold; any requested text is spelled correctly")
RULES = ("A requirement is met only if you can actually see it in the image; anything you cannot see "
         "counts as not met. Quantities too large to count exactly (stars, crowds) are judged by eye, as a "
         "typical viewer would. Extra details that don't contradict the prompt are fine. Ignore image "
         "quality and artistic style unless the prompt asks for them.")

ABS_TEMPLATE = (
    "You are checking whether an AI-generated image matches its text prompt.\n\n"
    "Prompt: {prompt}\n\n"
    "First describe in one or two sentences what the image actually shows, as if you had not read the "
    "prompt. Then check every requirement of the prompt against what you described: " + CHECKS + ". " +
    RULES + "\n\n"
    "Then score the match:\n"
    "5 = every requirement is met\n"
    "4 = all the main requirements are met; one minor detail is off\n"
    "3 = about half of the requirements are met\n"
    "2 = only a few requirements are met\n"
    "1 = nothing or almost nothing in the prompt is shown\n\n"
    'Reply with JSON only: {{"seen": what the image shows, "issues": [one short phrase per requirement that '
    'is not met; nothing that is fine], "score": 1-5}}'
)

PAIR_TEMPLATE = (
    "Two AI-generated images for the same text prompt are shown: image 1 first, then image 2.\n\n"
    "Prompt: {prompt}\n\n"
    "Which image matches the prompt more closely? Check every requirement: " + CHECKS + ". Ignore "
    "image quality and artistic style unless the prompt asks for them. Answer \"same\" only if both "
    "match equally well or equally badly.\n\n"
    'Reply with JSON only: {{"reason": one short sentence, "better": "1", "2" or "same"}}'
)

ABS_SCHEMA = {"type": "object", "properties": {
    "seen": {"type": "string"},
    "issues": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    "score": {"type": "integer", "minimum": 1, "maximum": 5}},
    "required": ["seen", "issues", "score"], "additionalProperties": False}
PAIR_SCHEMA = {"type": "object", "properties": {
    "reason": {"type": "string"}, "better": {"type": "string", "enum": ["1", "2", "same"]}},
    "required": ["reason", "better"], "additionalProperties": False}
MAX_TOKENS = {"abs": 400, "pair": 160}
# Frozen after the dev rounds (D-028 revision 3): absolute r3, pairwise r1. The test round runs
# only with this template.
FROZEN_TEMPLATE = "b9e98ed8"


def template_id() -> str:
    blob = json.dumps([ABS_TEMPLATE, PAIR_TEMPLATE, ABS_SCHEMA, PAIR_SCHEMA, MAX_SIDE], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def quote(prompt: str) -> str:
    return json.dumps(prompt, ensure_ascii=False)


# ---------- what to ask ----------

@dataclass
class Request:
    key: str                      # unique within a run
    kind: str                     # "abs" or "pair"
    mode: str                     # "greedy" or "sample"
    prompt_id: str
    prompt: str
    images: list[str]             # image paths; "grey:<path>" = a grey image of that image's size
    meta: dict[str, Any] = field(default_factory=dict)
    template: str | None = None   # another question with the absolute answer format (genprobe/story.py)

    def text(self) -> str:
        t = self.template or (ABS_TEMPLATE if self.kind == "abs" else PAIR_TEMPLATE)
        return t.format(prompt=quote(self.prompt))


@dataclass
class Bench:
    prompts: dict[str, Any]
    images: dict[str, dict]                          # sample_id -> manifest record
    by_prompt: dict[str, list[str]]                  # prompt_id -> sample_ids
    means: dict[str, dict[str, float]]               # prompt_id -> sample_id -> mean rating
    pairs: dict[str, list[tuple[str, str, int]]]     # meta_eval.make_pairs


def load_bench(prompts_path: str = "data/prompts/genai_bench_1600.jsonl",
               manifest: str = "data/manifests/ext_genai_bench.jsonl", folder: str | Path = FOLDER) -> Bench:
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    images = {sid: r for sid, r in latest_by_id(read_many([manifest])).items()
              if r.get("status") == "ok" and r.get("prompt_id") in prompts}
    ratings, _ = load_ratings(folder)
    by_prompt: dict[str, list[str]] = defaultdict(list)
    means: dict[str, dict[str, float]] = defaultdict(dict)
    for sid in sorted(images):
        r = images[sid]
        by_prompt[r["prompt_id"]].append(sid)
        v = ratings.get((r["prompt_id"], r["model"]), [])
        if v:
            means[r["prompt_id"]][sid] = sum(v) / len(v)
    pairs, _, _ = make_pairs(means)
    return Bench(prompts, images, dict(by_prompt), dict(means), pairs)


def round_prompts(bench: Bench, rnd: str) -> list[str]:
    want = "dev" if rnd == "dev" else "test"
    return sorted(pid for pid, p in bench.prompts.items() if p.split == want and pid in bench.by_prompt)


def h3_pairs(bench: Bench, seed: int = 0, n: int = N_H3_PAIRS) -> list[tuple[str, str, str, int]]:
    """The pre-registered H3 sample: n pairs drawn at random (seed 0) from the within-prompt pairs
    with different mean ratings, prompts outside the dev split."""
    pool = [(pid, a, b, s) for pid in round_prompts(bench, "test") for a, b, s in bench.pairs.get(pid, [])]
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def mismatch_map(bench: Bench, sids: list[str], seed: int = 0) -> dict[str, str]:
    """For each image, the same model's image for another prompt of the round: prompts are put in a
    random cycle (seed) and each image borrows from the next prompt in it."""
    pids = sorted({bench.images[s]["prompt_id"] for s in sids})
    order = pids[:]
    random.Random(seed).shuffle(order)
    nxt = {p: order[(i + 1) % len(order)] for i, p in enumerate(order)}
    out = {}
    for s in sids:
        r = bench.images[s]
        other = [x for x in bench.by_prompt.get(nxt[r["prompt_id"]], []) if bench.images[x]["model"] == r["model"]]
        if other and len(pids) > 1:
            out[s] = other[0]
    return out


def plan(bench: Bench, rnd: str, seed: int = 0) -> list[Request]:
    pids = round_prompts(bench, rnd)
    sids = [s for p in pids for s in bench.by_prompt[p]]
    if rnd == "dev":
        pairs = [(pid, a, b, s) for pid in pids for a, b, s in bench.pairs.get(pid, [])]
        blind = sids
    else:
        pairs = h3_pairs(bench, seed)
        blind = sorted(random.Random(seed).sample(sids, min(N_TEST_BLIND, len(sids))))
    swap = mismatch_map(bench, blind, seed)
    img = lambda s: bench.images[s]["path"]  # noqa: E731
    reqs: list[Request] = []

    def add_abs(s: str, variant: str, mode: str, images: list[str], extra: dict | None = None) -> None:
        r = bench.images[s]
        reqs.append(Request(f"abs:{variant}:{mode}:{s}", "abs", mode, r["prompt_id"],
                            bench.prompts[r["prompt_id"]].prompt, images,
                            {"sample_id": s, "model": r["model"], "variant": variant, **(extra or {})}))

    for s in sids:
        add_abs(s, "real", "greedy", [img(s)])
    for s in blind:
        add_abs(s, "grey", "greedy", [f"grey:{img(s)}"])
        if s in swap:
            add_abs(s, "mismatch", "greedy", [img(swap[s])], {"image_from": swap[s]})
        add_abs(s, "real", "sample", [img(s)])
    for pid, a, b, sign in pairs:
        pair_id = f"{a}|{b}"
        for order, (x, y) in ((1, (a, b)), (2, (b, a))):
            reqs.append(Request(f"pair:{order}:greedy:{pair_id}", "pair", "greedy", pid, bench.prompts[pid].prompt,
                                [img(x), img(y)], {"pair_id": pair_id, "order": order, "first": x, "second": y,
                                                   "human": sign}))
        reqs.append(Request(f"pair:1:sample:{pair_id}", "pair", "sample", pid, bench.prompts[pid].prompt,
                            [img(a), img(b)], {"pair_id": pair_id, "order": 1, "first": a, "second": b,
                                               "human": sign}))
    return reqs


# ---------- reading the answers ----------

SCORE_KEY = re.compile(r'"score"\s*:\s*$')


def score_digit_distribution(token_texts: list[str], tops: list[dict[str, float]]) -> dict[str, float] | None:
    """Probabilities of "1".."5" at the token right after `"score":` (renormalized over the five
    digits), from the top-k logprobs at each generated token. None if the key never appears."""
    text = ""
    for t, top in zip(token_texts, tops):
        if t.strip() and SCORE_KEY.search(text):          # skip the whitespace token(s) after the colon
            p = {k: 0.0 for k in "12345"}
            for tok, lp in top.items():
                k = tok.strip()
                if k in p:
                    p[k] += math.exp(lp)
            z = sum(p.values())
            return {k: v / z for k, v in p.items()} if z > 0 else None
        text += t
    return None


def parse_abs(text: str) -> tuple[int | None, list[str], bool]:
    try:
        d = json.loads(text)
        s = d.get("score")
        if isinstance(s, int) and 1 <= s <= 5:
            return s, [str(x) for x in d.get("issues", [])], True
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"score"\s*:\s*([1-5])\b', text)
    return (int(m.group(1)) if m else None), [], False


def _field(text: str, key: str) -> str | None:
    try:
        v = json.loads(text).get(key)
        return None if v is None else str(v)
    except (json.JSONDecodeError, AttributeError):
        return None


def parse_pair(text: str) -> tuple[str | None, str, bool]:
    """-> ("first" | "second" | "same" | None, reason, parsed_as_json)."""
    names = {"1": "first", "2": "second", "same": "same"}
    try:
        d = json.loads(text)
        b = str(d.get("better", "")).strip().lower()
        if b in names:
            reason = d["reason"] if "reason" in d else json.dumps(
                {"issues_1": d.get("issues_1", []), "issues_2": d.get("issues_2", [])}, ensure_ascii=False)
            return names[b], str(reason), True
    except (json.JSONDecodeError, AttributeError):
        pass
    m = re.search(r'"better"\s*:\s*"(1|2|same)"', text, re.I)
    return (names[m.group(1).lower()] if m else None), "", False


@dataclass
class Raw:
    text: str
    finish_reason: str | None = None
    token_texts: list[str] | None = None
    tops: list[dict[str, float]] | None = None


def to_records(reqs: list[Request], outs: list[list[Raw]], version: str) -> list[dict]:
    recs = []
    for req, samples in zip(reqs, outs):
        for i, raw in enumerate(samples):
            rec = {"key": req.key, "kind": req.kind, "mode": req.mode, "sample_idx": i, "prompt_id": req.prompt_id,
                   **req.meta, "text": raw.text, "finish_reason": raw.finish_reason, "judge_version": version}
            if req.kind == "abs":
                score, issues, ok = parse_abs(raw.text)
                rec["seen"] = _field(raw.text, "seen")
                dist = score_digit_distribution(raw.token_texts, raw.tops) if raw.tops else None
                rec.update(score=score, issues=issues, parse_ok=ok,
                           p_score=dist, expected_score=(sum(int(k) * v for k, v in dist.items()) if dist else None))
            else:
                verdict, reason, ok = parse_pair(raw.text)
                rec.update(verdict=verdict, reason=reason, parse_ok=ok)
            recs.append(rec)
    return recs


# ---------- the vLLM side (judge env, GPU) ----------

def image_url(path: str, max_side: int = MAX_SIDE) -> str:
    from PIL import Image
    grey = path.startswith("grey:")
    with Image.open(path[5:] if grey else path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        if grey:
            im = Image.new("RGB", im.size, GREY)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_memory_utilization(want: float, floor: float = 0.75, margin: float = 0.03, query=None, log=print) -> float:
    """vLLM reserves `want` x the GPU's memory at start-up and refuses to start when less is free. On a
    shared node another process can hold some of it (story job 94926 found 79 of 93 GiB free), so ask
    for what is free minus a margin, never less than `floor` (Gemma 4 31B's weights alone need ~62 GB).
    Memory set aside for the KV cache does not change what the judge answers for a given batch."""
    try:
        if query is None:
            query = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=30, check=True).stdout
        free, total = (float(x) for x in query.strip().splitlines()[0].split(","))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return want
    have = round(free / total - margin, 3)
    if have >= want:
        return want
    if have < floor:
        raise RuntimeError(f"only {free / 1024:.1f} of {total / 1024:.1f} GiB free on the GPU; the judge needs about "
                           f"{floor:.0%} of it")
    log(f"[judge] {free / 1024:.1f} of {total / 1024:.1f} GiB free: gpu_memory_utilization {want} -> {have}")
    return have


class VLLMJudge:
    def __init__(self, judge: str = DEFAULT_JUDGE, max_model_len: int = 8192, chunk: int = 256,
                 workers: int = 8, max_images: int = 2, version_extra: str = "") -> None:
        """max_images: images per request (genprobe/story.py shows up to 4 frames); version_extra is
        appended to the version string by callers that ask their own questions."""
        import importlib.metadata as md

        from vllm import LLM

        from .utils.hf import repo_cache_dir, require_cached
        cfg = JUDGES[judge]
        sha = require_cached(cfg["repo"], cfg["files"], download_hint(judge))
        path = str(repo_cache_dir(cfg["repo"]) / "snapshots" / sha)
        self.gpu_memory_utilization = fit_memory_utilization(cfg["gpu_memory_utilization"])
        self.llm = LLM(model=path, tokenizer=path, dtype="bfloat16", max_model_len=max_model_len,
                       gpu_memory_utilization=self.gpu_memory_utilization, limit_mm_per_prompt={"image": max_images},
                       seed=0, **cfg["llm"])
        extra = f"/{cfg['tag']}" if cfg.get("tag") else ""
        self.version = (f"vllm=={md.version('vllm')}/{cfg['short']}@{sha[:7]}/bf16{extra}/img{MAX_SIDE}/"
                        f"tmpl-{template_id()}{version_extra}")
        self.chat_kwargs = cfg["chat"]
        self.chunk, self.workers = chunk, workers

    def params(self, kind: str, mode: str):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        so = StructuredOutputsParams(json=ABS_SCHEMA if kind == "abs" else PAIR_SCHEMA)
        if mode == "greedy":
            return SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS[kind], structured_outputs=so,
                                  logprobs=20 if kind == "abs" else None)
        return SamplingParams(temperature=SAMPLE_TEMPERATURE, n=N_SAMPLES, seed=0, max_tokens=MAX_TOKENS[kind],
                              structured_outputs=so)

    def generate(self, reqs: list[Request], log=print) -> list[list[Raw]]:
        out: dict[str, list[Raw]] = {}
        groups: dict[tuple[str, str], list[Request]] = defaultdict(list)
        for r in reqs:
            groups[(r.kind, r.mode)].append(r)
        with ThreadPoolExecutor(self.workers) as pool:
            for (kind, mode), group in sorted(groups.items()):
                params = self.params(kind, mode)
                t0 = time.perf_counter()
                for i in range(0, len(group), self.chunk):
                    part = group[i:i + self.chunk]
                    urls = list(pool.map(lambda r: [image_url(p) for p in r.images], part))
                    convs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}} for u in us]
                               + [{"type": "text", "text": r.text()}]}] for r, us in zip(part, urls)]
                    try:
                        results = self.llm.chat(convs, params, use_tqdm=False, **self.chat_kwargs)
                    except TypeError:            # a vLLM without the use_tqdm keyword
                        results = self.llm.chat(convs, params, **self.chat_kwargs)
                    for r, res in zip(part, results):
                        out[r.key] = [self._raw(o) for o in res.outputs]
                log(f"[judge] {kind}/{mode}: {len(group)} requests in {time.perf_counter() - t0:.0f} s")
        return [out[r.key] for r in reqs]

    @staticmethod
    def _raw(o) -> Raw:
        texts = tops = None
        if o.logprobs:
            texts, tops = [], []
            for tid, lp in zip(o.token_ids, o.logprobs):
                chosen = lp.get(tid)
                texts.append((chosen.decoded_token or "") if chosen else "")
                tops.append({(v.decoded_token or ""): v.logprob for v in lp.values()})
        return Raw(o.text, o.finish_reason, texts, tops)


def run(rnd: str, tag: str, out_dir: str = "data/judge", engine=None, bench: Bench | None = None,
        log=print, judge: str = DEFAULT_JUDGE) -> dict[str, Any]:
    bench = bench or load_bench()
    reqs = plan(bench, rnd)
    counts = defaultdict(int)
    for r in reqs:
        counts[f"{r.kind}/{r.mode}/{r.meta.get('variant', 'pair')}"] += 1
    log(f"[judge] round {rnd}: {len(reqs)} requests " + json.dumps(dict(sorted(counts.items()))))
    t0 = time.perf_counter()
    engine = engine or VLLMJudge(judge)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    outs = engine.generate(reqs, log=log)
    gen_s = time.perf_counter() - t1
    recs = to_records(reqs, outs, engine.version)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / f"{tag}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    meta = {"tag": tag, "round": rnd, "judge": judge, "judge_version": engine.version, "template_id": template_id(),
            "requests": dict(sorted(counts.items())), "records": len(recs), "load_seconds": round(load_s, 1),
            "generate_seconds": round(gen_s, 1), "created_at": now_iso(), "env": fingerprint()}
    atomic_write_text(out / f"{tag}.meta.json", json.dumps(meta, indent=2, default=str))
    log(f"[judge] wrote {out / (tag + '.jsonl')} ({len(recs)} records), generation {gen_s:.0f} s")
    return meta


def grade_requests(images: list[dict]) -> list[Request]:
    """Absolute, greedy questions for a list of images (sample_id, prompt_id, prompt, path), each
    image once, always with its original prompt (module C's final grading, D-032)."""
    seen, reqs = set(), []
    for im in images:
        sid = im["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        reqs.append(Request(f"abs:real:greedy:{sid}", "abs", "greedy", im["prompt_id"], im["prompt"], [im["path"]],
                            {"sample_id": sid, "model": sid.split("__")[0], "variant": "real"}))
    return reqs


def grade(images_path: str, tag: str, out_dir: str = "data/judge", judge: str = "gemma4_31b", engine=None,
          log=print, group_prefix: str | None = None) -> dict[str, Any]:
    images = [json.loads(line) for line in Path(images_path).read_text().splitlines() if line.strip()]
    if group_prefix:                     # e.g. only the agent's images when the baselines are graded already
        images = [im for im in images if str(im.get("group", "")).startswith(group_prefix)]
    reqs = grade_requests(images)
    log(f"[judge] grading {len(reqs)} images from {images_path} with {judge}")
    t0 = time.perf_counter()
    engine = engine or VLLMJudge(judge)
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    recs = to_records(reqs, engine.generate(reqs, log=log), engine.version)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out / f"{tag}.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    meta = {"tag": tag, "kind": "grade", "images": images_path, "group_prefix": group_prefix, "judge": judge,
            "judge_version": engine.version,
            "template_id": template_id(), "records": len(recs), "load_seconds": round(load_s, 1),
            "generate_seconds": round(time.perf_counter() - t1, 1), "created_at": now_iso(), "env": fingerprint()}
    atomic_write_text(out / f"{tag}.meta.json", json.dumps(meta, indent=2, default=str))
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--round", choices=["dev", "test"], required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--out", default="data/judge")
    r.add_argument("--judge", choices=sorted(JUDGES), default=DEFAULT_JUDGE)
    g = sub.add_parser("grade", help="final grading of a list of images (module C, D-032)")
    g.add_argument("--images", required=True, help="JSONL with sample_id, prompt_id, prompt, path")
    g.add_argument("--tag", required=True)
    g.add_argument("--out", default="data/judge")
    g.add_argument("--judge", choices=sorted(JUDGES), default="gemma4_31b")
    g.add_argument("--only-group-prefix", default=None, help="grade only images whose group starts with this")
    args = ap.parse_args(argv)
    if args.cmd == "grade":
        if template_id() != FROZEN_TEMPLATE:
            print(f"template {template_id()} is not the frozen rubric {FROZEN_TEMPLATE}", file=sys.stderr)
            return 1
        if Path(args.out, f"{args.tag}.jsonl").exists():
            print(f"{args.out}/{args.tag}.jsonl exists; grades are written once", file=sys.stderr)
            return 1
        grade(args.images, args.tag, args.out, args.judge, group_prefix=args.only_group_prefix)
        return 0
    if args.round == "test" and template_id() != FROZEN_TEMPLATE:
        print(f"template {template_id()} is not the frozen rubric {FROZEN_TEMPLATE} (D-028); the test round "
              "only runs with the frozen rubric", file=sys.stderr)
        return 1
    if args.round == "test" and Path(args.out, f"{args.tag}.jsonl").exists():
        print(f"{args.out}/{args.tag}.jsonl exists; the test round runs once (pick a new tag only if it failed)",
              file=sys.stderr)
        return 1
    run(args.round, args.tag, args.out, judge=args.judge)
    return 0


if __name__ == "__main__":
    sys.exit(main())
