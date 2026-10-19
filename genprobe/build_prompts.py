"""Build prompt set v1 (guide §5): 400 GenAI-Bench prompts + 200 template prompts with answers.

    python -m genprobe.build_prompts --genai-bench data/external/genai_bench \
        --out data/prompts/prompts_v1.jsonl

Deterministic: same inputs and seed give a byte-identical file (tests check this). Run it
once, commit the output, and never rebuild after results exist (D-016).

GenAI-Bench part. genai_skills.json maps 10 fine skills to prompt indices; each prompt is in
the "basic" or the "advanced" tier (7 prompts have no skill and are skipped). Prompts are
picked rarest skill first until every skill has at least ``MIN_PER_SKILL`` prompts, then the
rest are drawn at random from what is left. ``prompt_id = gb{index:04d}`` keeps the GenAI-Bench
index, which module B uses to join the human ratings.

Template part (objects are COCO classes, so a detector can check them later):
counting 70 = 10 objects x counts 1-7; spatial 70 = 4 relations x random object pairs;
text 60 = 20 phrases per length tier (1-2, 3-4, 5-8 words) x 3 surface templates.

Split: 20 % dev / 80 % test inside every stratum (GenAI-Bench tier; template category and
sub-type), so dev is a miniature of test.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEED = 20261019
N_GENAI = 400
MIN_PER_SKILL = 60
DEV_FRACTION = 0.2

SKILL_NAMES = {  # GenAI-Bench key -> tag used here
    "attribute": "attribute", "scene": "scene", "spatial relation": "spatial_relation",
    "action relation": "action_relation", "part relation": "part_relation",
    "counting": "counting", "comparison": "comparison", "differentiation": "differentiation",
    "negation": "negation", "universal": "universal",
}
TIERS = ("basic", "advanced")

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
# (singular, plural). COCO classes that are easy to tell apart and to count.
COUNT_OBJECTS = [("apple", "apples"), ("orange", "oranges"), ("cup", "cups"), ("bottle", "bottles"),
                 ("book", "books"), ("chair", "chairs"), ("bicycle", "bicycles"), ("car", "cars"),
                 ("dog", "dogs"), ("bird", "birds")]
SPATIAL_OBJECTS = ["dog", "cat", "car", "bicycle", "chair", "cup", "bottle", "book", "clock",
                   "umbrella", "vase", "laptop", "bench", "teddy bear", "horse", "potted plant"]
RELATIONS = [("left_of", "to the left of"), ("right_of", "to the right of"),
             ("above", "above"), ("below", "below")]
COUNT_TEMPLATES = ["a photo of {n} {obj}", "a realistic photo of {n} {obj}"]
SPATIAL_TEMPLATES = ["a photo of {a} {rel} {b}", "{a} {rel} {b}"]
TEXT_TEMPLATES = ['a storefront sign that says "{t}"', 'a poster with the words "{t}"',
                  'a chalkboard that says "{t}"']
# None of these repeat the smoke set's phrases (D-010 keeps smoke separate).
TEXT_PHRASES = {
    "short": ["WELCOME", "SALE", "EXIT", "HELLO", "CLOSED", "BAKERY", "PIZZA", "FLOWERS", "TAXI",
              "STOP", "THANK YOU", "HOT COFFEE", "FREE WIFI", "QUIET PLEASE", "GOOD MORNING",
              "ICE CREAM", "LIVE MUSIC", "PUSH HERE", "FARM MARKET", "USED BOOKS"],
    "medium": ["BACK IN FIVE MINUTES", "FRESH FLOWERS DAILY", "NO DOGS ALLOWED", "HAPPY NEW YEAR",
               "PLEASE KNOCK FIRST", "MIND THE GAP", "COFFEE AND DONUTS", "SALE ENDS TODAY",
               "WORK IN PROGRESS", "KEEP THIS DOOR CLOSED", "CASH ONLY PLEASE", "SOUP OF THE DAY",
               "BEST PIZZA IN TOWN", "NOW HIRING FRIENDLY STAFF", "OPEN SEVEN DAYS",
               "FREE COFFEE REFILLS", "BIKES FOR RENT", "HANDMADE SOAP SOLD HERE", "TICKETS SOLD OUT",
               "WATCH YOUR HEAD"],
    "long": ["PLEASE TAKE OFF YOUR SHOES HERE", "ALL ITEMS ARE FIFTY PERCENT OFF",
             "THE LIBRARY CLOSES AT NINE TONIGHT", "WE ARE OPEN EVERY DAY OF THE WEEK",
             "DO NOT FEED THE ANIMALS PLEASE", "WASH YOUR HANDS BEFORE RETURNING TO WORK",
             "THE BEST COFFEE IN THE WHOLE CITY", "KEEP CALM AND READ A GOOD BOOK",
             "NO SMOKING WITHIN TWENTY FEET OF THIS DOOR", "PLEASE RING THE BELL FOR SERVICE",
             "FRESH BAKED COOKIES EVERY SINGLE MORNING", "THIS DOOR MUST REMAIN CLOSED AT ALL TIMES",
             "CHILDREN MUST BE WITH AN ADULT", "PARKING IS FOR CUSTOMERS ONLY",
             "HOMEMADE PIE AND HOT COFFEE INSIDE", "WATCH YOUR STEP WHEN LEAVING THE TRAIN",
             "SUMMER SALE STARTS THIS FRIDAY MORNING", "GARDEN TOOLS AND SEEDS SOLD HERE",
             "LOST AND FOUND IS AT THE FRONT DESK", "PLEASE SILENCE YOUR PHONE DURING THE SHOW"],
}
TEXT_TIERS = {"short": (1, 2), "medium": (3, 4), "long": (5, 8)}


def article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def load_genai_bench(folder: str | Path) -> tuple[dict[int, str], dict[str, list[int]]]:
    folder = Path(folder)
    images = json.loads((folder / "genai_image.json").read_text())
    skills = json.loads((folder / "genai_skills.json").read_text())
    prompts = {int(k): v["prompt"].strip() for k, v in images.items()}
    return prompts, {k: sorted(set(v)) for k, v in skills.items()}


def genai_tags(skills: dict[str, list[int]]) -> dict[int, list[str]]:
    """index -> [tier, fine skills...]; prompts without a fine skill are left out."""
    tier_of = {i: t for t in TIERS for i in skills.get(t, [])}
    fine: dict[int, list[str]] = defaultdict(list)
    for key, tag in SKILL_NAMES.items():
        for i in skills.get(key, []):
            fine[i].append(tag)
    return {i: [tier_of.get(i, "untiered")] + sorted(tags) for i, tags in fine.items()}


def sample_genai(tags: dict[int, list[str]], n: int, min_per_skill: int, rng: random.Random) -> list[int]:
    chosen: list[int] = []
    taken: set[int] = set()
    count: Counter = Counter()
    by_skill: dict[str, list[int]] = defaultdict(list)
    for i in sorted(tags):
        for t in tags[i][1:]:
            by_skill[t].append(i)
    for skill in sorted(by_skill, key=lambda s: (len(by_skill[s]), s)):   # rarest first
        pool = [i for i in by_skill[skill] if i not in taken]
        rng.shuffle(pool)
        for i in pool:
            if count[skill] >= min_per_skill or len(chosen) >= n:
                break
            chosen.append(i)
            taken.add(i)
            count.update(tags[i][1:])
    rest = [i for i in sorted(tags) if i not in taken]
    rng.shuffle(rest)
    chosen += rest[: n - len(chosen)]
    return sorted(chosen)


def template_prompts(rng: random.Random) -> list[dict[str, Any]]:
    out = []
    k = 0
    for singular, plural in COUNT_OBJECTS:
        for n in range(1, 8):
            k += 1
            obj = singular if n == 1 else plural
            out.append({"prompt_id": f"tc{k:04d}", "source": "template_count",
                        "prompt": rng.choice(COUNT_TEMPLATES).format(n=NUMBER_WORDS[n], obj=obj),
                        "tags": ["counting"], "spec": {"objects": [{"name": singular, "count": n}]},
                        "stratum": f"count_{n}"})
    k = 0
    pairs = [(a, b) for a in SPATIAL_OBJECTS for b in SPATIAL_OBJECTS if a != b]
    rng.shuffle(pairs)
    per_rel = [18, 18, 17, 17]
    it = iter(pairs)
    for (rel, phrase), m in zip(RELATIONS, per_rel):
        for _ in range(m):
            a, b = next(it)
            k += 1
            out.append({"prompt_id": f"ts{k:04d}", "source": "template_spatial",
                        "prompt": rng.choice(SPATIAL_TEMPLATES).format(a=article(a), rel=phrase, b=article(b)),
                        "tags": ["spatial"], "spec": {"relation": {"type": rel, "a": a, "b": b}},
                        "stratum": f"spatial_{rel}"})
    k = 0
    for tier, phrases in TEXT_PHRASES.items():
        lo, hi = TEXT_TIERS[tier]
        for t in phrases:
            n_words = len(t.split())
            if not lo <= n_words <= hi:
                raise ValueError(f"{t!r} has {n_words} words, outside the {tier} tier {lo}-{hi}")
            k += 1
            out.append({"prompt_id": f"tt{k:04d}", "source": "template_text",
                        "prompt": rng.choice(TEXT_TEMPLATES).format(t=t),
                        "tags": ["text"], "spec": {"text": t, "n_words": n_words, "length_tier": tier},
                        "stratum": f"text_{tier}"})
    return out


def assign_splits(rows: list[dict[str, Any]], rng: random.Random, dev_fraction: float = DEV_FRACTION) -> None:
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stratum[r["stratum"]].append(r)
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum], key=lambda r: r["prompt_id"])
        rng.shuffle(group)
        n_dev = round(len(group) * dev_fraction)
        for i, r in enumerate(group):
            r["split"] = "dev" if i < n_dev else "test"


def clip_token_counter():
    """open_clip's tokenizer when installed (on HPC3), else None: counts stay null here."""
    try:
        import open_clip
        tok = open_clip.get_tokenizer("ViT-L-14")
        return lambda text: len(tok.encode(text)) + 2
    except Exception:  # noqa: BLE001
        return None


def build(genai_folder: str | Path, seed: int = SEED, n_genai: int = N_GENAI,
          min_per_skill: int = MIN_PER_SKILL) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    texts, skills = load_genai_bench(genai_folder)
    tags = genai_tags(skills)
    picked = sample_genai(tags, n_genai, min_per_skill, rng)
    rows = [{"prompt_id": f"gb{i:04d}", "source": "genai_bench", "prompt": texts[i], "tags": tags[i],
             "spec": None, "stratum": f"genai_{tags[i][0]}"} for i in picked]
    rows += template_prompts(rng)
    assign_splits(rows, rng)
    count = clip_token_counter()
    for r in rows:
        r["n_clip_tokens"] = count(r["prompt"]) if count else None
    seen = Counter(r["prompt"].lower() for r in rows)
    dups = [p for p, c in seen.items() if c > 1]
    if dups:
        raise ValueError(f"duplicate prompts: {dups[:3]}")
    keys = ["prompt_id", "source", "prompt", "tags", "spec", "split", "stratum", "n_clip_tokens"]
    return [{k: r[k] for k in keys} for r in sorted(rows, key=lambda r: r["prompt_id"])]


def describe(rows: list[dict[str, Any]]) -> str:
    lines = [f"{len(rows)} prompts; split: {dict(Counter(r['split'] for r in rows))}"]
    lines.append("by stratum (dev/test): " + ", ".join(
        f"{s} {sum(r['split'] == 'dev' for r in rows if r['stratum'] == s)}/"
        f"{sum(r['split'] == 'test' for r in rows if r['stratum'] == s)}"
        for s in sorted({r['stratum'] for r in rows})))
    gb = [r for r in rows if r["source"] == "genai_bench"]
    skill_counts = Counter(t for r in gb for t in r["tags"][1:])
    lines.append("GenAI-Bench skills (prompts per tag, overlapping): " +
                 ", ".join(f"{k} {v}" for k, v in sorted(skill_counts.items())))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genai-bench", default="data/external/genai_bench")
    ap.add_argument("--out", default="data/prompts/prompts_v1.jsonl")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)
    rows = build(args.genai_bench, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    print(describe(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
