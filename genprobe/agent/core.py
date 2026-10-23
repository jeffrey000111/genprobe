"""Core of the generation-debugging agent (guide §10, D-031): per-prompt state, the checks that
decide pass / fail, diagnosis, the rule policy, rewrites, the GPU-second budget and the audit log.
Pure Python, no GPU; genprobe/agent/run.py drives it round by round.

Attempt 0 of every prompt is the SDXL seed-0 image of prompt set v1 (already generated and scored).
Each later attempt changes one thing (seed, prompt wording, cfg or model) of the best attempt so
far, is generated and checked, and replaces the best only if it is clearly better. The agent stops
when the image passes, when no action is left, or when the next action would exceed the budget.
Checks always use the ORIGINAL prompt and spec, never a rewrite (guide §10.6).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from pathlib import Path

from ..build_prompts import COUNT_OBJECTS, NUMBER_WORDS

ROOT = Path("data/agent")


def run_dir(run: str, root: Path = ROOT) -> Path:
    return Path(root) / run

BASE_MODEL = "sdxl"
BASE_SEED = 0
TAU_GENAI = 0.9          # VQAScore needed to count as passed on GenAI-Bench prompts (dev may change it)
TAU_TEMPLATE_VQA = 0.9   # rules r2: template prompts also need this VQAScore besides their checker
DELTA = 0.02             # a VQAScore gain smaller than this is noise (guide §10.6; dev may change it)
CFG_HIGH = 7.5           # SDXL's default is 5.0
SEED_OFFSET = 1000       # reseeded attempts use seeds 1000+, never v1's 0-3
MAX_ATTEMPTS = 12        # hard stop besides the budget (rules r2: the budget is the real limit)
RULESET = "r2"
# Strongest model per category on the DEV split of prompt set v1 (results/prompts_v1_dev): sd35m
# for counting (detector 0.589) and spatial (0.732); flux_schnell for text (OCR 0.912, and ~4x
# cheaper than sd35m's 0.896) and GenAI-Bench (VQAScore 0.802 basic, 0.539 advanced).
ROUTE = {"counting": "sd35m", "spatial": "sd35m", "text": "flux_schnell", "genai": "flux_schnell"}
CHECK_METRICS = {"counting": ("detect", "vqascore"), "spatial": ("detect", "vqascore"),
                 "text": ("ocr", "vqascore"), "genai": ("vqascore",)}
OCR_PASS = 0.85          # D-017 revision

# Rule policy: ordered actions per diagnosis; an action is skipped when it no longer changes
# anything (e.g. the model is already the routed one) or does not fit the budget. When the list is
# used up and budget is left, the best settings so far get new seeds until the budget runs out.
# Rules r2 (D-031 revision 1, after dev run devrule1): switching to the category's routed model
# comes first; it was the action behind most improvements in devrule1.
RULES = {
    "wrong_count": ["switch_model", "rewrite", "reseed"],
    "missing_object": ["switch_model", "rewrite", "reseed"],
    "wrong_relation": ["switch_model", "rewrite", "reseed"],
    "misspelled_text": ["switch_model", "reseed", "rewrite"],
    "missing_text": ["switch_model", "rewrite", "reseed"],
    "low_alignment": ["switch_model", "raise_cfg", "reseed"],
}
POLICIES = ("rule", "rule_sdxl", "llm")   # rule_sdxl: the ablation that never switches model
ACTIONS = ("reseed", "rewrite", "raise_cfg", "switch_model")


def category_of(tags: list[str]) -> str:
    for c in ("counting", "spatial", "text"):
        if c in tags[:1]:
            return c
    return "genai"


# ---------- checks ----------

def checker_ok(category: str, checks: dict[str, Any]) -> bool:
    if category in ("counting", "spatial"):
        return checks.get("detect") == 1.0
    if category == "text":
        return (checks.get("ocr") or 0.0) >= OCR_PASS
    return True


def passed(category: str, checks: dict[str, Any], tau_genai: float = TAU_GENAI) -> bool:
    """Rules r2: template prompts need their checker AND VQAScore >= 0.9 (in devrule1, 12 final
    images passed the checker alone but not the judge); GenAI-Bench prompts need VQAScore >= 0.9."""
    v = checks.get("vqascore") or 0.0
    if category == "genai":
        return v >= tau_genai
    return checker_ok(category, checks) and v >= TAU_TEMPLATE_VQA


def closeness(category: str, checks: dict[str, Any]) -> float:
    """How near a failed check came (higher is nearer); a tie-break between failed attempts."""
    x = checks.get("detect_extra") or {}
    if category == "counting":
        if "count_pred" in x:
            return -abs(int(x["count_pred"]) - int(x["count_target"]))
        return -99.0
    if category == "spatial":
        if checks.get("detect") == 1.0:
            return 0.0
        return -2.0 if x.get("missing") else -1.0
    if category == "text":
        return float(checks.get("ocr") or 0.0)
    return float(checks.get("vqascore") or 0.0)


def better(new: dict, cur: dict, category: str, delta: float = DELTA, tau_genai: float = TAU_GENAI) -> bool:
    """Does attempt `new` (its checks) clearly beat `cur`? Pass beats fail; then a nearer miss;
    then a VQAScore gain of at least delta."""
    pn, pc = passed(category, new, tau_genai), passed(category, cur, tau_genai)
    if pn != pc:
        return pn
    if category != "genai":
        cn, cc = closeness(category, new), closeness(category, cur)
        if cn != cc:
            return cn > cc
    return (new.get("vqascore") or 0.0) >= (cur.get("vqascore") or 0.0) + delta


def diagnose(category: str, checks: dict[str, Any]) -> str:
    x = checks.get("detect_extra") or {}
    if category != "genai" and checker_ok(category, checks):
        return "low_alignment"            # the checker is satisfied, VQAScore is not
    if category == "counting":
        return "missing_object" if x.get("count_pred", 0) == 0 else "wrong_count"
    if category == "spatial":
        return "missing_object" if x.get("missing") else "wrong_relation"
    if category == "text":
        return "missing_text" if (checks.get("ocr") or 0.0) < 0.3 else "misspelled_text"
    return "low_alignment"


def describe(category: str, checks: dict[str, Any]) -> str:
    """One line for the audit log / an LLM: what the checkers saw."""
    x = checks.get("detect_extra") or {}
    v = checks.get("vqascore")
    vq = f"VQAScore {v:.2f}" if isinstance(v, (int, float)) else "VQAScore –"
    if category == "counting" and "count_pred" in x:
        return f"detector counted {x['count_pred']} (wanted {x['count_target']}); {vq}"
    if category == "spatial":
        if x.get("missing"):
            return f"detector found no {' / '.join(x['missing'])} object; {vq}"
        return f"detector: relation {'holds' if checks.get('detect') == 1.0 else 'does not hold'}; {vq}"
    if category == "text":
        return f"OCR match {checks.get('ocr', 0):.2f}, read {checks.get('ocr_read', '')!r}; {vq}"
    return vq


# ---------- rewrites (rule policy: templates from the prompt's own spec) ----------

def _plural(name: str) -> str:
    return dict(COUNT_OBJECTS).get(name, name + "s")


def _article(noun: str) -> str:
    return ("an " if noun[0] in "aeiou" else "a ") + noun


def rewrite_rule(category: str, prompt: str, spec: dict | None) -> str | None:
    """Make the failing requirement explicit without adding anything else (guide §10.6)."""
    spec = spec or {}
    if category == "counting" and spec.get("objects"):
        o = spec["objects"][0]
        n = int(o["count"])
        word = NUMBER_WORDS.get(n, str(n))
        noun = o["name"] if n == 1 else _plural(o["name"])
        return f"a photo of exactly {word} {noun}, {word} in total, each one fully visible and clearly separated"
    if category == "spatial" and spec.get("relation"):
        r = spec["relation"]
        a, b = _article(r["a"]), _article(r["b"])
        side = {"left_of": (f"{a} on the left side of the image", f"{b} on the right side"),
                "right_of": (f"{a} on the right side of the image", f"{b} on the left side"),
                "above": (f"{a} in the upper part of the image", f"{b} in the lower part, directly below it"),
                "below": (f"{a} in the lower part of the image", f"{b} in the upper part, directly above it")}
        first, second = side[r["type"]]
        return f"{first} and {second}, both clearly visible"
    if category == "text" and spec.get("text"):
        return f'{prompt}, the exact text "{spec["text"]}" written once in large, clear, correctly spelled letters'
    return None


# ---------- state ----------

@dataclass
class Attempt:
    k: int
    model: str
    gen_prompt: str
    cfg: float | None
    seed: int
    sample_id: str
    action: str
    reason: str
    status: str = "planned"            # planned -> ok / error
    checks: dict[str, Any] = field(default_factory=dict)
    gen_seconds: float | None = None
    score_seconds: float | None = None
    llm_seconds: float = 0.0
    path: str | None = None

    def cost(self) -> float:
        return float(self.gen_seconds or 0.0) + float(self.score_seconds or 0.0) + float(self.llm_seconds or 0.0)


@dataclass
class PromptState:
    prompt_id: str
    prompt: str
    category: str
    spec: dict | None
    tags: list[str]
    budget: float
    policy: str = "rule"
    attempts: list[Attempt] = field(default_factory=list)
    best: int = 0
    done: bool = False
    stop_reason: str = ""
    used: dict[str, int] = field(default_factory=dict)     # diagnosis -> index into its rule list

    def spent(self) -> float:
        return sum(a.cost() for a in self.attempts)

    def best_attempt(self) -> Attempt:
        return self.attempts[self.best]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptState":
        d = dict(d)
        d["attempts"] = [Attempt(**a) for a in d["attempts"]]
        return cls(**d)


def attempt_prompt_id(prompt_id: str, k: int, run: str) -> str:
    """Unique per run, so two runs never share a sample_id (scores are reused by sample_id)."""
    return f"{prompt_id}-{run}-a{k}"


def expected_cost(model: str, costs: dict) -> float:
    return float(costs["gen"].get(model, costs["gen"][BASE_MODEL]))


def score_cost(category: str, costs: dict) -> float:
    return sum(float(costs["score"].get(m, 0.0)) for m in CHECK_METRICS[category])


def budget_for(category: str, costs: dict, n: int = 4) -> float:
    """Best-of-n with the base model: n generations plus n checks (guide §10.4)."""
    return n * (expected_cost(BASE_MODEL, costs) + score_cost(category, costs))


def propose(state: PromptState, action: str, costs: dict) -> tuple[dict, str] | None:
    """The parameters `action` would give the next attempt (from the best one so far), with a
    reason; None if the action would change nothing."""
    b = state.best_attempt()
    nxt = {"model": b.model, "gen_prompt": b.gen_prompt, "cfg": b.cfg, "seed": b.seed}
    if action == "reseed":
        nxt["seed"] = SEED_OFFSET + len(state.attempts)
        return nxt, "new seed, same everything else"
    if action == "rewrite":
        new = rewrite_rule(state.category, state.prompt, state.spec)
        if not new or new == b.gen_prompt:
            return None
        nxt["gen_prompt"] = new
        return nxt, "state the failing requirement explicitly"
    if action == "raise_cfg":
        if b.model != BASE_MODEL or (b.cfg or 0) >= CFG_HIGH:
            return None
        nxt["cfg"] = CFG_HIGH
        return nxt, f"follow the prompt more closely (cfg {CFG_HIGH})"
    if action == "switch_model":
        target = ROUTE[state.category]
        if b.model == target or state.policy == "rule_sdxl":
            return None
        nxt.update(model=target, cfg=None)
        return nxt, f"{target} is the strongest model for {state.category} prompts on dev"
    raise ValueError(action)


def next_rule_action(state: PromptState, costs: dict) -> tuple[str, dict, str] | None:
    """The next affordable action in the rule list of the current diagnosis, or None."""
    diag = diagnose(state.category, state.best_attempt().checks)
    rules = RULES[diag]
    left = state.budget - state.spent()
    i = state.used.get(diag, 0)
    while i < len(rules):
        action = rules[i]
        i += 1
        p = propose(state, action, costs)
        if p is None:
            continue
        params, reason = p
        if expected_cost(params["model"], costs) + score_cost(state.category, costs) > left + 1e-9:
            continue
        state.used[diag] = i
        return action, params, f"{diag}: {reason}"
    state.used[diag] = i
    p = propose(state, "reseed", costs)                  # rest of the budget: new seeds, best settings
    if p and expected_cost(p[0]["model"], costs) + score_cost(state.category, costs) <= left + 1e-9:
        return "reseed", p[0], f"{diag}: rule list used up; a new seed with the best settings so far"
    return None


def audit(state: PromptState, a: Attempt, decision: str) -> dict:
    return {"prompt_id": state.prompt_id, "step": a.k, "action": a.action, "policy": state.policy,
            "params": {"model": a.model, "gen_prompt": a.gen_prompt, "cfg": a.cfg, "seed": a.seed},
            "sample_id": a.sample_id, "checks": {k: v for k, v in a.checks.items() if k != "detect_extra"},
            "passed": passed(state.category, a.checks) if a.checks else None,
            "diagnosis": diagnose(state.category, a.checks) if a.checks else None,
            "decision_reason": a.reason, "result": decision,
            "gpu_seconds": round(a.cost(), 3), "budget_left": round(state.budget - state.spent(), 3)}


def round_up(x: float) -> float:
    return math.ceil(x * 1000) / 1000
