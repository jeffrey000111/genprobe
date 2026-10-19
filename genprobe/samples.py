"""Sample identity, prompt loading, and sample construction.

A *sample* is one (model, prompt, seed) triple. Its ``sample_id`` is the join key
for every manifest and score table in the project:

    sample_id = f"{model}__{prompt_id}__s{seed}"
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SEP = "__"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_part(kind: str, value: str) -> None:
    if not _SAFE_NAME.match(value) or SEP in value:
        raise ValueError(
            f"{kind}={value!r} must match {_SAFE_NAME.pattern} and must not contain {SEP!r}"
        )


def make_sample_id(model: str, prompt_id: str, seed: int) -> str:
    _check_part("model", model)
    _check_part("prompt_id", prompt_id)
    if int(seed) < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return f"{model}{SEP}{prompt_id}{SEP}s{int(seed)}"


def parse_sample_id(sample_id: str) -> tuple[str, str, int]:
    model, prompt_id, s = sample_id.split(SEP)
    if not s.startswith("s"):
        raise ValueError(f"bad sample_id {sample_id!r}")
    return model, prompt_id, int(s[1:])


def image_path(root: str | Path, model: str, prompt_id: str, seed: int, ext: str = "png") -> Path:
    """Canonical location of a generated image: {root}/images/{model}/{prompt_id}_s{seed}.png"""
    return Path(root) / "images" / model / f"{prompt_id}_s{int(seed)}.{ext}"


@dataclass
class Prompt:
    prompt_id: str
    prompt: str
    source: str = "unknown"
    tags: list[str] = field(default_factory=list)
    spec: dict[str, Any] | None = None
    split: str | None = None
    stratum: str | None = None
    n_clip_tokens: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prompt":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        p = cls(**known)
        _check_part("prompt_id", p.prompt_id)
        return p


def load_prompts(
    path: str | Path, split: str | None = None, limit: int | None = None
) -> list[Prompt]:
    """Load a prompts JSONL file. Sorted by prompt_id so ``limit`` is deterministic."""
    prompts: list[Prompt] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            p = Prompt.from_dict(json.loads(line))
            if p.prompt_id in seen:
                raise ValueError(f"{path}:{lineno}: duplicate prompt_id {p.prompt_id!r}")
            seen.add(p.prompt_id)
            if split is None or p.split == split:
                prompts.append(p)
    prompts.sort(key=lambda p: p.prompt_id)
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


@dataclass
class Sample:
    sample_id: str
    model: str
    prompt_id: str
    prompt: str
    seed: int
    path: str
    tags: list[str] = field(default_factory=list)
    spec: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def build_samples(
    prompts: Iterable[Prompt], model: str, seeds: Iterable[int], root: str | Path
) -> list[Sample]:
    out = []
    for p in prompts:
        for seed in seeds:
            out.append(
                Sample(
                    sample_id=make_sample_id(model, p.prompt_id, seed),
                    model=model,
                    prompt_id=p.prompt_id,
                    prompt=p.prompt,
                    seed=int(seed),
                    path=str(image_path(root, model, p.prompt_id, seed)),
                    tags=list(p.tags),
                    spec=p.spec,
                )
            )
    return out
