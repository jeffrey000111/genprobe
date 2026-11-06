"""What a worker runs: GenProbe's metric plugins, unchanged, behind one small interface (D-040).

``MetricScorer`` wraps any plugin from ``genprobe.metrics`` that scores an image against a
free-text prompt (CLIPScore, VQAScore, PickScore, HPSv2.1). The service therefore reports the
same metric versions as the research pipeline, and moving CLIPScore on a CPU to VQAScore on a GPU
is a change of worker settings, not of code. ``FakeScorer`` is a deterministic stand-in for tests
and for load tests of the service itself.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Item:
    key: str          # the task id, as text
    image_path: str
    prompt: str


@dataclass(frozen=True)
class ScoreResult:
    value: float
    extra: dict[str, Any] = field(default_factory=dict)


class Scorer(Protocol):
    name: str
    version: str

    def score(self, items: list[Item]) -> list[ScoreResult]:
        """One result per item, in order. Raises if the batch cannot be scored."""


class MetricScorer:
    def __init__(self, name: str, device: str = "cpu"):
        from ..metrics import REGISTRY

        if name not in REGISTRY:
            raise ValueError(f"unknown metric {name!r}; known: {', '.join(sorted(REGISTRY))}")
        cls = REGISTRY[name]
        if set(cls.needs) != {"image", "prompt"}:
            raise ValueError(f"{name} needs {sorted(cls.needs)}; the service scores an image against a "
                             "free-text prompt only (no template specs, no video)")
        self.metric = cls()
        self.metric.setup(device)
        self.name = name
        self.version = self.metric.version

    def score(self, items: list[Item]) -> list[ScoreResult]:
        from ..samples import Sample

        samples = [Sample(sample_id=it.key, model="service", prompt_id="request", prompt=it.prompt,
                          seed=0, path=it.image_path) for it in items]
        by_id = {r["sample_id"]: r for r in self.metric.score_batch(samples)}
        out = []
        for it in items:
            r = by_id.get(it.key)
            if r is None:
                raise RuntimeError(f"{self.name} returned no score for task {it.key}")
            v = r.get("value")
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                raise ValueError(f"{self.name} returned {v!r} for task {it.key}")
            out.append(ScoreResult(float(v), dict(r.get("extra") or {})))
        return out


class FakeScorer:
    """Scores from a hash of the image bytes and the prompt, after sleeping
    ``base_seconds + per_item_seconds * len(batch)`` (a model's fixed and per-image cost).
    A prompt containing ``[fail]`` makes its whole batch raise, to exercise retries and the
    worker's isolation of a bad item."""

    name = "fake"

    def __init__(self, base_seconds: float = 0.0, per_item_seconds: float = 0.0, version: str = "fake-1"):
        self.base_seconds = base_seconds
        self.per_item_seconds = per_item_seconds
        self.version = version

    def score(self, items: list[Item]) -> list[ScoreResult]:
        time.sleep(self.base_seconds + self.per_item_seconds * len(items))
        bad = [it.key for it in items if "[fail]" in it.prompt]
        if bad:
            raise RuntimeError(f"injected failure for task(s) {', '.join(bad)}")
        out = []
        for it in items:
            with open(it.image_path, "rb") as f:
                h = hashlib.sha256(f.read() + it.prompt.encode("utf-8")).hexdigest()
            out.append(ScoreResult(int(h[:8], 16) / 0xFFFFFFFF, {"fake": True}))
        return out


def load_scorer(name: str, device: str = "cpu") -> Scorer:
    if name == "fake":
        return FakeScorer(float(os.environ.get("GENPROBE_FAKE_BASE_SECONDS", "0")),
                          float(os.environ.get("GENPROBE_FAKE_ITEM_SECONDS", "0")))
    return MetricScorer(name, device)
