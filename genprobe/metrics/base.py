"""Metric plugin interface (guide §7.1).

Each metric lives in its own file, subclasses ``Metric``, and registers itself with
``@register``. The scoring runner (``genprobe.score``) handles sharding, resume and
writing records; a metric only loads its model and scores batches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..samples import Sample

# Required keys in every score record (guide §4.3). The runner adds the bookkeeping
# fields (metric_version, status, created_at, run_id, ...); metrics return the rest.
SCORE_RECORD_KEYS = frozenset({"sample_id", "metric", "metric_version", "value", "extra", "seconds", "status"})


class Metric(ABC):
    name: ClassVar[str]                     # "vqascore"
    # "t2v_metrics==3.0/qwen2.5-vl-7b@cc59489/sdpa/tmpl-…". Resume matches on it, so anything
    # that changes scores belongs in it. Plugins may set it per instance in __init__ (cheap
    # lookups only: no torch import).
    version: str
    needs: ClassVar[frozenset[str]] = frozenset({"image", "prompt"})  # add "spec" for template prompts
    env: ClassVar[str] = "eval"             # virtualenv it runs in: "eval" or "aux"
    higher_is_better: ClassVar[bool] = True
    default_batch_size: ClassVar[int] = 16
    # Filled by setup(): what was actually loaded (weights revision, dtype, attention, templates).
    # The runner stores it in data/runs/score_*.json next to the package versions.
    info: dict[str, Any] = {}

    def applies_to(self, sample: Sample) -> bool:
        """Skip samples this metric can't score (e.g. OCR only on text-rendering prompts)."""
        if "spec" in self.needs and not sample.spec:
            return False
        return True

    @abstractmethod
    def setup(self, device: str) -> None:
        """Load models once per job."""

    @abstractmethod
    def score_batch(self, samples: list[Sample]) -> list[dict[str, Any]]:
        """Return one dict per sample, same order, each with at least
        ``{"sample_id", "value", "extra"}``; ``extra`` holds metric-specific details
        (e.g. the VQA question, a ``truncated`` flag, detected counts)."""


REGISTRY: dict[str, type[Metric]] = {}


def register(cls: type[Metric]) -> type[Metric]:
    if cls.name in REGISTRY and REGISTRY[cls.name] is not cls:
        raise ValueError(f"metric {cls.name!r} registered twice")
    REGISTRY[cls.name] = cls
    return cls


def validate_record(rec: dict[str, Any]) -> list[str]:
    """Problems with a score record; empty list means it's well-formed."""
    problems = [f"missing key {k!r}" for k in SCORE_RECORD_KEYS - rec.keys()]
    if rec.get("status") == "ok":
        v = rec.get("value")
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v != v:  # v != v catches NaN
            problems.append(f"value must be a finite number when status=ok, got {v!r}")
    if "extra" in rec and not isinstance(rec["extra"], dict):
        problems.append("extra must be a dict")
    return problems
