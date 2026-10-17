"""Validated configs. Every record carries ``config_hash`` so a changed parameter is visible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenConfig(BaseModel):
    """One text-to-image model at fixed, documented settings (guide §6.1)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="short model name used in sample_id, e.g. 'sdxl'")
    repo: str = Field(description="Hugging Face repo id")
    revision: str | None = Field(None, description="pin a commit hash once the model is validated")
    variant: str | None = Field(None, description="weight variant to download/load, e.g. 'fp16'")
    dtype: Literal["fp32", "fp16", "bf16"]
    steps: int = Field(gt=0)
    cfg: float = Field(ge=0)
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    batch_size: int = Field(1, gt=0)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    load_kwargs: dict[str, Any] = Field(default_factory=dict, description="extra from_pretrained kwargs")
    call_kwargs: dict[str, Any] = Field(default_factory=dict, description="extra pipe(...) kwargs")
    cpu_offload: bool = False
    notes: str = ""

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str) -> str:
        if "__" in v or not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("name must be alphanumeric/underscore/dash and must not contain '__'")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds_ok(cls, v: list[int]) -> list[int]:
        if not v or len(set(v)) != len(v) or min(v) < 0:
            raise ValueError("seeds must be a non-empty list of distinct non-negative ints")
        return v

    def config_hash(self) -> str:
        return stable_hash(self.model_dump(exclude={"notes"}))


def stable_hash(obj: Any, n: int = 12) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_gen_config(path: str | Path) -> GenConfig:
    return GenConfig.model_validate(load_yaml(path))
