"""Image generation backends.

- ``FakeBackend``: deterministic noise images on CPU. Used by tests and by the
  ``--backend fake`` smoke run so sharding/resume/manifests can be checked on a login
  node without spending GPU hours.
- ``DiffusersBackend``: the real thing (guide §6.2). Imports torch/diffusers lazily.

Both return a list of PIL images and a timing dict for one batch.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from PIL import Image

from .config import GenConfig
from .samples import Sample


def is_blank(img: Image.Image, std_threshold: float = 2.0) -> bool:
    """True for all-black / single-colour images (safety-checker blackouts, VAE NaNs)."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return float(arr.std()) < std_threshold


class FakeBackend:
    name = "fake"

    def __init__(self, cfg: GenConfig, size: int = 64, blank_ids: set[str] | None = None,
                 fail_ids: set[str] | None = None):
        self.cfg = cfg
        self.size = size
        self.blank_ids = blank_ids or set()
        self.fail_ids = fail_ids or set()
        self.calls: list[str] = []  # sample_ids actually generated, for tests

    def scheduler_name(self) -> str:
        return "FakeScheduler"

    def generate(self, batch: list[Sample]) -> tuple[list[Image.Image], dict[str, Any]]:
        t0 = time.perf_counter()
        images = []
        for s in batch:
            if s.sample_id in self.fail_ids:
                raise RuntimeError(f"injected failure for {s.sample_id}")
            self.calls.append(s.sample_id)
            if s.sample_id in self.blank_ids:
                arr = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            else:
                seed = int(hashlib.sha256(f"{s.prompt}|{s.seed}".encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(seed)
                arr = rng.integers(0, 256, size=(self.size, self.size, 3), dtype=np.uint8)
            images.append(Image.fromarray(arr))
        sec = (time.perf_counter() - t0) / max(len(batch), 1)
        return images, {"sec_per_img": sec, "peak_mem_gb": 0.0}


class DiffusersBackend:
    """Loads one diffusers pipeline per job and generates batches with per-sample generators."""

    name = "diffusers"

    def __init__(self, cfg: GenConfig):
        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.cfg = cfg
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.dtype]

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available on this node; refusing to run the diffusers backend.")
        if cfg.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise SystemExit(
                f"{torch.cuda.get_device_name(0)} does not support bf16 (P100/V100?). "
                "Request an A100/H100 via --gres (guide §3.6)."
            )

        kwargs = dict(cfg.load_kwargs)
        if cfg.variant:
            kwargs["variant"] = cfg.variant
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        # `torch_dtype` is accepted across diffusers releases; newer releases may prefer
        # `dtype`. Either way we verify the loaded dtype below instead of trusting the kwarg,
        # because an ignored kwarg would silently load fp32 weights.
        self.pipe = DiffusionPipeline.from_pretrained(cfg.repo, torch_dtype=dtype, **kwargs)
        if cfg.cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        loaded = getattr(self.pipe, "dtype", None)
        if loaded is not None and loaded != dtype:
            raise SystemExit(
                f"pipeline loaded as {loaded}, expected {dtype}. Check the dtype kwarg name for "
                "your diffusers version (pip show diffusers) and update DiffusersBackend."
            )

    def scheduler_name(self) -> str:
        return type(self.pipe.scheduler).__name__

    def generate(self, batch: list[Sample]) -> tuple[list[Image.Image], dict[str, Any]]:
        torch = self.torch
        cfg = self.cfg
        gens = [torch.Generator("cuda").manual_seed(s.seed) for s in batch]
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        images = self.pipe(
            prompt=[s.prompt for s in batch],
            generator=gens,
            num_inference_steps=cfg.steps,
            guidance_scale=cfg.cfg,
            height=cfg.height,
            width=cfg.width,
            **cfg.call_kwargs,
        ).images
        end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000 / len(batch)
        return images, {"sec_per_img": sec, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}


def make_backend(kind: str, cfg: GenConfig, **kwargs):
    if kind == "fake":
        return FakeBackend(cfg, **kwargs)
    if kind == "diffusers":
        return DiffusersBackend(cfg)
    raise ValueError(f"unknown backend {kind!r}")
