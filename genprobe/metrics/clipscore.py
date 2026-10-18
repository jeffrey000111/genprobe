"""CLIPScore baseline (guide §7.3): cosine similarity of CLIP ViT-L/14 image and text embeddings.

Implemented directly on open_clip rather than through t2v_metrics so the exact weights,
activation and text truncation are under our control and batched (D-013).

- Weights: OpenAI CLIP ViT-L/14 (224 px), loaded by open_clip from the HF repo
  ``timm/vit_large_patch14_clip_224.openai`` (download_models.sh clip_l14).
- OpenAI weights were trained with QuickGELU. open_clip 3.x no longer switches the activation
  for the ``openai`` tag by itself (it only warns), so we force it and check the result.
  Without it the scores are silently worse.
- ``value`` is the raw cosine. The CLIPScore of Hessel et al. (2021) is ``2.5 * max(cos, 0)``;
  it is stored in ``extra["clipscore_w2.5"]``. Both rank samples identically when cos > 0.
- CLIP's text encoder sees at most 77 tokens; longer prompts are truncated and flagged with
  ``extra["truncated"]`` (guide §7.3).
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

from ..utils.hf import require_cached
from .base import Metric, register

HF_REPO = "timm/vit_large_patch14_clip_224.openai"
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh clip_l14"'


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


def n_clip_tokens(tokenizer: Any, text: str) -> int | None:
    """Token count including start/end tokens, or None if the tokenizer can't tell us."""
    enc = getattr(tokenizer, "encode", None)
    if enc is None:
        return None
    try:
        return len(enc(text)) + 2
    except Exception:  # noqa: BLE001 - counting is diagnostic only
        return None


@register
class CLIPScore(Metric):
    name = "clipscore"
    arch = "ViT-L-14"
    pretrained = "openai"
    needs = frozenset({"image", "prompt"})
    env = "eval"
    higher_is_better = True
    default_batch_size = 32

    def __init__(self) -> None:
        self.version = f"open_clip=={_pkg_version('open-clip-torch')}/{self.arch}:{self.pretrained}/cos"
        self.info: dict[str, Any] = {}

    def setup(self, device: str) -> None:
        import warnings

        import open_clip
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; CLIPScore runs inside a GPU job, not on a login node")
        rev = require_cached(HF_REPO, ("open_clip_model.safetensors",), DOWNLOAD_HINT)

        cfg = open_clip.get_pretrained_cfg(self.arch, self.pretrained)
        if not cfg:
            raise RuntimeError(f"open_clip has no pretrained tag {self.pretrained!r} for {self.arch}")
        hub = str(cfg.get("hf_hub", "")).strip("/")
        if hub != HF_REPO:
            raise RuntimeError(f"open_clip now loads {self.arch}:{self.pretrained} from {hub!r}, "
                               f"not {HF_REPO!r}; update HF_REPO and download_models.sh")
        quick = bool(cfg.get("quick_gelu", False))
        with warnings.catch_warnings():
            # open_clip warns (instead of failing) when the activation doesn't match the weights
            warnings.filterwarnings("error", message=".*quick.?gelu")
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.arch, pretrained=self.pretrained, device=device, force_quick_gelu=quick)
        has_quick = any(type(m).__name__ == "QuickGELU" for m in model.modules())
        if has_quick != quick:
            raise RuntimeError(f"activation mismatch: weights want quick_gelu={quick}, model has {has_quick}")

        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.arch)
        self.context_length = int(getattr(self.tokenizer, "context_length", 77))
        self.device = device
        self.info = {"hf_repo": HF_REPO, "revision": rev, "quick_gelu": quick,
                     "context_length": self.context_length, "precision": "fp32",
                     "preprocess": str(preprocess)}

    def score_batch(self, samples):
        import torch
        import torch.nn.functional as F
        from PIL import Image

        images = []
        for s in samples:
            with Image.open(s.path) as im:
                images.append(self.preprocess(im.convert("RGB")))
        texts = [s.prompt for s in samples]
        with torch.inference_mode():
            img = self.model.encode_image(torch.stack(images).to(self.device))
            txt = self.model.encode_text(self.tokenizer(texts).to(self.device))
            cos = (F.normalize(img.float(), dim=-1) * F.normalize(txt.float(), dim=-1)).sum(-1)
        out = []
        for s, c in zip(samples, cos.tolist()):
            n = n_clip_tokens(self.tokenizer, s.prompt)
            out.append({"sample_id": s.sample_id, "value": float(c),
                        "extra": {"clipscore_w2.5": 2.5 * max(float(c), 0.0), "n_tokens": n,
                                  "truncated": None if n is None else n > self.context_length}})
        return out
