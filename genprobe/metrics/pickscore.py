"""PickScore (guide §7.7, D-021): a CLIP-H model fine-tuned on human preferences (Pick-a-Pic).

``value`` = ``logit_scale.exp() * cos(image, prompt)`` as in the PickScore README (typical
values around 20-23); ``extra["cos"]`` keeps the raw cosine. It rewards what people prefer to
look at, not whether the image follows the prompt, so it is reported next to the alignment
metrics, never averaged with them.

Weights and processor both come from ``yuvalkirstain/PickScore_v1`` (its preprocessor and
tokenizer files match the ``laion/CLIP-ViT-H-14-laion2B-s32B-b79K`` processor the README
loads: 224 px bicubic resize + centre crop, OpenAI mean/std, CLIP BPE). fp32. Prompts over 77
tokens are truncated by the text encoder and flagged in ``extra["truncated"]``.
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

from ..utils.hf import cached_revision, require_cached
from .base import Metric, register

HF_REPO = "yuvalkirstain/PickScore_v1"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors")
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh pickscore"'


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


@register
class PickScore(Metric):
    name = "pickscore"
    needs = frozenset({"image", "prompt"})
    env = "eval"
    higher_is_better = True
    default_batch_size = 16

    def __init__(self) -> None:
        rev = (cached_revision(HF_REPO) or "nocache")[:7]
        self.version = f"transformers=={_pkg_version('transformers')}/pickscore-v1@{rev}/fp32"
        self.info: dict[str, Any] = {}

    def setup(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; run on a GPU node or pass --device cpu")
        rev = require_cached(HF_REPO, REQUIRED_FILES, DOWNLOAD_HINT)
        self.processor = AutoProcessor.from_pretrained(HF_REPO)
        model, load = AutoModel.from_pretrained(HF_REPO, torch_dtype=torch.float32, output_loading_info=True)
        if load.get("missing_keys") or load.get("mismatched_keys"):
            raise RuntimeError(f"PickScore loaded partially: missing {load.get('missing_keys')[:5]}, "
                               f"mismatched {load.get('mismatched_keys')[:5]}")
        self.model = model.to(device).eval()
        self.device = device
        self.info = {"hf_repo": HF_REPO, "revision": rev, "dtype": "float32", "device": device,
                     "logit_scale": float(self.model.logit_scale.exp().item()),
                     "unexpected_keys": load.get("unexpected_keys")}

    def score_batch(self, samples):
        import torch
        from PIL import Image

        images = []
        for s in samples:
            with Image.open(s.path) as im:
                images.append(im.convert("RGB"))
        prompts = [s.prompt for s in samples]
        tok = self.processor.tokenizer
        lengths = [len(tok(p)["input_ids"]) for p in prompts]
        img_in = self.processor(images=images, return_tensors="pt").to(self.device)
        txt_in = self.processor(text=prompts, padding=True, truncation=True, max_length=77,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            ie = self.model.get_image_features(**img_in)
            te = self.model.get_text_features(**txt_in)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            cos = (ie * te).sum(dim=-1)
            scores = self.model.logit_scale.exp() * cos
        return [{"sample_id": s.sample_id, "value": float(v),
                 "extra": {"cos": round(float(c), 6), "n_tokens": n, "truncated": n > 77}}
                for s, v, c, n in zip(samples, scores.tolist(), cos.tolist(), lengths)]
