"""HPSv2.1 (guide §7.7, D-021): Human Preference Score v2.1, an OpenCLIP ViT-H/14 fine-tuned on
human preference pairs (HPDv2).

``value`` = cosine between the normalized image and prompt embeddings, which is what the
``hpsv2`` package returns (typical values around 0.25-0.32).

Implemented on mainstream open_clip (already in the eval env) instead of the ``hpsv2``
package, which bundles its own copy of open_clip and downloads its checkpoint at run time:
the ViT-H-14 architecture from open_clip, the weights ``HPS_v2.1_compressed.pt`` from
``xswu/HPSv2`` (download_models.sh hpsv2), loaded with ``weights_only=True``; every model
weight must be present in the checkpoint. Images go through open_clip's ViT-H-14 eval transform
(224 px bicubic resize + centre crop, OpenAI mean/std), as in hpsv2. One difference, recorded
in the version: fp32 here, where hpsv2 runs under CUDA autocast (fp16).
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

from ..utils.hf import cached_revision, repo_cache_dir, require_cached
from .base import Metric, register

HF_REPO = "xswu/HPSv2"
CHECKPOINT = "HPS_v2.1_compressed.pt"
ARCH = "ViT-H-14"
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh hpsv2"'


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


def extract_state_dict(ckpt: Any) -> dict:
    """HPS checkpoints are ``{"state_dict": {...}}``; strip a DataParallel ``module.`` prefix."""
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict) or not sd:
        raise RuntimeError(f"{CHECKPOINT} does not hold a state dict (got {type(ckpt).__name__})")
    if all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


@register
class HPSv2(Metric):
    name = "hpsv2"
    needs = frozenset({"image", "prompt"})
    env = "eval"
    higher_is_better = True
    default_batch_size = 16

    def __init__(self) -> None:
        rev = (cached_revision(HF_REPO) or "nocache")[:7]
        self.version = (f"open_clip=={_pkg_version('open-clip-torch')}/{ARCH}/hps-v2.1@{rev}/fp32/cos")
        self.info: dict[str, Any] = {}

    def setup(self, device: str) -> None:
        import open_clip
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; run on a GPU node or pass --device cpu")
        rev = require_cached(HF_REPO, (CHECKPOINT,), DOWNLOAD_HINT)
        path = repo_cache_dir(HF_REPO) / "snapshots" / rev / CHECKPOINT
        model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=None, device="cpu")
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"could not load {path} with weights_only=True: {e}") from e
        missing, unexpected = model.load_state_dict(extract_state_dict(ckpt), strict=False)
        if missing:
            raise RuntimeError(f"HPSv2.1 checkpoint is missing {len(missing)} weights, e.g. {missing[:5]}")
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(ARCH)
        self.device = device
        self.info = {"hf_repo": HF_REPO, "revision": rev, "checkpoint": CHECKPOINT, "arch": ARCH,
                     "dtype": "float32", "device": device, "unexpected_keys": list(unexpected)}

    def score_batch(self, samples):
        import torch
        from PIL import Image

        ims = []
        for s in samples:
            with Image.open(s.path) as im:
                ims.append(self.preprocess(im.convert("RGB")))
        image = torch.stack(ims).to(self.device)
        text = self.tokenizer([s.prompt for s in samples]).to(self.device)
        with torch.inference_mode():
            ie = self.model.encode_image(image, normalize=True)
            te = self.model.encode_text(text, normalize=True)
            cos = (ie * te).sum(dim=-1)
        return [{"sample_id": s.sample_id, "value": float(c), "extra": {}} for s, c in zip(samples, cos.tolist())]
