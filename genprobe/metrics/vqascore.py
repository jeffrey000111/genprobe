"""VQAScore (guide §7.4): probability that a VLM answers "Yes" to 'Does this figure show "{prompt}"?'.

Runs t2v_metrics with the Qwen2.5-VL-7B backbone (D-014). Things this plugin pins down that
t2v_metrics would otherwise leave implicit:

- **Attention.** t2v_metrics 3.0 hard-codes ``attn_implementation="flash_attention_2"`` for
  the Qwen models, and flash-attn is not in the eval env (it has to be compiled against the
  exact torch/CUDA). We switch the entry to PyTorch SDPA before the model is built (3.1 on
  GitHub uses SDPA by default too) and check what transformers actually loaded.
- **Weights.** t2v_metrics doesn't pass its ``cache_dir`` to the Qwen ``from_pretrained``
  calls, so weights come from the normal HF cache ($HF_HOME/hub, filled on coe-hpc1 by
  ``download_models.sh qwen25vl7b``). The cached commit hash goes into ``metric_version``.
- **Template.** The question/answer templates are passed explicitly and hashed into
  ``metric_version``: changing them changes the scores (guide §7.4).
- **Pairs.** ``scorer(images, texts)`` scores every image against every text; we only want
  each image with its own prompt, so samples are scored one pair at a time (the Qwen
  backbone has no batching anyway).
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import inspect
from typing import Any

from ..utils.hf import cached_revision, hub_cache_dir, require_cached
from .base import Metric, register

BACKBONE = "qwen2.5-vl-7b"
HF_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
QUESTION_TEMPLATE = 'Does this figure show "{}"? Please answer Yes or No.'
ANSWER_TEMPLATE = "Yes"
ATTN = "sdpa"
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh qwen25vl7b"'


def template_id(question: str = QUESTION_TEMPLATE, answer: str = ANSWER_TEMPLATE) -> str:
    return hashlib.sha1(f"{question}\x00{answer}".encode()).hexdigest()[:8]


def force_attention(models: dict[str, Any], backbone: str, impl: str = ATTN) -> str | None:
    """Set the attention implementation of one t2v_metrics model entry in place.

    Returns the previous value. Raises if the entry doesn't have the layout we expect, so a
    t2v_metrics upgrade can't silently ignore the override.
    """
    if backbone not in models:
        raise KeyError(f"t2v_metrics has no backbone {backbone!r}; available: {sorted(models)}")
    entry = models[backbone].get("model")
    if not isinstance(entry, dict) or "attn_implementation" not in entry:
        raise KeyError(f"unexpected t2v_metrics config layout for {backbone!r}: {models[backbone]!r}")
    old = entry["attn_implementation"]
    entry["attn_implementation"] = impl
    return old


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


@register
class VQAScore(Metric):
    name = "vqascore"
    needs = frozenset({"image", "prompt"})
    env = "eval"
    higher_is_better = True
    default_batch_size = 8

    def __init__(self) -> None:
        rev = (cached_revision(HF_REPO) or "nocache")[:7]
        self.version = (f"t2v_metrics=={_pkg_version('t2v-metrics')}/{BACKBONE}@{rev}/{ATTN}"
                        f"/tmpl-{template_id()}")
        self.info: dict[str, Any] = {}

    def setup(self, device: str) -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; VQAScore runs inside a GPU job, not on a login node")
        try:
            import t2v_metrics
        except RuntimeError as e:
            if "ffmpeg" in str(e):
                raise RuntimeError(
                    "t2v_metrics won't import without an `ffmpeg` on PATH. Link the one bundled "
                    "with imageio-ffmpeg into the env: rerun scripts/setup_envs.sh eval on coe-hpc1, "
                    "or see docs/hpc_notes.md (eval env).") from e
            raise
        from t2v_metrics.models.vqascore_models import qwen2vl_model

        repo = qwen2vl_model.QWEN2_VL_MODELS.get(BACKBONE, {}).get("model", {}).get("path")
        if repo != HF_REPO:
            raise RuntimeError(f"t2v_metrics loads {BACKBONE} from {repo!r}, expected {HF_REPO!r}; "
                               "update HF_REPO and download_models.sh")
        rev = require_cached(HF_REPO, ("config.json", "preprocessor_config.json", "tokenizer_config.json"),
                             DOWNLOAD_HINT)
        old_attn = force_attention(qwen2vl_model.QWEN2_VL_MODELS, BACKBONE, ATTN)

        self.scorer = t2v_metrics.VQAScore(model=BACKBONE, device=device, cache_dir=str(hub_cache_dir()))

        params = inspect.signature(self.scorer.model.forward).parameters
        missing = [p for p in ("question_template", "answer_template") if p not in params]
        if missing:
            raise RuntimeError(f"{BACKBONE}.forward() has no {missing} argument(s); can't fix the template")
        self.kwargs = {"question_template": QUESTION_TEMPLATE, "answer_template": ANSWER_TEMPLATE}

        hf_model = getattr(self.scorer.model, "model", None)
        cfg = getattr(hf_model, "config", None)
        loaded_attn = getattr(cfg, "_attn_implementation", None)
        if loaded_attn not in (None, ATTN):
            raise RuntimeError(f"model loaded with attention {loaded_attn!r}, expected {ATTN!r}")
        dtype = str(next(hf_model.parameters()).dtype) if hf_model is not None else None
        self.info = {"backbone": BACKBONE, "hf_repo": HF_REPO, "revision": rev,
                     "attn_implementation": loaded_attn, "attn_overridden_from": old_attn,
                     "dtype": dtype, "question_template": QUESTION_TEMPLATE,
                     "answer_template": ANSWER_TEMPLATE, "template_id": template_id()}

    def score_batch(self, samples):
        import torch

        out = []
        with torch.inference_mode():
            for s in samples:
                scores = self.scorer(images=[str(s.path)], texts=[s.prompt], **self.kwargs)
                out.append({"sample_id": s.sample_id, "value": float(scores.reshape(-1)[0]),
                            "extra": {"question": QUESTION_TEMPLATE.format(s.prompt)}})
        return out
