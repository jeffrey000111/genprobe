"""VQAScore (guide §7.4): probability that a VLM answers "Yes" to 'Does this figure show "{prompt}"?'.

Backbone: Qwen2.5-VL-7B-Instruct (bf16, PyTorch SDPA attention). Implemented directly on
transformers + qwen-vl-utils, as a line-for-line port of the scoring path in t2v_metrics'
``Qwen2VLModel.forward`` (D-014). t2v_metrics itself is not imported: its 3.0 PyPI package
won't import without three git-only dependencies (LLaVA-NeXT, OpenAI CLIP, a pytorchvideo
fork; LLaVA-NeXT pins torch 2.1) that the Qwen path never uses.

What the port reproduces from t2v_metrics:
- message ``[{"role": "user", "content": [{"type": "image", "image": PIL}, {"type": "text",
  "text": question}]}]``, ``apply_chat_template(..., add_generation_prompt=True)``, images
  resized by ``qwen_vl_utils.process_vision_info`` and passed with ``do_resize=False``;
- ``generate(max_new_tokens=1, do_sample=False, temperature=1.0, output_scores=True)`` and
  ``softmax(scores[0])`` over the full vocabulary at the "Yes" token. ``generate`` applies the
  model's generation_config processors (Qwen ships ``repetition_penalty=1.05``), exactly as in
  t2v_metrics, so the scores match its definition rather than raw forward logits.

What it pins down explicitly: the template (hashed into ``metric_version``), the weights
revision (also in ``metric_version``), the attention implementation, and that every weight
in the checkpoint was actually loaded (a silent partial load would give fluent-looking but
meaningless scores). Each image is scored only against its own prompt, one at a time.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
from typing import Any

from ..utils.hf import cached_revision, require_cached
from .base import Metric, register

BACKBONE = "qwen2.5-vl-7b"
HF_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
QUESTION_TEMPLATE = 'Does this figure show "{}"? Please answer Yes or No.'   # t2v_metrics default
ANSWER_TEMPLATE = "Yes"
NEGATIVE_ANSWER = "No"   # only for the diagnostic p_no in extra
ATTN = "sdpa"
IMPL = "port-of-t2v_metrics-qwen2vl"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                  "chat_template.json", "model.safetensors.index.json")
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh qwen25vl7b"'


def template_id(question: str = QUESTION_TEMPLATE, answer: str = ANSWER_TEMPLATE) -> str:
    return hashlib.sha1(f"{question}\x00{answer}".encode()).hexdigest()[:8]


def build_messages(image: Any, question: str) -> list[dict[str, Any]]:
    """The exact chat structure t2v_metrics sends for a single image (no system message)."""
    return [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": question}]}]


def single_token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"answer {text!r} is {len(ids)} tokens {ids}; this implementation scores "
                         "single-token answers only")
    return int(ids[0])


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
    dtype_name = "bfloat16"   # the check script may switch to float32 on CPU; jobs never do

    def __init__(self) -> None:
        rev = (cached_revision(HF_REPO) or "nocache")[:7]
        self.version = (f"vqascore-{IMPL}/transformers=={_pkg_version('transformers')}/{BACKBONE}@{rev}"
                        f"/{ATTN}/tmpl-{template_id()}")
        self.info: dict[str, Any] = {}

    def setup(self, device: str) -> None:
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; VQAScore runs inside a GPU job, not on a login node")
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        rev = require_cached(HF_REPO, REQUIRED_FILES, DOWNLOAD_HINT)
        dtype = getattr(torch, self.dtype_name)
        model, loading = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            HF_REPO, torch_dtype=dtype, attn_implementation=ATTN, output_loading_info=True)
        problems = {k: v for k, v in loading.items() if v and k in ("missing_keys", "mismatched_keys")}
        if problems:
            raise RuntimeError(f"{HF_REPO}@{rev[:7]} did not load cleanly with transformers "
                               f"{_pkg_version('transformers')}: { {k: list(v)[:5] for k, v in problems.items()} }")
        loaded_attn = getattr(model.config, "_attn_implementation", None)
        if loaded_attn != ATTN:
            raise RuntimeError(f"model loaded with attention {loaded_attn!r}, expected {ATTN!r}")
        self.model = model.to(device).eval()
        self.processor = AutoProcessor.from_pretrained(HF_REPO)
        self.process_vision_info = process_vision_info
        self.yes_id = single_token_id(self.processor.tokenizer, ANSWER_TEMPLATE)
        self.no_id = single_token_id(self.processor.tokenizer, NEGATIVE_ANSWER)
        self.device = device
        gen = self.model.generation_config
        self.info = {"implementation": IMPL, "backbone": BACKBONE, "hf_repo": HF_REPO, "revision": rev,
                     "dtype": self.dtype_name, "attn_implementation": loaded_attn,
                     "question_template": QUESTION_TEMPLATE, "answer_template": ANSWER_TEMPLATE,
                     "template_id": template_id(), "answer_token_id": self.yes_id,
                     "unexpected_keys": len(loading.get("unexpected_keys", [])),
                     "generation_config": {k: getattr(gen, k, None) for k in
                                           ("repetition_penalty", "temperature", "top_k", "top_p", "do_sample")}}

    def score_one(self, image: Any, prompt: str) -> tuple[float, dict[str, Any]]:
        import torch

        question = QUESTION_TEMPLATE.format(prompt)
        messages = build_messages(image, question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt", do_resize=False).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=1, temperature=1.0, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        probs = torch.softmax(out.scores[0][0].float(), dim=-1)
        top = int(probs.argmax())
        extra = {"question": question, "p_no": float(probs[self.no_id]),
                 "top_token": self.processor.tokenizer.decode([top]),
                 "n_image_tokens": int(inputs["image_grid_thw"].prod().item() // 4)
                 if "image_grid_thw" in inputs else None}
        return float(probs[self.yes_id]), extra

    def score_batch(self, samples):
        import torch
        from PIL import Image

        out = []
        with torch.inference_mode():
            for s in samples:
                with Image.open(s.path) as im:
                    image = im.convert("RGB")
                value, extra = self.score_one(image, s.prompt)
                out.append({"sample_id": s.sample_id, "value": value, "extra": extra})
        return out
