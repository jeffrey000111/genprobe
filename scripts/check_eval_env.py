"""Check the eval env and everything the scoring plugins rely on, without a GPU or loading weights.

Runs on a CPU node through scripts/check_eval_env.sh, so problems show up in minutes instead
of after a long wait for an H100. Exits 1 if any required check fails.

Covers: package versions; open_clip's config for the CLIP weights; the Qwen2.5-VL classes,
processor, chat template and image preprocessing on a real smoke image; that the checkpoint's
weight names match what this transformers version builds (a mismatch would load silently
half-random weights); that the answers are single tokens; and the HF cache.
"""

from __future__ import annotations

import glob
import importlib.metadata as md
import json
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # for `genprobe`
FAILED: list[str] = []


def check(name: str, fn, required: bool = True) -> None:
    try:
        detail = fn()
        print(f"OK    {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        if required:
            FAILED.append(name)
        print(f"{'FAIL' if required else 'INFO'}  {name}: {type(e).__name__}: {e}")
        if required:
            traceback.print_exc(limit=4)


def versions() -> str:
    out = []
    for p in ("torch", "transformers", "accelerate", "huggingface-hub", "qwen-vl-utils", "open-clip-torch",
              "t2v-metrics", "flash-attn"):
        try:
            out.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            out.append(f"{p}: not installed")
    return ", ".join(out)


def open_clip_cfg() -> str:
    import open_clip

    from genprobe.metrics.clipscore import HF_REPO
    cfg = open_clip.get_pretrained_cfg("ViT-L-14", "openai")
    hub = str(cfg.get("hf_hub", "")).strip("/")
    assert hub == HF_REPO, f"open_clip loads from {hub!r}, plugin expects {HF_REPO!r}"
    assert cfg.get("quick_gelu") is True, f"quick_gelu={cfg.get('quick_gelu')!r}"
    return f"ViT-L-14:openai -> {hub}, quick_gelu=True"


def weights_cached() -> str:
    from genprobe.metrics import clipscore, vqascore
    from genprobe.utils.hf import require_cached
    a = require_cached(clipscore.HF_REPO, ("open_clip_model.safetensors",), clipscore.DOWNLOAD_HINT)
    b = require_cached(vqascore.HF_REPO, vqascore.REQUIRED_FILES, vqascore.DOWNLOAD_HINT)
    return f"{clipscore.HF_REPO}@{a[:7]}, {vqascore.HF_REPO}@{b[:7]}"


def qwen_weight_names() -> str:
    """Build the model on the meta device (no memory, no weights) and compare its parameter
    names with the checkpoint index."""
    import torch
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration

    from genprobe.metrics.vqascore import ATTN, HF_REPO
    from genprobe.utils.hf import cached_revision, repo_cache_dir
    cfg = AutoConfig.from_pretrained(HF_REPO)
    cfg._attn_implementation = ATTN
    with torch.device("meta"):
        model = Qwen2_5_VLForConditionalGeneration._from_config(cfg, attn_implementation=ATTN,
                                                                torch_dtype=torch.bfloat16)
    expected = set(model.state_dict().keys())
    index = repo_cache_dir(HF_REPO) / "snapshots" / cached_revision(HF_REPO) / "model.safetensors.index.json"
    stored = set(json.loads(index.read_text())["weight_map"])
    missing, unexpected = sorted(expected - stored), sorted(stored - expected)
    assert not missing, f"{len(missing)} model weights not in the checkpoint, e.g. {missing[:3]}"
    note = f", {len(unexpected)} unused checkpoint tensors e.g. {unexpected[:2]}" if unexpected else ""
    return (f"{len(expected)} weights all present in the checkpoint{note}; "
            f"attn={model.config._attn_implementation}")


def qwen_inputs() -> str:
    """Processor, chat template and vision preprocessing exactly as the plugin runs them."""
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    from genprobe.metrics.vqascore import (ANSWER_TEMPLATE, HF_REPO, NEGATIVE_ANSWER, QUESTION_TEMPLATE,
                                           build_messages, single_token_id)
    proc = AutoProcessor.from_pretrained(HF_REPO)
    yes, no = single_token_id(proc.tokenizer, ANSWER_TEMPLATE), single_token_id(proc.tokenizer, NEGATIVE_ANSWER)
    # first image of each model (512 px for SD1.5, 1024 px for the others)
    imgs = [sorted(glob.glob(f"{d}/*.png"))[0] for d in sorted(glob.glob(str(ROOT / "data/images/*")))
            if glob.glob(f"{d}/*.png")]
    sizes = []
    for path in imgs:
        with Image.open(path) as im:
            image = im.convert("RGB")
        messages = build_messages(image, QUESTION_TEMPLATE.format("a photo of 2 dogs"))
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        assert text.rstrip().endswith("assistant"), f"generation prompt missing: ...{text[-60:]!r}"
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                      return_tensors="pt", do_resize=False)
        sizes.append(f"{Path(path).parent.name} {image.size}->grid {inputs['image_grid_thw'].tolist()[0]}, "
                     f"{inputs['input_ids'].shape[1]} tokens")
    if not imgs:
        sizes.append("no smoke images found under data/images, preprocessing not exercised")
    return f"'Yes'={yes}, 'No'={no}; " + "; ".join(sizes)


def t2v_metrics_import() -> str:
    import t2v_metrics  # noqa: F401
    return "importable (not used by the plugins)"


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("no ffmpeg on PATH (only t2v_metrics needs it)")
    return exe


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(versions())
    check("scoring weights in the HF cache", weights_cached)
    check("open_clip config for CLIPScore weights", open_clip_cfg)
    check("Qwen2.5-VL weight names match this transformers", qwen_weight_names)
    check("Qwen2.5-VL processor, chat template, image preprocessing", qwen_inputs)
    check("ffmpeg on PATH", ffmpeg, required=False)
    check("import t2v_metrics", t2v_metrics_import, required=False)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
