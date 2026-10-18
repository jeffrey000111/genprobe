"""Check the eval env and the t2v_metrics internals the scoring plugins rely on, without a GPU.

Runs on a CPU node through scripts/check_eval_env.sh, so problems show up in minutes instead
of after a long wait for an H100. Loads no model weights. Exits 1 if any check fails.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for `genprobe`
FAILED: list[str] = []


def check(name: str, fn) -> None:
    try:
        detail = fn()
        print(f"OK    {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        FAILED.append(name)
        print(f"FAIL  {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)


def versions() -> str:
    out = []
    for p in ("torch", "transformers", "accelerate", "huggingface-hub", "t2v-metrics", "open-clip-torch",
              "qwen-vl-utils", "decord", "imageio-ffmpeg", "flash-attn"):
        try:
            out.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            out.append(f"{p}: not installed")
    return ", ".join(out)


def ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("no `ffmpeg` on PATH (see docs/hpc_notes.md, eval env)")
    first = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=30, check=True).stdout
    return f"{exe} ({first.splitlines()[0] if first else '?'})"


def import_open_clip() -> str:
    import open_clip

    from genprobe.metrics.clipscore import HF_REPO
    cfg = open_clip.get_pretrained_cfg("ViT-L-14", "openai")
    hub = str(cfg.get("hf_hub", "")).strip("/")
    assert hub == HF_REPO, f"open_clip loads from {hub!r}, plugin expects {HF_REPO!r}"
    assert cfg.get("quick_gelu") is True, f"quick_gelu={cfg.get('quick_gelu')!r}"
    return f"ViT-L-14:openai -> {hub}, quick_gelu=True"


def import_t2v_metrics() -> str:
    import t2v_metrics  # noqa: F401
    return t2v_metrics.__file__


def qwen_entry() -> str:
    from t2v_metrics.models.vqascore_models import qwen2vl_model as q

    from genprobe.metrics.vqascore import BACKBONE, HF_REPO
    entry = q.QWEN2_VL_MODELS[BACKBONE]
    model = entry["model"]
    assert model["path"] == HF_REPO, f"path {model['path']!r} != {HF_REPO!r}"
    assert "attn_implementation" in model, f"no attn_implementation key in {model!r}"
    params = inspect.signature(q.Qwen2VLModel.forward).parameters
    missing = [p for p in ("question_template", "answer_template") if p not in params]
    assert not missing, f"Qwen2VLModel.forward lacks {missing}"
    default_q = params["question_template"].default
    return (f"{BACKBONE}: path ok, attn={model['attn_implementation']!r} (plugin switches to sdpa), "
            f"dtype={model.get('torch_dtype')}, default question={default_q!r}")


def score_forwards_kwargs() -> str:
    import t2v_metrics

    params = inspect.signature(t2v_metrics.VQAScore.forward).parameters
    assert any(p.kind is p.VAR_KEYWORD for p in params.values()), f"VQAScore.forward{tuple(params)} drops kwargs"
    return f"VQAScore.forward{tuple(params)}"


def weights_cached() -> str:
    from genprobe.metrics import clipscore, vqascore
    from genprobe.utils.hf import require_cached
    a = require_cached(clipscore.HF_REPO, ("open_clip_model.safetensors",), clipscore.DOWNLOAD_HINT)
    b = require_cached(vqascore.HF_REPO, ("config.json", "preprocessor_config.json", "tokenizer_config.json",
                                          "model.safetensors.index.json"), vqascore.DOWNLOAD_HINT)
    return f"{clipscore.HF_REPO}@{a[:7]}, {vqascore.HF_REPO}@{b[:7]}"


def main() -> int:
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    print(versions())
    check("ffmpeg on PATH", ffmpeg)
    check("import open_clip + CLIP weights config", import_open_clip)
    check("import t2v_metrics", import_t2v_metrics)
    if "import t2v_metrics" not in FAILED:
        check("t2v_metrics qwen2.5-vl-7b entry and forward()", qwen_entry)
        check("VQAScore.forward passes template kwargs", score_forwards_kwargs)
    check("scoring weights in the HF cache", weights_cached)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {', '.join(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
