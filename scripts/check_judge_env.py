"""Probe the judge env on an H100 (slurm/judge_probe.sbatch): can vLLM load Qwen3-VL-8B-Instruct
offline from the HF cache and answer about one image, about two images in one prompt, with
JSON-constrained output, and how fast is it? Each step reports on its own, so one job shows
everything that works and everything that doesn't before genprobe/judge.py is written.

Images are drawn here (shapes on white), so nothing from data/ is needed.
"""

from __future__ import annotations

import base64
import importlib.metadata as md
import io
import json
import sys
import time
import traceback

REPO = "Qwen/Qwen3-VL-8B-Instruct"


def data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def shapes(n: int, color: str, kind: str = "circle", size: int = 448):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (size, size), "white")
    d = ImageDraw.Draw(img)
    step = size // (n + 1)
    for i in range(n):
        x = step * (i + 1)
        box = (x - 30, size // 2 - 30, x + 30, size // 2 + 30)
        (d.ellipse if kind == "circle" else d.rectangle)(box, fill=color)
    return img


def msg(text: str, *imgs) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": data_url(im)}} for im in imgs]
    return [{"role": "user", "content": content + [{"type": "text", "text": text}]}]


def step(name: str, fn) -> bool:
    print(f"\n== {name}", flush=True)
    t0 = time.perf_counter()
    try:
        fn()
        print(f"   ok ({time.perf_counter() - t0:.1f} s)", flush=True)
        return True
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print(f"   FAILED ({time.perf_counter() - t0:.1f} s)", flush=True)
        return False


def main() -> int:
    sys.path.insert(0, ".")
    from genprobe.utils.hf import repo_cache_dir, require_cached

    for p in ("vllm", "torch", "transformers", "triton", "flashinfer-python", "xgrammar", "outlines-core",
              "huggingface-hub"):
        try:
            print(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            print(f"{p}: not installed")
    sha = require_cached(REPO, ("config.json", "model.safetensors.index.json"),
                         'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh judge"')
    path = str(repo_cache_dir(REPO) / "snapshots" / sha)
    print(f"{REPO}@{sha[:7]} -> snapshot dir found")

    from vllm import LLM, SamplingParams
    st: dict = {}

    def load():
        st["llm"] = LLM(model=path, tokenizer=path, dtype="bfloat16", max_model_len=8192,
                        gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 2}, seed=0)

    if not step("load model (offline, bf16, max_model_len 8192)", load):
        return 1
    llm = st["llm"]
    greedy = SamplingParams(temperature=0.0, max_tokens=64)

    def one_image():
        out = llm.chat(msg("How many red circles are in this image? Answer with a number only.",
                           shapes(3, "red")), greedy)
        print("   answer (expected 3):", repr(out[0].outputs[0].text))

    def two_images():
        out = llm.chat(msg("Image 1 and image 2 are shown in that order. Which one shows a blue square? "
                           "Answer 1 or 2 only.", shapes(1, "red", "square"), shapes(1, "blue", "square")), greedy)
        print("   answer (expected 2):", repr(out[0].outputs[0].text))
        out = llm.chat(msg("Image 1 and image 2 are shown in that order. Which one shows a blue square? "
                           "Answer 1 or 2 only.", shapes(1, "blue", "square"), shapes(1, "red", "square")), greedy)
        print("   swapped (expected 1):", repr(out[0].outputs[0].text))

    schema = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5},
                                               "issues": {"type": "array", "items": {"type": "string"}}},
              "required": ["score", "issues"], "additionalProperties": False}
    question = ('Prompt: "four green circles". Rate from 1 to 5 how well the image matches the prompt and '
                'list what is wrong. Reply as JSON {"score": <1-5>, "issues": [...]}.')

    def json_constrained():
        params = None
        try:
            from vllm.sampling_params import StructuredOutputsParams
            params = SamplingParams(temperature=0.0, max_tokens=200,
                                    structured_outputs=StructuredOutputsParams(json=schema))
            print("   API: SamplingParams(structured_outputs=StructuredOutputsParams(json=...))")
        except (ImportError, TypeError):
            from vllm.sampling_params import GuidedDecodingParams
            params = SamplingParams(temperature=0.0, max_tokens=200, guided_decoding=GuidedDecodingParams(json=schema))
            print("   API: SamplingParams(guided_decoding=GuidedDecodingParams(json=...))")
        out = llm.chat(msg(question, shapes(3, "red")), params)
        text = out[0].outputs[0].text
        print("   output:", repr(text))
        print("   parsed:", json.loads(text))

    def json_free():
        out = llm.chat(msg(question, shapes(3, "red")), SamplingParams(temperature=0.0, max_tokens=200))
        print("   unconstrained output:", repr(out[0].outputs[0].text))

    def logprobs():
        out = llm.chat(msg("Is there a red circle in this image? Answer Yes or No.", shapes(1, "red")),
                       SamplingParams(temperature=0.0, max_tokens=1, logprobs=5))
        lp = out[0].outputs[0].logprobs[0]
        print("   top-5 first-token logprobs:", {v.decoded_token: round(v.logprob, 3) for v in lp.values()})

    def throughput():
        imgs = [shapes(1 + i % 5, ["red", "blue", "green"][i % 3]) for i in range(64)]
        convs = [msg("Describe this image in one short sentence.", im) for im in imgs]
        t0 = time.perf_counter()
        outs = llm.chat(convs, SamplingParams(temperature=0.0, max_tokens=48))
        dt = time.perf_counter() - t0
        print(f"   64 single-image requests in {dt:.1f} s ({dt / 64:.3f} s/request, 448 px images)")
        print("   first:", repr(outs[0].outputs[0].text))

    results = [step(n, f) for n, f in (("one image", one_image), ("two images in one prompt", two_images),
                                       ("JSON-constrained output", json_constrained),
                                       ("free-form output", json_free), ("first-token logprobs", logprobs),
                                       ("throughput, batch of 64", throughput))]
    print("\nall steps ok" if all(results) else "\nSOME STEPS FAILED (see above)")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
