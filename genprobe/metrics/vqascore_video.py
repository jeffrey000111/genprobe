"""VQAScore on a whole video (guide §11.3, D-035): the probability that Qwen2.5-VL-7B answers "Yes"
to the same question as the image VQAScore, with the video given as a video.

A port of t2v_metrics' ``Qwen2VLModel.forward`` for video paths (the same model and weights as
genprobe/metrics/vqascore.py): ``{"type": "video", "video": path, "max_pixels": 360 * 420,
"fps": 8.0}`` (t2v_metrics' settings for qwen2.5-vl-7b), ``process_vision_info(...,
return_video_kwargs=True)``, the processor called with ``do_resize=False`` and the returned
``fps``, one generated token, ``softmax`` at "Yes". The question template stays the image one
("Does this figure show ..."), as t2v_metrics uses it for video too. Frames are read with decord
(pinned through qwen-vl-utils' FORCE_QWENVL_VIDEO_READER); fps, pixel cap, reader and the
qwen-vl-utils version are part of ``metric_version``.
"""

from __future__ import annotations

import os
from typing import Any

from .base import register
from .vqascore import QUESTION_TEMPLATE, VQAScore, _pkg_version

FPS = 8.0
MAX_PIXELS = 360 * 420
READER = "decord"
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@register
class VQAScoreVideo(VQAScore):
    name = "vqascore_video"
    needs = frozenset({"video", "prompt"})
    default_batch_size = 1

    def __init__(self) -> None:
        super().__init__()
        self.version = (self.version.replace("vqascore-", "vqascore_video-", 1) +
                        f"/fps{FPS:g}/maxpix{MAX_PIXELS}/reader-{READER}/qwen-vl-utils=={_pkg_version('qwen-vl-utils')}")

    def applies_to(self, sample) -> bool:
        return str(sample.path).lower().endswith(VIDEO_EXT)

    def setup(self, device: str) -> None:
        os.environ["FORCE_QWENVL_VIDEO_READER"] = READER
        super().setup(device)
        self.info.update({"input": "video", "fps": FPS, "max_pixels": MAX_PIXELS, "video_reader": READER,
                          "qwen_vl_utils": _pkg_version("qwen-vl-utils")})

    def score_video(self, path: str, prompt: str) -> tuple[float, dict[str, Any]]:
        import torch

        question = QUESTION_TEMPLATE.format(prompt)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": str(path), "max_pixels": MAX_PIXELS, "fps": FPS},
            {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self.process_vision_info(messages, return_video_kwargs=True)
        if isinstance(video_kwargs.get("fps"), list):
            video_kwargs["fps"] = video_kwargs["fps"][0] if video_kwargs["fps"] else None
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                return_tensors="pt", do_resize=False,
                                **(video_kwargs if video_inputs is not None else {})).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=1, temperature=1.0, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        probs = torch.softmax(out.scores[0][0].float(), dim=-1)
        top = int(probs.argmax())
        grid = inputs.get("video_grid_thw")
        extra = {"question": question, "p_no": float(probs[self.no_id]),
                 "top_token": self.processor.tokenizer.decode([top]),
                 "video_grid_thw": grid[0].tolist() if grid is not None else None,
                 "n_video_tokens": int(grid.prod().item() // 4) if grid is not None else None,
                 "fps_used": video_kwargs.get("fps")}
        return float(probs[self.yes_id]), extra

    def score_batch(self, samples):
        import torch

        out = []
        with torch.inference_mode():
            for s in samples:
                value, extra = self.score_video(s.path, s.prompt)
                out.append({"sample_id": s.sample_id, "value": value, "extra": extra})
        return out
