"""Detector check for counting and spatial template prompts (guide §7.5, D-018), with
Grounding DINO (tiny), one text query per object name.

Counting (spec ``{"objects": [{"name", "count"}]}``): boxes with score >= BOX_THRESHOLD,
duplicates merged (NMS at IoU NMS_IOU), and "group" boxes dropped (a box that contains two or
more other kept boxes is the detector outlining the whole group); value = 1 if the number of
boxes left equals the requested count, else 0.

Spatial (spec ``{"relation": {"type", "a", "b"}}``): the highest-scoring box for a and the
highest-scoring box for b that is not the same box (IoU < SAME_BOX_IOU); value = 1 if the box
centres satisfy the relation by more than MARGIN of the image size (left/right on x,
above/below on y, from the viewer, like the human rule in D-015), else 0. A missing object
gives 0.

The thresholds were fixed before any v1 result existed (D-018); they may only be re-tuned on
the dev split. The decisions are plain Python below, so they are unit-tested without the model.
"""

from __future__ import annotations

import importlib.metadata as md
from typing import Any

from ..utils.hf import cached_revision, require_cached
from .base import Metric, register

HF_REPO = "IDEA-Research/grounding-dino-tiny"
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
KEEP_THRESHOLD = 0.2       # boxes stored in extra for diagnosis; decisions use BOX_THRESHOLD
NMS_IOU = 0.5
GROUP_CONTAIN = 0.9        # a box "contains" another when it covers >= 90% of the other's area
SAME_BOX_IOU = 0.8
MARGIN = 0.05
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh grounding_dino_tiny"'

Box = tuple[float, float, float, float]   # x0, y0, x1, y1 in pixels
Det = tuple[Box, float]


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Det], iou_threshold: float = NMS_IOU) -> list[Det]:
    kept: list[Det] = []
    for box, score in sorted(dets, key=lambda d: -d[1]):
        if all(iou(box, k[0]) < iou_threshold for k in kept):
            kept.append((box, score))
    return kept


def drop_group_boxes(dets: list[Det]) -> list[Det]:
    """Remove boxes that contain at least two of the other boxes."""
    def contains(outer: Box, inner: Box) -> bool:
        return area(inner) > 0 and intersection(outer, inner) / area(inner) >= GROUP_CONTAIN
    return [d for i, d in enumerate(dets)
            if sum(contains(d[0], o[0]) for j, o in enumerate(dets) if j != i) < 2]


def decide_count(dets: list[Det], target: int) -> dict[str, Any]:
    kept = drop_group_boxes(nms([d for d in dets if d[1] >= BOX_THRESHOLD]))
    return {"value": float(len(kept) == target), "count_pred": len(kept), "count_target": target,
            "scores": [round(s, 3) for _, s in kept]}


def centre(box: Box, w: int, h: int) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2 / w, (box[1] + box[3]) / 2 / h)


def relation_holds(rel: str, a: tuple[float, float], b: tuple[float, float], margin: float = MARGIN) -> bool:
    (ax, ay), (bx, by) = a, b
    if rel == "left_of":
        return ax + margin < bx
    if rel == "right_of":
        return ax > bx + margin
    if rel == "above":
        return ay + margin < by
    if rel == "below":
        return ay > by + margin
    raise ValueError(f"unknown relation {rel!r}")


def decide_relation(dets_a: list[Det], dets_b: list[Det], rel: str, w: int, h: int) -> dict[str, Any]:
    a_sorted = sorted((d for d in dets_a if d[1] >= BOX_THRESHOLD), key=lambda d: -d[1])
    b_sorted = sorted((d for d in dets_b if d[1] >= BOX_THRESHOLD), key=lambda d: -d[1])
    missing = [n for n, d in (("a", a_sorted), ("b", b_sorted)) if not d]
    if missing:
        return {"value": 0.0, "missing": missing}
    a = a_sorted[0]
    b = next((d for d in b_sorted if iou(d[0], a[0]) < SAME_BOX_IOU), None)
    if b is None:
        return {"value": 0.0, "missing": ["b"], "note": "b only found on a's box"}
    ca, cb = centre(a[0], w, h), centre(b[0], w, h)
    return {"value": float(relation_holds(rel, ca, cb)), "missing": [],
            "centre_a": [round(v, 3) for v in ca], "centre_b": [round(v, 3) for v in cb],
            "score_a": round(a[1], 3), "score_b": round(b[1], 3)}


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


@register
class DetectScore(Metric):
    name = "detect"
    needs = frozenset({"image", "spec"})
    env = "eval"
    higher_is_better = True
    default_batch_size = 16
    decision_threshold = 1.0

    def __init__(self) -> None:
        rev = (cached_revision(HF_REPO) or "nocache")[:7]
        self.version = (f"transformers=={_pkg_version('transformers')}/grounding-dino-tiny@{rev}/"
                        f"box{BOX_THRESHOLD}-text{TEXT_THRESHOLD}-nms{NMS_IOU}-group{GROUP_CONTAIN}"
                        f"-margin{MARGIN}")
        self.info: dict[str, Any] = {}

    def applies_to(self, sample) -> bool:
        spec = sample.spec or {}
        return bool(spec.get("objects")) or bool(spec.get("relation"))

    def setup(self, device: str) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; run on a GPU node or pass --device cpu")
        rev = require_cached(HF_REPO, REQUIRED_FILES, DOWNLOAD_HINT)
        self.processor = AutoProcessor.from_pretrained(HF_REPO)
        self.model, load = AutoModelForZeroShotObjectDetection.from_pretrained(HF_REPO, output_loading_info=True)
        if load.get("missing_keys") or load.get("mismatched_keys"):
            raise RuntimeError(f"Grounding DINO loaded partially: missing {load.get('missing_keys')[:5]}, "
                               f"mismatched {load.get('mismatched_keys')[:5]}")
        self.model = self.model.to(device).eval()
        self.device = device
        import inspect
        params = inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self.threshold_kw = "threshold" if "threshold" in params else "box_threshold"
        self.info = {"hf_repo": HF_REPO, "revision": rev, "device": device,
                     "transformers": _pkg_version("transformers"), "box_threshold": BOX_THRESHOLD,
                     "text_threshold": TEXT_THRESHOLD, "nms_iou": NMS_IOU, "group_contain": GROUP_CONTAIN,
                     "same_box_iou": SAME_BOX_IOU, "margin": MARGIN, "unexpected_keys": load.get("unexpected_keys")}

    def detect(self, image, name: str) -> list[Det]:
        """All boxes for one object name (query "<name>."), score >= KEEP_THRESHOLD."""
        import torch

        inputs = self.processor(images=image, text=name.lower() + ".", return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], text_threshold=TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)], **{self.threshold_kw: KEEP_THRESHOLD})[0]
        return [(tuple(float(v) for v in b), float(s))
                for b, s in zip(res["boxes"].tolist(), res["scores"].tolist())]

    def score_batch(self, samples):
        from PIL import Image

        out = []
        for s in samples:
            with Image.open(s.path) as im:
                image = im.convert("RGB")
            spec = s.spec
            if spec.get("objects"):
                obj = spec["objects"][0]
                r = decide_count(self.detect(image, obj["name"]), int(obj["count"]))
                r["check"] = "count"
            else:
                rel = spec["relation"]
                r = decide_relation(self.detect(image, rel["a"]), self.detect(image, rel["b"]),
                                    rel["type"], image.width, image.height)
                r["check"] = rel["type"]
            out.append({"sample_id": s.sample_id, "value": r.pop("value"), "extra": r})
        return out
