"""OCR check for text-rendering prompts (guide §7.6, D-017): does the image show the requested text?

EasyOCR (CRAFT detector + english_g2 recognizer) reads all text in the image; the boxes are
put in reading order (grouped into lines top to bottom, each line left to right) and the
requested text is matched against the best-matching stretch of what was read
(genprobe/metrics/textmatch.py).

``value`` = 1 - edit distance / length of the requested text, in [0, 1] (1 = spelled exactly,
ignoring case, punctuation and spaces); ``extra`` has ``exact``, ``word_recall``, the
normalized strings and every OCR box with its confidence. An image counts as having the text
right when ``value`` >= 0.85 (``decision_threshold``; chosen on dev labels, D-017 revision).

Runs in the ``aux`` env so EasyOCR's dependencies (opencv, scikit-image, ...) stay out of the
eval env. Weights come from ``download_models.sh easyocr`` (on coe-hpc1) in
$EASYOCR_MODULE_PATH/model; EasyOCR itself checks their md5 and never downloads in a job.
Quantization is off, so CPU and GPU runs use the same fp32 recognizer.
"""

from __future__ import annotations

import importlib.metadata as md
import os
from pathlib import Path
from typing import Any

from .base import Metric, register
from .textmatch import text_match

DETECTOR, RECOGNIZER = "craft", "english_g2"
WEIGHT_FILES = ("craft_mlt_25k.pth", "english_g2.pth")
MATCH_VERSION = "semiglobal-nospace-v1"
DOWNLOAD_HINT = 'ssh sjsu "cd ~/genprobe && bash scripts/download_models.sh easyocr"'


def model_dir() -> Path:
    base = os.environ.get("EASYOCR_MODULE_PATH") or os.path.expanduser("~/.EasyOCR")
    return Path(base) / "model"


def reading_order(results: list) -> list:
    """EasyOCR results ``[(box, text, conf)]`` grouped into lines (top to bottom), each line
    left to right. A box joins the current line when its vertical centre is within half a
    text height of the line's centre."""
    items = []
    for r in results:
        ys = [float(p[1]) for p in r[0]]
        xs = [float(p[0]) for p in r[0]]
        items.append(((min(ys) + max(ys)) / 2, max(max(ys) - min(ys), 1.0), min(xs), r))
    items.sort(key=lambda it: it[0])
    lines: list[list] = []
    for it in items:
        if lines:
            line = lines[-1]
            yc = sum(i[0] for i in line) / len(line)
            h = max(max(i[1] for i in line), it[1])
            if abs(it[0] - yc) <= 0.5 * h:
                line.append(it)
                continue
        lines.append([it])
    return [it[3] for line in lines for it in sorted(line, key=lambda i: i[2])]


def _pkg_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "none"


@register
class OCRScore(Metric):
    name = "ocr"
    needs = frozenset({"image", "spec"})
    env = "aux"
    higher_is_better = True
    default_batch_size = 16
    decision_threshold = 0.85   # D-017 revision: 1.0 (exact) until the dev check of 2026-10-19

    def __init__(self) -> None:
        self.version = f"easyocr=={_pkg_version('easyocr')}/{DETECTOR}+{RECOGNIZER}/{MATCH_VERSION}"
        self.info: dict[str, Any] = {}

    def applies_to(self, sample) -> bool:
        return bool(sample.spec) and isinstance(sample.spec.get("text"), str)

    def setup(self, device: str) -> None:
        import easyocr
        import torch

        use_gpu = device.startswith("cuda")
        if use_gpu and not torch.cuda.is_available():
            raise RuntimeError("no CUDA device visible; run on a GPU node or pass --device cpu")
        mdir = model_dir()
        missing = [f for f in WEIGHT_FILES if not (mdir / f).is_file()]
        if missing:
            raise FileNotFoundError(f"EasyOCR weights missing in {mdir}: {missing}. Download on coe-hpc1: "
                                    f"{DOWNLOAD_HINT}")
        self.reader = easyocr.Reader(["en"], gpu=use_gpu, model_storage_directory=str(mdir),
                                     detect_network=DETECTOR, recog_network=RECOGNIZER,
                                     download_enabled=False, quantize=False, verbose=False)
        self.info = {"easyocr": _pkg_version("easyocr"), "torch": _pkg_version("torch"),
                     "model_dir": str(mdir), "device": "cuda" if use_gpu else "cpu",
                     "detector": DETECTOR, "recognizer": RECOGNIZER, "match": MATCH_VERSION,
                     "quantize": False}

    def score_batch(self, samples):
        out = []
        for s in samples:
            results = reading_order(self.reader.readtext(str(s.path)))
            ocr_text = " ".join(str(r[1]) for r in results)
            m = text_match(s.spec["text"], ocr_text)
            boxes = [{"text": str(r[1]), "conf": round(float(r[2]), 3)} for r in results]
            out.append({"sample_id": s.sample_id, "value": m.pop("value"), "extra": {**m, "boxes": boxes}})
        return out
