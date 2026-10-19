"""Download EasyOCR model weights into a folder, checked against EasyOCR's own md5 sums.

Called by ``scripts/download_models.sh easyocr`` on coe-hpc1 (the only machine with internet).
The URLs and md5 sums are read from the installed package's ``easyocr/config.py`` (the aux
env's copy), so they always match the EasyOCR version the jobs run. That file is loaded on its
own, without importing easyocr (whose torch/opencv wheels are built for HPC3, not coe-hpc1).
Standard library only.

    python scripts/easyocr_download.py <path to easyocr/config.py> <model dir> craft english_g2
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


def load_config(path: Path):
    spec = importlib.util.spec_from_file_location("easyocr_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_model(cfg, name: str) -> dict:
    if name in cfg.detection_models:
        return cfg.detection_models[name]
    for gen in cfg.recognition_models.values():
        if name in gen:
            return gen[name]
    raise KeyError(f"{name!r} is not in this EasyOCR's config (detection: {sorted(cfg.detection_models)})")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(model: dict, outdir: Path) -> str:
    target = outdir / model["filename"]
    if target.is_file() and md5(target) == model["md5sum"]:
        return f"{target.name}: already present, md5 ok"
    with tempfile.TemporaryDirectory(dir=outdir) as tmp:
        zpath = Path(tmp) / "model.zip"
        with urllib.request.urlopen(model["url"], timeout=120) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zpath) as z:
            member = next((n for n in z.namelist() if Path(n).name == model["filename"]), None)
            if member is None:
                raise RuntimeError(f"{model['url']} has no {model['filename']} (contents: {z.namelist()})")
            z.extract(member, tmp)
        got = Path(tmp) / member
        digest = md5(got)
        if digest != model["md5sum"]:
            raise RuntimeError(f"{model['filename']}: md5 {digest} != expected {model['md5sum']}")
        got.replace(target)
    return f"{target.name}: downloaded, md5 ok ({target.stat().st_size / 1e6:.1f} MB)"


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    cfg = load_config(Path(argv[0]))
    outdir = Path(argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in argv[2:]:
        try:
            print(f"{name}: {fetch(find_model(cfg, name), outdir)}")
        except Exception as e:  # noqa: BLE001
            print(f"{name}: FAILED {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(name)
    if failed:
        return 1
    print("check: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
