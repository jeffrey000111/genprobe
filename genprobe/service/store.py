"""Uploaded images: validation and a content-addressed store (D-040).

An image is stored once under the sha256 of its bytes, so the same upload twice takes no extra
space and its hash is the cache key. Files are written to a temporary name and renamed into
place, so a worker never reads a half-written image. The API and the workers share the
directory (a docker volume); on several hosts it would be an object store behind the same two
methods.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})


class ImageRejected(ValueError):
    """The upload is not an image the service will score. ``status`` is the HTTP status."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ImageInfo:
    sha256: str
    format: str
    width: int
    height: int


def inspect_image(data: bytes, max_pixels: int) -> ImageInfo:
    """Check that ``data`` is a PNG, JPEG or WebP no larger than ``max_pixels`` and decodes.

    The size is read from the header before anything is decoded, so a small file that expands
    to a huge bitmap (a decompression bomb) is refused without allocating it.
    """
    from PIL import Image, UnidentifiedImageError

    if not data:
        raise ImageRejected(400, "empty image")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
            if fmt not in ALLOWED_FORMATS:
                raise ImageRejected(415, f"unsupported image format {fmt}; use PNG, JPEG or WebP")
            if w * h > max_pixels:
                raise ImageRejected(413, f"image is {w}x{h}; at most {max_pixels} pixels")
            im.verify()                      # structure and checksums
        with Image.open(io.BytesIO(data)) as im:
            im.load()                        # and the pixel data itself
    except ImageRejected:
        raise
    except Image.DecompressionBombError as e:  # PIL's own guard, in case max_pixels is set high
        raise ImageRejected(413, str(e)) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise ImageRejected(400, f"not a readable image: {e}") from None
    return ImageInfo(hashlib.sha256(data).hexdigest(), fmt, w, h)


class ImageStore:
    def __init__(self, root: str | Path):
        self.root = Path(root) / "images"

    def path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError(f"not a sha256 hex digest: {sha256!r}")
        return self.root / sha256[:2] / sha256

    def put(self, data: bytes, sha256: str | None = None) -> str:
        sha256 = sha256 or hashlib.sha256(data).hexdigest()
        p = self.path(sha256)
        if p.exists():
            return sha256
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)               # atomic on one file system
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
        return sha256

    def exists(self, sha256: str) -> bool:
        return self.path(sha256).exists()
