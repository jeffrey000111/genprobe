"""Text matching for the OCR check (guide §7.6, D-017). Pure Python, no dependencies.

The requested text is compared with everything OCR read in the image. Other text in the
picture (a second sign, a logo) must not count against the image, so the requested text is
aligned against its best-matching stretch of the OCR output (semi-global edit distance):
skipping OCR text before and after is free, every edit inside costs 1.

Case, punctuation and spaces are ignored for the score: the human rule (D-015) ignores case
and punctuation, and OCR often merges or splits words ("THANKYOU"), which says nothing about
the spelling. Word-level recall is kept as a diagnostic.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def normalize(text: str) -> str:
    """Upper case, letters and digits only, words separated by single spaces."""
    return " ".join(_NON_ALNUM.sub(" ", text.upper()).split())


def semiglobal_distance(target: str, text: str) -> int:
    """Smallest edit distance between ``target`` and any substring of ``text``."""
    if not target:
        return 0
    prev = [0] * (len(text) + 1)            # an empty target matches anywhere for free
    for i, tc in enumerate(target, 1):
        cur = [i] + [0] * len(text)
        for j, xc in enumerate(text, 1):
            cur[j] = min(prev[j] + 1,                     # target char missing from the image
                         cur[j - 1] + 1,                  # extra char inside the match
                         prev[j - 1] + (tc != xc))        # match / misspelling
        prev = cur
    return min(prev)


def text_match(target: str, ocr_text: str) -> dict:
    """``value`` = 1 - distance / len(target) on space-free normalized strings, in [0, 1]."""
    t, x = normalize(target), normalize(ocr_text)
    t_chars, x_chars = t.replace(" ", ""), x.replace(" ", "")
    d = semiglobal_distance(t_chars, x_chars)
    words = set(x.split())
    target_words = t.split()
    return {
        "value": max(0.0, 1.0 - d / max(len(t_chars), 1)),
        "distance": d,
        "exact": d == 0,
        "word_recall": sum(w in words for w in target_words) / max(len(target_words), 1),
        "target_norm": t,
        "ocr_norm": x,
    }
