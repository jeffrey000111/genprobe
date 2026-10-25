"""Blind check of the multi-shot videos (genprobe/story_check.py, D-038): items, blindness and the
analysis of labels. No video is encoded (join_shots is replaced)."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from genprobe import story_check as sc


def fake(n_briefs=24):
    seqs, text, score = {}, {}, {}
    for i in range(n_briefs):
        b = f"st{i:02d}"
        text[b] = f"brief {b}"
        for arm in "ABCD":
            seqs[(b, arm)] = [f"{arm}model__{b}s{k}__s0" for k in range(1, 4)]
            score[(b, arm)] = 5 if (arm == "B" and i % 3 == 0) else 3
    return seqs, text, score


class TestStoryCheck(unittest.TestCase):
    def test_items(self):
        seqs, text, score = fake()
        public, key = sc.build(seqs, text, score)
        self.assertEqual((public, key), sc.build(seqs, text, score))
        kinds = [(k["q"], k["kind"]) for k in key]
        self.assertEqual(kinds.count(("story", "main")), 24)
        self.assertEqual(kinds.count(("consist", "main")), 24)
        self.assertEqual(kinds.count(("story", "gold")), sc.N_GOLD)
        self.assertEqual(kinds.count(("story", "repeat")), sc.N_REPEAT)
        self.assertEqual(kinds.count(("consist", "repeat")), sc.N_REPEAT)
        qs = [k["q"] for k in key]
        self.assertEqual(qs, sorted(qs, key=lambda q: q != "story"))              # part 1 first
        for k in key:
            if k["kind"] == "main":
                self.assertEqual({k["left"][1], k["right"][1]}, {"A", "C"})
                self.assertEqual(k["left"][0], k["right"][0])
            if k["kind"] == "gold":
                self.assertEqual(score[tuple(k["real"])], 5)                        # the real one passed the judge
                other = k["right"] if k["left"] == k["real"] else k["left"]
                self.assertNotEqual(other[0], k["real"][0])
        self.assertEqual(len({k["brief_id"] for k in key if k["kind"] == "gold"}), sc.N_GOLD)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(sc, "join_shots", lambda paths, out: Path(out).write_bytes(b"x" + "".join(paths).encode())):
            man = {s: {"path": f"/v/{s}.mp4"} for v in seqs.values() for s in v}
            z = sc.write_package(public, key, man, out=Path(d))
            names = zipfile.ZipFile(z).namelist()
            self.assertIn("story_check/index.html", names)
            self.assertEqual(sum(n.endswith(".mp4") for n in names), 2 * len(public))
            js = zipfile.ZipFile(z).read("story_check/items.js").decode()
            for word in ("Amodel", "Cmodel", "__s0", '"C"', "brief_id"):
                self.assertNotIn(word, js)

    def test_analyze(self):
        seqs, text, score = fake()
        public, key = sc.build(seqs, text, score)
        labels = {}
        for k in key:
            if k["kind"] == "gold":
                labels[k["item_id"]] = "A" if k["left"] == k["real"] else "B"
            elif k["q"] == "story":                     # always the agent
                labels[k["item_id"]] = "A" if k["left"][1] == "C" else "B"
            else:                                       # consistency: always the same
                labels[k["item_id"]] = "same"
        judge = {f"st{i:02d}": {"A": 3, "C": 4 if i % 2 else 2} for i in range(24)}
        res = sc.analyze(labels, key, judge, n_boot=100)
        self.assertTrue(res["gold"]["passed"])
        self.assertEqual(res["story"]["prefers_agent"]["value"], 1.0)
        self.assertEqual(res["story"]["repeat"]["same_answer"], 1.0)
        self.assertEqual(res["story"]["agrees_with_judge"]["share"], 0.5)
        self.assertEqual(res["consist"]["answers"], {"same": 24})
        self.assertEqual(res["consist"]["prefers_agent"]["n"], 0)
        self.assertIn("tells the brief", sc.render(res))
        json.dumps(res)


if __name__ == "__main__":
    unittest.main()
