"""Blind labelling sets (genprobe/label_set.py): balanced choice, files, and safe finalizing."""

import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from genprobe import label_set
from genprobe.backends import FakeBackend
from genprobe.generate import run_generation
from genprobe.samples import Prompt, Sample
from tests.helpers import make_cfg, quiet


def fake_samples():
    out = []
    for model in ("a", "b", "c", "d"):
        for i in range(6):
            for kind, spec in (("count", {"objects": [{"name": "cup", "count": 2}]}),
                               ("spatial", {"relation": {"type": "above", "a": "cup", "b": "book"}}),
                               ("text", {"text": "SALE"})):
                out.append(Sample(f"{model}__{kind}{i}__s0", model, f"{kind}{i}", "p", 0, "x.png", spec=spec))
    return out


class TestChoose(unittest.TestCase):
    def test_balanced_and_deterministic(self):
        picked = label_set.choose(fake_samples(), {"count": 8, "spatial": 4, "text": 6}, seed=0)
        self.assertEqual(Counter(k for k, _ in picked), {"count": 8, "spatial": 4, "text": 6})
        self.assertEqual(Counter(s.model for k, s in picked if k == "count"), {"a": 2, "b": 2, "c": 2, "d": 2})
        self.assertEqual(picked, label_set.choose(fake_samples(), {"count": 8, "spatial": 4, "text": 6}, seed=0))
        self.assertEqual(len({s.sample_id for _, s in picked}), 18)

    def test_kind_of(self):
        self.assertEqual(label_set.kind_of(Sample("x", "m", "p", "t", 0, "", spec={"text": "A"})), "text")
        self.assertIsNone(label_set.kind_of(Sample("x", "m", "p", "t", 0, "", spec=None)))


class TestMakeAndFinalize(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prompts = [Prompt("tc0001", "a photo of two cups", source="template_count", tags=["counting"],
                              spec={"objects": [{"name": "cup", "count": 2}]}, split="dev"),
                       Prompt("tt0001", 'a sign that says "SALE"', source="template_text", tags=["text"],
                              spec={"text": "SALE"}, split="dev"),
                       Prompt("tt0002", 'a sign that says "EXIT"', source="template_text", tags=["text"],
                              spec={"text": "EXIT"}, split="test")]
            ppath = d / "p.jsonl"
            ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
            cfg = make_cfg(name="toy", seeds=[0, 1], batch_size=2)
            run_generation(cfg=cfg, prompts=prompts, out=d, backend=FakeBackend(cfg), backend_kind="fake", log=quiet)
            out = d / "review"
            files = label_set.make(str(ppath), str(d / "manifests" / "gen_*.jsonl"), str(out), "dev",
                                   {"count": 2, "spatial": 0, "text": 5}, seed=0)
            self.assertIn(out / "sheet_01.jpg", files)
            with open(out / "labels.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)                      # 2 count + only 2 dev text images exist
            self.assertNotIn("toy", "".join(r["prompt"] for r in rows))
            with self.assertRaises(SystemExit):                 # never replaces a label file
                label_set.make(str(ppath), str(d / "manifests" / "gen_*.jsonl"), str(out), "dev",
                               {"count": 1}, seed=0)
            with self.assertRaises(SystemExit):                 # unfilled labels are refused
                label_set.finalize(str(out), str(d / "labels.csv"))
            for r in rows:
                r["correct"] = "y" if r["kind"] == "count" else "？"
                r["true_count"] = "2" if r["kind"] == "count" else ""
            with open(out / "labels.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=label_set.LABEL_FIELDS)
                w.writeheader()
                w.writerows(rows)
            self.assertEqual(label_set.finalize(str(out), str(d / "labels.csv")), 4)
            with open(d / "labels.csv", newline="") as f:
                final = list(csv.DictReader(f))
            self.assertEqual({r["correct"] for r in final}, {"y", "?"})
            self.assertTrue(all(r["sample_id"].startswith("toy__") for r in final))
            with self.assertRaises(SystemExit):                 # second finalize would overwrite
                label_set.finalize(str(out), str(d / "labels.csv"))


if __name__ == "__main__":
    unittest.main()
