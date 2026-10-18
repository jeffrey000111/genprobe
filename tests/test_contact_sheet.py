"""Contact sheets: one JPEG per category with every model's image, and a label template that is
never overwritten (it may hold a person's labels)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from genprobe.backends import FakeBackend
from genprobe.contact_sheet import LABEL_COLUMNS, make_sheets
from genprobe.generate import run_generation
from tests.helpers import make_cfg, make_prompts, quiet


class TestContactSheet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        prompts = make_prompts(3)
        prompts[2].tags = ["spatial"]
        self.ppath = self.out / "prompts.jsonl"
        self.ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
        for name in ("toy", "toy2"):
            cfg = make_cfg(name=name, seeds=[0], batch_size=2)
            run_generation(cfg=cfg, prompts=prompts, out=self.out, backend=FakeBackend(cfg),
                           backend_kind="fake", log=quiet)
        d = self.out / "scores" / "clipscore"
        d.mkdir(parents=True)
        d.joinpath("t.jsonl").write_text(json.dumps(
            {"sample_id": "toy__p000__s0", "metric": "clipscore", "metric_version": "v", "value": 0.31,
             "extra": {}, "seconds": 0.1, "status": "ok", "created_at": "2026-10-18"}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def sheets(self):
        return make_sheets(prompts_path=str(self.ppath), manifests=str(self.out / "manifests" / "gen_*.jsonl"),
                           data=str(self.out), metrics=["clipscore", "vqascore"], out=str(self.out / "review"))

    def test_one_sheet_per_category_and_labels(self):
        written = self.sheets()
        names = sorted(p.name for p in written)
        self.assertEqual(names, ["prompts_counting.jpg", "prompts_labels.csv", "prompts_spatial.jpg"])
        with Image.open(self.out / "review" / "prompts_counting.jpg") as im:
            self.assertGreater(im.size[0], 2 * 256)          # two model columns
            self.assertGreater(im.size[1], 2 * 256)          # two counting prompts
        with open(self.out / "review" / "prompts_labels.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(list(rows[0]), LABEL_COLUMNS)
        self.assertEqual(len(rows), 6)

    def test_labels_never_overwritten(self):
        self.sheets()
        labels = self.out / "review" / "prompts_labels.csv"
        labels.write_text("my labels\n")
        written = self.sheets()
        self.assertEqual(labels.read_text(), "my labels\n")
        self.assertNotIn(labels, written)


if __name__ == "__main__":
    unittest.main()
