"""Blind DPO check (genprobe/dpo_check.py, D-037): items, blindness and the analysis of labels."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from genprobe import dpo_check as dc


def fake_images(img: str, base="dpo_base", new="dpo_s500", per=30):
    rows = []
    for st in dc.STRATA:
        for j in range(per):
            pid = f"{st[:2]}{j:03d}"
            for arm in (base, new):
                rows.append({"sample_id": f"{arm}__{pid}__s0", "prompt_id": pid, "prompt": f"prompt {pid}", "path": img,
                             "group": arm, "seed": 0, "category": st})
    return rows


class TestDPOCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.img = str(self.d / "x.png")
        Image.new("RGB", (32, 32), (10, 200, 10)).save(self.img)
        self.images = fake_images(self.img)

    def tearDown(self):
        self.tmp.cleanup()

    def test_items(self):
        public, key = dc.build(self.images, "dpo_base", "dpo_s500")
        self.assertEqual((public, key), dc.build(self.images, "dpo_base", "dpo_s500"))
        kinds = [(k["q"], k["kind"]) for k in key]
        self.assertEqual(kinds.count(("text", "main")), 60)
        self.assertEqual(kinds.count(("quality", "main")), 60)
        self.assertEqual(kinds.count(("text", "gold")), dc.N_GOLD)
        self.assertEqual(kinds.count(("text", "repeat")) + kinds.count(("quality", "repeat")), 2 * dc.N_REPEAT)
        # part 1 comes first, part 2 after
        qs = [k["q"] for k in key]
        self.assertEqual(qs, sorted(qs, key=lambda q: q != "text"))
        by = {k["item_id"]: k for k in key}
        for k in key:
            if k["kind"] == "repeat":
                o = by[k["repeat_of"]]
                self.assertEqual((o["q"], {o["left"], o["right"]}), (k["q"], {k["left"], k["right"]}))
        for k in key:
            if k["kind"] == "main":
                self.assertEqual({k["left"].split("__")[0], k["right"].split("__")[0]}, {"dpo_base", "dpo_s500"})
        z = dc.write_package(public, key, "dpo_base", "dpo_s500", out=self.d / "out")
        js = zipfile.ZipFile(z).read("dpo_check/items.js").decode()
        for word in ("dpo_base", "dpo_s500", "sample_id", "__s0"):
            self.assertNotIn(word, js)

    def test_analyze(self):
        public, key = dc.build(self.images, "dpo_base", "dpo_s500")
        labels = {}
        for k in key:
            if k["kind"] == "gold":
                labels[k["item_id"]] = "A" if k["left"] == k["real"] else "B"
            elif k["q"] == "text":                     # always picks the DPO image
                labels[k["item_id"]] = "A" if k["left"].startswith("dpo_s500__") else "B"
            else:                                      # always picks the base image
                labels[k["item_id"]] = "A" if k["left"].startswith("dpo_base__") else "B"
        res = dc.analyze(labels, key, "dpo_s500", n_boot=100)
        self.assertTrue(res["gold"]["passed"])
        self.assertEqual(res["text"]["prefers_new"]["value"], 1.0)
        self.assertEqual(res["quality"]["prefers_new"]["value"], 0.0)
        self.assertEqual(res["text"]["repeat"]["same_answer"], 1.0)
        self.assertIn("looks better", dc.render(res))
        json.dumps(res)


if __name__ == "__main__":
    unittest.main()
