"""Own human evaluation (genprobe/human_eval.py): what gets shown, that nothing in the package
reveals a model, and the analysis of a label file."""

import csv
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from genprobe import human_eval as he

MODELS = ["sd15", "sdxl", "sd35m", "flux_schnell"]


def fake_rows(n_per_cat: int, split: str, flag: set[str] = frozenset()):
    rows = []
    cats = ["counting", "spatial", "text", "basic", "advanced"]
    for c in cats:
        for j in range(n_per_cat):
            pid = f"{c[:2]}{split[0]}{j:03d}"
            for s in range(4):
                for k, m in enumerate(MODELS):
                    sid = f"{m}__{pid}__s{s}"
                    rows.append({"sample_id": sid, "model": m, "prompt_id": pid, "seed": str(s), "category": c,
                                 "prompt": f"prompt {pid}", "status": "ok", "vqascore": str(k / 10),
                                 "clipscore": "0.25", "pickscore": str(20 - k), "hpsv2": "", "detect": "",
                                 "ocr": "", "nsfw_sd": "1.0" if sid in flag else "0.0", "nsfw_vit": "0.0"})
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        flagged = {"sd15__cot000__s0", "sdxl__spt001__s2"}
        write_csv(self.d / "test.csv", fake_rows(30, "test", flagged))
        write_csv(self.d / "dev.csv", fake_rows(6, "dev"))
        self.test_rows, self.dev_rows = he.load_rows(self.d / "test.csv"), he.load_rows(self.d / "dev.csv")
        self.flagged = flagged

    def tearDown(self):
        self.tmp.cleanup()

    def test_items(self):
        public, key = he.build(self.test_rows, self.dev_rows)
        self.assertEqual((public, key), he.build(self.test_rows, self.dev_rows))       # deterministic
        kinds = [k["kind"] for k in key]
        self.assertEqual((kinds.count("practice"), kinds.count("main"), kinds.count("gold"), kinds.count("repeat")),
                         (20, 200, 20, 40))
        self.assertEqual([i["item_id"] for i in public[:2]], ["p001", "p002"])
        self.assertEqual(public[-1]["item_id"], "q260")
        main = [k for k in key if k["kind"] == "main"]
        self.assertEqual({s: sum(k["stratum"] == s for k in main) for s in he.STRATA},
                         {s: 50 for s in he.STRATA})
        shown = {k["left"] for k in key} | {k["right"] for k in key}
        self.assertFalse(shown & self.flagged)                                           # safety-flagged never shown
        pos = {k["item_id"]: n for n, k in enumerate(key)}
        by = {k["item_id"]: k for k in key}
        for k in key:
            if k["kind"] == "repeat":
                o = by[k["repeat_of"]]
                self.assertEqual(o["kind"], "main")
                self.assertEqual({o["left"], o["right"]}, {k["left"], k["right"]})
                self.assertGreaterEqual(pos[k["item_id"]] - pos[o["item_id"]], 50)
            if k["kind"] == "gold":
                decoy = k["left"] if k["right"] == k["real"] else k["right"]
                self.assertNotEqual(decoy.split("__")[1], k["prompt_id"])
                self.assertEqual(decoy.split("__")[0], k["real"].split("__")[0])         # same model
            if k["kind"] in ("main", "practice"):
                self.assertEqual(k["left"].split("__")[1:], k["right"].split("__")[1:])  # same prompt and seed
                self.assertNotEqual(k["left"].split("__")[0], k["right"].split("__")[0])
        self.assertTrue(all(k["part"] == "practice" for k in key[:20]))
        lefts = sum(k["left"] == min(k["left"], k["right"]) for k in main)
        self.assertTrue(70 < lefts < 130)                                                # sides are mixed

    def test_package_reveals_no_model(self):
        public, key = he.build(self.test_rows, self.dev_rows)
        img = self.d / "x.png"
        Image.new("RGB", (64, 32), "red").save(img)
        paths = {sid: str(img) for k in key for sid in (k["left"], k["right"])}
        z = he.write_package(public, key, paths, out=self.d / "out")
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            items_js = zf.read("human_eval/items.js").decode()
            text = items_js + zf.read("human_eval/index.html").decode()
            with zf.open("human_eval/img/q001_A.jpg") as f:
                self.assertEqual(Image.open(f).size, (he.SIDE, he.SIDE))
        for m in MODELS:
            self.assertNotIn(m, text)
            self.assertFalse(any(m in n for n in names))
        for word in ("gold", "repeat", "kind", "real", "stratum"):                      # no hint of the item type
            self.assertNotIn(word, items_js)
            self.assertFalse(any(word in n for n in names))
        self.assertEqual(len([n for n in names if n.endswith(".jpg")]), 2 * len(public))
        self.assertTrue((self.d / "out" / "key.json").is_file())


class TestAnalyze(unittest.TestCase):
    def test_analysis(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            write_csv(d / "test.csv", fake_rows(30, "test"))
            write_csv(d / "dev.csv", fake_rows(6, "dev"))
            public, key = he.build(he.load_rows(d / "test.csv"), he.load_rows(d / "dev.csv"))
            scores = he.load_scores([str(d / "test.csv")])
            # annotator: always the image with the higher VQAScore (model index), gold always right
            labels = {}
            for k in key:
                if k["kind"] == "gold":
                    labels[k["item_id"]] = "A" if k["left"] == k["real"] else "B"
                elif k["kind"] == "practice":
                    labels[k["item_id"]] = "same"                                        # not analysed
                else:
                    a, b = scores[k["left"]]["vqascore"], scores[k["right"]]["vqascore"]
                    labels[k["item_id"]] = "A" if a > b else "B"
            with open(d / "labels.csv", "w") as f:
                f.write("item_id,part,label,answered_at,seconds\n")
                for i, (iid, lab) in enumerate(labels.items()):
                    f.write(f"{iid},x,{lab if i != 5 else ''},t,1\n")                    # one left blank
            res = he.analyze(he.read_labels(d / "labels.csv"), key, scores, n_boot=100)
            self.assertTrue(res["gold"]["passed"])
            self.assertEqual(res["repeat"]["same_answer"], 1.0)
            self.assertEqual(res["repeat"]["cohen_kappa"], 1.0)
            rows = {(r["metric"], r["stratum"]): r for r in res["agreement"]}
            self.assertEqual(rows[("vqascore", "all")]["agreement"], 1.0)
            self.assertEqual(rows[("clipscore", "all")]["agreement"], 0.5)                 # constant -> ties
            self.assertEqual(rows[("pickscore", "all")]["agreement"], 0.0)                 # reversed
            self.assertNotIn(("hpsv2", "all"), rows)                                       # no scores
            self.assertIn("passed", he.render(res))

    def test_kappa(self):
        self.assertEqual(he.cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]), 1.0)
        self.assertAlmostEqual(he.cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
