"""Blind human check of the judge (genprobe/judge_check.py, D-033): what gets shown, that nothing in
the package reveals a group or model, and the analysis of a label file."""

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from genprobe import judge_check as jc

CATS = {"genai": 320, "counting": 56, "spatial": 56, "text": 48}
GROUP_MODEL = {"single": "sdxl", "best_of_4": "sdxl", "routed_best_of_k": "flux_schnell", "agent_rule": "sd35m"}


def fake_final(img: str, split: str = "t", run: str = "testrule", scale: int = 1):
    rows = []
    for c, n in CATS.items():
        for j in range(n // scale):
            pid = f"{c[:2]}{split}{j:03d}"
            for g, m in GROUP_MODEL.items():
                # the agent kept the base image for every 5th prompt, as when its checks passed at once
                if g == "agent_rule" and j % 5 == 0:
                    sid = f"sdxl__{pid}__s0"
                elif g == "agent_rule":
                    sid = f"{m}__{pid}-{run}-a1__s0"
                elif g == "single":
                    sid = f"sdxl__{pid}__s0"
                elif g == "best_of_4":
                    sid = f"sdxl__{pid}__s{j % 4}"
                else:
                    sid = f"{m}__{pid}__s{j % 3}"
                rows.append({"prompt_id": pid, "group": g, "sample_id": sid, "path": img, "prompt": f"prompt {pid}",
                             "category": c, "gpu_seconds": 1.0})
    return rows


class TestJudgeCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.img = str(self.d / "x.png")
        Image.new("RGB", (64, 64), (200, 10, 10)).save(self.img)
        self.final = fake_final(self.img)
        self.dev = fake_final(self.img, split="d", run="devrule2", scale=4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_allocation(self):
        self.assertEqual(jc.allocate({"genai": 320, "counting": 56, "spatial": 56, "text": 48}, 50),
                         {"genai": 33, "counting": 6, "spatial": 6, "text": 5})

    def test_items(self):
        public, key = jc.build(self.final, self.dev)
        self.assertEqual((public, key), jc.build(self.final, self.dev))        # deterministic
        kinds = [k["kind"] for k in key]
        self.assertEqual((kinds.count("practice"), kinds.count("gold"), kinds.count("repeat")),
                         (jc.N_PRACTICE, jc.N_GOLD, jc.N_REPEAT))
        main = [k for k in key if k["kind"] == "main"]
        prompts = {k["prompt_id"] for k in main}
        self.assertEqual(len(prompts), jc.N_PROMPTS)
        cats = {c: len({k["prompt_id"] for k in main if k["category"] == c}) for c in jc.CATEGORIES}
        self.assertEqual(cats, {"genai": 33, "counting": 6, "spatial": 6, "text": 5})
        # every image once; every (prompt, group) of the three groups covered exactly once
        sids = [k["sample_id"] for k in main]
        self.assertEqual(len(sids), len(set(sids)))
        covered = sorted((k["prompt_id"], g) for k in main for g in k["groups"])
        self.assertEqual(covered, sorted((p, g) for p in prompts for g in jc.GROUPS))
        self.assertLess(len(main), 3 * jc.N_PROMPTS)                           # shared images shown once
        # attention checks: an image of another prompt (other category), never a main image
        cat_of = {r["prompt_id"]: r["category"] for r in self.final}
        for k in key:
            if k["kind"] == "gold":
                self.assertNotIn(k["sample_id"], sids)
                self.assertNotEqual(cat_of[k["image_prompt_id"]], k["category"])
        # repeats show a main image again, later
        pos = {k["item_id"]: i for i, k in enumerate(key)}
        by = {k["item_id"]: k for k in key}
        for k in key:
            if k["kind"] == "repeat":
                o = by[k["repeat_of"]]
                self.assertEqual((o["kind"], o["sample_id"]), ("main", k["sample_id"]))
                self.assertGreater(pos[k["item_id"]] - pos[o["item_id"]], 30)
        # practice from the dev run
        self.assertTrue(all("d" in k["prompt_id"][2:3] for k in key if k["part"] == "practice"))

    def test_package_is_blind(self):
        public, key = jc.build(self.final, self.dev)
        z = jc.write_package(public, key, self.d / "out")
        with zipfile.ZipFile(z) as f:
            names = f.namelist()
            items_js = f.read("judge_check/items.js").decode()
            page = f.read("judge_check/index.html").decode()
        self.assertEqual(len([n for n in names if n.endswith(".jpg")]), len(public))
        for word in ("sdxl", "flux", "sd35m", "agent", "routed", "best_of", "score", "testrule", "sample_id", "group"):
            self.assertNotIn(word, items_js)
        self.assertIn('data-label="unsure"', page)
        self.assertEqual(set(public[0]), {"item_id", "part", "prompt", "image"})

    def test_analyze(self):
        public, key = jc.build(self.final, self.dev)
        main = [k for k in key if k["kind"] == "main"]
        # judge: passes everything that is not an agent-only image; person agrees except on 3 images
        grades = {k["sample_id"]: {"sample_id": k["sample_id"], "score": 5 if k["groups"] != ["agent_rule"] else 3}
                  for k in main}
        labels = {}
        flip = {main[0]["sample_id"], main[1]["sample_id"], main[2]["sample_id"]}
        for k in key:
            if k["part"] == "practice":
                labels[k["item_id"]] = "yes"
            elif k["kind"] == "gold":
                labels[k["item_id"]] = "no"
            else:
                j = grades[k["sample_id"]]["score"] == 5
                labels[k["item_id"]] = ("yes" if j else "no") if k["sample_id"] not in flip else ("no" if j else "yes")
        res = jc.analyze(labels, key, grades, n_boot=200)
        self.assertTrue(res["gold"]["passed"])
        self.assertEqual(res["repeat"]["same_answer"], 1.0)
        n = len(main)
        self.assertAlmostEqual(res["agreement"]["accuracy"]["value"], (n - 3) / n)
        self.assertLess(res["agreement"]["cohen_kappa"]["value"], 1.0)
        # judge passes routed images always; agent-only images never
        self.assertEqual(res["pass_rate"]["routed_best_of_k"]["judge"]["value"], 1.0)
        d = res["paired_difference"]["agent_rule - routed_best_of_k"]["judge"]["value"]
        self.assertLess(d, 0)
        self.assertIn("Judge vs person", jc.render(res))
        # a label file round trip
        p = self.d / "labels.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "part", "label", "answered_at", "seconds"])
            for iid, lab in labels.items():
                w.writerow([iid, "main", lab, "", ""])
        self.assertEqual(jc.read_labels(p), labels)
        with open(p, "a", newline="") as f:
            csv.writer(f).writerow(["q999", "main", "maybe", "", ""])
        with self.assertRaises(ValueError):
            jc.read_labels(p)
        json.dumps(res)                                                        # serialisable


if __name__ == "__main__":
    unittest.main()
