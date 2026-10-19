"""Prompt set v1 builder: deterministic, right sizes, every skill covered, stratified split."""

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from genprobe.build_prompts import SKILL_NAMES, build
from genprobe.samples import Prompt


def fake_genai_bench(folder: Path, n: int = 300, seed: int = 1) -> None:
    rng = random.Random(seed)
    images, skills = {}, {k: [] for k in ["basic", "advanced", *SKILL_NAMES]}
    basic = ["attribute", "scene", "spatial relation", "action relation", "part relation"]
    adv = ["counting", "comparison", "differentiation", "negation", "universal"]
    for i in range(n):
        images[f"{i:05d}"] = {"id": f"{i:05d}", "prompt": f"prompt number {i}", "prompt in Chinese": "",
                              "models": {"DALLE_3": [3, 3, 3]}}
        if i % 50 == 49:
            continue                                    # no tier, no skill: must be skipped
        tier, pool = ("basic", basic) if i % 2 else ("advanced", adv)
        skills[tier].append(i)
        for s in rng.sample(pool, rng.randint(1, 3)):
            skills[s].append(i)
    folder.mkdir(parents=True)
    (folder / "genai_image.json").write_text(json.dumps(images))
    (folder / "genai_skills.json").write_text(json.dumps(skills))


class TestBuildPrompts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gb = Path(self.tmp.name) / "gb"
        fake_genai_bench(self.gb)

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self, **kw):
        return build(self.gb, n_genai=100, min_per_skill=12, **kw)

    def test_deterministic(self):
        self.assertEqual(self.rows(), self.rows())
        self.assertNotEqual(self.rows(), self.rows(seed=1))

    def test_sizes_and_ids(self):
        rows = self.rows()
        src = Counter(r["source"] for r in rows)
        self.assertEqual(src, {"genai_bench": 100, "template_count": 70, "template_spatial": 70,
                               "template_text": 60})
        ids = [r["prompt_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(Prompt.from_dict(r).prompt_id for r in rows))   # ids pass the sample-id check
        gb_idx = {int(r["prompt_id"][2:]) for r in rows if r["source"] == "genai_bench"}
        self.assertFalse({i for i in gb_idx if i % 50 == 49})                # untagged ones skipped

    def test_every_skill_reaches_minimum(self):
        rows = self.rows()
        counts = Counter(t for r in rows if r["source"] == "genai_bench" for t in r["tags"][1:])
        self.assertEqual(set(counts), set(SKILL_NAMES.values()))
        self.assertGreaterEqual(min(counts.values()), 12)

    def test_split_is_stratified(self):
        rows = self.rows()
        for stratum in {r["stratum"] for r in rows}:
            group = [r for r in rows if r["stratum"] == stratum]
            dev = sum(r["split"] == "dev" for r in group)
            self.assertEqual(dev, round(len(group) * 0.2), stratum)

    def test_template_specs(self):
        rows = {r["prompt_id"]: r for r in self.rows()}
        counts = [r for r in rows.values() if r["source"] == "template_count"]
        self.assertEqual(sorted({r["spec"]["objects"][0]["count"] for r in counts}), list(range(1, 8)))
        for r in counts:
            n = r["spec"]["objects"][0]["count"]
            self.assertIn(["one", "two", "three", "four", "five", "six", "seven"][n - 1], r["prompt"])
        for r in rows.values():
            if r["source"] == "template_text":
                self.assertIn(f'"{r["spec"]["text"]}"', r["prompt"])
                self.assertEqual(r["spec"]["n_words"], len(r["spec"]["text"].split()))
            if r["source"] == "template_spatial":
                rel = r["spec"]["relation"]
                self.assertNotEqual(rel["a"], rel["b"])
                self.assertTrue(r["prompt"].index(rel["a"]) < r["prompt"].index(rel["b"]))


if __name__ == "__main__":
    unittest.main()
