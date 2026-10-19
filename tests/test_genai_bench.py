"""GenAI-Bench preparation (genprobe/genai_bench.py) on a tiny fake copy of the dataset."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from genprobe import genai_bench
from genprobe.score import load_samples


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = Path(self.tmp.name)
        folder = d / "gb"
        folder.mkdir()
        images = {f"{i:05d}": {"id": f"{i:05d}", "prompt": f"prompt {i}", "prompt in Chinese": "",
                               "models": {m: [3, 4, 5] for m in genai_bench.MODELS}} for i in range(4)}
        images["00001"]["models"]["DALLE_3"] = [4, 41, 5]          # out-of-range rating
        (folder / "genai_image.json").write_text(json.dumps(images))
        (folder / "genai_skills.json").write_text(json.dumps({"basic": [0, 1], "advanced": [2], "counting": [0, 2],
                                                              "spatial relation": [1]}))
        for m in genai_bench.MODELS:
            for i in range(4):
                if (m, i) != ("SDXL_Turbo", 3):                     # one image missing
                    p = genai_bench.image_path(folder, m, i)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8)).save(p, format="JPEG")
        v1 = d / "v1.jsonl"
        v1.write_text(json.dumps({"prompt_id": "gb0001", "prompt": "prompt 1", "source": "genai_bench",
                                  "tags": ["basic"], "split": "dev"}) + "\n")
        self.folder, self.v1 = folder, v1

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, **kw):
        d = self.d
        return genai_bench.prepare(self.folder, str(self.v1), str(d / "p.jsonl"), str(d / "smoke.jsonl"),
                                   str(d / "m.jsonl"), **kw)

    def test_prepare(self):
        res = self.run_prepare(verify=True)
        self.assertEqual((res["prompts"], res["dev"], res["images"], res["missing"], res["unreadable"]),
                         (4, 1, 24, 1, 0))
        self.assertEqual(res["dropped_ratings"], [{"prompt_id": "gb0001", "model": "DALLE_3", "rating": 41}])
        prompts = [json.loads(l) for l in (self.d / "p.jsonl").read_text().splitlines()]
        self.assertEqual(prompts[0]["tags"], ["basic", "counting"])
        self.assertEqual(prompts[3]["tags"], ["untagged"])
        self.assertEqual([p["split"] for p in prompts], ["test", "dev", "test", "test"])
        samples = load_samples(str(self.d / "m.jsonl"), str(self.d / "p.jsonl"))
        self.assertEqual(len(samples), 23)                          # the missing image is not a sample
        self.assertIn("DALLE_3__gb0000__s0", {s.sample_id for s in samples})
        ratings, _ = genai_bench.load_ratings(self.folder)
        self.assertEqual(ratings[("gb0001", "DALLE_3")], [4.0, 5.0])

    def test_smoke_split_mix(self):
        self.run_prepare()
        smoke = [json.loads(l) for l in (self.d / "smoke.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(p["split"] for p in smoke), ["dev", "test", "test", "test"])


if __name__ == "__main__":
    unittest.main()
