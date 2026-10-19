"""Example panels (genprobe/examples.py) on fake images."""

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from genprobe import examples
from genprobe.backends import FakeBackend
from genprobe.generate import run_generation
from genprobe.samples import Prompt
from tests.helpers import make_cfg, quiet


class TestExamples(unittest.TestCase):
    def test_choose_spreads_models(self):
        rows = [{"sample_id": f"{m}__p{i}__s0", "model": m, "ok": i % 2} for m in ("a", "b", "c") for i in range(6)]
        got = examples.choose(rows, lambda r: r["ok"] == 1, 3, random.Random(0))
        self.assertEqual(sorted(r["model"] for r in got), ["a", "b", "c"])
        self.assertEqual(examples.choose(rows, lambda r: False, 3, random.Random(0)), [])

    def test_prompts_about_children_are_left_out(self):
        for p in ("Two children sitting on the bed", "a baby in a crib", "A girl with a kite", "teenagers at a mall",
                  "In the classroom, two students raise their hands"):
            self.assertTrue(examples.NO_PUBLIC.search(p), p)
        for p in ("a kidney bean", "a cowboy hat", "a photo of three birds"):
            self.assertFalse(examples.NO_PUBLIC.search(p), p)

    def test_run_writes_panels_and_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            prompts = [Prompt("tc0001", "a photo of two cups", source="template_count", tags=["counting"],
                              spec={"objects": [{"name": "cup", "count": 2}]}, split="test"),
                       Prompt("tt0001", 'a sign that says "SALE"', source="template_text", tags=["text"],
                              spec={"text": "SALE"}, split="test"),
                       Prompt("gb0001", "a crystal tree", source="genai_bench", tags=["basic", "scene"], split="test")]
            ppath = d / "p.jsonl"
            ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
            rows = []
            for model, good in (("sd15", False), ("flux_schnell", True)):
                cfg = make_cfg(name=model, seeds=[0, 1], batch_size=2, height=64, width=64)
                run_generation(cfg=cfg, prompts=prompts, out=d, backend=FakeBackend(cfg), backend_kind="fake", log=quiet)
                for p in prompts:
                    for s in (0, 1):
                        rows.append({"sample_id": f"{model}__{p.prompt_id}__s{s}", "model": model, "prompt_id": p.prompt_id,
                                     "category": p.tags[0], "prompt": p.prompt, "status": "ok",
                                     "vqascore": 0.99 if good else 0.01,
                                     "detect": (1.0 if good else 0.0) if p.tags == ["counting"] else "",
                                     "ocr": (1.0 if good else 0.2) if p.tags == ["text"] else "",
                                     "pickscore": 23.0 if not good else 20.0})
            res = d / "res"
            res.mkdir()
            with open(res / "per_sample.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            written = examples.run(str(res), str(ppath), str(d / "manifests" / "gen_*.jsonl"), str(d / "ex"))
            names = {p.name for p in written}
            self.assertTrue({"counting.jpg", "text.jpg", "genai_bench.jpg", "pretty_but_wrong.jpg", "index.md"} <= names)
            self.assertNotIn("spatial.jpg", names)                       # no spatial prompts: no panel
            idx = (d / "ex" / "index.md").read_text()
            self.assertIn("`flux_schnell__tc0001__s", idx)
            self.assertIn("`sd15__gb0001__s", idx)                        # low VQAScore row


if __name__ == "__main__":
    unittest.main()
