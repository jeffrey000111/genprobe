"""The agent's HTML gallery (genprobe/gallery.py, guide §10.5): which attempt is kept, which prompts
are drawn, and that the page is self-contained."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from genprobe import gallery as G


class TestGallery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_kept_step(self):
        steps = [{"step": 0, "result": "start"}, {"step": 1, "result": "new best"},
                 {"step": 2, "result": "kept attempt 1"}, {"step": 3, "result": "new best; passed"}]
        self.assertEqual(G.kept_step(steps), 3)
        self.assertEqual(G.kept_step(steps[:3]), 1)
        self.assertEqual(G.kept_step([{"step": 0, "result": "start (passed)"}]), 0)

    def test_build(self):
        Path("data/prompts").mkdir(parents=True)
        cats = {"counting": "tc", "spatial": "ts", "text": "tt", "genai": "gb"}
        prompts, audit, man = [], [], []
        Image.new("RGB", (64, 64), (200, 30, 30)).save("x.png")
        for c, pre in cats.items():
            for i in range(5):
                pid = f"{pre}{i:04d}"
                prompts.append({"prompt_id": pid, "prompt": f"prompt <{pid}>", "tags": [c if c != "genai" else "basic"],
                                "split": "test"})
                n = 1 if i == 0 else 3                          # tc0000 etc.: no repair, never drawn
                for s in range(n):
                    sid = f"sdxl__{pid}-a{s}__s{s}"
                    audit.append({"prompt_id": pid, "step": s, "action": "base" if s == 0 else "reseed",
                                  "params": {"model": "sdxl", "seed": s, "cfg": None, "gen_prompt": "p"},
                                  "sample_id": sid, "checks": {"vqascore": 0.1 * s}, "passed": s == 2,
                                  "diagnosis": "low_alignment", "decision_reason": "why",
                                  "result": "start" if s == 0 else ("new best; passed" if s == 2 else "kept attempt 0"),
                                  "gpu_seconds": 3.9, "budget_left": 10.0 - s})
                    man.append({"sample_id": sid, "path": "x.png", "status": "ok"})
        Path(G.V1).write_text("".join(json.dumps(r) + "\n" for r in prompts))
        Path("results/agent/run1").mkdir(parents=True)
        Path("results/agent/run1/audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in audit))
        Path("data/agent/run1/manifests").mkdir(parents=True)
        Path("data/agent/run1/manifests/gen_r1_sdxl.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
        Path("data/judge").mkdir(parents=True)
        Path("data/judge/g.jsonl").write_text("".join(json.dumps({"sample_id": r["sample_id"], "score": 5}) + "\n"
                                                      for r in audit if r["step"] == 2))
        res = G.build("run1", "g")
        self.assertEqual(len(res["prompts"]), 4 * G.PER_CATEGORY)
        self.assertNotIn("tc0000", res["prompts"])
        self.assertEqual(res, {**res, "prompts": G.build("run1", "g")["prompts"]})   # same draw again
        page = Path("results/examples/agent_gallery.html").read_text()
        self.assertEqual(page.count('class="tile kept"'), 4 * G.PER_CATEGORY)
        self.assertIn("data:image/jpeg;base64,", page)
        self.assertIn("5 of 5 — pass", page)
        self.assertIn("prompt &lt;", page)                                           # prompts are escaped
        self.assertNotIn("x.png", page)


if __name__ == "__main__":
    unittest.main()
