"""Pipeline check of a scoring run (genprobe/score_check.py)."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from genprobe import score_check


class TestScoreCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.d = Path(self.tmp.name)
        prompts = [{"prompt_id": f"gb{i:04d}", "prompt": f"p {i}", "source": "genai_bench", "tags": ["basic"],
                    "split": "test"} for i in range(3)]
        (d / "p.jsonl").write_text("".join(json.dumps(p) + "\n" for p in prompts))
        recs = [{"sample_id": f"{m}__gb{i:04d}__s0", "model": m, "prompt_id": f"gb{i:04d}", "prompt": f"p {i}",
                 "seed": 0, "path": "x.jpeg", "status": "ok"} for m in ("A", "B") for i in range(3)]
        (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        self.recs = recs

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, metric, recs, error_on=()):
        (self.d / "scores" / metric).mkdir(parents=True, exist_ok=True)
        with open(self.d / "scores" / metric / "t_shard000of001.jsonl", "w") as f:
            for k, r in enumerate(recs):
                err = r["sample_id"] in error_on
                f.write(json.dumps({"sample_id": r["sample_id"], "metric_version": "v", "value": None if err else k / 10,
                                    "seconds": 0.5, "status": "error" if err else "ok",
                                    "error": "RuntimeError: boom" if err else None, "created_at": "t"}) + "\n")

    def run_check(self, *metrics):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = score_check.main([*metrics, "--samples", str(self.d / "m.jsonl"), "--prompts",
                                     str(self.d / "p.jsonl"), "--tag", "t", "--data", str(self.d), "--project", "7200"])
        return code, buf.getvalue()

    def test_healthy(self):
        self.write("clipscore", self.recs)
        code, out = self.run_check("clipscore")
        self.assertEqual(code, 0, out)
        self.assertIn("expected 6, scored 6, ok 6, error 0, missing 0", out)
        self.assertIn("projected for 7200 images: 1.00 h", out)
        self.assertIn("by model (mean s/image): A 0.500, B 0.500", out)

    def test_errors_and_missing(self):
        self.write("clipscore", self.recs[:5], error_on={self.recs[0]["sample_id"]})
        code, out = self.run_check("clipscore")
        self.assertEqual(code, 1)
        self.assertIn("ok 4, error 1, missing 1", out)
        self.assertIn("error ×1: RuntimeError: boom", out)


if __name__ == "__main__":
    unittest.main()
