"""Test 3 (guide §4.6): each metric plugin, run on 3 fake images, emits well-formed score records.

Real plugins registered in genprobe.metrics are exercised too; ones whose models or packages
aren't available here (no GPU, no t2v_metrics) are skipped rather than failed.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from genprobe.metrics import REGISTRY, Metric, validate_record
from genprobe.score import load_samples, run_scoring
from genprobe.generate import run_generation
from genprobe.backends import FakeBackend
from tests.helpers import make_cfg, make_prompts, quiet


class MeanBrightness(Metric):
    """Trivial metric used only to test the runner and record format."""

    name = "mean_brightness"
    version = "test/1"
    needs = frozenset({"image"})
    env = "eval"
    default_batch_size = 2

    def setup(self, device):
        self.ready = True

    def score_batch(self, samples):
        out = []
        for s in samples:
            with Image.open(s.path) as im:
                out.append({"sample_id": s.sample_id, "value": float(np.asarray(im).mean() / 255),
                            "extra": {"size": list(im.size)}})
        return out


class SpecOnly(MeanBrightness):
    name = "spec_only"
    needs = frozenset({"image", "spec"})


class Broken(MeanBrightness):
    name = "broken"

    def score_batch(self, samples):
        raise RuntimeError("model exploded")


class TestMetricInterface(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        cfg = make_cfg(seeds=[0], batch_size=2)
        run_generation(cfg=cfg, prompts=make_prompts(3), out=self.out, backend=FakeBackend(cfg),
                       backend_kind="fake", log=quiet)
        self.samples = load_samples(str(self.out / "manifests" / "gen_toy_*.jsonl"))
        self.assertEqual(len(self.samples), 3)

    def tearDown(self):
        self.tmp.cleanup()

    def score(self, metric, **kw):
        return run_scoring(metric=metric, samples=self.samples, out=self.out, tag="toy",
                           device="cpu", log=quiet, **kw)

    def read_scores(self, name):
        from genprobe.utils.manifest import read_many
        return read_many((self.out / "scores" / name).glob("*.jsonl"))

    def test_record_format(self):
        counts = self.score(MeanBrightness())
        self.assertEqual(counts["ok"], 3)
        recs = self.read_scores("mean_brightness")
        self.assertEqual(len(recs), 3)
        for r in recs:
            self.assertEqual(validate_record(r), [], r)
            self.assertEqual(r["metric_version"], "test/1/bs2")
            self.assertTrue(0.0 <= r["value"] <= 1.0)

    def test_resume_skips_scored(self):
        self.score(MeanBrightness())
        counts = self.score(MeanBrightness())
        self.assertEqual(counts, {"ok": 0, "error": 0, "skipped": 3})
        # A different batch size is a different metric_version -> rescored.
        counts = self.score(MeanBrightness(), batch_size=3)
        self.assertEqual(counts["ok"], 3)

    def test_failures_are_recorded_not_raised(self):
        counts = self.score(Broken())
        self.assertEqual(counts["error"], 3)
        for r in self.read_scores("broken"):
            self.assertEqual(r["status"], "error")
            self.assertIn("model exploded", r["error"])
            self.assertEqual(validate_record(r), [])

    def test_spec_metrics_need_prompts_join(self):
        # Manifests don't carry specs: without --prompts a spec metric scores nothing, loudly.
        msgs = []
        counts = run_scoring(metric=SpecOnly(), samples=self.samples, out=self.out, tag="toy",
                             device="cpu", log=msgs.append)
        self.assertEqual(counts["ok"], 0)
        self.assertTrue(any("forget --prompts" in m for m in msgs))

        # With the prompts file joined, only prompts that have a spec are scored.
        prompts = make_prompts(3)
        prompts[0].spec = None
        ppath = self.out / "prompts.jsonl"
        ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
        joined = load_samples(str(self.out / "manifests" / "gen_toy_*.jsonl"), str(ppath))
        counts = run_scoring(metric=SpecOnly(), samples=joined, out=self.out, tag="toy",
                             device="cpu", log=quiet)
        self.assertEqual(counts["ok"], 2)

    def test_validate_record_catches_problems(self):
        good = {"sample_id": "a", "metric": "m", "metric_version": "v", "value": 0.5,
                "extra": {}, "seconds": 0.1, "status": "ok"}
        self.assertEqual(validate_record(good), [])
        self.assertTrue(validate_record({**good, "value": float("nan")}))
        self.assertTrue(validate_record({**good, "value": None}))
        self.assertTrue(validate_record({**good, "extra": []}))
        self.assertTrue(validate_record({k: v for k, v in good.items() if k != "metric_version"}))

    def test_registered_plugins(self):
        for name, cls in sorted(REGISTRY.items()):
            with self.subTest(metric=name):
                metric = cls()
                try:
                    metric.setup("cuda")
                except Exception as e:  # noqa: BLE001
                    self.skipTest(f"{name}: setup unavailable here ({type(e).__name__}: {e})")
                batch = [s for s in self.samples if metric.applies_to(s)]
                for r in metric.score_batch(batch):
                    rec = {"metric": name, "metric_version": metric.version, "seconds": 0.0,
                           "status": "ok", **r}
                    self.assertEqual(validate_record(rec), [])


if __name__ == "__main__":
    unittest.main()
