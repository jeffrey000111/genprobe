"""Metric sanity checks (genprobe/sanity.py): prompt edits, statistics, and a run with a toy metric."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from genprobe import sanity
from genprobe.backends import FakeBackend
from genprobe.generate import run_generation
from genprobe.metrics.base import REGISTRY, Metric, register
from genprobe.samples import Prompt, load_prompts
from tests.helpers import make_cfg, quiet

ROOT = Path(__file__).resolve().parents[1]


class TestCounterfactual(unittest.TestCase):
    def test_count(self):
        p, spec, what = sanity.counterfactual("a photo of three apples",
                                              {"objects": [{"name": "apple", "count": 3}]})
        self.assertEqual((p, spec["objects"][0]["count"], what), ("a photo of five apples", 5, "count 3->5"))
        p, spec, _ = sanity.counterfactual("a realistic photo of one cup", {"objects": [{"name": "cup", "count": 1}]})
        self.assertEqual(p, "a realistic photo of three cups")
        p, spec, _ = sanity.counterfactual("a photo of seven birds", {"objects": [{"name": "bird", "count": 7}]})
        self.assertEqual(p, "a photo of five birds")

    def test_relation(self):
        p, spec, _ = sanity.counterfactual("a photo of a chair to the left of an umbrella",
                                           {"relation": {"type": "left_of", "a": "chair", "b": "umbrella"}})
        self.assertEqual((p, spec["relation"]["type"]), ("a photo of a chair to the right of an umbrella", "right_of"))
        p, spec, _ = sanity.counterfactual("a teddy bear above a bench",
                                           {"relation": {"type": "above", "a": "teddy bear", "b": "bench"}})
        self.assertEqual((p, spec["relation"]["type"]), ("a teddy bear below a bench", "below"))

    def test_text(self):
        p, spec, _ = sanity.counterfactual('a chalkboard that says "SALE"', {"text": "SALE"})
        self.assertEqual((p, spec["text"]), ('a chalkboard that says "EXIT"', "EXIT"))

    def test_none_for_free_text(self):
        self.assertIsNone(sanity.counterfactual("a crystal tree", None))

    def test_every_v1_template_prompt_has_one(self):
        n = 0
        for p in load_prompts(ROOT / "data/prompts/prompts_v1.jsonl"):
            if p.source.startswith("template"):
                c = sanity.counterfactual(p.prompt, p.spec)
                self.assertIsNotNone(c, p.prompt)
                self.assertNotEqual(c[0], p.prompt)
                self.assertNotEqual(c[1], p.spec)
                n += 1
        self.assertEqual(n, 200)


class TestStats(unittest.TestCase):
    def test_own_higher(self):
        r = sanity.own_higher([1.0, 0.5, 0.2, 0.0], [0.0, 0.5, 0.3, 0.0])
        self.assertEqual((r["higher"], r["ties"], r["lower"], r["rate"]), (1, 2, 1, 0.5))

    def test_repeat(self):
        r = sanity.repeat_stats([0.1, 0.2], [0.1, 0.25])
        self.assertEqual(r["identical"], 0.5)
        self.assertAlmostEqual(r["max_abs_diff"], 0.05)


if "toy_prompt_len" not in REGISTRY:
    @register
    class ToyPromptLen(Metric):
        """Prefers the image's own prompt: 1 if the prompt is the one the image was made from."""
        name = "toy_prompt_len"
        version = "toy"
        needs = frozenset({"image", "prompt"})
        default_batch_size = 4

        def setup(self, device):
            pass

        def score_batch(self, samples):
            return [{"sample_id": s.sample_id, "value": float(s.prompt == self.truth[s.sample_id]), "extra": {}}
                    for s in samples]


class TestRun(unittest.TestCase):
    def test_run_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prompts = [Prompt(f"tc{i:04d}", f"a photo of {w} apples", source="template_count", tags=["counting"],
                              spec={"objects": [{"name": "apple", "count": k}]}, split="dev")
                       for i, (k, w) in enumerate([(2, "two"), (3, "three"), (4, "four"), (5, "five")])]
            ppath = out / "p.jsonl"
            ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
            cfg = make_cfg(name="toy", seeds=[0, 1], batch_size=2)
            run_generation(cfg=cfg, prompts=prompts, out=out, backend=FakeBackend(cfg), backend_kind="fake", log=quiet)
            truth = {f"toy__{p.prompt_id}__s{s}": p.prompt for p in prompts for s in (0, 1)}
            with mock.patch.object(sanity, "OUT_DIR", out / "sanity"), \
                    mock.patch.object(REGISTRY["toy_prompt_len"], "truth", truth, create=True):
                r = sanity.run("toy_prompt_len", str(ppath), str(out / "manifests" / "gen_*.jsonl"),
                               data=str(out), n=8, n_repeat=4, device="cpu")
                md = sanity.report()
            self.assertEqual(r["n_images"], 8)
            self.assertEqual((r["mismatch"]["rate"], r["mismatch"]["n"]), (1.0, 8))
            self.assertEqual(r["counterfactual"]["count"]["higher"], 8)
            self.assertEqual(r["repeat_in_job"]["identical"], 1.0)
            self.assertEqual(r["repeat_vs_stored"]["n"], 0)          # never scored by a scoring run
            self.assertEqual(len(r["per_image"]), 8)
            self.assertTrue(all("cf" in x and "mismatch" in x for x in r["per_image"]))
            self.assertTrue((out / "sanity" / "toy_prompt_len.json").is_file())
            self.assertIn("| toy_prompt_len | 8 | 1.00 (8↑ 0= 0↓ of 8)", md)

            labels = out / "labels.csv"
            labels.write_text("sample_id,correct,note,true_count\n"
                              "toy__tc0000__s0,y,,2\ntoy__tc0001__s1,y,,3\ntoy__tc0002__s0,n,,\n")
            with mock.patch.object(sanity, "OUT_DIR", out / "sanity"), \
                    mock.patch.object(REGISTRY["toy_prompt_len"], "truth", truth, create=True):
                r2 = sanity.run_labelled("toy_prompt_len", str(ppath), str(out / "manifests" / "gen_*.jsonl"),
                                         str(labels), device="cpu")
                md = sanity.report()
                again = sanity.run("toy_prompt_len", str(ppath), str(out / "manifests" / "gen_*.jsonl"),
                                   data=str(out), n=8, n_repeat=4, device="cpu")
            c = r2["counterfactual_labelled"]
            self.assertEqual((c["all"]["n"], c["all"]["higher"]), (2, 2))     # only the two labelled y
            self.assertEqual(r2["n_images"], 8)                                 # merged into the same file
            self.assertIn("On images labelled correct", md)
            self.assertIn("counterfactual_labelled", again)                     # a rerun keeps it


if __name__ == "__main__":
    unittest.main()
