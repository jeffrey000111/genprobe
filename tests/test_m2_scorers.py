"""PickScore, HPSv2.1 and the two safety filters: the logic that runs without the models."""

import unittest
from types import SimpleNamespace

from genprobe.metrics import REGISTRY
from genprobe.metrics.hpsv2 import extract_state_dict
from genprobe.metrics.safety import sd_checker_decision


class TestRegistry(unittest.TestCase):
    def test_registered(self):
        for name in ("pickscore", "hpsv2", "nsfw_sd", "nsfw_vit"):
            self.assertIn(name, REGISTRY)
            self.assertEqual(REGISTRY[name].env, "eval")
        self.assertTrue(REGISTRY["pickscore"].higher_is_better)
        self.assertFalse(REGISTRY["nsfw_sd"].higher_is_better)
        self.assertFalse(REGISTRY["nsfw_vit"].higher_is_better)

    def test_safety_scores_every_image(self):
        genai = SimpleNamespace(spec=None, prompt="a cat", sample_id="m__p__s0")
        for name in ("nsfw_sd", "nsfw_vit", "pickscore", "hpsv2"):
            self.assertTrue(REGISTRY[name]().applies_to(genai), name)

    def test_versions_name_weights(self):
        self.assertIn("pickscore-v1@", REGISTRY["pickscore"]().version)
        self.assertIn("hps-v2.1@", REGISTRY["hpsv2"]().version)
        self.assertIn("sd-safety-checker@", REGISTRY["nsfw_sd"]().version)
        self.assertIn("falconsai-nsfw-vit@", REGISTRY["nsfw_vit"]().version)


class TestSDCheckerRule(unittest.TestCase):
    def test_clean_image(self):
        flagged, margin = sd_checker_decision([0.1, 0.1], [0.2, 0.2], [0.15, 0.1], [0.2, 0.3])
        self.assertFalse(flagged)
        self.assertAlmostEqual(margin, -0.05)

    def test_concept_over_threshold(self):
        self.assertTrue(sd_checker_decision([0.0], [0.5], [0.31], [0.3])[0])

    def test_special_care_lowers_the_bar(self):
        # a concept 0.005 below its threshold fires once a special-care concept has fired (+0.01)
        self.assertFalse(sd_checker_decision([0.1], [0.2], [0.295], [0.3])[0])
        self.assertTrue(sd_checker_decision([0.25], [0.2], [0.295], [0.3])[0])

    def test_special_care_alone_does_not_flag(self):
        self.assertFalse(sd_checker_decision([0.9], [0.2], [0.1], [0.3])[0])

    def test_rounding_at_zero(self):
        # 0.3004 - 0.3 rounds to 0.0, which is not > 0
        self.assertFalse(sd_checker_decision([], [], [0.3004], [0.3])[0])


class TestHPSCheckpoint(unittest.TestCase):
    def test_state_dict_unwrapping(self):
        self.assertEqual(extract_state_dict({"state_dict": {"a": 1}, "epoch": 3}), {"a": 1})
        self.assertEqual(extract_state_dict({"module.a": 1, "module.b": 2}), {"a": 1, "b": 2})
        self.assertEqual(extract_state_dict({"a": 1}), {"a": 1})
        with self.assertRaises(RuntimeError):
            extract_state_dict([1, 2])
        with self.assertRaises(RuntimeError):
            extract_state_dict({"state_dict": {}})


if __name__ == "__main__":
    unittest.main()
