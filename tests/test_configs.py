"""Every shipped config and prompt file parses and follows the project's rules."""

import unittest
from pathlib import Path

from genprobe.config import load_gen_config
from genprobe.samples import load_prompts

ROOT = Path(__file__).resolve().parent.parent


class TestConfigs(unittest.TestCase):
    def test_gen_configs_validate(self):
        paths = sorted((ROOT / "configs" / "gen").glob("*.yaml"))
        self.assertEqual(len(paths), 4)
        hashes = set()
        for p in paths:
            with self.subTest(config=p.name):
                cfg = load_gen_config(p)
                self.assertEqual(cfg.name, p.stem, "config name must match its filename")
                self.assertEqual(cfg.seeds, [0, 1, 2, 3])
                hashes.add(cfg.config_hash())
        self.assertEqual(len(hashes), 4)

    def test_model_specific_rules(self):
        flux = load_gen_config(ROOT / "configs/gen/flux_schnell.yaml")
        self.assertEqual(flux.cfg, 0.0, "FLUX schnell must run with guidance 0")
        self.assertLessEqual(flux.call_kwargs.get("max_sequence_length", 0), 256)
        sd15 = load_gen_config(ROOT / "configs/gen/sd15.yaml")
        self.assertIsNone(sd15.load_kwargs.get("safety_checker", "missing"))

    def test_unknown_config_keys_rejected(self):
        from pydantic import ValidationError
        from genprobe.config import GenConfig
        with self.assertRaises(ValidationError):
            GenConfig(name="x", repo="r", dtype="fp16", steps=1, cfg=1, height=8, width=8, stepz=3)

    def test_smoke_prompts(self):
        prompts = load_prompts(ROOT / "data/prompts/smoke.jsonl")
        self.assertEqual(len(prompts), 20)
        self.assertTrue(all(p.split == "dev" and p.spec for p in prompts))
        by_tag = {}
        for p in prompts:
            for t in p.tags:
                by_tag[t] = by_tag.get(t, 0) + 1
        self.assertEqual(by_tag, {"counting": 7, "spatial": 7, "text": 6})
        for p in prompts:
            if "text" in p.tags:
                self.assertIn(f'"{p.spec["text"]}"', p.prompt)


if __name__ == "__main__":
    unittest.main()
