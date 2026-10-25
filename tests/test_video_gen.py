"""Wan2.2 prompt selection and the video evaluation summary (genprobe/video_gen.py,
genprobe/video_eval.py, D-036). Nothing here loads a model or decodes a video."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from genprobe import genai_bench_video as gv
from genprobe import video_eval as ve
from genprobe import video_gen as vg
from genprobe.samples import make_sample_id


class TestVideoGen(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data/prompts").mkdir(parents=True)
        rows = [{"prompt_id": f"gv{i:04d}", "prompt": f"p{i}", "tags": ["basic" if i < 500 else "advanced", "scene"],
                 "split": "test", "key": f"{i:05d}"} for i in range(800)]
        Path(vg.SOURCE_PROMPTS).write_text("".join(json.dumps(r) + "\n" for r in rows))

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_select(self):
        res = vg.select()
        self.assertEqual(res, {"prompts": 160, "by_tier": {"advanced": 60, "basic": 100},
                               "smoke": res["smoke"]})
        self.assertEqual(res, vg.select())                                     # deterministic
        rows = [json.loads(x) for x in Path(vg.PROMPTS).read_text().splitlines()]
        self.assertEqual([r["prompt_id"] for r in rows], sorted(r["prompt_id"] for r in rows))
        self.assertEqual(res["smoke"], [r["prompt_id"] for r in rows[:3]])
        self.assertEqual(vg.model_name(121), "wan22_f121")

    def test_analyze(self):
        vg.select()
        prompts = ve.wan_prompts()
        wan = "wan22_f121"
        models = [wan] + list(gv.MODELS)
        d = Path("data/scores/vqascore_video")
        d.mkdir(parents=True)
        with open(d / "x.jsonl", "w") as f:
            for i, m in enumerate(models):
                for p in prompts:
                    f.write(json.dumps({"sample_id": make_sample_id(m, p["prompt_id"], 0), "metric": "vqascore_video",
                                        "metric_version": "v", "value": 0.9 - 0.1 * i, "status": "ok",
                                        "created_at": "2026-10-23"}) + "\n")
        Path("data/video/vbench").mkdir(parents=True)
        for i, m in enumerate(models):
            rows = [{"model": m, "prompt_id": p["prompt_id"], **{dim: 0.5 + 0.01 * i for dim in ve.DIMENSIONS}}
                    for p in prompts[:-1 if m == "Gen2" else None]]
            (Path("data/video/vbench") / f"{m}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        res = ve.analyze(wan, n_boot=100)
        self.assertAlmostEqual(res["table"]["vqascore_video"][wan]["value"], 0.9)
        self.assertAlmostEqual(res["paired_difference"][f"vqascore_video|{wan} - Modelscope"]["value"], 0.4)
        self.assertEqual(res["table"]["aesthetic_quality"]["Gen2"]["n"], 159)
        self.assertAlmostEqual(res["paired_difference"][f"imaging_quality|{wan} - Floor33"]["value"], -0.01)
        md = Path("results/video/summary.md").read_text()
        self.assertIn("(n=159)", md)

    def test_extra_model(self):
        """A second run of ours (LTX-Video, D-039) becomes a column after Wan2.2, with its timing."""
        vg.select()
        prompts = ve.wan_prompts()
        wan, ltx = "wan22_f49", "ltxv095_f49"
        self.assertEqual(vg.model_name(49, "ltxv095"), ltx)
        models = [wan, ltx] + list(gv.MODELS)
        d = Path("data/scores/vqascore_video")
        d.mkdir(parents=True)
        with open(d / "x.jsonl", "w") as f:
            for i, m in enumerate(models):
                for p in prompts:
                    f.write(json.dumps({"sample_id": make_sample_id(m, p["prompt_id"], 0), "metric": "vqascore_video",
                                        "metric_version": "v", "value": 0.9 - 0.1 * i, "status": "ok",
                                        "created_at": "2026-10-25"}) + "\n")
        mdir = Path("data/video/manifests")
        mdir.mkdir(parents=True)
        for m, sec in ((wan, 65.0), (ltx, 12.0)):
            (mdir / f"gen_{m}_shard000of001.jsonl").write_text("".join(json.dumps(
                {"sample_id": make_sample_id(m, p["prompt_id"], 0), "model": m, "prompt_id": p["prompt_id"],
                 "path": f"v/{m}/{p['prompt_id']}.mp4", "status": "ok", "gen_seconds": sec, "frames": 49,
                 "width": 1280 if m == wan else 704, "height": 704 if m == wan else 480}) + "\n" for p in prompts))
        self.assertEqual(len(ve.video_paths(ltx, prompts)), len(prompts))
        res = ve.analyze(wan, n_boot=50, extra=(ltx,))
        self.assertEqual(res["models"][:2], [wan, ltx])
        self.assertAlmostEqual(res["paired_difference"][f"vqascore_video|{wan} - {ltx}"]["value"], 0.1)
        self.assertAlmostEqual(res["generation"][ltx]["mean_gen_seconds"], 12.0)
        self.assertIn("704 x 480", Path("results/video/summary.md").read_text())
        with self.assertRaises(ValueError):
            vg.generate(48, model="ltxv095")                                     # LTX needs 8k + 1 frames


if __name__ == "__main__":
    unittest.main()
