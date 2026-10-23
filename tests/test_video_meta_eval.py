"""Video meta-evaluation (genprobe/genai_bench_video.py, genprobe/video_meta_eval.py, D-035): the
prompt and video manifests, the frame layout, per-video score sets, H4/H5 and the refusal to compute
on partial scores. No video is decoded here (the frames step needs PyAV and runs on HPC3)."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from genprobe import genai_bench_video as gv
from genprobe import video_meta_eval as vme
from genprobe.meta_eval import IncompleteScores
from genprobe.samples import make_sample_id


class TestVideoMetaEval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.folder = Path("data/external/genai_bench_video")
        self.folder.mkdir(parents=True)
        n = 30
        data = {}
        for i in range(n):
            key = f"{i:05d}"
            # human ratings: Floor33 best, then Gen2, Pika_v1, Modelscope; prompt 0 has a tie
            r = {"Floor33": [5, 5, 4], "Gen2": [4, 4, 3], "Pika_v1": [3, 3, 2], "Modelscope": [2, 1, 1]}
            if i == 0:
                r["Gen2"] = [5, 5, 4]
            if i == 1:
                r["Modelscope"] = [2, 1, 7]                    # an out-of-range rating is dropped
            data[key] = {"id": key, "prompt": f"video prompt {i}", "models": r}
            for m in gv.MODELS:
                (self.folder / m).mkdir(exist_ok=True)
                (self.folder / m / f"{key}.mp4").write_bytes(b"x")
        (self.folder / "genai_video.json").write_text(json.dumps(data))
        (self.folder / "genai_skills.json").write_text(json.dumps(
            {"basic": list(range(15)), "advanced": list(range(15, 30)), "counting": list(range(10)),
             "attribute": list(range(10, 30))}))
        Path("data/prompts").mkdir(parents=True)
        Path("data/manifests").mkdir(parents=True)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def write_scores(self, metric, recs, version="v1"):
        d = Path("data/scores") / metric
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "x.jsonl", "w") as f:
            for sid, v in recs.items():
                f.write(json.dumps({"sample_id": sid, "metric": metric, "metric_version": version, "value": v,
                                    "status": "ok", "created_at": "2026-10-23T00:00:00"}) + "\n")

    def test_frame_indices(self):
        self.assertEqual(gv.frame_indices(36), [0, 4, 9, 13, 18, 22, 26, 31, 35])
        self.assertEqual(gv.frame_indices(9), list(range(9)))
        self.assertEqual(gv.frame_indices(0), [])

    def test_prepare_and_run(self):
        res = gv.prepare()
        self.assertEqual((res["prompts"], res["videos"], res["missing"]), (30, 120, 0))
        self.assertEqual(len(res["dropped_ratings"]), 1)
        rows = [json.loads(x) for x in Path(gv.PROMPTS).read_text().splitlines()]
        self.assertEqual(rows[0]["prompt_id"], "gv0000")
        self.assertEqual(rows[0]["tags"][0], "basic")
        videos = [json.loads(x) for x in Path(gv.MANIFEST).read_text().splitlines()]
        # frames manifests as `frames` would write them (no decoding here)
        fr = []
        for v in videos:
            for k in range(gv.N_FRAMES):
                fr.append({"sample_id": make_sample_id(v["model"], v["prompt_id"], gv.FRAME_SEED + k), "model": v["model"],
                           "prompt_id": v["prompt_id"], "status": "ok", "video": v["sample_id"], "frame": k})
        rank = {"Floor33": 4, "Gen2": 3, "Pika_v1": 2, "Modelscope": 1}
        # vqascore_video orders videos like people; clipscore frames reversed; mid-frame VQAScore flat
        self.write_scores("vqascore_video", {v["sample_id"]: rank[v["model"]] / 10 for v in videos})
        mid = {f["sample_id"]: 0.5 for f in fr if f["frame"] == gv.MID}
        self.write_scores("vqascore", mid)
        self.write_scores("clipscore", {f["sample_id"]: (5 - rank[f["model"]]) / 10 + f["frame"] / 100 for f in fr})
        self.write_scores("pickscore", {f["sample_id"]: 20.0 for f in fr})
        partial = {f["sample_id"]: 0.3 for f in fr[:-1]}                      # one frame missing
        self.write_scores("hpsv2", partial)
        with self.assertRaises(IncompleteScores):
            vme.run(n_boot=100)
        self.write_scores("hpsv2", {f["sample_id"]: 0.3 for f in fr})
        out = vme.run(n_boot=100)
        h4, h5 = out["hypotheses"]["H4"], out["hypotheses"]["H5"]
        self.assertAlmostEqual(h4["agreement_a"], 1.0)
        self.assertAlmostEqual(h4["agreement_b"], 0.0)
        self.assertEqual(h4["verdict"], "supported")
        self.assertAlmostEqual(h5["agreement_b"], 0.5)                       # equal scores count 1/2
        self.assertEqual(h5["verdict"], "supported")
        self.assertEqual(out["pairs_total"], 30 * 6)
        self.assertEqual(out["pairs_tied_excluded"], 1)
        self.assertTrue((vme.OUT / "hypotheses.md").exists())
        self.assertIn("H4", (vme.OUT / "hypotheses.md").read_text())


if __name__ == "__main__":
    unittest.main()
