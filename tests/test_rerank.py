"""Re-ranking curve (genprobe/rerank.py, D-034): which image best-of-N picks, what it costs, which
prompts need new seeds, and the analysis against module C's groups."""

import json
import tempfile
import unittest
from pathlib import Path

from genprobe import rerank as rr
from genprobe.agent import core
from genprobe.samples import load_prompts, make_sample_id

PROMPTS = "data/prompts/prompts_v1.jsonl"
COSTS = {"gen": {"sdxl": 3.0, "sd35m": 4.6, "flux_schnell": 1.1}, "score": {"vqascore": 0.85, "detect": 5.5, "ocr": 2.3}}


class TestPick(unittest.TestCase):
    def test_pick(self):
        sc = [0.5, 0.7, 0.7, 0.2, 0.9, 0.1, 0.1, 0.1, 0.95]
        self.assertEqual([rr.pick(sc, n) for n in rr.NS], [0, 1, 1, 8])     # ties keep the lower seed

    def test_cost(self):
        self.assertAlmostEqual(rr.cost(1, "sdxl", COSTS), 3.0)               # nothing to score at N = 1
        self.assertAlmostEqual(rr.cost(4, "sdxl", COSTS), 4 * 3.85)
        self.assertAlmostEqual(rr.cost(9, "flux_schnell", COSTS), 9 * 1.95)

    def test_prompt_files(self):
        with tempfile.TemporaryDirectory() as d:
            counts = rr.write_prompt_files(PROMPTS, per_category=5, out=Path(d))
            self.assertEqual(counts, {"flux_schnell": 10, "sd35m": 10, "sdxl": 20})
            ps = load_prompts(Path(d) / "prompts_sd35m.jsonl")
            self.assertTrue(all(core.category_of(p.tags) in ("counting", "spatial") for p in ps))
            self.assertTrue(all(p.split == "test" for p in ps))
            full = rr.write_prompt_files(PROMPTS, out=Path(d))
            self.assertEqual(full["sdxl"], 480)
            self.assertEqual(full["sd35m"] + full["flux_schnell"], 480)


class TestAnalyze(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.prompts = rr.test_prompts(PROMPTS, per_category=5)

    def tearDown(self):
        self.tmp.cleanup()

    def fake(self):
        paths, vqa = {}, {}
        for i, p in enumerate(self.prompts):
            for curve in rr.CURVES:
                m = rr.model_for(curve, core.category_of(p.tags))
                for s in range(9):
                    sid = make_sample_id(m, p.prompt_id, s)
                    paths[sid] = f"img/{sid}.png"
                    vqa[sid] = ((i * 7 + s * 3) % 10) / 10 + (0.05 if m != "sdxl" else 0)
        return paths, vqa

    def test_picks_and_analysis(self):
        paths, vqa = self.fake()
        rows, missing = rr.make_picks(self.prompts, paths, vqa)
        self.assertEqual(missing, [])
        self.assertEqual(len(rows), len(self.prompts) * len(rr.CURVES) * len(rr.NS))
        for r in rows:                                        # the pick is the best of the first N
            sids = [make_sample_id(r["model"], r["prompt_id"], s) for s in range(r["n"])]
            self.assertEqual(r["vqascore"], max(vqa[s] for s in sids))
            self.assertLess(r["seed"], r["n"])
        del vqa[make_sample_id("sdxl", self.prompts[0].prompt_id, 7)]
        _, missing = rr.make_picks(self.prompts, paths, vqa)
        self.assertEqual(len(missing), 1)
        vqa[make_sample_id("sdxl", self.prompts[0].prompt_id, 7)] = 0.0

        data, root = self.d / "data", self.d / "data" / "agent"
        (data / "judge").mkdir(parents=True)
        picks = data / "picks.jsonl"
        picks.write_text("".join(json.dumps(r) + "\n" for r in rows))
        # judge: passes every routed image, and SDXL images of seeds 2 and up
        grades = {r["sample_id"]: 5 if (r["curve"] == "routed" or r["seed"] >= 2) else 2 for r in rows}
        expect = {f"{c}@{n}": sum(grades[r["sample_id"]] == 5 for r in rows if r["group"] == f"{c}@{n}") / len(self.prompts)
                  for c in rr.CURVES for n in rr.NS}
        version = "vllm/gemma-4-31b/test"
        (data / "judge" / "g.jsonl").write_text("".join(json.dumps({"sample_id": s, "score": v}) + "\n"
                                                        for s, v in grades.items()))
        (data / "judge" / "g.meta.json").write_text(json.dumps({"judge_version": version}))
        for run, tag in rr.AGENT_RUNS.items():
            (root / run).mkdir(parents=True)
            (root / run / "state.json").write_text(json.dumps({"costs": COSTS, "states": []}))
            final, g = [], []
            for p in self.prompts:
                for grp in ("single", "best_of_4", "routed_best_of_k", "agent_rule" if run == "testrule" else "agent_rule_sdxl"):
                    sid = f"{grp}__{p.prompt_id}"
                    final.append({"prompt_id": p.prompt_id, "group": grp, "sample_id": sid, "category":
                                  core.category_of(p.tags), "gpu_seconds": 10.0})
                    g.append({"sample_id": sid, "score": 5 if grp == "agent_rule" else 1})
            (root / run / "final.jsonl").write_text("".join(json.dumps(x) + "\n" for x in final))
            (data / "judge" / f"{tag}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in g))
            (data / "judge" / f"{tag}.meta.json").write_text(json.dumps({"judge_version": version}))
        res = rr.analyze(str(picks), "g", data=str(data), out=str(self.d / "out"), root=root, n_boot=200, figure=False)
        for grp, v in expect.items():
            self.assertAlmostEqual(res["pass_rate"][grp]["all"]["value"], v)
        self.assertEqual(res["pass_rate"]["sdxl@1"]["all"]["value"], 0.0)
        self.assertEqual(res["pass_rate"]["routed@1"]["all"]["value"], 1.0)
        self.assertEqual(res["pass_rate"]["agent_rule"]["all"]["value"], 1.0)
        self.assertEqual(res["pass_rate"]["best_of_4"]["all"]["value"], 0.0)
        self.assertAlmostEqual(res["gpu_seconds"]["sdxl@4"], 4 * 3.85)
        self.assertAlmostEqual(res["paired_difference"]["sdxl@9 - sdxl@1|all"]["value"], expect["sdxl@9"])
        self.assertAlmostEqual(res["paired_difference"]["agent_rule - sdxl@4|all"]["value"], 1 - expect["sdxl@4"])
        self.assertAlmostEqual(res["paired_difference"]["routed@1 - sdxl@1|all"]["value"], 1.0)
        md = (self.d / "out" / "summary.md").read_text()
        self.assertIn("routed@9", md)
        # a different judge version for module C is refused
        (data / "judge" / "grade_testrule_gemma4_31b.meta.json").write_text(json.dumps({"judge_version": "other"}))
        with self.assertRaises(SystemExit):
            rr.analyze(str(picks), "g", data=str(data), out=str(self.d / "out2"), root=root, n_boot=50, figure=False)


if __name__ == "__main__":
    unittest.main()
