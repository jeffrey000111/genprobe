"""Diffusion-DPO data and evaluation (genprobe/dpo.py, D-037): no evaluation prompt is trained on,
which pairs are kept, which checkpoint dev picks, and H6. No model is loaded."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from genprobe import dpo
from genprobe.samples import load_prompts, make_sample_id

REPO = Path(__file__).resolve().parents[1]


class TestDPO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        Path("data/prompts").mkdir(parents=True)
        v1 = (REPO / dpo.V1).read_text()
        Path(dpo.V1).write_text(v1)
        v1_ids = [json.loads(x)["prompt_id"] for x in v1.splitlines() if x.strip()]
        gb = [i for i in v1_ids if i.startswith("gb")]
        extra = [f"gb{i:04d}" for i in range(1600) if f"gb{i:04d}" not in set(gb)][:60]
        rows = [{"prompt_id": p, "prompt": f"prompt {p}", "tags": ["basic", "scene"], "split": "test"} for p in gb + extra]
        Path(dpo.GENAI_1600).write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.extra = extra

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_train_prompts_exclude_v1(self):
        res = dpo.write_train_prompts()
        self.assertEqual(res["prompts"], 60)
        ids = {p.prompt_id for p in load_prompts(dpo.train_prompts_path(False))}
        self.assertEqual(ids, set(self.extra))
        self.assertFalse(ids & {p.prompt_id for p in load_prompts(dpo.V1)})
        self.assertEqual(dpo.write_train_prompts(smoke=True)["prompts"], dpo.SMOKE_N)

    def test_pairs(self):
        images = {"a": [("s0", 0.2, "p0"), ("s1", 0.9, "p1"), ("s2", 0.5, "p2")],
                  "b": [("t0", 0.50, "q0"), ("t1", 0.55, "q1")],          # gap 0.05 < 0.1: dropped
                  "c": [("u0", 0.3, "r0")]}                                 # one image: dropped
        pairs, stats = dpo.make_pairs(images)
        self.assertEqual([(p["prompt_id"], p["win"], p["lose"]) for p in pairs], [("a", "s1", "s0")])
        self.assertEqual((stats["pairs"], stats["dropped"]), (1, 2))

    def write_eval(self, arms, split, seeds, limit=None):
        prompts = dpo.eval_prompts(split, limit)
        d = Path("data/dpo/eval/manifests")
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"gen_{split}.jsonl", "a") as f:
            for arm in arms:
                for p in prompts:
                    for s in seeds:
                        f.write(json.dumps({"sample_id": make_sample_id(arm, p.prompt_id, s), "model": arm,
                                            "prompt_id": p.prompt_id, "prompt": p.prompt, "seed": s,
                                            "path": f"img/{arm}/{p.prompt_id}_{s}.png", "status": "ok"}) + "\n")
        return prompts

    def write_grades(self, tag, scores):
        Path("data/judge").mkdir(parents=True, exist_ok=True)
        Path(f"data/judge/{tag}.jsonl").write_text("".join(
            json.dumps({"sample_id": s, "score": v, "expected_score": float(v)}) + "\n" for s, v in scores.items()))

    def test_select_and_analyze(self):
        arms = ["dpo_base", "dpo_s250", "dpo_s500", "dpo_s1000", "dpo_s2000"]
        dev = self.write_eval(arms, "dev", [0], limit=3)
        # s500 and s1000 tie at the top: fewer steps wins
        rate = {"dpo_base": 0.0, "dpo_s250": 0.25, "dpo_s500": 0.5, "dpo_s1000": 0.5, "dpo_s2000": 0.0}
        grades = {}
        for arm in arms:
            for i, p in enumerate(dev):
                grades[make_sample_id(arm, p.prompt_id, 0)] = 5 if i < rate[arm] * len(dev) else 3
        self.write_grades("gdev", grades)
        res = dpo.select("gdev", arms, limit=3, results=Path("results/dpo"))
        self.assertEqual(res["selected"], "dpo_s500")
        self.assertEqual(json.loads(Path("data/dpo/selected.json").read_text())["selected"], "dpo_s500")

        test = self.write_eval(["dpo_base", "dpo_s500"], "test", [0, 1, 2, 3], limit=3)
        g = {}
        for i, p in enumerate(test):
            g[make_sample_id("dpo_base", p.prompt_id, 0)] = 5 if i % 3 == 0 else 2
            g[make_sample_id("dpo_s500", p.prompt_id, 0)] = 5 if i % 3 != 2 else 2
        self.write_grades("gtest", g)
        sd = Path("data/dpo/eval/scores/vqascore")
        sd.mkdir(parents=True)
        with open(sd / "x.jsonl", "w") as f:
            for arm, v in (("dpo_base", 0.4), ("dpo_s500", 0.7)):
                for p in test:
                    for s in range(4):
                        f.write(json.dumps({"sample_id": make_sample_id(arm, p.prompt_id, s), "metric": "vqascore",
                                            "metric_version": "v", "value": v, "status": "ok",
                                            "created_at": "2026-10-23"}) + "\n")
        out = dpo.analyze("gtest", "dpo_base", "dpo_s500", limit=3, results=Path("results/dpo"), n_boot=200)
        n = len(test)
        base = sum(1 for i in range(n) if i % 3 == 0) / n
        new = sum(1 for i in range(n) if i % 3 != 2) / n
        self.assertAlmostEqual(out["pass_rate"]["dpo_base"]["all"]["value"], base)
        self.assertAlmostEqual(out["H6"]["value"], new - base)
        self.assertAlmostEqual(out["metrics"]["vqascore"]["difference"]["value"], 0.3)
        self.assertIn("H6", Path("results/dpo/summary_smoke.md").read_text())

    def test_training_record(self):
        Path("data/dpo").mkdir(parents=True, exist_ok=True)
        Path("data/dpo/pairs.meta.json").write_text(json.dumps(
            {"prompts": 4, "candidates": 16, "scored": 16, "pairs": 3, "dropped": 1, "min_gap": 0.1, "mean_gap": 0.4,
             "vqascore_version": "v", "pairs_list": []}))
        lines = ["Steps:   0%|          | 0/4 [00:00<?, ?it/s]"]
        for step, acc, loss in ((1, 0.5, 0.69), (2, 0.75, 0.6), (3, 1.0, 0.5), (4, 0.5, 0.7)):
            lines.append(f"Steps:  {25 * step}%|##        | {step}/4 [00:0{step}<00:01, 2.10s/it, implicit_acc={acc}, "
                         f"loss={loss}, lr=1e-5, raw_model_loss=0.1, ref_loss=0.1]")
        Path("train.log").write_text("\r".join(lines) + "\nINFO done\n")
        res = dpo.train_log("train.log", window=2, results=Path("results/dpo"))
        self.assertEqual((res["steps"], res["logged"], res["pairs"]["pairs"]), (4, 4, 3))
        self.assertEqual([w["steps"] for w in res["windows"]], [[1, 2], [3, 4]])
        self.assertAlmostEqual(res["windows"][0]["implicit_acc"], 0.625)
        self.assertAlmostEqual(res["windows"][1]["loss"], 0.6)
        self.assertIn("| 3–4 | 2 |", Path("results/dpo/training.md").read_text())


if __name__ == "__main__":
    unittest.main()
