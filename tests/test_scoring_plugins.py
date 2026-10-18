"""CLIPScore / VQAScore plugins, the HF cache helpers, and the summary step.

The real models only run in a GPU job on HPC3. These tests cover everything around them that
can go wrong silently: the attention override, template versioning, cache checks, token
counting, sample selection, and the summary tables.
"""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

from genprobe.backends import FakeBackend, _version_tuple, dtype_kwarg_name
from genprobe.generate import run_generation
from genprobe.metrics import REGISTRY
from genprobe.metrics import clipscore, vqascore
from genprobe.score import load_samples
from genprobe.summarize import bootstrap_mean_ci, pick_scores, spearman, summarize
from genprobe.utils.hf import cached_revision, require_cached
from tests.helpers import make_cfg, make_prompts, quiet


class TestRegistry(unittest.TestCase):
    def test_plugins_registered(self):
        self.assertIn("clipscore", REGISTRY)
        self.assertIn("vqascore", REGISTRY)

    def test_import_stays_light(self):
        # Importing the metrics package must not pull in torch/open_clip/t2v_metrics.
        code = ("import sys, genprobe.metrics, genprobe.score; "
                "print(sorted(m for m in ('torch', 'open_clip', 't2v_metrics') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(out.stdout.strip(), "[]")


class TestVQAScoreConfig(unittest.TestCase):
    def test_template_is_part_of_version(self):
        a = vqascore.template_id()
        self.assertEqual(a, vqascore.template_id(vqascore.QUESTION_TEMPLATE, vqascore.ANSWER_TEMPLATE))
        self.assertNotEqual(a, vqascore.template_id("Is this {}?", "Yes"))
        self.assertNotEqual(a, vqascore.template_id(vqascore.QUESTION_TEMPLATE, "yes"))
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"HF_HOME": d}, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            v = vqascore.VQAScore().version
        self.assertTrue(v.startswith("vqascore-port-of-t2v_metrics-qwen2vl/transformers=="), v)
        self.assertIn("qwen2.5-vl-7b@nocache", v)
        self.assertIn(f"/sdpa/tmpl-{a}", v)

    def test_messages_match_t2v_metrics(self):
        img = object()
        self.assertEqual(vqascore.build_messages(img, "Q?"),
                         [{"role": "user", "content": [{"type": "image", "image": img},
                                                       {"type": "text", "text": "Q?"}]}])
        self.assertEqual(vqascore.QUESTION_TEMPLATE.format("two dogs"),
                         'Does this figure show "two dogs"? Please answer Yes or No.')

    def test_single_token_answers_only(self):
        tok = types.SimpleNamespace(encode=lambda t, add_special_tokens: [7] if t == "Yes" else [1, 2])
        self.assertEqual(vqascore.single_token_id(tok, "Yes"), 7)
        with self.assertRaisesRegex(ValueError, "single-token"):
            vqascore.single_token_id(tok, "Absolutely yes")

    def test_score_one_takes_softmax_of_first_generated_step(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed here")
        m = vqascore.VQAScore()
        logits = torch.full((1, 10), -1e4)
        logits[0, 3], logits[0, 4] = 0.0, 0.0          # 'Yes' and 'No' equally likely
        calls = {}

        class Inputs(dict):
            def to(self, device):
                return self

        def generate(**kw):
            calls.update(kw)
            return types.SimpleNamespace(scores=(logits,))

        m.model = types.SimpleNamespace(generate=generate)
        m.processor = mock.Mock()
        m.processor.apply_chat_template.return_value = "<prompt>assistant\n"
        m.processor.return_value = Inputs(input_ids=torch.zeros(1, 5), image_grid_thw=torch.tensor([[1, 4, 4]]))
        m.processor.tokenizer.decode.return_value = "Yes"
        m.process_vision_info = lambda messages: (["img"], None)
        m.yes_id, m.no_id, m.device = 3, 4, "cpu"
        value, extra = m.score_one("PIL", "two dogs")
        self.assertAlmostEqual(value, 0.5, places=5)
        self.assertAlmostEqual(extra["p_no"], 0.5, places=5)
        self.assertEqual(extra["n_image_tokens"], 4)
        self.assertEqual((calls["max_new_tokens"], calls["do_sample"], calls["output_scores"]), (1, False, True))
        kw = m.processor.call_args.kwargs
        self.assertEqual((kw["do_resize"], kw["images"], kw["videos"]), (False, ["img"], None))


def fake_hf_cache(home: Path, repo: str, files: tuple[str, ...], sha: str = "cc5948981374") -> None:
    d = home / "hub" / f"models--{repo.replace('/', '--')}"
    (d / "refs").mkdir(parents=True)
    (d / "refs" / "main").write_text(sha)
    snap = d / "snapshots" / sha
    snap.mkdir(parents=True)
    for f in files:
        (snap / f).write_text("{}")


def fake_torch(cuda: bool = True) -> types.ModuleType:
    t = types.ModuleType("torch")
    t.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    t.bfloat16, t.float32 = "bf16", "fp32"
    return t


class TestVQAScoreSetupWithFakes(unittest.TestCase):
    """setup() against stand-ins for transformers / qwen_vl_utils: every check must fire when
    its assumption breaks, and nothing loads before the cache check passes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"HF_HOME": str(self.home)})
        self.env.start()
        os.environ.pop("HF_HUB_CACHE", None)
        self.loading = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        self.attn = "sdpa"
        self.loads = []
        test = self

        class FakeModel:
            def __init__(self):
                self.config = types.SimpleNamespace(_attn_implementation=test.attn)
                self.generation_config = types.SimpleNamespace(repetition_penalty=1.05, temperature=0.1,
                                                               top_k=1, top_p=0.001, do_sample=True)

            def to(self, device):
                return self

            def eval(self):
                return self

        class FakeQwen:
            @classmethod
            def from_pretrained(cls, repo, **kw):
                test.loads.append((repo, kw))
                return FakeModel(), test.loading

        tokenizer = types.SimpleNamespace(encode=lambda t, add_special_tokens: {"Yes": [9454], "No": [2753]}[t])
        tf = types.ModuleType("transformers")
        tf.Qwen2_5_VLForConditionalGeneration = FakeQwen
        tf.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda repo: types.SimpleNamespace(tokenizer=tokenizer))
        qvu = types.ModuleType("qwen_vl_utils")
        qvu.process_vision_info = lambda messages: ([], None)
        self.mods = mock.patch.dict(sys.modules, {"torch": fake_torch(), "transformers": tf, "qwen_vl_utils": qvu})
        self.mods.start()

    def tearDown(self):
        self.mods.stop()
        self.env.stop()
        self.tmp.cleanup()

    def cache(self):
        fake_hf_cache(self.home, vqascore.HF_REPO, vqascore.REQUIRED_FILES)

    def test_loads_bf16_sdpa_and_records_info(self):
        self.cache()
        m = vqascore.VQAScore()
        m.setup("cuda")
        repo, kw = self.loads[0]
        self.assertEqual(repo, vqascore.HF_REPO)
        self.assertEqual((kw["torch_dtype"], kw["attn_implementation"], kw["output_loading_info"]),
                         ("bf16", "sdpa", True))
        self.assertEqual((m.yes_id, m.no_id), (9454, 2753))
        self.assertEqual(m.info["revision"], "cc5948981374")
        self.assertEqual(m.info["generation_config"]["repetition_penalty"], 1.05)

    def test_missing_weights_say_how_to_download(self):
        with self.assertRaisesRegex(FileNotFoundError, "download_models.sh qwen25vl7b"):
            vqascore.VQAScore().setup("cuda")
        self.assertEqual(self.loads, [])

    def test_partial_load_fails(self):
        self.cache()
        self.loading["missing_keys"] = ["model.layers.0.mlp.up_proj.weight"]
        with self.assertRaisesRegex(RuntimeError, "did not load cleanly"):
            vqascore.VQAScore().setup("cuda")

    def test_unexpected_keys_are_only_recorded(self):
        self.cache()
        self.loading["unexpected_keys"] = ["some.extra.tensor"]
        m = vqascore.VQAScore()
        m.setup("cuda")
        self.assertEqual(m.info["unexpected_keys"], 1)

    def test_wrong_attention_fails(self):
        self.cache()
        self.attn = "eager"
        with self.assertRaisesRegex(RuntimeError, "attention 'eager'"):
            vqascore.VQAScore().setup("cuda")

    def test_refuses_without_gpu(self):
        sys.modules["torch"] = fake_torch(cuda=False)
        with self.assertRaisesRegex(RuntimeError, "login node"):
            vqascore.VQAScore().setup("cuda")


class TestCLIPScoreSetupWithFakes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)
        self.env = mock.patch.dict(os.environ, {"HF_HOME": str(self.home)})
        self.env.start()
        os.environ.pop("HF_HUB_CACHE", None)
        fake_hf_cache(self.home, clipscore.HF_REPO, ("open_clip_model.safetensors",))
        self.cfg = {"hf_hub": clipscore.HF_REPO + "/", "quick_gelu": True}
        self.build_quick = None   # None: follow force_quick_gelu; True/False: ignore it
        self.warn = None
        test = self

        class QuickGELU:
            pass

        class GELU:
            pass

        class Model:
            def __init__(self, quick):
                self.quick = quick

            def modules(self):
                return [self, QuickGELU() if self.quick else GELU()]

            def eval(self):
                return self

        def create(arch, pretrained, device, force_quick_gelu):
            if test.warn:
                warnings.warn(test.warn, UserWarning)
            q = force_quick_gelu if test.build_quick is None else test.build_quick
            return Model(q), None, "preprocess"

        oc = types.ModuleType("open_clip")
        oc.get_pretrained_cfg = lambda arch, tag: self.cfg
        oc.create_model_and_transforms = create
        oc.get_tokenizer = lambda arch: types.SimpleNamespace(context_length=77)
        self.mods = mock.patch.dict(sys.modules, {"torch": fake_torch(), "open_clip": oc})
        self.mods.start()

    def tearDown(self):
        self.mods.stop()
        self.env.stop()
        self.tmp.cleanup()

    def test_forces_quick_gelu_for_openai_weights(self):
        m = clipscore.CLIPScore()
        m.setup("cuda")
        self.assertTrue(m.info["quick_gelu"])
        self.assertEqual(m.context_length, 77)

    def test_activation_mismatch_fails(self):
        self.build_quick = False
        with self.assertRaisesRegex(RuntimeError, "activation mismatch"):
            clipscore.CLIPScore().setup("cuda")

    def test_open_clip_quickgelu_warning_is_an_error(self):
        self.warn = "These pretrained weights were trained with QuickGELU activation but the model config does not have that enabled."
        with self.assertRaises(UserWarning):
            clipscore.CLIPScore().setup("cuda")

    def test_weights_repo_change_fails(self):
        self.cfg["hf_hub"] = "timm/some_other_repo/"
        with self.assertRaisesRegex(RuntimeError, "update HF_REPO"):
            clipscore.CLIPScore().setup("cuda")


class TestCLIPScoreHelpers(unittest.TestCase):
    def test_token_count_includes_start_end(self):
        tok = mock.Mock()
        tok.encode = lambda text: text.split()
        self.assertEqual(clipscore.n_clip_tokens(tok, "a photo of two dogs"), 7)

    def test_token_count_unknown(self):
        self.assertIsNone(clipscore.n_clip_tokens(object(), "x"))

    def test_version(self):
        v = clipscore.CLIPScore().version
        self.assertTrue(v.startswith("open_clip=="))
        self.assertIn("ViT-L-14:openai/cos", v)


class TestHFCache(unittest.TestCase):
    def test_cached_revision_and_require(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"HF_HOME": d}, clear=False):
            os.environ.pop("HF_HUB_CACHE", None)
            self.assertIsNone(cached_revision("org/model"))
            with self.assertRaisesRegex(FileNotFoundError, "download_models.sh x"):
                require_cached("org/model", ("config.json",), "download_models.sh x")
            repo = Path(d) / "hub" / "models--org--model"
            (repo / "refs").mkdir(parents=True)
            (repo / "refs" / "main").write_text("abc123\n")
            snap = repo / "snapshots" / "abc123"
            snap.mkdir(parents=True)
            self.assertEqual(cached_revision("org/model"), "abc123")
            with self.assertRaisesRegex(FileNotFoundError, "config.json"):
                require_cached("org/model", ("config.json",))
            (snap / "config.json").write_text("{}")
            self.assertEqual(require_cached("org/model", ("config.json",)), "abc123")


class TestDiffusersDtypeKwarg(unittest.TestCase):
    def test_kwarg_name(self):
        self.assertEqual(dtype_kwarg_name(_version_tuple("0.41.0")), "dtype")
        self.assertEqual(dtype_kwarg_name(_version_tuple("0.42.0.dev0")), "dtype")
        self.assertEqual(dtype_kwarg_name(_version_tuple("1.0.0")), "dtype")
        self.assertEqual(dtype_kwarg_name(_version_tuple("0.39.2")), "torch_dtype")


class TestStats(unittest.TestCase):
    def test_bootstrap_constant(self):
        mean, lo, hi = bootstrap_mean_ci({"a": [0.5], "b": [0.5, 0.5]})
        self.assertEqual((mean, lo, hi), (0.5, 0.5, 0.5))

    def test_bootstrap_averages_seeds_first(self):
        # prompt a: seeds 0 and 1 -> mean 0.5; prompt b: 1.0. Mean over prompts = 0.75.
        mean, lo, hi = bootstrap_mean_ci({"a": [0.0, 1.0], "b": [1.0]})
        self.assertAlmostEqual(mean, 0.75)
        self.assertTrue(0.5 <= lo <= mean <= hi <= 1.0)

    def test_spearman(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
        self.assertTrue(math.isnan(spearman([1, 1, 1], [1, 2, 3])))  # constant input

    def test_pick_scores_uses_one_version(self):
        recs = [
            {"sample_id": "s1", "metric_version": "v1", "status": "ok", "value": 0.1, "created_at": "2026-01-01"},
            {"sample_id": "s1", "metric_version": "v2", "status": "ok", "value": 0.9, "created_at": "2026-02-01"},
            {"sample_id": "s2", "metric_version": "v1", "status": "ok", "value": 0.2, "created_at": "2026-01-01"},
            {"sample_id": "s3", "metric_version": "v2", "status": "ok", "value": 0.3, "created_at": "2026-02-01"},
        ]
        ok, version, info = pick_scores(recs, {"s1", "s2", "s3"})
        self.assertEqual(version, "v2")
        self.assertEqual({k: r["value"] for k, r in ok.items()}, {"s1": 0.9, "s3": 0.3})
        self.assertEqual(info["versions_seen"], 2)


class TestSelectionAndSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        prompts = make_prompts(4)
        prompts[2].tags = ["spatial"]
        self.ppath = self.out / "prompts.jsonl"
        self.ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
        for name in ("toy", "toy2"):
            cfg = make_cfg(name=name, seeds=[0], batch_size=2)
            run_generation(cfg=cfg, prompts=prompts, out=self.out, backend=FakeBackend(cfg),
                           backend_kind="fake", log=quiet)
        self.manifests = str(self.out / "manifests" / "gen_*.jsonl")
        # a second prompt set generated later must not leak into this one
        other = make_prompts(6)[4:]
        cfg = make_cfg(name="toy", seeds=[0], batch_size=2)
        run_generation(cfg=cfg, prompts=other, out=self.out, backend=FakeBackend(cfg),
                       backend_kind="fake", log=quiet)

    def tearDown(self):
        self.tmp.cleanup()

    def test_prompts_file_restricts_samples(self):
        self.assertEqual(len(load_samples(self.manifests)), 10)
        sel = load_samples(self.manifests, str(self.ppath))
        self.assertEqual(len(sel), 8)
        self.assertEqual(sel, sorted(sel, key=lambda s: s.sample_id))
        self.assertEqual({s.tags[0] for s in sel}, {"counting", "spatial"})

    def write_scores(self, metric, value_fn, version="v1"):
        d = self.out / "scores" / metric
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "t_shard000of001.jsonl", "w") as f:
            for s in load_samples(self.manifests):
                f.write(json.dumps({"sample_id": s.sample_id, "metric": metric, "metric_version": version,
                                    "value": value_fn(s), "extra": {"truncated": s.prompt_id == "p000"},
                                    "seconds": 0.1, "status": "ok", "created_at": "2026-10-18T00:00:00"}) + "\n")

    def test_summary_tables(self):
        self.write_scores("clipscore", lambda s: 0.2 if s.model == "toy" else 0.3)
        self.write_scores("vqascore", lambda s: 0.5 + int(s.prompt_id[1:]) / 10)
        res = summarize(prompts_path=str(self.ppath), manifests=self.manifests, data=str(self.out),
                        metrics=["clipscore", "vqascore"], out=str(self.out / "res"), full_run_images=2400)
        self.assertEqual(res["models"], ["toy", "toy2"])
        self.assertEqual(res["metrics"]["clipscore"]["n_ok"], 8)
        self.assertEqual(res["metrics"]["clipscore"]["n_truncated"], 2)
        self.assertAlmostEqual(res["by_model"]["clipscore"]["toy"]["mean"], 0.2)
        self.assertAlmostEqual(res["by_model"]["clipscore"]["toy2"]["mean"], 0.3)
        self.assertEqual(res["by_model"]["vqascore"]["toy"]["n_prompts"], 4)
        self.assertAlmostEqual(res["by_model_category"]["vqascore"]["toy"]["spatial"]["mean"], 0.7)
        self.assertEqual(res["generation_cost"]["toy"]["n_images"], 4)
        for f in ("summary.md", "summary.json", "per_sample.csv"):
            self.assertTrue((self.out / "res" / f).is_file(), f)
        with open(self.out / "res" / "per_sample.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 8)
        self.assertEqual({r["clip_truncated"] for r in rows}, {"True", "False"})
        md = (self.out / "res" / "summary.md").read_text()
        self.assertIn("| toy | 0.200 [0.200, 0.200] (n=4)", md)

    def test_summary_flags_missing_scores(self):
        self.write_scores("clipscore", lambda s: 0.2)
        res = summarize(prompts_path=str(self.ppath), manifests=self.manifests, data=str(self.out),
                        metrics=["clipscore", "vqascore"], out=str(self.out / "res"), full_run_images=2400)
        self.assertEqual(res["metrics"]["vqascore"]["n_ok"], 0)
        self.assertIsNone(res["metrics"]["vqascore"]["metric_version"])
        self.assertIn("–", (self.out / "res" / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()
