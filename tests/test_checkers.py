"""OCR and detector checkers (D-017, D-018): text matching, reading order, box decisions,
the plugins with fake models, job-script env selection, and checker verdicts in summaries.

The real EasyOCR / Grounding DINO models only run in a Slurm job; everything here is the
plain-Python logic that decides the scores.
"""

import csv
import json
import re
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from genprobe.metrics import REGISTRY, detect, ocr
from genprobe.metrics.textmatch import normalize, semiglobal_distance, text_match
from genprobe.summarize import decision_agreement, summarize
from genprobe.backends import FakeBackend
from genprobe.generate import run_generation
from genprobe.score import load_samples
from tests.helpers import make_cfg, make_prompts, quiet

ROOT = Path(__file__).resolve().parents[1]


class TestTextMatch(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Hot  coffee!"), "HOT COFFEE")
        self.assertEqual(normalize("open-24/7"), "OPEN 24 7")
        self.assertEqual(normalize("  "), "")

    def test_semiglobal(self):
        self.assertEqual(semiglobal_distance("SALE", "BIGSALETODAY"), 0)
        self.assertEqual(semiglobal_distance("SALE", "BIGSAIETODAY"), 1)     # one misread letter
        self.assertEqual(semiglobal_distance("SALE", "SAE"), 1)              # one letter missing
        self.assertEqual(semiglobal_distance("SALE", ""), 4)
        self.assertEqual(semiglobal_distance("", "ANYTHING"), 0)

    def test_text_match(self):
        m = text_match("THANK YOU", "thankyou")                 # merged words still count
        self.assertEqual((m["value"], m["exact"], m["word_recall"]), (1.0, True, 0.0))
        m = text_match("FREE WIFI", "Cafe - Free WiFi - open")   # other text around it is ignored
        self.assertTrue(m["exact"])
        self.assertEqual(m["word_recall"], 1.0)
        m = text_match("BAKERY", "BAKREY")
        self.assertAlmostEqual(m["value"], 1 - 2 / 6)
        self.assertFalse(m["exact"])
        self.assertEqual(text_match("SALE", "")["value"], 0.0)
        self.assertEqual(text_match("SALE", "xx")["value"], 0.0)   # never below 0


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


class TestReadingOrder(unittest.TestCase):
    def test_lines_then_left_to_right(self):
        res = [(box(300, 105, 400, 140), "SEATED", 0.9), (box(0, 0, 100, 40), "PLEASE", 0.9),
               (box(0, 100, 120, 142), "TO BE", 0.9), (box(120, 2, 250, 38), "WAIT HERE", 0.9)]
        self.assertEqual([r[1] for r in ocr.reading_order(res)], ["PLEASE", "WAIT HERE", "TO BE", "SEATED"])

    def test_big_and_small_text_on_separate_lines(self):
        res = [(box(0, 120, 50, 130), "small", 0.9), (box(0, 0, 400, 100), "BIG", 0.9)]
        self.assertEqual([r[1] for r in ocr.reading_order(res)], ["BIG", "small"])


def det(x0, y0, x1, y1, s):
    return ((float(x0), float(y0), float(x1), float(y1)), s)


class TestDetectDecisions(unittest.TestCase):
    def test_iou_and_nms(self):
        self.assertAlmostEqual(detect.iou((0, 0, 2, 2), (1, 0, 3, 2)), 1 / 3)
        self.assertEqual(detect.iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)
        kept = detect.nms([det(0, 0, 10, 10, 0.9), det(1, 1, 10, 10, 0.8), det(50, 50, 60, 60, 0.7)])
        self.assertEqual([s for _, s in kept], [0.9, 0.7])

    def test_group_box_is_dropped(self):
        apples = [det(0, 0, 10, 10, 0.8), det(20, 0, 30, 10, 0.8), det(40, 0, 50, 10, 0.8)]
        group = det(-1, -1, 51, 11, 0.5)
        self.assertEqual(len(detect.drop_group_boxes(apples + [group])), 3)
        # a box around a single other box is not a group
        self.assertEqual(len(detect.drop_group_boxes([det(0, 0, 100, 100, 0.6), det(10, 10, 20, 20, 0.6)])), 2)

    def test_count(self):
        dets = [det(0, 0, 10, 10, 0.9), det(20, 0, 30, 10, 0.5), det(40, 0, 50, 10, 0.3)]   # last below 0.35
        r = detect.decide_count(dets, 2)
        self.assertEqual((r["value"], r["count_pred"]), (1.0, 2))
        self.assertEqual(detect.decide_count(dets, 3)["value"], 0.0)
        self.assertEqual(detect.decide_count([], 0)["value"], 1.0)

    def test_relation(self):
        a, b = [det(0, 40, 20, 60, 0.9)], [det(70, 40, 90, 60, 0.8)]     # a left of b
        self.assertEqual(detect.decide_relation(a, b, "left_of", 100, 100)["value"], 1.0)
        self.assertEqual(detect.decide_relation(a, b, "right_of", 100, 100)["value"], 0.0)
        top, bottom = [det(40, 0, 60, 20, 0.9)], [det(40, 70, 60, 90, 0.9)]   # image y grows downwards
        self.assertEqual(detect.decide_relation(top, bottom, "above", 100, 100)["value"], 1.0)
        self.assertEqual(detect.decide_relation(bottom, top, "below", 100, 100)["value"], 1.0)
        # centres closer than the margin: not left of
        near = [det(3, 40, 23, 60, 0.8)]
        self.assertEqual(detect.decide_relation(a, near, "left_of", 100, 100)["value"], 0.0)

    def test_relation_missing_or_same_box(self):
        a = [det(0, 0, 20, 20, 0.9)]
        r = detect.decide_relation(a, [det(70, 0, 90, 20, 0.2)], "left_of", 100, 100)
        self.assertEqual((r["value"], r["missing"]), (0.0, ["b"]))
        r = detect.decide_relation(a, [det(0, 0, 20, 21, 0.6)], "left_of", 100, 100)
        self.assertEqual((r["value"], r["missing"]), (0.0, ["b"]))
        with self.assertRaises(ValueError):
            detect.relation_holds("near", (0, 0), (1, 1))


def spec_sample(spec, sid="m__p__s0", path=""):
    return SimpleNamespace(spec=spec, sample_id=sid, path=path)


class TestPlugins(unittest.TestCase):
    def test_registered_with_env_and_threshold(self):
        self.assertEqual(REGISTRY["ocr"].env, "aux")
        self.assertEqual(REGISTRY["detect"].env, "eval")
        self.assertEqual(REGISTRY["ocr"].decision_threshold, 1.0)
        self.assertIsNone(REGISTRY["vqascore"].decision_threshold)

    def test_applies_to(self):
        o, d = REGISTRY["ocr"](), REGISTRY["detect"]()
        text = spec_sample({"text": "SALE"})
        count = spec_sample({"objects": [{"name": "cup", "count": 2}]})
        rel = spec_sample({"relation": {"type": "above", "a": "cup", "b": "book"}})
        genai = spec_sample(None)
        self.assertEqual([o.applies_to(s) for s in (text, count, rel, genai)], [True, False, False, False])
        self.assertEqual([d.applies_to(s) for s in (text, count, rel, genai)], [False, True, True, False])

    def test_versions_carry_settings(self):
        v = REGISTRY["detect"]().version
        for part in ("grounding-dino-tiny@", f"box{detect.BOX_THRESHOLD}", f"margin{detect.MARGIN}",
                     "transformers=="):
            self.assertIn(part, v)
        self.assertIn(ocr.MATCH_VERSION, REGISTRY["ocr"]().version)

    def test_ocr_with_fake_reader(self):
        m = REGISTRY["ocr"]()

        class FakeReader:
            def readtext(self, path):
                return [(box(0, 50, 80, 90), "Coffee", 0.8), (box(0, 0, 80, 40), "HOT", 0.95)]
        m.reader = FakeReader()
        r = m.score_batch([spec_sample({"text": "HOT COFFEE"}, sid="a")])[0]
        self.assertEqual((r["sample_id"], r["value"]), ("a", 1.0))
        self.assertEqual(r["extra"]["ocr_norm"], "HOT COFFEE")
        self.assertEqual([b["text"] for b in r["extra"]["boxes"]], ["HOT", "Coffee"])

    def test_ocr_missing_weights_say_how_to_download(self):
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.dict("os.environ", {"EASYOCR_MODULE_PATH": d}):
            try:
                import easyocr  # noqa: F401
            except ImportError:
                self.skipTest("easyocr not installed here")
            with self.assertRaisesRegex(FileNotFoundError, "download_models.sh easyocr"):
                REGISTRY["ocr"]().setup("cpu")

    def test_detect_with_fake_detector(self):
        m = REGISTRY["detect"]()
        boxes = {"cup": [det(0, 40, 20, 60, 0.9), det(30, 40, 50, 60, 0.7)], "book": [det(70, 40, 90, 60, 0.6)]}
        m.detect = lambda image, name: boxes.get(name, [])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.png"
            Image.new("RGB", (100, 100)).save(p)
            out = m.score_batch([
                spec_sample({"objects": [{"name": "cup", "count": 2}]}, "c", p),
                spec_sample({"relation": {"type": "left_of", "a": "cup", "b": "book"}}, "l", p),
                spec_sample({"relation": {"type": "left_of", "a": "book", "b": "cup"}}, "r", p),
                spec_sample({"relation": {"type": "above", "a": "cup", "b": "dog"}}, "m", p)])
        self.assertEqual([r["value"] for r in out], [1.0, 1.0, 0.0, 0.0])
        self.assertEqual(out[0]["extra"]["check"], "count")
        self.assertEqual(out[3]["extra"]["missing"], ["b"])


class TestJobScripts(unittest.TestCase):
    """Each Slurm scoring script must activate the env a metric declares (Metric.env)."""

    def env_map(self, script):
        text = (ROOT / script).read_text()
        m = re.search(r'case "\$METRIC" in(.*?)esac', text, re.S)
        self.assertIsNotNone(m, f"{script} has no metric -> env case")
        mapping, default = {}, None
        for pat, env in re.findall(r"([\w|*]+)\)\s*DEFAULT_ENV=(\w+)", m.group(1)):
            for name in pat.split("|"):
                if name == "*":
                    default = env
                else:
                    mapping[name] = env
        return mapping, default

    def test_env_matches_plugins(self):
        for script in ("slurm/score.sbatch", "slurm/score_cpu.sbatch"):
            mapping, default = self.env_map(script)
            for name, cls in REGISTRY.items():
                self.assertEqual(mapping.get(name, default), cls.env, f"{script}: {name}")

    def test_cpu_script_runs_on_cpu(self):
        text = (ROOT / "slurm/score_cpu.sbatch").read_text()
        self.assertIn("#SBATCH --partition=cpuqs", text)
        self.assertIn("--device cpu", text)
        self.assertNotIn("--gres", text)


class TestVerdicts(unittest.TestCase):
    def test_decision_agreement(self):
        values = {"a": 1.0, "b": 0.0, "c": 1.0, "d": 0.5, "e": 1.0}
        labels = {"a": "y", "b": "n", "c": "n", "d": "y", "e": "?"}
        prompt_of = {"a": "p1", "b": "p1", "c": "p2", "d": "p2", "e": "p2"}
        r = decision_agreement(values, labels, prompt_of, 1.0)
        self.assertEqual((r["images"], r["tp"], r["tn"], r["fp"], r["fn"]), (4, 1, 1, 1, 1))
        self.assertEqual(r["accuracy"], 0.5)
        self.assertLessEqual(r["ci95"][0], 0.5)
        self.assertEqual(decision_agreement({}, {}, {}, 1.0)["images"], 0)

    def test_summary_counts_only_applicable_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prompts = make_prompts(4)           # counting prompts: detect applies, ocr doesn't
            ppath = out / "prompts.jsonl"
            ppath.write_text("".join(json.dumps(p.__dict__) + "\n" for p in prompts))
            cfg = make_cfg(name="toy", seeds=[0], batch_size=2)
            run_generation(cfg=cfg, prompts=prompts, out=out, backend=FakeBackend(cfg), backend_kind="fake",
                           log=quiet)
            manifests = str(out / "manifests" / "gen_*.jsonl")
            samples = load_samples(manifests, str(ppath))
            d = out / "scores" / "detect"
            d.mkdir(parents=True)
            with open(d / "t.jsonl", "w") as f:
                for i, s in enumerate(samples):
                    f.write(json.dumps({"sample_id": s.sample_id, "metric": "detect", "metric_version": "v",
                                        "value": float(i % 2), "extra": {}, "seconds": 0.1, "status": "ok",
                                        "created_at": "2026-10-19T00:00:00"}) + "\n")
            labels = out / "labels.csv"
            labels.write_text("sample_id,correct,note\n" +
                              "".join(f"{s.sample_id},{'y' if i % 2 else 'n'},\n" for i, s in enumerate(samples)))
            res = summarize(prompts_path=str(ppath), manifests=manifests, data=str(out),
                            metrics=["detect", "ocr"], out=str(out / "res"), full_run_images=2400,
                            labels_path=str(labels))
            self.assertEqual((res["metrics"]["detect"]["n_ok"], res["metrics"]["detect"]["n_expected"]), (4, 4))
            self.assertEqual(res["metrics"]["ocr"]["n_expected"], 0)
            v = res["human_agreement"]["by_metric"]["detect"]["verdict"]
            self.assertEqual((v["images"], v["accuracy"]), (4, 1.0))
            md = (out / "res" / "summary.md").read_text()
            self.assertIn("| detect | 1.0 | 4 | 1.00", md)
            with open(out / "res" / "per_sample.csv", newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 4)


class TestEasyOCRDownload(unittest.TestCase):
    """scripts/easyocr_download.py: reads EasyOCR's config without importing easyocr, checks md5."""

    def setUp(self):
        import hashlib
        import importlib.util
        import zipfile
        spec = importlib.util.spec_from_file_location("easyocr_download", ROOT / "scripts" / "easyocr_download.py")
        self.dl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.dl)
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        payload = b"fake weights"
        with zipfile.ZipFile(d / "craft.zip", "w") as z:
            z.writestr("craft_mlt_25k.pth", payload)
        good, bad = hashlib.md5(payload).hexdigest(), "0" * 32
        (d / "config.py").write_text(
            "import os\n"
            f"detection_models = {{'craft': {{'filename': 'craft_mlt_25k.pth', 'url': '{(d / 'craft.zip').as_uri()}', "
            f"'md5sum': '{good}'}}}}\n"
            f"recognition_models = {{'gen2': {{'english_g2': {{'filename': 'craft_mlt_25k.pth', "
            f"'url': '{(d / 'craft.zip').as_uri()}', 'md5sum': '{bad}'}}}}}}\n")
        self.dir = d

    def tearDown(self):
        self.tmp.cleanup()

    def test_download_and_md5(self):
        out = self.dir / "model"
        self.assertEqual(self.dl.main([str(self.dir / "config.py"), str(out), "craft"]), 0)
        self.assertEqual((out / "craft_mlt_25k.pth").read_bytes(), b"fake weights")
        self.assertIn("already present", self.dl.fetch(self.dl.find_model(
            self.dl.load_config(self.dir / "config.py"), "craft"), out))

    def test_md5_mismatch_and_unknown_model_fail(self):
        out = self.dir / "model2"
        with unittest.mock.patch("sys.stderr"):
            self.assertEqual(self.dl.main([str(self.dir / "config.py"), str(out), "english_g2"]), 1)
            self.assertEqual(self.dl.main([str(self.dir / "config.py"), str(out), "nope"]), 1)
        self.assertFalse((out / "craft_mlt_25k.pth").exists())


if __name__ == "__main__":
    unittest.main()
