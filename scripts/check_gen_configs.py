"""Before queueing generation on H100s: every generator config resolves offline to its pinned
weights (model_index.json, every component folder, the fp16 variant files), and the prompt
set loads. CPU only, loads no weights. Run through scripts/check_gen_env.sh (gen env, cpuqs).

Resolution goes through ``DiffusionPipeline.download``, the exact call ``from_pretrained``
makes, because huggingface_hub >= 1.22 checks offline snapshots for completeness against the
file patterns the caller asks for: a bare ``snapshot_download`` reports our deliberately
partial downloads (no README, no duplicate single-file checkpoints) as incomplete.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from diffusers import DiffusionPipeline

    from genprobe.config import load_gen_config
    from genprobe.samples import load_prompts

    print(", ".join(f"{p}=={md.version(p)}" for p in ("torch", "diffusers", "transformers", "huggingface-hub")))
    failed = []
    for path in sorted((ROOT / "configs" / "gen").glob("*.yaml")):
        cfg = load_gen_config(path)
        try:
            skipped = {k: v for k, v in cfg.load_kwargs.items() if v is None}   # e.g. safety_checker
            snap = Path(DiffusionPipeline.download(cfg.repo, revision=cfg.revision, variant=cfg.variant,
                                                   local_files_only=True, **skipped))
            index = json.loads((snap / "model_index.json").read_text())
            parts = [k for k, v in index.items() if isinstance(v, list) and v[0] is not None]
            missing = [k for k in parts if not (snap / k).is_dir()]
            if cfg.load_kwargs.get("safety_checker", "x") is None:
                missing = [k for k in missing if k not in ("safety_checker", "feature_extractor")]
            assert not missing, f"component folders missing: {missing}"
            if cfg.variant:
                n = len(list(snap.rglob(f"*.{cfg.variant}.safetensors")))
                assert n, f"no *.{cfg.variant}.safetensors files"
            print(f"OK    {cfg.name}: {cfg.repo}@{cfg.revision[:7] if cfg.revision else 'main'}, "
                  f"{len(parts)} components, {cfg.steps} steps, cfg {cfg.cfg}, batch {cfg.batch_size}")
        except Exception as e:  # noqa: BLE001
            failed.append(cfg.name)
            print(f"FAIL  {cfg.name}: {type(e).__name__}: {e}")
    for prompts in sys.argv[1:] or ["data/prompts/prompts_v1.jsonl"]:
        ps = load_prompts(ROOT / prompts)
        splits = {s: sum(p.split == s for p in ps) for s in ("dev", "test")}
        print(f"OK    {prompts}: {len(ps)} prompts {splits}")
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
