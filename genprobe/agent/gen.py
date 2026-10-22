"""Generate the attempts the agent planned for one round (gen env, GPU; slurm/agent.sbatch).

    python -m genprobe.agent.gen --run devrule --round 1 [--backend fake]

Reads data/agent/<run>/round<k>/requests.jsonl (sample_id, model, gen_prompt, cfg, seed, path) and
writes data/agent/<run>/manifests/gen_r<k>_<model>.jsonl. Each model's pipeline is loaded once,
with its own configs/gen/<model>.yaml; a request's cfg (if set) replaces the config's guidance
for that image. The record's `prompt` is the ORIGINAL prompt (checks use it, guide §10.6) and
`gen_prompt` is the text the image was generated from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from ..backends import is_blank, make_backend
from ..config import load_gen_config
from ..generate import _generate_batch
from ..samples import Sample
from ..utils.env import fingerprint, now_iso, run_id
from ..utils.io import atomic_save_image
from ..utils.manifest import ManifestWriter, done_ids, read_many
from ..utils.shard import chunks
from .run import ROOT, run_dir


def generate_round(run: str, rnd: int, backend_kind: str = "diffusers", root: Path = ROOT, log=print,
                   backend_factory=None) -> dict[str, int]:
    d = run_dir(run, root)
    reqs = [json.loads(line) for line in (d / f"round{rnd}" / "requests.jsonl").read_text().splitlines() if line]
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in reqs:
        by_model[r["model"]].append(r)
    counts = {"ok": 0, "blank": 0, "error": 0, "skipped": 0}
    fp, rid = fingerprint(), run_id()
    for model, items in sorted(by_model.items()):
        cfg = load_gen_config(f"configs/gen/{model}.yaml")
        mpath = d / "manifests" / f"gen_r{rnd}_{model}.jsonl"
        done = done_ids(read_many([mpath]) if mpath.exists() else [], require_file=True)
        todo = [r for r in items if r["sample_id"] not in done]
        counts["skipped"] += len(items) - len(todo)
        if not todo:
            continue
        backend = (backend_factory or make_backend)(backend_kind, cfg)
        log(f"[agent.gen] {model}: {len(todo)} images")
        first = True
        with ManifestWriter(mpath) as w:
            groups: dict[float, list[dict]] = defaultdict(list)
            for r in todo:
                groups[cfg.cfg if r.get("cfg") is None else float(r["cfg"])].append(r)
            for guidance, rs in sorted(groups.items()):
                backend.cfg = cfg.model_copy(update={"cfg": guidance})
                for batch in chunks(rs, cfg.batch_size):
                    samples = [Sample(sample_id=r["sample_id"], model=model, prompt_id=r["prompt_id"],
                                      prompt=r["gen_prompt"], seed=int(r["seed"]), path=r["path"]) for r in batch]
                    for s, r, (_, img, timing, err) in zip(samples, batch, _generate_batch(backend, samples)):
                        rec = {"sample_id": s.sample_id, "model": model, "prompt_id": s.prompt_id,
                               "prompt": r["prompt"], "gen_prompt": r["gen_prompt"], "seed": s.seed, "path": s.path,
                               "steps": cfg.steps, "cfg": guidance, "height": cfg.height, "width": cfg.width,
                               "config_hash": cfg.config_hash(), "backend": backend_kind, "gpu": fp.get("gpu"),
                               "git_commit": fp["git_commit"], "run_id": rid, "created_at": now_iso(), "warmup": first}
                        if err is not None:
                            rec.update(status="error", error=err)
                        else:
                            atomic_save_image(img, s.path)
                            rec.update(status="blank" if is_blank(img) else "ok",
                                       gen_seconds=round(timing["sec_per_img"], 4))
                        counts[rec["status"]] += 1
                        w.write(rec)
                    first = False
        del backend
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - no torch in tests
            pass
    log(f"[agent.gen] round {rnd}: " + json.dumps(counts))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--backend", choices=["diffusers", "fake"], default="diffusers")
    args = ap.parse_args(argv)
    counts = generate_round(args.run, args.round, args.backend)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
