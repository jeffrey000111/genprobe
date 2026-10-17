"""Batch image generation with sharding and resume (guide §4.4, §6.2).

Examples
--------
GPU-free smoke run (fake images, safe on a login node):
    python -m genprobe.generate --config configs/gen/sdxl.yaml \
        --prompts data/prompts/smoke.jsonl --backend fake --seeds 0

Real run, one shard of a Slurm array:
    python -m genprobe.generate --config configs/gen/sdxl.yaml \
        --prompts data/prompts/smoke.jsonl --shard $SLURM_ARRAY_TASK_ID --num-shards 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .backends import is_blank, make_backend
from .config import load_gen_config
from .samples import Sample, build_samples, load_prompts
from .utils.env import fingerprint, now_iso, run_id
from .utils.io import atomic_save_image, atomic_write_text, clean_tmp
from .utils.manifest import ManifestWriter, done_ids, read_many
from .utils.shard import chunks, select_shard


def manifest_dir(out: str | Path) -> Path:
    return Path(out) / "manifests"


def manifest_path(out: str | Path, model: str, shard: int, num_shards: int) -> Path:
    # shard index *and* count in the name: changing num_shards never mixes two runs' files
    return manifest_dir(out) / f"gen_{model}_shard{shard:03d}of{num_shards:03d}.jsonl"


def run_generation(
    *,
    cfg,
    prompts,
    out: str | Path,
    shard: int = 0,
    num_shards: int = 1,
    seeds: list[int] | None = None,
    backend=None,
    backend_kind: str = "diffusers",
    log=print,
) -> dict[str, int]:
    """Generate every not-yet-done sample in this shard. Returns counts by status."""
    out = Path(out)
    seeds = seeds if seeds is not None else cfg.seeds
    chash = cfg.config_hash()
    all_samples = build_samples(prompts, cfg.name, seeds, out)
    mine = select_shard(all_samples, shard, num_shards, key=lambda s: s.sample_id)

    removed = clean_tmp(out / "images" / cfg.name)
    if removed:
        log(f"removed {removed} leftover .tmp file(s)")

    # Done = latest record across ALL of this model's manifests is terminal, was produced
    # with the same config_hash, and its image file exists.
    existing = read_many(manifest_dir(out).glob(f"gen_{cfg.name}_*.jsonl"))
    done = done_ids(existing, match={"config_hash": chash}, require_file=True)
    stale = {r["sample_id"] for r in existing if r.get("config_hash") != chash}
    todo = [s for s in mine if s.sample_id not in done]
    if stale & {s.sample_id for s in todo}:
        log(f"WARNING: {len(stale & {s.sample_id for s in todo})} sample(s) were made with a different "
            f"config; they will be regenerated and their images overwritten")
    log(f"[{cfg.name}] shard {shard}/{num_shards}: {len(mine)} samples, {len(mine) - len(todo)} done, "
        f"{len(todo)} to generate (config_hash={chash})")
    counts = {"ok": 0, "blank": 0, "error": 0, "skipped": len(mine) - len(todo)}
    if not todo:
        return counts

    rid = run_id()
    fp = fingerprint()
    atomic_write_text(out / "runs" / f"gen_{cfg.name}_{rid}.json",
                      json.dumps({"run_id": rid, "config": cfg.model_dump(), "config_hash": chash,
                                  "backend": backend_kind, "env": fp}, indent=2, default=str))
    if backend is None:
        backend = make_backend(backend_kind, cfg)
    scheduler = backend.scheduler_name()

    base = {
        "model": cfg.name, "model_repo": cfg.repo, "revision": cfg.revision, "dtype": cfg.dtype,
        "steps": cfg.steps, "cfg": cfg.cfg, "height": cfg.height, "width": cfg.width,
        "batch_size": cfg.batch_size, "cpu_offload": cfg.cpu_offload, "scheduler": scheduler,
        "backend": backend_kind, "gpu": fp.get("gpu"), "git_commit": fp["git_commit"],
        "config_hash": chash, "run_id": rid,
    }

    first_batch = True
    with ManifestWriter(manifest_path(out, cfg.name, shard, num_shards)) as w:
        for batch in chunks(todo, cfg.batch_size):
            results = _generate_batch(backend, batch)
            for s, img, timing, err in results:
                rec: dict[str, Any] = {
                    **base, "sample_id": s.sample_id, "prompt_id": s.prompt_id, "prompt": s.prompt,
                    "seed": s.seed, "path": s.path, "created_at": now_iso(),
                    # First batch of every job includes CUDA init/compilation: keep the number
                    # but flag it so aggregate.py can exclude it from timing statistics.
                    "warmup": first_batch,
                }
                if err is not None:
                    rec.update(status="error", error=err)
                else:
                    atomic_save_image(img, s.path)
                    rec.update(status="blank" if is_blank(img) else "ok",
                               gen_seconds=round(timing["sec_per_img"], 4),
                               peak_mem_gb=round(timing["peak_mem_gb"], 2),
                               image_size=list(img.size))
                counts[rec["status"]] += 1
                w.write(rec)
            first_batch = False
    log(f"[{cfg.name}] shard {shard}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return counts


def _generate_batch(backend, batch: list[Sample]):
    """Try the whole batch; on failure retry one by one so one bad prompt can't sink the rest."""
    try:
        images, timing = backend.generate(batch)
        return [(s, img, timing, None) for s, img in zip(batch, images)]
    except Exception as e:  # noqa: BLE001 - we record and continue
        if len(batch) == 1:
            return [(batch[0], None, None, f"{type(e).__name__}: {e}"[:2000])]
    out = []
    for s in batch:
        try:
            images, timing = backend.generate([s])
            out.append((s, images[0], timing, None))
        except Exception as e:  # noqa: BLE001
            out.append((s, None, None, f"{type(e).__name__}: {e}"[:2000]))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/gen/<model>.yaml")
    ap.add_argument("--prompts", required=True, help="prompts JSONL")
    ap.add_argument("--split", default=None, help="only prompts with this split (dev/test)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts after sorting by prompt_id")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override config seeds")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=None, help="output root (default: data, or data/_fake for --backend fake)")
    ap.add_argument("--backend", choices=["diffusers", "fake"], default="diffusers")
    args = ap.parse_args(argv)

    cfg = load_gen_config(args.config)
    prompts = load_prompts(args.prompts, split=args.split, limit=args.limit)
    if not prompts:
        print("no prompts selected", file=sys.stderr)
        return 2
    out = args.out or ("data/_fake" if args.backend == "fake" else "data")
    counts = run_generation(cfg=cfg, prompts=prompts, out=out, shard=args.shard,
                            num_shards=args.num_shards, seeds=args.seeds, backend_kind=args.backend)
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
