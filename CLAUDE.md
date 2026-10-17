# GenProbe — working notes for Claude Code

GenProbe evaluates open text-to-image models, meta-evaluates automatic metrics against human
ratings, and later uses the trusted metrics in a generation-debugging agent. The full plan
(in Chinese) is in `docs/private/guide.md` when present; read the section for the current
step instead of loading the whole file. Progress and choices live in `docs/decisions.md`.

## Where things run

- **This laptop:** editing, `make test`, `make smoke-fake` (CPU, fake images). No GPU here.
- **SJSU CoE HPC3** (SSH alias `sjsu3`, login node g17, reached through coe-hpc1; repo at
  `~/genprobe`): all real generation and scoring, submitted with Slurm. Reach it with
  `ssh sjsu3 "cd ~/genprobe && <command>"`. coe-hpc1 has no working Slurm any more.
- Sync with `scripts/hpc_sync.sh push` (code -> HPC; requires committed code) and
  `scripts/hpc_sync.sh pull` (results/, logs/, env locks, docs/hpc_notes.md -> laptop).
  Images, manifests and scores stay on the HPC under `data/`; never copy them back wholesale.
- If `scripts/hpc_sync.sh check` fails, stop and ask the user to connect the VPN and run
  `ssh sjsu3` once in a terminal. Never ask for, type, or store their password.

## HPC rules (shared cluster)

- Nothing heavy on the login node: no generation, scoring, or model loading there. Installing
  packages and running unit tests is fine. Everything else goes through `sbatch`.
- Submit from the repo root (`cd ~/genprobe`). Job scripts use partition `gpuqs` and
  `--gres=gpu:a100:1` (D-009). Always pass an explicit `--time`; prefer short jobs.
- Storage: `/home` is shared and ~99% full — keep our footprint small there (code, envs,
  `data/`). Model weights and pip caches go to `/scratch/$USER`, which is purged
  periodically, so never keep anything there that can't be re-downloaded.
- Before any job that needs more than ~1 GPU-hour, run the 20-prompt smoke version first and
  show the user the result. Propose the full submission and wait for approval.
- Use `squeue -u $USER`, `sacct -j <id>` and `logs/` to follow jobs. Don't poll in tight loops.
- Check installed versions before writing code against a library:
  `ssh sjsu3 "source ~/miniforge3/etc/profile.d/conda.sh && conda activate eval && pip show <pkg>"`.
  If unsure about an API, say which README to check instead of guessing.

## Research rules

- Never invent results, numbers, or citations. Every number in REPORT.md or the README comes
  from a file in `results/`. Bad results are reported as they are.
- Thresholds, rubrics and agent rules are tuned on the dev split only; test is for reporting.
- The judge that grades a method must differ from the metric that method optimizes.
- Bootstrap confidence intervals resample prompts, not images.
- Record every parameter, threshold or trade-off in `docs/decisions.md` with its reason.
- Pre-registered hypotheses in `docs/decisions.md` are not edited after results exist.

## Code conventions

- One sample = `{model}__{prompt_id}__s{seed}`; all tables join on `sample_id`.
- New metrics: one file in `genprobe/metrics/`, subclass `Metric`, `@register`, import it in
  `genprobe/metrics/__init__.py`; heavy imports go inside `setup()`.
- Keep `make test` green; add a test when changing sharding, resume, or record formats.
- Commit before `hpc_sync.sh push` so manifests record a clean commit.
- This repo is public: no personal identifiers, credentials, weights, images, or `data/`.
