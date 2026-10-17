#!/bin/bash
# Keep a laptop checkout and the HPC checkout in step, over an SSH alias.
#
#   scripts/hpc_sync.sh check   # is the SSH connection up? (no password prompt if it is)
#   scripts/hpc_sync.sh push    # laptop -> HPC: code, configs, prompt sets, .git
#   scripts/hpc_sync.sh pull    # HPC -> laptop: results/, logs/, env locks, docs/hpc_notes.md
#
# Ownership is one-way so nothing gets overwritten:
#   written on the laptop : genprobe/ configs/ slurm/ scripts/ tests/ data/prompts/ docs/ (except hpc_notes.md) ...
#   written on the HPC    : data/ (never synced: images, manifests, scores) results/ logs/ env/*.lock.txt docs/hpc_notes.md
#
# push refuses to run with uncommitted code changes, because every manifest records the git
# commit it was produced with; set FORCE=1 for a quick experiment (records will say -dirty).
#
# Env: HPC_HOST (default: sjsu3, an alias in ~/.ssh/config)  HPC_DIR (default: genprobe, relative to remote $HOME)
#
# Expected ~/.ssh/config entries on the laptop. HPC3 (login node g17) is only reachable through
# coe-hpc1, and /home is shared between them. Log in once per day with `ssh sjsu3`; later ssh,
# rsync and Claude Code reuse that connection without prompting:
#   Host sjsu
#       HostName coe-hpc1.sjsu.edu
#       User <your SJSU ID>
#       ControlMaster auto
#       ControlPath ~/.ssh/cm-%C
#       ControlPersist 8h
#       ServerAliveInterval 60
#   Host sjsu3
#       HostName coe-hpc3
#       User <your SJSU ID>
#       ProxyJump sjsu
#       IdentityFile ~/.ssh/id_ed25519
#       ControlMaster auto
#       ControlPath ~/.ssh/cm-%C
#       ControlPersist 8h
#       ServerAliveInterval 60
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${HPC_HOST:-sjsu3}
DIR=${HPC_DIR:-genprobe}
CMD=${1:-}
# DRY_RUN=1 lists what would be transferred without changing anything (do this the first time).
RS=(rsync -az)
# (-v rather than --itemize-changes: macOS's bundled openrsync only supports the common flags)
[[ "${DRY_RUN:-0}" == 1 ]] && RS=(rsync -az --dry-run -v)

check() {
  if ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" true 2>/dev/null; then
    echo "connected to $HOST"
  else
    cat >&2 <<EOF
Cannot reach '$HOST' without a password prompt.
  1. Connect the SJSU VPN if you are off campus.
  2. In a terminal, run:  ssh $HOST   and type your password once (keep that window open,
     or rely on ControlPersist in ~/.ssh/config). Then retry.
EOF
    exit 3
  fi
}

case "$CMD" in
  check)
    check
    ;;
  push)
    check
    CODE_PATHS=(genprobe configs slurm scripts data/prompts pyproject.toml)
    existing=(); for p in "${CODE_PATHS[@]}"; do [[ -e $p ]] && existing+=("$p"); done
    if git rev-parse --git-dir >/dev/null 2>&1 && [[ -n "$(git status --porcelain -- "${existing[@]}")" ]] \
       && [[ "${FORCE:-0}" != 1 ]]; then
      echo "Uncommitted changes in code paths; commit first (or FORCE=1):" >&2
      git status --short -- "${existing[@]}" >&2
      exit 4
    fi
    [[ "${DRY_RUN:-0}" == 1 ]] || ssh "$HOST" "mkdir -p '$DIR/data/prompts' '$DIR/logs'"
    # No --delete anywhere: a sync can add or update files on the other side, never remove them.
    "${RS[@]}" \
      --exclude='/data/' --exclude='/results/' --exclude='/logs/' \
      --exclude='/env/*.lock.txt' --exclude='/docs/hpc_notes.md' --exclude='/docs/private/' \
      --exclude='__pycache__/' --exclude='*.egg-info/' --exclude='.DS_Store' \
      ./ "$HOST:$DIR/"
    "${RS[@]}" data/prompts/ "$HOST:$DIR/data/prompts/"
    echo "${DRY_RUN:+[dry run] }pushed $(git rev-parse --short HEAD 2>/dev/null || echo 'working tree') to $HOST:$DIR"
    ;;
  pull)
    check
    mkdir -p results logs env docs
    "${RS[@]}" "$HOST:$DIR/results/" results/ 2>/dev/null || echo "(no results/ on $HOST yet)"
    "${RS[@]}" "$HOST:$DIR/logs/" logs/ 2>/dev/null || echo "(no logs/ on $HOST yet)"
    "${RS[@]}" --include='*.lock.txt' --exclude='*' "$HOST:$DIR/env/" env/ 2>/dev/null || true
    "${RS[@]}" "$HOST:$DIR/docs/hpc_notes.md" docs/hpc_notes.md 2>/dev/null || true
    echo "pulled results/, logs/, env locks and docs/hpc_notes.md from $HOST:$DIR"
    ;;
  *)
    sed -n '2,35p' "$0"
    exit 2
    ;;
esac
