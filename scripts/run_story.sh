#!/bin/bash
# Multi-shot videos from a brief (genprobe/story.py, D-038): submits slurm/story.sbatch's stages in
# order, each after the previous one succeeded. Run on HPC3's login node (it only calls sbatch):
#
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_story.sh smoke"   1 dev brief, all arms, tau_v 0.9 (pipeline check)
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_story.sh dev"     8 dev briefs, arm B -> tau_v (results/story/dev_tau.md)
#   ssh sjsu3 "cd ~/genprobe && bash scripts/run_story.sh test"    24 test briefs, all arms (after approval)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: run_story.sh smoke|dev|test}
command -v sbatch >/dev/null || { echo "sbatch not found: run this on HPC3 (ssh sjsu3), not coe-hpc1" >&2; exit 2; }
mkdir -p logs
case "$NAME" in
  smoke) NS=1; T_KF=00:45:00; T_V0=01:00:00; T_V12=01:00:00; T_J=00:45:00; V12=1 ;;
  dev)   NS=1; T_KF=00:45:00; T_V0=01:15:00; T_V12=;        T_J=00:45:00; V12=0 ;;
  test)  NS=2; T_KF=01:00:00; T_V0=03:30:00; T_V12=02:30:00; T_J=01:00:00; V12=1
         [[ -f data/story/tau_v.json ]] || { echo "data/story/tau_v.json missing: run the dev stage first" >&2; exit 2; } ;;
  *) echo "usage: run_story.sh smoke|dev|test" >&2; exit 2 ;;
esac
sub() { local id; id=$(sbatch --parsable "$@"); echo "${id%%;*}"; }
a=$(sub --time="$T_KF" slurm/story.sbatch kf "$NAME")
b=$(sub --dependency="afterok:$a" --array=0-$((NS - 1)) --time="$T_V0" slurm/story.sbatch vid0 "$NAME" "$NS")
last=$b; msg="keyframes $a -> videos $b"
if [[ $V12 == 1 ]]; then
  c=$(sub --dependency="afterok:$b" --time="$T_V12" slurm/story.sbatch vid12 "$NAME")
  last=$c; msg="$msg -> redo $c"
fi
d=$(sub --dependency="afterok:$last" --time="$T_J" slurm/story.sbatch judge "$NAME")
echo "submitted story $NAME: $msg -> judge $d"
squeue -u "$USER" -o '%.10i %.20j %.8T %.10M %.10l %R'
