#!/bin/bash
# VBench for module D (guide §11.3, D-036), on **coe-hpc1** (internet): the source at a pinned commit
# into ~/envs/vbench-src (the vbench env puts it on its path) and the weights its six custom-input
# dimensions load in local mode into $HF_HOME/vbench (jobs set VBENCH_CACHE_DIR to it). URLs and file
# layout are VBench's own (vbench/utils.py init_submodules, vbench/aesthetic_quality.py).
#
#   ssh sjsu "cd ~/genprobe && bash scripts/download_vbench.sh"
#
# Re-running skips what is there.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/paths.sh
has_internet || { echo "No internet on $(hostname); run on coe-hpc1: ssh sjsu \"cd ~/genprobe && bash $0\"" >&2; exit 2; }
VBENCH_COMMIT=fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490          # main, 2026-08-21
SRC="$GENPROBE_ENVS/vbench-src"
C="$HF_HOME/vbench"
if [[ ! -d "$SRC/.git" ]]; then
  git clone -q https://github.com/Vchitect/VBench.git "$SRC"
fi
git -C "$SRC" fetch -q --depth 1 origin "$VBENCH_COMMIT" 2>/dev/null || true
git -C "$SRC" checkout -q "$VBENCH_COMMIT"
echo "VBench source: $SRC @ $(git -C "$SRC" rev-parse --short HEAD)"

get() {   # url dest
  if [[ -s "$2" ]]; then echo "have $2"; return 0; fi
  mkdir -p "$(dirname "$2")"
  curl -sSfL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" && echo "got  $2 ($(du -h "$2" | cut -f1))"
}
get https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt "$C/clip_model/ViT-B-32.pt"
get https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt "$C/clip_model/ViT-L-14.pt"
get https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth "$C/amt_model/amt-s.pth"
get https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth "$C/dino_model/dino_vitbase16_pretrain.pth"
get https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth "$C/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
get https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth "$C/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
if [[ ! -s "$C/raft_model/models/raft-things.pth" ]]; then
  get https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip "$C/raft_model/models.zip"
  unzip -q -o "$C/raft_model/models.zip" -d "$C/raft_model/" && rm -f "$C/raft_model/models.zip"
fi
[[ -s "$C/raft_model/models/raft-things.pth" ]] && echo "have $C/raft_model/models/raft-things.pth"
if [[ ! -d "$C/dino_model/facebookresearch_dino_main/.git" ]]; then
  git clone -q https://github.com/facebookresearch/dino "$C/dino_model/facebookresearch_dino_main"
fi
echo "dino repo @ $(git -C "$C/dino_model/facebookresearch_dino_main" rev-parse --short HEAD)"
(cd "$C" && find . -type f \( -name '*.pt' -o -name '*.pth' \) -exec sha256sum {} + | sort -k2 > SHA256SUMS)
echo "weights in $C (sha256 in $C/SHA256SUMS):"
du -sh "$C"
