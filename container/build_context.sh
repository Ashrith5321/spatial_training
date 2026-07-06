#!/usr/bin/env bash
# Stage a Docker build context at ~/longnav-build by plain-tarring the two
# working conda envs + every external source tree their editable installs point
# to. We do NOT use conda-pack (it refuses editable installs); instead we keep
# identical absolute paths in the image, so all .pth / finder files resolve as-is.
#
# Editable source map (discovered from the envs' site-packages):
#   longnav_vlm: longnav,verl -> repo ;  habitat(0.2.3) -> ~/habitat-lab-v0.2.3
#   vln:         longnav      -> repo ;  habitat(0.2.4) -> ~/vlfm/habitat-lab-v0.2.4
#                (+ groundingdino/vlfm, small, kept so sys.path stays clean)
#   both:        habitat_sim 0.2.1 build -> ~/habitat-sim/build
set -euo pipefail

BUILD="$HOME/longnav-build"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$BUILD"

pack() {  # pack <dest-name> -C <parent> <item...>
  local out="$BUILD/$1"; shift
  echo ">> $out"
  tar --warning=no-file-changed -czf "$out" "$@"
}

echo ">> packing conda envs (plain tar, ~10G each, several min)"
pack longnav_vlm.tar.gz -C "$HOME/miniconda3/envs" longnav_vlm
pack vln.tar.gz         -C "$HOME/miniconda3/envs" vln

echo ">> packing habitat-sim 0.2.1 build (4G)"
pack habitat-sim.tar.gz -C "$HOME" habitat-sim/build

echo ">> packing external habitat source trees"
pack habitat-lab-v023.tar.gz -C "$HOME" habitat-lab-v0.2.3
pack vlfm-src.tar.gz -C "$HOME" \
    vlfm/habitat-lab-v0.2.4 vlfm/GroundingDINO_src vlfm/vlfm

echo ">> packing repo source (excluding data/dump/outputs/.git)"
tar --warning=no-file-changed -C "$HOME/Documents" -czf "$BUILD/repo.tar.gz" \
    --exclude='spatial_training/data' \
    --exclude='spatial_training/dump' \
    --exclude='spatial_training/outputs' \
    --exclude='spatial_training/.git' \
    --exclude='*/__pycache__' \
    --exclude='*.pyc' \
    spatial_training

cp "$HERE/Dockerfile" "$BUILD/Dockerfile"
echo; echo ">> build context ready at $BUILD"; du -sh "$BUILD"/*.tar.gz
echo ">> next:  sudo docker build -t longnav-rl:latest $BUILD"
