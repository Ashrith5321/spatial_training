#!/bin/bash
# LongNav R1 RL training on Great Lakes, from the SFT checkpoint, HM3D v1.
# Submit:  export WANDB_API_KEY=<key> && sbatch container/train.sbatch
#SBATCH --job-name=longnav-rl
#SBATCH --account=maanigj0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

SIF=/scratch/maanigj_root/maanigj0/ashed/containers/longnav-rl.sif
CKPT=/scratch/maanigj_root/maanigj0/ashed/checkpoints/longnav_sft
DATA=/scratch/maanigj_root/maanigj0/ashed/hm3d_v0.1
OUT=/scratch/maanigj_root/maanigj0/ashed/outputs
mkdir -p "$OUT"

# --no-home keeps the baked-in /home/ashed intact; HOME is set writable inside.
# WANDB_API_KEY is passed through explicitly (--env HOME is blocked, so HOME is
# exported inside the shell instead).
singularity exec --nv --no-home \
  --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
  --bind "$CKPT":/mnt/ckpt \
  --bind "$DATA":/home/ashed/Documents/spatial_training/data \
  --bind "$OUT":/home/ashed/Documents/spatial_training/outputs \
  "$SIF" bash -lc '
    source /home/ashed/miniconda3/etc/profile.d/conda.sh
    conda activate longnav_vlm
    cd /home/ashed/Documents/spatial_training
    export HOME=/tmp/home WANDB_DIR=/tmp/wandb HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    mkdir -p /tmp/home /tmp/wandb
    # --nv injects libEGL_nvidia.so.0 but not its glvnd ICD -> EGL falls back to
    # Mesa and habitat_sim cant create a GPU context. Register NVIDIA as the EGL
    # vendor so habitat renders on the GPU.
    printf "%s" "{\"file_format_version\":\"1.0.0\",\"ICD\":{\"library_path\":\"/.singularity.d/libs/libEGL_nvidia.so.0\"}}" > /tmp/nvidia_egl_icd.json
    export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/nvidia_egl_icd.json __GLX_VENDOR_LIBRARY_NAME=nvidia
    python -m longnav.scripts.train_rl \
      +checkpoint=sft \
      +dataset=hm3d_train \
      +experiment=train \
      +resources=single \
      +training=hapo \
      vlm.model_id=/mnt/ckpt \
      sim.workspace=/home/ashed/Documents/spatial_training \
      resources.object_spilling_directory=/tmp/ray_spill \
      task.output_dir=/home/ashed/Documents/spatial_training/outputs \
      task.run_name=longnav_rl_gl
  '
