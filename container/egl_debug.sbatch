#!/bin/bash
# Debug the HabitatEnvActor SIGSEGV at sim init: enumerate EGL devices and
# attempt a minimal habitat_sim.Simulator, first as-is (reproduce), then with
# the NVIDIA EGL vendor ICD forced via __EGL_VENDOR_LIBRARY_FILENAMES (fix).
#
# Put egl_debug.py next to this file, then:  sbatch egl_debug.sbatch
#SBATCH --job-name=egl-debug
#SBATCH --account=maanigj0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out

# no `set -e`: run 1 is EXPECTED to crash; we want run 2 to happen anyway.
set -uo pipefail

SIF=/scratch/maanigj_root/maanigj0/ashed/containers/longnav-rl.sif
DBG=/scratch/maanigj_root/maanigj0/ashed/jobs

echo "########## host: driver + EGL vendor ICDs on the compute node ##########"
nvidia-smi -L
ls -l /usr/share/glvnd/egl_vendor.d/ 2>&1
cat /usr/share/glvnd/egl_vendor.d/10_nvidia.json 2>/dev/null

singularity exec --nv --no-home --bind "$DBG":/mnt/dbg "$SIF" bash -lc '
  export HOME=/tmp/home && mkdir -p "$HOME"
  source /home/ashed/miniconda3/etc/profile.d/conda.sh
  conda activate vln
  export MAGNUM_LOG=verbose HABITAT_SIM_LOG=debug

  echo "########## container: what the loader can see ##########"
  echo "--- --nv injected libs (EGL/GLX related) ---"
  ls /.singularity.d/libs/ 2>/dev/null | grep -i "egl\|glx\|opengl"
  echo "--- vendor ICD dirs inside container ---"
  ls -l /usr/share/glvnd/egl_vendor.d/ /etc/glvnd/egl_vendor.d/ 2>&1

  echo
  echo "########## RUN 1: as-is (expected to reproduce the segfault) ##########"
  python /mnt/dbg/egl_debug.py
  echo "run 1 exit code: $?"

  echo
  echo "########## RUN 2: NVIDIA ICD forced via /tmp json ##########"
  printf "{ \"file_format_version\": \"1.0.0\", \"ICD\": { \"library_path\": \"libEGL_nvidia.so.0\" } }\n" > /tmp/10_nvidia.json
  export __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/10_nvidia.json
  python /mnt/dbg/egl_debug.py
  echo "run 2 exit code: $?"
'
