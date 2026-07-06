# LongNav R1 training container.
#
# The two working conda envs (`longnav_vlm`, `vln`), the from-source habitat-sim
# 0.2.1 build, and every external editable source tree are reconstructed at their
# ORIGINAL absolute paths (/home/ashed/...), so all baked-in `.pth` / editable
# finder files resolve untouched. Ray launches actors by conda-env NAME, so a
# real miniconda is present with both envs registered under envs/.
#
# Host is Ubuntu 24.04 / glibc 2.39 -> base image matches so the prebuilt
# habitat-sim .so files load. NVIDIA GL/EGL/CUDA driver libs are injected at run
# time by `singularity exec --nv`; below are only the vendor-neutral loaders.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 ca-certificates wget git \
      libglvnd0 libgl1 libglx0 libegl1 libgles2 libopengl0 libgl1-mesa-dri \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
      libx11-6 libxi6 libxrandr2 libxxf86vm1 libxcursor1 libxinerama1 \
      libgomp1 libjpeg-turbo8 libpng16-16 \
  && rm -rf /var/lib/apt/lists/*

# Base miniconda at the SAME prefix as the host so Ray can `conda activate` the
# named envs and the envs' hardcoded shebangs/prefixes stay valid.
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
  && bash /tmp/mc.sh -b -p /home/ashed/miniconda3 \
  && rm /tmp/mc.sh
ENV PATH=/home/ashed/miniconda3/bin:$PATH

# Conda envs (ADD auto-extracts; tarballs hold `longnav_vlm/` and `vln/`).
ADD longnav_vlm.tar.gz /home/ashed/miniconda3/envs/
ADD vln.tar.gz         /home/ashed/miniconda3/envs/

# habitat-sim 0.2.1 build (holds `habitat-sim/build/...`).
ADD habitat-sim.tar.gz /home/ashed/

# External editable source trees, at identical paths.
# habitat-lab-v023 -> longnav_vlm's habitat 0.2.3
ADD habitat-lab-v023.tar.gz /home/ashed/
# vlfm-src -> vln's habitat 0.2.4 (plus small groundingdino/vlfm extras)
ADD vlfm-src.tar.gz         /home/ashed/

# LongNav repo (holds spatial_training/) -> editable longnav/verl target.
ADD repo.tar.gz /home/ashed/Documents/

# Mount point for HM3D data (bind-mounted at run time).
RUN mkdir -p /home/ashed/Documents/spatial_training/data

WORKDIR /home/ashed/Documents/spatial_training
ENV HF_HUB_ENABLE_HF_TRANSFER=1
