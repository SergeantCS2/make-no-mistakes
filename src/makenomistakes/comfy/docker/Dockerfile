# syntax=docker/dockerfile:1
# ComfyUI as mnm builds it: an official Ubuntu base, ComfyUI at a pinned release
# checked by commit, PyTorch from PyTorch's own index (its wheels carry CUDA's libraries; the
# NVIDIA container runtime brings the driver), ffmpeg for video nodes. Nothing here runs as root
# at run time: `docker run --user` gives the user's uid. Approved node packs are a layer on
# top. mnm renders the build from its code; `mnm comfyui install` runs it.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG COMFYUI_REF=v0.42.1
ARG COMFYUI_COMMIT=63be2f9b82ad0dac88a7ceaa28a9cf3154125a18
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev git ca-certificates ffmpeg build-essential \
      libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

# A proxy's certificate, when the build runs behind one (`--secret id=ca,src=<file>`); unused
# otherwise. It never reaches the image.
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export GIT_SSL_CAINFO=/run/secrets/ca; fi; \
    git clone --depth 1 --branch "${COMFYUI_REF}" https://github.com/Comfy-Org/ComfyUI /opt/comfyui \
 && test "$(git -C /opt/comfyui rev-parse HEAD)" = "${COMFYUI_COMMIT}" \
 && rm -rf /opt/comfyui/.git

COPY constraints.txt /opt/constraints.txt
# umask 022: every file pip writes is world-readable as written (ComfyUI runs as the user's
# uid), so the venv needs no `chmod -R` after: that walk once took 397 s of a 12-minute build.
RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --only-binary :all: --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple \
      -c /opt/constraints.txt torch torchvision torchaudio \
 && pip install --only-binary :all: -c /opt/constraints.txt -r /opt/comfyui/requirements.txt

# What ComfyUI writes beside its code goes to the mounted folders (`--user-directory` and the
# rest); the models folder it lists is mnm's links, read-only, through extra_model_paths.
RUN chmod -R a+rX /opt/comfyui && mkdir -p /opt/comfyui/models && chmod a+rwx /opt/comfyui/models

WORKDIR /opt/comfyui
ENTRYPOINT ["python", "/opt/comfyui/main.py"]
