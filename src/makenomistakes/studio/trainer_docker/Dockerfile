# syntax=docker/dockerfile:1
# The studio's trainer as mnm builds it (as it builds ComfyUI): an official Ubuntu base,
# Applio at a pinned release checked by commit (MIT; its training scripts only are run), every
# Python package pinned by version and sha256 (`requirements-<variant>-<arch>.txt`, torch from
# PyTorch's own index: `cpu`, or `cu130` for an NVIDIA card with driver 580+), Applio's weights
# fetched once here and checked by sha256 (`weights.sha256`). At run time a training is one
# `docker run --rm --init --network none --read-only` as the user's uid over one job folder
# (the dataset mnm copied in, the work Applio writes, the result): nothing reaches the network.
# mnm renders the build from its code; `mnm studio trainer install` runs it.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG TARGETARCH
ARG VARIANT=cpu
ARG APPLIO_REF
ARG APPLIO_COMMIT
ARG HUB_REVISION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp \
    TORCH_HOME=/tmp/torch \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

# A proxy's certificate, when the build runs behind one (`--secret id=ca,src=<file>`); unused
# otherwise. It never reaches the image.
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export GIT_SSL_CAINFO=/run/secrets/ca; fi; \
    git -c advice.detachedHead=false clone --depth 1 --branch "${APPLIO_REF}" \
      https://github.com/IAHispano/Applio /opt/applio \
 && test "$(git -C /opt/applio rev-parse HEAD)" = "${APPLIO_COMMIT}" \
 && echo "${APPLIO_COMMIT}" > /opt/applio-commit \
 && rm -rf /opt/applio/.git

COPY requirements-*.txt /opt/
# umask 022: the user's uid reads what pip wrote.
RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --require-hashes --only-binary :all: -r "/opt/requirements-${VARIANT}-${TARGETARCH}.txt"

COPY weights.sha256 /opt/weights.sha256
RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    set -e; ca=""; if [ -s /run/secrets/ca ]; then ca="--cacert /run/secrets/ca"; fi; \
    hub="https://huggingface.co/IAHispano/Applio/resolve/${HUB_REVISION}/Resources"; \
    grep -v '^#' /opt/weights.sha256 | while read -r sum source dest; do \
      [ -n "$sum" ] || continue; \
      mkdir -p "$(dirname "/opt/applio/$dest")"; \
      curl -fsSL $ca -o "/opt/applio/$dest" "$hub/$source"; \
      echo "$sum  /opt/applio/$dest" | sha256sum -c -; \
    done; \
    chmod -R a+rX /opt/applio

COPY --chmod=755 mnm_train.py /usr/local/bin/mnm-train
# The training scripts' packages import, or the build fails here rather than at the first voice.
RUN python -c "import torch, torchaudio, librosa, soundfile, soxr, faiss, sklearn, transformers, torchcrepe, torchfcpe, onnxruntime, matplotlib, einops" \
 && python -c "from torch.utils.tensorboard import SummaryWriter" \
 && mnm-train versions

WORKDIR /opt/applio
ENTRYPOINT ["/opt/venv/bin/python", "/usr/local/bin/mnm-train"]
