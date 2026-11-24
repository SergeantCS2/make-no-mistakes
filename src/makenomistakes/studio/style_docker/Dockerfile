# syntax=docker/dockerfile:1
# The studio's style trainer as mnm builds it (as it builds ComfyUI): an official Ubuntu
# base, ACE-Step 1.5's own trainer (`acestep.training_v2`, MIT) at a commit checked by hash and
# run from its source, every Python package pinned by version and sha256
# (`requirements-<variant>-<arch>.txt`, torch from PyTorch's own index: `cpu`, or `cu130` for an
# NVIDIA card with driver 580+), FFmpeg's libraries and libpython for torchcodec (its FFmpeg 6
# core links both). No weights are baked in: the
# ACE-Step checkpoint mnm downloads is mounted read-only at run time. A training is one
# `docker run --init --network none --read-only` as the user's uid over one job folder (the
# songs mnm copied in, the tensors and checkpoints the trainer writes, the adapter): nothing
# reaches the network. mnm renders the build from its code; `mnm studio trainer install
# --for styles` runs it.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG TARGETARCH
ARG VARIANT=cpu
ARG ACESTEP_COMMIT

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/acestep \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HOME=/tmp \
    TORCH_HOME=/tmp/torch \
    TRITON_CACHE_DIR=/tmp/triton \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv git ca-certificates \
      libavcodec60 libavformat60 libavutil58 libswresample4 libavdevice60 libavfilter9 \
      libpython3.12t64 \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

# A proxy's certificate, when the build runs behind one (`--secret id=ca,src=<file>`); unused
# otherwise. It never reaches the image. The commit is fetched by its hash and checked.
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export GIT_SSL_CAINFO=/run/secrets/ca; fi; \
    git init -q /opt/acestep \
 && git -C /opt/acestep fetch -q --depth 1 https://github.com/ace-step/ACE-Step-1.5 "${ACESTEP_COMMIT}" \
 && git -C /opt/acestep -c advice.detachedHead=false checkout -q FETCH_HEAD \
 && test "$(git -C /opt/acestep rev-parse HEAD)" = "${ACESTEP_COMMIT}" \
 && echo "${ACESTEP_COMMIT}" > /opt/acestep-commit \
 && rm -rf /opt/acestep/.git \
 && chmod -R a+rX /opt/acestep

COPY requirements-*.txt /opt/
# umask 022: the user's uid reads what pip wrote.
RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    test -f "/opt/requirements-${VARIANT}-${TARGETARCH}.txt" \
 && pip install --require-hashes --only-binary :all: -r "/opt/requirements-${VARIANT}-${TARGETARCH}.txt"

COPY --chmod=755 mnm_style.py /usr/local/bin/mnm-style
# The trainer's own modules import, or the build fails here rather than at the first style.
RUN python -c "import acestep.training_v2.trainer_fixed, acestep.training_v2.preprocess, acestep.training_v2.cli.common, peft, lycoris, lightning, torchcodec" \
 && mnm-style versions

WORKDIR /job
ENTRYPOINT ["/opt/venv/bin/python", "/usr/local/bin/mnm-style"]
