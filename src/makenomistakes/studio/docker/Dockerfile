# syntax=docker/dockerfile:1
# The studio's tools as mnm builds them (as it builds ComfyUI): an official Ubuntu base,
# every Python package pinned by version and sha256 (`requirements-<arch>.txt`, CPU torch from
# PyTorch's own index; basic-pitch by its wheel's sha256), the scorer's, the beat tracker's and
# the speaker model's weights fetched once here and checked by sha256 (Basic Pitch's ONNX model
# is inside its wheel). At run time a job is `docker run --rm --network none --read-only` as the
# owner's uid over one run folder, mounted read-only: nothing reaches the network then.
# mnm renders the build from its code; `mnm studio install` runs it.
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG TARGETARCH
# Audiobox Aesthetics (CC-BY-4.0): facebook/audiobox-aesthetics at this revision
ARG SCORER_REVISION
ARG SCORER_SHA256
ARG SCORER_CONFIG_SHA256
# Beat This! (MIT): CPJKU's `final0` checkpoint from their cloud space
ARG BEATS_URL
ARG BEATS_SHA256
# SpeechBrain's ECAPA-TDNN (Apache-2.0): speechbrain/spkrec-ecapa-voxceleb at this revision
ARG ECAPA_REVISION
ARG ECAPA_SHA256

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m venv /opt/venv

COPY requirements-amd64.txt requirements-arm64.txt requirements-nodeps.txt /opt/
# A proxy's certificate, when the build runs behind one (`--secret id=ca,src=<file>`); unused
# otherwise. It never reaches the image. umask 022: the user's uid reads what pip wrote.
# basic-pitch goes in after, without its dependencies (its TensorFlow pin has no Python 3.12
# build; its ONNX model runs on the onnxruntime pinned above): `pip check` would name it.
RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --require-hashes --only-binary :all: -r "/opt/requirements-${TARGETARCH}.txt" \
 && pip install --no-deps --require-hashes --only-binary :all: -r /opt/requirements-nodeps.txt

RUN --mount=type=secret,id=ca,required=false \
    umask 022; \
    set -e; ca=""; if [ -s /run/secrets/ca ]; then ca="--cacert /run/secrets/ca"; fi; \
    mkdir -p /opt/weights/audiobox-aesthetics /opt/weights/beat_this /opt/weights/ecapa; \
    hub="https://huggingface.co/facebook/audiobox-aesthetics/resolve/${SCORER_REVISION}"; \
    curl -fsSL $ca -o /opt/weights/audiobox-aesthetics/model.safetensors "$hub/model.safetensors"; \
    curl -fsSL $ca -o /opt/weights/audiobox-aesthetics/config.json "$hub/config.json"; \
    curl -fsSL $ca -o /opt/weights/beat_this/final0.ckpt "${BEATS_URL}"; \
    ecapa="https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/${ECAPA_REVISION}"; \
    curl -fsSL $ca -o /opt/weights/ecapa/embedding_model.ckpt "$ecapa/embedding_model.ckpt"; \
    echo "${SCORER_SHA256}  /opt/weights/audiobox-aesthetics/model.safetensors" | sha256sum -c -; \
    echo "${SCORER_CONFIG_SHA256}  /opt/weights/audiobox-aesthetics/config.json" | sha256sum -c -; \
    echo "${BEATS_SHA256}  /opt/weights/beat_this/final0.ckpt" | sha256sum -c -; \
    echo "${ECAPA_SHA256}  /opt/weights/ecapa/embedding_model.ckpt" | sha256sum -c -; \
    chmod -R a+rX /opt/weights

COPY --chmod=755 mnm_tools.py /usr/local/bin/mnm-tools
COPY mnm_tune.py /usr/local/bin/mnm_tune.py
# Every package imports, or the build fails here rather than at the first song.
RUN python -c "import torch, torchaudio, librosa, soundfile, audiobox_aesthetics, beat_this, speechbrain, matchering, onnxruntime, torchcrepe" \
 && python -c "from basic_pitch.inference import predict" \
 && mnm-tools versions

WORKDIR /work
ENTRYPOINT ["/opt/venv/bin/python", "/usr/local/bin/mnm-tools"]
