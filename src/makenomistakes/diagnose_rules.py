"""The vocabulary of `diagnose`'s rules (the backends' own rules are in `diagnose_backends`):
the stable `Cause` codes
auto-fix keys on, a `Diagnosis`, and one rule scoped to the engines and backends whose logs it
may fire on (like `log_notes.Note.engines`)."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel

from makenomistakes.backend import CUDA, Backend
from makenomistakes.hub.facts import Engine


class Cause(StrEnum):
    """Stable ids for the rules so code (auto-fix) can key on them, not on English."""

    NO_GPU_RUNTIME = "no-gpu-runtime"
    CUDA_DRIVER = "cuda-driver"
    OUT_OF_MEMORY = "out-of-memory"
    SD_WRONG_FAMILY = "sd-wrong-family"
    SD_LOAD_FAILED = "sd-load-failed"
    PORT_TAKEN = "port-taken"
    IMAGE_TOO_OLD = "image-too-old"
    QUANT_DROPPED = "quant-dropped"  # A tensor type the engine no longer reads
    CTX_TOO_LARGE = "ctx-too-large"
    PROJECTOR_FAILED = "projector-failed"
    MISSING_FILE = "missing-file"
    HUB_REFUSED = "hub-refused"
    KV_NEEDS_FLASH_ATTN = "kv-needs-flash-attn"
    LORA_NOT_APPLIED = "lora-not-applied"
    VOICE_LOAD_FAILED = "voice-load-failed"
    STYLE_MISMATCH = "style-mismatch"  # An ACE-Step style shaped for another model
    VOICE_NEEDS_ESPEAK = "voice-needs-espeak"
    # The backends' own (`diagnose_backends`); out of memory stays OUT_OF_MEMORY
    NO_GPU_DEVICE = "no-gpu-device"  # the engine sees no ROCm or Vulkan device
    GPU_DEVICE_DENIED = "gpu-device-denied"  # /dev/kfd or /dev/dri refused: the groups
    SOFTWARE_RENDERING = "software-rendering"  # Vulkan on lavapipe: slow, a driver missing


class Diagnosis(BaseModel):
    cause: str
    remedy: str
    cause_code: Cause


ALL_ENGINES = frozenset(Engine)
DOWNLOAD_ONLY: frozenset[Engine] = frozenset()  # fires only without an engine: Hub download errors
ALL_BACKENDS = frozenset(Backend)
CUDA_ONLY: frozenset[Backend] = CUDA  # NVIDIA's runtime and driver lines


class Rule(BaseModel):
    code: Cause
    pattern: re.Pattern[str]
    cause: str
    remedy: str
    engines: frozenset[Engine] = ALL_ENGINES  # which engine's log the rule may fire on
    backends: frozenset[Backend] = ALL_BACKENDS  # and which backend's image
    # Causes this one explains when both match: their wording would send the user elsewhere.
    supersedes: frozenset[Cause] = frozenset()


def make_rule(
    code: Cause,
    pattern: str,
    cause: str,
    remedy: str,
    engines: frozenset[Engine] = ALL_ENGINES,
    supersedes: frozenset[Cause] = frozenset(),
    backends: frozenset[Backend] = ALL_BACKENDS,
) -> Rule:
    return Rule(
        code=code,
        pattern=re.compile(pattern, re.IGNORECASE),
        cause=cause,
        remedy=remedy,
        engines=engines,
        supersedes=supersedes,
        backends=backends,
    )


OOM_PATTERN = (
    r"cudaMalloc failed|\bout of memory\b|failed to allocate|"
    r"ggml_backend_cuda_buffer_type_alloc_buffer: allocating .* failed|"
    r"not enough memory|(?-i:\bOOM\b)|"  # capitals only: `zimage-oom` is a name
    # HIP's and Vulkan's own (ggml-cuda's `ROCm error: out of memory` is matched above)
    r"hipErrorOutOfMemory|ErrorOutOfDeviceMemory|OutOfDeviceMemoryError"
)
