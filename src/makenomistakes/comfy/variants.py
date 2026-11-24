"""Which ComfyUI build a box gets: chosen by the box's backend at `mnm comfyui install`
and recorded in the ledger (`Ledger.variant`), so the doctor can say when the box moved on
("built for CUDA 13, this box runs Vulkan: `mnm comfyui install`").

- CUDA 13 → PyTorch's cu130 wheels.
- CUDA 12 (a card older than Turing, or a driver below 580) → the cu126 wheels with
  `--disable-cuda-malloc` (`Variant.CUDA_LEGACY`). The same torch version: cu126 still carries
  Pascal's and Volta's code (`comfy.image`). Not Blackwell's: a 50-series card on a driver
  below 580 builds it and fails at its first run; the doctor's driver line says to update.
- ROCm, or Vulkan on an AMD card → the ROCm wheels: PyTorch has no Vulkan build, and an AMD
  card runs ROCm's wheels with no ROCm on the host. Planned, unverified.
- CPU, or Vulkan without an AMD card (Intel, the software device) → the CPU wheels."""

from __future__ import annotations

from makenomistakes.backend import Backend
from makenomistakes.comfy.image import Variant
from makenomistakes.hardware.fingerprint import MachineFingerprint, Vendor


def variant_for(backend: Backend, box: MachineFingerprint | None) -> Variant:
    if backend is Backend.CUDA13:
        return Variant.CUDA
    if backend is Backend.CUDA12:
        return Variant.CUDA_LEGACY
    if backend is Backend.ROCM:
        return Variant.ROCM
    amd = box is not None and any(c.vendor is Vendor.AMD for c in box.cards)
    if backend is Backend.VULKAN and amd:
        return Variant.ROCM
    return Variant.CPU


def install_variant(cpu: bool, cards_seen: bool) -> Variant:
    """What `mnm comfyui install` builds: the CPU when asked (`--cpu`); else the box's
    backend's variant, the box as last recorded; before any box was recorded, CUDA 13 with a
    card visible and the CPU without."""
    from makenomistakes.backend import BackendError  # noqa: PLC0415 - keeps this module a leaf
    from makenomistakes.backend_now import box_backend, box_now  # noqa: PLC0415

    if cpu:
        return Variant.CPU
    box = box_now()
    if box is None:
        return Variant.CUDA if cards_seen else Variant.CPU
    try:
        return variant_for(box_backend(), box)
    except BackendError:
        return Variant.CUDA if cards_seen else Variant.CPU
