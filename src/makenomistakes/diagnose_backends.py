"""What ROCm's and Vulkan's images print when they cannot reach the card, each rule
scoped to its backend like `log_notes.Note.engines`; out of memory is the shared
`OUT_OF_MEMORY` (its pattern carries HIP's and Vulkan's words, `diagnose_rules.OOM_PATTERN`),
whose remedy and auto-fix are the same on every backend. The CPU backend's out of memory is
the kernel's (exit 137, `engine_exits`).

Where each pattern comes from:
- Vulkan, recorded on 2026-10-27 (no card; `server-vulkan` b8xxx and
  `whisper.cpp:main-vulkan`): without `GGML_VK_VISIBLE_DEVICES` ggml-vulkan skips Mesa's
  software device and prints `ggml_vulkan: No devices found.`; with it the device is
  `llvmpipe (LLVM …)` (whisper.cpp adds "Device type is CPU");
  `docker run --device /dev/dri` on a host without it fails with "error gathering device
  information while adding custom device "/dev/dri": no such file or directory".
- ROCm, from upstream's sources (not run on an AMD card; `server-rocm` is 7.7 GB): HIP's
  `hipErrorNoDevice` reads "no ROCm-capable device is detected" (ggml-cuda's init prints it
  after "failed to initialize ROCm"); rocminfo's "Unable to open /dev/kfd read-write:
  Permission denied" and its "is not member of "render" group" (ROCm's rocminfo.cc); the
  same Docker line for `/dev/kfd` as for `/dev/dri`.
"""

from __future__ import annotations

from makenomistakes.backend import Backend
from makenomistakes.diagnose_rules import Cause, Rule, make_rule

ROCM = frozenset({Backend.ROCM})
VULKAN = frozenset({Backend.VULKAN})
AMD_BACKENDS = ROCM | VULKAN
_GROUPS = (
    "Add yourself to the render and video groups (`sudo usermod -aG render,video $USER`), "
    "then log out and back in; `mnm doctor` checks the device files and the groups."
)

BACKEND_RULES: tuple[Rule, ...] = (
    make_rule(
        Cause.GPU_DEVICE_DENIED,
        r"unable to open /dev/(?:kfd|dri)\S* read-write|"
        r"/dev/(?:kfd|dri)\S*:? permission denied|"
        r"permission denied[^\n]{0,80}/dev/(?:kfd|dri)|"
        r"is not member of \"render\" group",
        "The engine may not open the card's device files (/dev/kfd, /dev/dri).",
        _GROUPS,
        backends=AMD_BACKENDS,
    ),
    make_rule(
        Cause.NO_GPU_DEVICE,
        r"no ROCm-capable device|hipErrorNoDevice|No HIP devices|"
        r"failed to initialize ROCm|"
        r"error gathering device information while adding custom device \"/dev/kfd\"",
        "The engine sees no ROCm device: /dev/kfd is missing or the amdgpu driver is not loaded.",
        "Check `ls /dev/kfd` and `lsmod | grep amdgpu` (a reboot after a kernel or firmware "
        "update often brings them back); unset `MNM_BACKEND` to run the Vulkan images, or "
        "`mnm doctor`.",
        backends=ROCM,
        supersedes=frozenset({Cause.MISSING_FILE}),
    ),
    make_rule(
        Cause.NO_GPU_DEVICE,
        # not llama-server's "no usable GPU found": every CPU build prints it at a healthy start
        r"ggml_vulkan: No devices found|No Vulkan devices found|"
        r"Available devices:\s*\(none\)|"
        r"error gathering device information while adding custom device \"/dev/dri\"",
        "The engine sees no Vulkan device: /dev/dri is missing, or the card's Vulkan driver "
        "is not installed.",
        "Check `ls /dev/dri` and the card's driver (`lsmod | grep amdgpu`, Mesa's "
        "`mesa-vulkan-drivers`), then `mnm doctor`; on a box without a card unset "
        "`MNM_BACKEND`: the CPU images run there.",
        backends=VULKAN,
    ),
    make_rule(
        Cause.SOFTWARE_RENDERING,
        r"\bllvmpipe\b|\blavapipe\b|Device type is CPU",
        "Vulkan runs on Mesa's software device (lavapipe), not a card: slow, a driver is missing.",
        "Install the card's Vulkan driver (`mesa-vulkan-drivers` for AMD and Intel) and check "
        "`ls /dev/dri`; on a box without a card the CPU backend is faster (unset "
        "`MNM_BACKEND`).",
        backends=VULKAN,
    ),
)
