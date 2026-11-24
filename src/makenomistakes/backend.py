"""The box's backend: which build of each engine runs here and what Docker is
told to reach the cards. Decided once from the fingerprint (`decide`) and the user's choice
(`MNM_BACKEND`, read like `MNM_GPU`); a choice the cards cannot run is refused with the reason,
never a silent fallback.

- NVIDIA cards: `cuda13` when the driver is 580 or newer and every card is Turing (7.5) or
  newer (`hardware.cards`' floors: the CUDA 13 images drop Pascal and Volta), else `cuda12`.
- AMD cards: `vulkan` (the floor, every RDNA card and the MI50 alike), `rocm` when the
  user chose it and the card's gfx target is one `server-rocm` carries.
- Intel or other cards: `vulkan`. No card: `cpu`. `sycl` is reserved, never chosen.

The device arguments live here too (`device_argv`, `recipes/cards.py` renders them): CUDA
`--gpus` and `CUDA_DEVICE_ORDER`, as one card always rendered them;
ROCm `/dev/kfd` plus each chosen card's render node with the `render` and `video` groups by
number (a name does not resolve inside the image) and `HIP_VISIBLE_DEVICES` within them
(ROCm's Docker how-to); Vulkan `/dev/dri` whole with `GGML_VK_VISIBLE_DEVICES`
in the Vulkan loader's order (`hardware.vulkan`); CPU nothing. llama.cpp names a backend's
devices `CUDA<k>`, `ROCm<k>`, `Vulkan<k>` (`ggml/include/ggml-cuda.h` lines 10-18:
`GGML_CUDA_NAME` is "ROCm" under `GGML_USE_HIP`; `ggml/include/ggml-vulkan.h` line 10,
`GGML_VK_NAME`; read at llama.cpp 19ff817 on 2026-10-27), `k` the place among the devices the
container sees."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from makenomistakes.hardware.fingerprint import CardId, MachineFingerprint, MemoryKind, Vendor
from makenomistakes.hardware.gpu_select import device_env, device_spec, parse_choice


class Backend(StrEnum):
    CUDA12 = "cuda12"
    CUDA13 = "cuda13"
    ROCM = "rocm"
    VULKAN = "vulkan"
    CPU = "cpu"
    SYCL = "sycl"  # reserved: Intel's oneAPI images exist, nothing chooses them yet


CUDA = frozenset({Backend.CUDA12, Backend.CUDA13})
LABELS = {
    Backend.CUDA12: "CUDA 12",
    Backend.CUDA13: "CUDA 13",
    Backend.ROCM: "ROCm",
    Backend.VULKAN: "Vulkan",
    Backend.CPU: "CPU",
    Backend.SYCL: "SYCL",
}
# The prefix llama.cpp gives a backend's devices (`-ot …=<name>`, `--device`): see above.
DEVICE_PREFIX = {
    Backend.CUDA12: "CUDA",
    Backend.CUDA13: "CUDA",
    Backend.ROCM: "ROCm",
    Backend.VULKAN: "Vulkan",
}
CUDA13_DRIVER = 580  # CUDA 13.0's Linux driver floor (NVIDIA's CUDA 13.0 release notes)
TURING = 7.5  # the CUDA 13 images' lowest compute capability (`hardware.cards`)
# The gfx targets `server-rocm` is built for (llama.cpp's `.devops/rocm.Dockerfile`,
# read 2026-10-26): a card outside it runs the Vulkan image.
ROCM_TARGETS = frozenset(
    {
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1150",
        "gfx1151",
        "gfx1200",
        "gfx1201",
    }
)
# Not in the GUIDE: with this set, `MNM_BACKEND=vulkan` on a box without a card runs on Mesa's
# software device, which is how the Vulkan images are checked without a card.
SOFTWARE_SEAM = "MNM_VULKAN_SOFTWARE"
CHOICES = "cuda12, cuda13, rocm, vulkan or cpu (empty: mnm decides from the cards)"


class BackendError(ValueError):
    """`MNM_BACKEND` names a backend these cards cannot run, or nothing mnm knows."""


def parse_backend(text: str) -> Backend | None:
    """`MNM_BACKEND`'s value: empty for mnm's own choice; a ValueError naming the choices."""
    value = text.strip().lower()
    if not value:
        return None
    try:
        found = Backend(value)
    except ValueError:
        raise BackendError(f"MNM_BACKEND={text.strip()} is not a backend: {CHOICES}") from None
    if found is Backend.SYCL:
        raise BackendError(f"MNM_BACKEND=sycl is reserved and not built yet: {CHOICES}")
    return found


def label(backend: Backend) -> str:
    return LABELS[backend]


def _capability(card: CardId) -> float | None:
    try:
        return float(card.compute_cap) if card.compute_cap else None
    except ValueError:
        return None


def _driver(found: MachineFingerprint) -> int | None:
    majors = [c.driver_major for c in found.cards if c.driver_major is not None]
    if majors:
        return min(majors)
    head = (found.driver_version or "").split(".")[0]
    return int(head) if head.isdigit() else None


def cuda13_problem(found: MachineFingerprint) -> str | None:
    """Why these NVIDIA cards cannot run the CUDA 13 images; None when they can."""
    nvidia = [c for c in found.cards if c.vendor is Vendor.NVIDIA]
    driver = _driver(found)
    if driver is None:
        return "the NVIDIA driver's version could not be read"
    if driver < CUDA13_DRIVER:
        return f"the NVIDIA driver is {driver}, and CUDA 13 needs {CUDA13_DRIVER} or newer"
    old = [c for c in nvidia if (cc := _capability(c)) is None or cc < TURING]
    if old:
        names = ", ".join(f"{c.short} ({c.compute_cap or 'compute unknown'})" for c in old)
        return f"{names} is older than Turing (7.5), which CUDA 13 dropped"
    return None


def natural(found: MachineFingerprint) -> Backend:
    """What mnm picks for these cards without a choice from the user."""
    vendors = {c.vendor for c in found.cards}
    if Vendor.NVIDIA in vendors:
        return Backend.CUDA12 if cuda13_problem(found) else Backend.CUDA13
    if vendors:
        return Backend.VULKAN  # AMD, Intel or anything else with a Vulkan driver
    return Backend.CPU


APU_VULKAN = (
    "an APU runs Vulkan until a tune measures ROCm faster on it (`mnm tune <name>`; the "
    "guides' default for Strix Halo is Vulkan)"
)


def not_taken(
    found: MachineFingerprint, chosen: str | None, gpu: str | None, decided: Backend
) -> str | None:
    """Why the user's `MNM_BACKEND` is not the backend taken (the APU rule), for the lines
    that name the backend; None when it was taken or none was asked."""
    wanted = parse_backend(chosen or "")
    if wanted is Backend.ROCM and decided is Backend.VULKAN and apu_only(used_cards(found, gpu)):
        return f"MNM_BACKEND=rocm is not taken yet: {APU_VULKAN}"
    return None


def decide(
    found: MachineFingerprint,
    chosen: str | None,
    gpu: str | None = None,
    rocm_measured: bool = False,
) -> Backend:
    """The backend for this box: the user's `MNM_BACKEND` when the cards can run it, else
    mnm's own (`natural`). A choice they cannot run is a BackendError with the reason.

    The cards are the ones `MNM_GPU` (`gpu`) selects, so on a box with an NVIDIA and an
    AMD card each runs its own backend (`MNM_GPU=0` CUDA, `MNM_GPU=1` Vulkan); a choice of both
    is refused (an instance runs on one maker's cards). On an APU `MNM_BACKEND=rocm` stays
    Vulkan until `rocm_measured` (a tune on this box adopted ROCm): the APU rule."""
    used = used_cards(found, gpu)
    makers = {c.vendor for c in used.cards}
    if len(makers) > 1:
        raise BackendError(
            f"MNM_GPU={gpu or '0'} names NVIDIA and AMD cards together, and an instance runs on "
            "one maker's cards (their images differ): set MNM_GPU to one maker's cards, and "
            "pin instances to the others with `mnm set <name> gpus=`"
        )
    wanted = parse_backend(chosen or "")
    if wanted is None:
        return natural(used)
    problem = refusal(used, wanted)
    if problem is not None:
        raise BackendError(f"MNM_BACKEND={wanted.value} cannot run on this box: {problem}")
    if wanted is Backend.ROCM and apu_only(used) and not rocm_measured:
        return Backend.VULKAN
    return wanted


def used_cards(found: MachineFingerprint, gpu: str | None) -> MachineFingerprint:
    """The box as `MNM_GPU` uses it: its cards narrowed to the chosen ones (the first card
    when unset, as every reading is). A box of one maker's cards is returned whole; a value
    that is not a card list leaves every card (the doctor says why)."""
    if len({c.vendor for c in found.cards}) < 2:
        return found
    try:
        choice = parse_choice(gpu or "0")
    except ValueError:
        return found
    wanted = set(choice.resolve([c.index for c in found.cards]))
    return found.model_copy(update={"cards": [c for c in found.cards if c.index in wanted]})


def apu_only(found: MachineFingerprint) -> bool:
    """Every card is an APU planning in its GTT."""
    return bool(found.cards) and all(c.memory is MemoryKind.CARVE_OUT for c in found.cards)


def refusal(found: MachineFingerprint, wanted: Backend) -> str | None:
    """Why `wanted` cannot run on these cards; None when it can."""
    check = _CHECKS.get(wanted)
    return check(found) if check is not None else f"{label(wanted)} is not built yet"


def _vendors(found: MachineFingerprint) -> set[Vendor]:
    return {c.vendor for c in found.cards}


def _cuda12(found: MachineFingerprint) -> str | None:
    if Vendor.NVIDIA not in _vendors(found):
        return "it has no NVIDIA card, and the CUDA images need one"
    return None


def _cuda13(found: MachineFingerprint) -> str | None:
    return _cuda12(found) or cuda13_problem(found)


def _rocm(found: MachineFingerprint) -> str | None:
    amd = [c for c in found.cards if c.vendor is Vendor.AMD]
    if not amd:
        return "it has no AMD card, and the ROCm images need one"
    outside = [c for c in amd if c.gfx_target and c.gfx_target not in ROCM_TARGETS]
    if outside:
        names = ", ".join(f"{c.short} ({c.gfx_target})" for c in outside)
        return f"llama.cpp's ROCm image is not built for {names}: the Vulkan image runs it"
    return None


def _vulkan(found: MachineFingerprint) -> str | None:
    vendors = _vendors(found)
    if Vendor.NVIDIA in vendors:
        return "its NVIDIA cards run the CUDA images, which are faster there"
    if not vendors and os.environ.get(SOFTWARE_SEAM) != "1":
        return (
            "it has no card: Vulkan would run on Mesa's software device, slower than the CPU "
            "backend"
        )
    return None


_CHECKS: dict[Backend, Callable[[MachineFingerprint], str | None]] = {
    Backend.CPU: lambda _found: None,  # every box runs the CPU images
    Backend.CUDA12: _cuda12,
    Backend.CUDA13: _cuda13,
    Backend.ROCM: _rocm,
    Backend.VULKAN: _vulkan,
}


def software(found: MachineFingerprint | None, backend: Backend) -> bool:
    """Vulkan on Mesa's software device (lavapipe): Vulkan on a box without a card."""
    return backend is Backend.VULKAN and found is not None and not found.cards


def words(backend: Backend, found: MachineFingerprint | None = None) -> str:
    """`CUDA 13`, `Vulkan on Mesa's software device: slow`, `CPU (no card)`."""
    if software(found, backend):
        return "Vulkan on Mesa's software device: slow"
    if backend is Backend.CPU:
        return "CPU (no card)" if found is not None and not found.cards else "CPU"
    return label(backend)


def device_name(backend: Backend, k: int) -> str:
    """llama.cpp's name for the `k`th device the container sees: `CUDA0`, `ROCm1`, `Vulkan0`."""
    return f"{DEVICE_PREFIX.get(backend, 'CUDA')}{k}"


@dataclass(frozen=True)
class HostDevices:
    """What the ROCm and Vulkan arguments read from the host (`hardware.devices`): each AMD
    card's render node in bus order, the `render` and `video` groups' numbers, and the Vulkan
    loader's device order (indices of the physical devices for each card, in card order)."""

    # `/dev/dri/renderD128`, … for each card in the probe's order (from the probe; an
    # NVIDIA card's is empty), else every AMD card's by PCI address (read from sysfs)
    render_nodes: tuple[str, ...] = ()
    group_ids: tuple[int, ...] = ()  # render's, then video's: what `/etc/group` says
    vulkan_indices: tuple[int, ...] | None = None  # None: the loader was not asked
    software: bool = False  # Vulkan without a card: Mesa's software device (lavapipe)
    software_index: int = 0  # its place among the loader's devices


def device_argv(
    backend: Backend, cards: tuple[int, ...] | None, host: HostDevices | None = None
) -> list[str]:
    """What `docker run` is told to reach the chosen cards (`cards`: their indices in the
    probe's order; None for every card)."""
    host = host or HostDevices()
    if backend in CUDA:
        return ["--gpus", device_spec(cards), *device_env(cards)]
    if backend is Backend.ROCM:
        return _rocm_argv(cards, host)
    if backend is Backend.VULKAN:
        return _vulkan_argv(cards, host)
    return []  # CPU (and SYCL, never chosen): the image needs no device


def _rocm_argv(cards: tuple[int, ...] | None, host: HostDevices) -> list[str]:
    nodes = host.render_nodes
    wanted = cards if cards is not None else tuple(range(len(nodes)))
    chosen = [nodes[i] for i in wanted if i < len(nodes) and nodes[i]]
    argv = ["--device", "/dev/kfd"]
    for node in chosen or ["/dev/dri"]:  # nodes unread: every card (the whole folder)
        argv += ["--device", node]
    for gid in host.group_ids:
        argv += ["--group-add", str(gid)]
    count = len(chosen) if chosen else len(cards or ())
    if count:  # the passed nodes are the container's cards 0..n-1, in bus order
        argv += ["--env", "HIP_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(count))]
    return argv


def _vulkan_argv(cards: tuple[int, ...] | None, host: HostDevices) -> list[str]:
    if host.software:
        # no card (the software device): nothing to map; lavapipe is a CPU-type device, which
        # ggml-vulkan uses only when named (`ggml-vulkan.cpp`, the visible-devices branch)
        return ["--env", f"GGML_VK_VISIBLE_DEVICES={host.software_index}"]
    indices = vulkan_order(cards, host.vulkan_indices)
    argv = ["--device", "/dev/dri"]
    for gid in host.group_ids:  # The card's device files are the render/video groups'
        argv += ["--group-add", str(gid)]
    if indices:
        argv += ["--env", "GGML_VK_VISIBLE_DEVICES=" + ",".join(str(i) for i in indices)]
    return argv


def vulkan_order(cards: Sequence[int] | None, loader: Sequence[int] | None) -> list[int]:
    """The Vulkan device index of each chosen card: the loader's order where it was read
    (`loader[i]` is card i's), else the cards' own indices (PCI order, usually the loader's)."""
    if cards is None:
        return list(loader) if loader else []
    if loader is None:
        return list(cards)
    return [loader[i] for i in cards if i < len(loader)]
