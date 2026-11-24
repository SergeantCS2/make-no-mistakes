"""What the ROCm and Vulkan device arguments read from the host (`backend.HostDevices`):
each AMD card's render node from sysfs, the `render` and `video` groups' numbers from
`/etc/group`, and the Vulkan loader's device order from `vulkaninfo --summary`.

- **Render nodes** (the kernel's DRM sysfs): `/sys/class/drm/renderD*/device`
  links to the card's PCI device; its `vendor` file reads `0x1002` for AMD. The nodes are sorted
  by PCI address, the order the cards are numbered in (`CUDA_DEVICE_ORDER=PCI_BUS_ID`'s
  equivalent: pass the nodes in bus order, the container sees them as cards 0..n-1).
- **Groups:** ROCm's Docker how-to adds `--group-add render --group-add video`; the images do
  not know the names, so the numbers come from the host's `/etc/group`.
- **The Vulkan order** may differ from PCI order (an iGPU beside the card):
  `GGML_VK_VISIBLE_DEVICES` indexes the loader's physical devices
  (`ggml-vulkan.cpp`, `enumeratePhysicalDevices`). `vulkaninfo --summary` lists them as `GPU0:`,
  `GPU1:` … with `vendorID` and `deviceType`; a card is matched to the first unused device of
  its vendor, in order. Without `vulkaninfo` (vulkan-tools, which setup installs) the
  cards' own order stands.

Every read goes through a `FileReader` or a `CommandRunner`: nothing here is cached; `backend_now`
caches what it renders with."""

from __future__ import annotations

import os
import re
from pathlib import Path

from pydantic import BaseModel

from makenomistakes.hardware.fingerprint import CardId, Vendor
from makenomistakes.hardware.probe import CommandRunner, FileReader, read_text_or_empty

DRM = "sys/class/drm"
GROUP_FILE = "etc/group"
VULKANINFO = ["vulkaninfo", "--summary"]
AMD_PCI = "0x1002"
VENDOR_IDS = {Vendor.AMD: 0x1002, Vendor.NVIDIA: 0x10DE, Vendor.INTEL: 0x8086}
GROUPS = ("render", "video")
_GPU = re.compile(r"^GPU(\d+):\s*$")
_FIELD = re.compile(r"^\s*(\w+)\s*=\s*(.*?)\s*$")


class VulkanDevice(BaseModel):
    index: int
    name: str = ""
    vendor_id: int | None = None
    device_type: str = ""  # PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, …_INTEGRATED_GPU, …_CPU
    driver: str = ""  # radv, llvmpipe, NVIDIA

    @property
    def software(self) -> bool:
        return self.device_type.endswith("_CPU")


def render_nodes(root: Path, read_text: FileReader = read_text_or_empty) -> tuple[str, ...]:
    """Each AMD card's `/dev/dri/renderD*`, by PCI address; empty when sysfs shows none."""
    found: list[tuple[str, str]] = []
    drm = root / DRM
    try:
        entries = sorted(drm.glob("renderD*"))
    except OSError:
        return ()
    for entry in entries:
        device = entry / "device"
        if read_text(device / "vendor").strip().lower() != AMD_PCI:
            continue
        address = os.path.basename(os.path.realpath(device))
        found.append((address, f"/dev/dri/{entry.name}"))
    return tuple(node for _, node in sorted(found))


def groups(text: str) -> dict[str, tuple[int, tuple[str, ...]]]:
    """The `render` and `video` groups of `/etc/group`'s text: each one's number and members
    (the doctor's and setup's reading of who may open the card)."""
    found: dict[str, tuple[int, tuple[str, ...]]] = {}
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and parts[0] in GROUPS and parts[2].isdigit():
            members = parts[3].split(",") if len(parts) > 3 else []
            found[parts[0]] = (int(parts[2]), tuple(m for m in members if m))
    return found


def group_ids(text: str) -> tuple[int, ...]:
    """The `render` and `video` groups' numbers from `/etc/group`'s text, in that order; a
    group the host lacks is left out."""
    found = groups(text)
    return tuple(found[name][0] for name in GROUPS if name in found)


def parse_vulkan_summary(text: str) -> list[VulkanDevice]:
    """`vulkaninfo --summary`'s Devices section, one entry per `GPUn:` block."""
    devices: list[VulkanDevice] = []
    current: dict[str, str] | None = None
    index = -1

    def close() -> None:
        if current is not None:
            vendor = current.get("vendorID", "")
            devices.append(
                VulkanDevice(
                    index=index,
                    name=current.get("deviceName", ""),
                    vendor_id=int(vendor, 16) if re.fullmatch(r"0x[0-9a-fA-F]+", vendor) else None,
                    device_type=current.get("deviceType", ""),
                    driver=current.get("driverName", ""),
                )
            )

    for line in text.splitlines():
        head = _GPU.match(line.strip())
        if head is not None:
            close()
            index, current = int(head.group(1)), {}
            continue
        field = _FIELD.match(line)
        if current is not None and field is not None:
            current.setdefault(field.group(1), field.group(2))
    close()
    return devices


def loader_indices(cards: list[CardId], devices: list[VulkanDevice]) -> tuple[int, ...]:
    """Each card's place in the loader's list: the first unused device of the card's vendor
    that is not a software one, in card order; a card with no match keeps its own index."""
    used: set[int] = set()
    found: list[int] = []
    for card in sorted(cards, key=lambda c: c.index):
        vendor = VENDOR_IDS.get(card.vendor)
        match = next(
            (
                d.index
                for d in devices
                if d.index not in used and not d.software and d.vendor_id == vendor
            ),
            None,
        )
        index = match if match is not None else card.index
        used.add(index)
        found.append(index)
    return tuple(found)


def software_index(devices: list[VulkanDevice]) -> int:
    """The software device's place (lavapipe), 0 when the loader was not asked."""
    return next((d.index for d in devices if d.software), 0)


def vulkan_devices(runner: CommandRunner) -> list[VulkanDevice] | None:
    """The loader's devices by `vulkaninfo --summary` on the host; None when the tool is
    missing or failed (the cards' own order stands then)."""
    try:
        code, out, _ = runner.run(VULKANINFO, timeout=20.0)
    except OSError:
        return None
    return parse_vulkan_summary(out) if code == 0 else None
