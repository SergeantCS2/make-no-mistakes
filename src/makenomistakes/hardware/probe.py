"""Probe the machine: GPU, memory, CPU, WSL networking, Docker, disk.

Parsing is separated from execution so every parser is unit-testable on captured
output, and `probe()` takes its command runner and file reader as arguments so tests
can drive it without touching the host.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, field_validator

from makenomistakes.hardware.fingerprint import Vendor
from makenomistakes.hardware.gpu_select import CardChoice, chosen, device_spec
from makenomistakes.hardware.proc_parsers import (
    DockerInfo,
    NetworkingMode,
    is_wsl_kernel,
    parse_cgroup_limit,
    parse_docker_info,
    parse_networking_mode,
    parse_ufw_enabled,
)
from makenomistakes.hardware.runner import SubprocessRunner
from makenomistakes.hardware.windows_ram import windows_ram
from makenomistakes.private_ids import without_card_ids

__all__ = [  # `proc_parsers`' and `runner`'s names, found here too
    "DockerInfo",
    "NetworkingMode",
    "SubprocessRunner",
    "is_wsl_kernel",
    "parse_cgroup_limit",
    "parse_docker_info",
    "parse_networking_mode",
    "parse_ufw_enabled",
]

MIB = 1024 * 1024
NVIDIA_SMI_CANDIDATES = ("nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/bin/nvidia-smi")
# Every card with its index, compute capability, bus id and uuid, and its PCIe link
# (width and generation, now and at most). An nvidia-smi too old for `compute_cap` refuses
# the whole query, and the earlier one is asked instead (the link fields are older still).
NVIDIA_SMI_QUERY = (
    "--query-gpu=index,name,memory.total,memory.used,driver_version,compute_cap,pci.bus_id,uuid,"
    "pcie.link.width.current,pcie.link.width.max,pcie.link.gen.current,pcie.link.gen.max,"
    "utilization.gpu"
)
NVIDIA_SMI_QUERY_OLD = "--query-gpu=name,memory.total,memory.used,driver_version"
NVIDIA_SMI_FORMAT = "--format=csv,noheader,nounits"
CGROUP_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
PROC_VERSION = Path("/proc/version")
PROC_MEMINFO = Path("/proc/meminfo")
UFW_CONF = Path("/etc/ufw/ufw.conf")
GPU_SMOKE_IMAGE = "nvidia/cuda:12.8.1-base-ubuntu24.04"
NO_NVIDIA_SMI = "nvidia-smi not found"


RAM_INDEX = -1  # The RAM's place among the cards (`MemorySnapshot.ram`); never a card's


class Gpu(BaseModel):
    """One card as nvidia-smi lists it (`index` is nvidia-smi's, or the card's place in
    the list when an older query had none)."""

    name: str
    vram_total_bytes: int
    vram_used_bytes: int
    driver_version: str | None = None
    index: int = 0
    compute_cap: str | None = None  # "8.9"; None from an nvidia-smi that cannot say
    bus_id: str | None = None  # "00000000:01:00.0"
    uuid: str | None = None
    # The PCIe link nvidia-smi reports (`pcie.link.width.current` …); None for `[N/A]`
    # (some read so under WSL) or an older query
    lanes: int | None = None  # x16, x8, x4, x1 now
    lanes_max: int | None = None
    pcie_gen: int | None = None  # 5 for PCIe 5.0, now (a card idles at a lower one)
    pcie_gen_max: int | None = None
    # `memory.total`/`used` answered `[N/A]` (a DGX Spark's GB10 shares the machine's
    # RAM): the card is kept, and its memory is read as the pool's (`unified_pool.pooled`:
    # MemTotal less the firm margin, MemTotal less MemAvailable in use); 0 bytes until then
    unified: bool = False
    # `utilization.gpu`, the percent of the last sample period a kernel ran (a
    # browser's video on Windows shows here); None for `[N/A]` or an older query
    busy_percent: int | None = None
    # An AMD card read from sysfs (`hardware.amd_probe`): its gfx target, the GTT (system
    # memory the card maps), the VRAM the CPU sees (Resizable BAR), its render node, the
    # driver's `unique_id` (a secret like a uuid); `carve_out` an APU whose VRAM figures are
    # its carve-out plus its GTT (the pool it plans in, shared with the RAM)
    vendor: Vendor = Vendor.NVIDIA
    gfx_target: str | None = None
    gtt_total_bytes: int | None = None
    gtt_used_bytes: int | None = None
    vis_vram_total_bytes: int | None = None
    render_node: str | None = None
    unique_id: str | None = None
    carve_out: bool = False

    @property
    def in_ram(self) -> bool:
        """Not a card: the RAM, where a model on the processor's image sits."""
        return self.index == RAM_INDEX

    @property
    def is_amd(self) -> bool:
        return self.vendor is Vendor.AMD

    @property
    def vram_free_bytes(self) -> int:
        return max(self.vram_total_bytes - self.vram_used_bytes, 0)


class MemoryInfo(BaseModel):
    total_bytes: int
    available_bytes: int | None = None


class DiskInfo(BaseModel):
    path: str
    exists: bool
    writable: bool
    total_bytes: int = 0
    free_bytes: int = 0


class ProbeError(BaseModel):
    message: str  # a probe that could not run: what the tool said


class SmokeResult(BaseModel):
    ok: bool
    output: str  # what the container printed, its cards' uuids left out (`--json` dumps it)

    @field_validator("output")
    @classmethod
    def _without_card_ids(cls, text: str) -> str:
        return without_card_ids(text)


class HardwareReport(BaseModel):
    is_wsl: bool
    wsl_networking_mode: NetworkingMode | None
    cpu_threads: int | None
    memory: MemoryInfo | None
    cgroup_limit_bytes: int | None
    gpus: list[Gpu] | ProbeError
    # The indices `MNM_GPU` names, in its order (first in `gpus`); empty: the first card
    chosen: tuple[int, ...] = ()
    unchosen: list[Gpu] | None = None  # as `MemorySnapshot.unchosen`
    docker: DockerInfo | ProbeError
    docker_gpu_smoke: SmokeResult | None
    disk: DiskInfo
    ufw_enabled: bool = False  # /etc/ufw/ufw.conf says ENABLED=yes (a native box)
    # Under WSL, the Windows PC's own RAM (the VM sees its share; the doctor asks no more)
    host_ram_bytes: int | None = None

    @property
    def box_cards(self) -> list[Gpu] | ProbeError:
        """As `MemorySnapshot.box_cards`."""
        return self.unchosen if self.unchosen is not None else self.gpus


class CommandRunner(Protocol):
    def run(self, argv: list[str], timeout: float = 15.0) -> tuple[int, str, str]: ...


class LineRunner(CommandRunner, Protocol):
    """A runner that also hands over a command's output as it comes (a silent 10-30 minute
    `docker build` looks hung); ComfyUI's engine streams its builds, `run`s the rest."""

    def stream(
        self,
        argv: list[str],
        on_line: Callable[[str | None], None],
        timeout: float,
        quiet: float = 60.0,
    ) -> int:
        """Run `argv` (stderr into stdout), `on_line` each line as printed and None after every
        `quiet` seconds without one; the exit code (124 at `timeout`, 127 if missing)."""
        ...


FileReader = Callable[[Path], str]


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# --- parsers (pure) ---------------------------------------------------------------


def parse_nvidia_smi(output: str) -> list[Gpu]:
    """Either query's lines: `index, name, total, used, driver, compute_cap, bus id, uuid`
    or the earlier `name, total, used, driver` (each card's place is its index)."""
    gpus: list[Gpu] = []
    for line in output.splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 8 and cells[0].isdigit():
            gpu = _card(cells[1:5], int(cells[0]))
            if gpu is not None:
                found = {**_extra(cells[5:8]), **_link(cells[8:12]), **_busy(cells[12:13])}
                gpus.append(gpu.model_copy(update=found))
            continue
        gpu = _card(cells, len(gpus)) if len(cells) >= 3 else None
        if gpu is not None:
            gpus.append(gpu)
    return gpus


def _card(cells: list[str], index: int) -> Gpu | None:
    unified = all(c.startswith("[") for c in cells[1:3])  # `[N/A]`: unified memory
    try:
        total, used = [0, 0] if unified else [int(float(c)) * MIB for c in cells[1:3]]
    except ValueError:
        return None
    driver = cells[3] if len(cells) > 3 and cells[3] else None
    return Gpu(
        name=cells[0],
        vram_total_bytes=total,
        vram_used_bytes=used,
        driver_version=driver,
        index=index,
        unified=unified,
    )


def _extra(cells: list[str]) -> dict[str, str | None]:
    """compute_cap, bus id and uuid; nvidia-smi's `[N/A]` is nothing."""
    values = [None if not c or c.startswith("[") else c for c in cells]
    return {"compute_cap": values[0], "bus_id": values[1], "uuid": values[2]}


def _link(cells: list[str]) -> dict[str, int | None]:
    """The PCIe link's four numbers; `[N/A]`, or a query without them, is None."""
    keys = ("lanes", "lanes_max", "pcie_gen", "pcie_gen_max")
    return {
        k: int(c) if c.isdigit() else None
        for k, c in zip(keys, [*cells, "", "", "", ""][:4], strict=True)
    }


def _busy(cells: list[str]) -> dict[str, int | None]:
    """`utilization.gpu`; `[N/A]`, or a query without it, is None."""
    return {"busy_percent": int(cells[0]) if cells and cells[0].isdigit() else None}


def parse_meminfo(text: str) -> MemoryInfo | None:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        number = rest.strip().split(" ")[0]
        if key in {"MemTotal", "MemAvailable"} and number.isdigit():
            values[key] = int(number) * 1024
    if "MemTotal" not in values:
        return None
    return MemoryInfo(total_bytes=values["MemTotal"], available_bytes=values.get("MemAvailable"))


class MemorySnapshot(BaseModel):
    """GPU and RAM occupancy right now: what the UI polls and what fit-now verdicts use."""

    gpus: list[Gpu] | ProbeError
    memory: MemoryInfo | None
    chosen: tuple[int, ...] = ()  # as `HardwareReport.chosen`
    # every card listed, when `MNM_GPU` could not choose among them (`gpus` then says why)
    unchosen: list[Gpu] | None = None
    # The RAM as one more place a model sits (`ram_room.with_ram`): a model on the
    # processor's image holds its room here, at `RAM_INDEX`, beside the cards' (never in
    # `gpus`, which stay as nvidia-smi read them); None when the RAM was not read
    ram: Gpu | None = None

    @property
    def box_cards(self) -> list[Gpu] | ProbeError:
        """The cards the box's fingerprint is read from: every card listed, whatever
        `MNM_GPU` says (a stale value must never make the box unreadable)."""
        return self.unchosen if self.unchosen is not None else self.gpus

    @property
    def gpu(self) -> Gpu | None:
        """The first chosen card: what a one-card box plans, reads and runs on."""
        return self.gpus[0] if isinstance(self.gpus, list) and self.gpus else None

    @property
    def cards(self) -> list[Gpu]:
        """Every chosen card in `MNM_GPU`'s order; the first card alone without a
        choice recorded."""
        return chosen_cards(self.gpus, self.chosen)

    def listed(self) -> list[Gpu]:
        """Every card nvidia-smi listed (none when it could not be read)."""
        return self.gpus if isinstance(self.gpus, list) else []

    def card(self, index: int) -> Gpu | None:
        """The card nvidia-smi lists at `index`, None when it lists none there; the RAM at
        `RAM_INDEX`."""
        if index == RAM_INDEX:
            return self.ram
        if not isinstance(self.gpus, list):
            return None
        return next((g for g in self.gpus if g.index == index), None)

    @property
    def gpu_error(self) -> str | None:
        """Why there is no GPU to read, for every surface that has to say so."""
        if isinstance(self.gpus, ProbeError):
            return self.gpus.message
        return None if self.gpus else "no GPU listed"


# --- probe (impure) ---------------------------------------------------------------


def probe_memory(
    runner: CommandRunner | None = None, read_text: FileReader = read_text_or_empty
) -> MemorySnapshot:
    """nvidia-smi and /proc/meminfo only: cheap enough to call on every page load. A unified
    card (a GB10) reads its memory as the pool's (`unified_pool.pooled`)."""
    from makenomistakes.hardware.unified_pool import pooled  # noqa: PLC0415 - it imports this

    runner = runner or SubprocessRunner()
    gpus, chosen, unchosen = _probe_cards(runner)
    memory = parse_meminfo(read_text(PROC_MEMINFO))
    return MemorySnapshot(
        gpus=pooled(gpus, memory, runner),
        chosen=chosen,
        memory=memory,
        unchosen=None if unchosen is None else pooled(unchosen, memory, runner),
    )


def probe(
    models_dir: Path,
    runner: CommandRunner | None = None,
    read_text: FileReader = read_text_or_empty,
    docker_gpu_smoke: bool = False,
    host_ram: bool = False,
) -> HardwareReport:
    """What this machine has. `docker_gpu_smoke` runs a container on the card; `host_ram`
    asks Windows for the PC's RAM under WSL (the doctor's; a second or so through interop)."""
    from makenomistakes.hardware import probe_parts as parts  # noqa: PLC0415 - it imports this
    from makenomistakes.hardware.unified_pool import pooled  # noqa: PLC0415 - it imports this

    runner = runner or SubprocessRunner()
    is_wsl = is_wsl_kernel(read_text(PROC_VERSION))
    docker = parts.probe_docker(runner)
    memory = parse_meminfo(read_text(PROC_MEMINFO))
    gpus, chosen, unchosen = _probe_cards(runner)
    return HardwareReport(
        is_wsl=is_wsl,
        wsl_networking_mode=networking_mode(runner) if is_wsl else None,
        cpu_threads=os.cpu_count(),
        memory=memory,
        cgroup_limit_bytes=parse_cgroup_limit(read_text(CGROUP_MEMORY_MAX)),
        gpus=pooled(gpus, memory, runner),  # A unified card's memory is the RAM's
        chosen=chosen,
        unchosen=None if unchosen is None else pooled(unchosen, memory, runner),
        docker=docker,
        docker_gpu_smoke=(
            parts.docker_gpu_smoke(runner)
            if docker_gpu_smoke and isinstance(docker, DockerInfo)
            else None
        ),
        disk=parts.probe_disk(models_dir),
        ufw_enabled=parse_ufw_enabled(read_text(UFW_CONF)),
        host_ram_bytes=windows_ram(runner) if host_ram and is_wsl else None,
    )


def networking_mode(runner: CommandRunner) -> NetworkingMode:
    """WSL's networking mode (`wslinfo`): the doctor's and the router manifest's one reading."""
    code, out, _ = runner.run(["wslinfo", "--networking-mode"])
    return parse_networking_mode(out) if code == 0 else NetworkingMode.UNKNOWN


def chosen_card() -> int:
    """The first card `MNM_GPU` names (0 when unset); a ValueError naming a wrong value."""
    return chosen().first


def gpu_device(cards: tuple[int, ...] | None = None) -> str:
    """`docker run --gpus` for an engine on `cards` (the first chosen card when not given):
    `device=N` for one card; several as `device_spec` words them."""
    return device_spec(cards if cards is not None else (chosen_card(),))


def chosen_cards(gpus: list[Gpu] | ProbeError, chosen: tuple[int, ...]) -> list[Gpu]:
    """The chosen cards among `gpus` in that order; the first listed without a choice."""
    if not isinstance(gpus, list) or not gpus:
        return []
    if not chosen:
        return gpus[:1]
    by_index = {g.index: g for g in gpus}
    return [by_index[i] for i in chosen if i in by_index]


def choose(gpus: list[Gpu], index: int | CardChoice) -> list[Gpu] | ProbeError:
    """The chosen cards first, in `MNM_GPU`'s order (what every reading and plan reads), the
    others after them; a card nvidia-smi does not list is said."""
    choice = index if isinstance(index, CardChoice) else CardChoice(indices=(index,))
    if len({g.index for g in gpus}) != len(gpus):  # built without indices: their places
        gpus = [g.model_copy(update={"index": i}) for i, g in enumerate(gpus)]
    listed = [g.index for g in gpus]
    wanted = choice.resolve(listed)
    missing = [i for i in wanted if i not in listed]
    if missing:
        return ProbeError(
            message=f"MNM_GPU={','.join(str(i) for i in wanted)}, but nvidia-smi lists "
            f"{len(gpus)} card{'s' if len(gpus) != 1 else ''} (indices "
            f"{', '.join(str(i) for i in listed) or 'none'})"  # AMD's need not follow on
        )
    first = [g for i in wanted for g in gpus if g.index == i]
    return first + [g for g in gpus if g.index not in wanted]


def listed_cards(runner: CommandRunner) -> list[Gpu] | ProbeError:
    """Every card nvidia-smi lists and every AMD card sysfs shows (after them),
    whatever `MNM_GPU` says; `NO_NVIDIA_SMI`: neither."""
    from makenomistakes.hardware.amd_probe import with_amd  # noqa: PLC0415 - it imports this

    return with_amd(_nvidia_cards(runner), runner)


def _nvidia_cards(runner: CommandRunner) -> list[Gpu] | ProbeError:
    last_error = NO_NVIDIA_SMI
    for binary in NVIDIA_SMI_CANDIDATES:
        if "/" not in binary and shutil.which(binary) is None:
            continue
        code, out, err = runner.run([binary, NVIDIA_SMI_QUERY, NVIDIA_SMI_FORMAT])
        if code != 0:  # an nvidia-smi without `compute_cap`: the earlier query
            code, out, err = runner.run([binary, NVIDIA_SMI_QUERY_OLD, NVIDIA_SMI_FORMAT])
        if code == 0:
            return parse_nvidia_smi(out)
        last_error = (err or out).strip() or f"{binary} exited with {code}"
    return ProbeError(message=last_error)


Cards = tuple[list[Gpu] | ProbeError, tuple[int, ...], list[Gpu] | None]


def _probe_cards(runner: CommandRunner) -> Cards:
    """Every card nvidia-smi lists, the chosen ones first, and the chosen indices; and the
    cards as listed when `MNM_GPU` could not choose among them (a stale value after a card
    went: the fingerprint is read from them), else None."""
    cards = listed_cards(runner)
    if isinstance(cards, ProbeError):
        return cards, (), None
    try:
        choice = chosen()
    except ValueError as exc:  # MNM_GPU is not a card list: every reading says so
        return ProbeError(message=str(exc)), (), cards
    found = choose(cards, choice)
    wanted = choice.resolve([g.index for g in cards])
    if isinstance(found, ProbeError):
        return found, (), cards
    return found, (wanted if choice.several else ()), None
