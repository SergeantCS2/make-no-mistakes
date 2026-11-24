"""What mnm knows about the boxes it ran on, in `<state>/machine.json` (schema-versioned): the
box as it was last read (`current`, its fingerprint and when it was taken), every box
this state folder has seen (`seen`, newest last, each with its first and last time and what
mnm measured on it), and the two measured numbers: the RAM bandwidth (`mnm doctor --deep`)
and the rate RAM-held experts decode at (`mnm calibration`, learned from MoE readings).

Schema 2 keeps each number under the fingerprint it was measured on; schema 1's two
numbers, which named no box, are kept `unclaimed` until the first fingerprint is taken, which
claims them. A changed box is recorded and said, never acted on here; what a plan reads
(`load`, `expert_rate`) is the current box's own figure, else the constant (another
box's figure never stands in).

The planner prices experts in RAM at the rate learned from this box's own readings, which
no probe caps (readings are the truth: a probe that read 3.0 GB/s would cap a learned 19.55
to it). Without readings it keeps `CPU_EXPERT_RATE` (an RTX 4070 box's 30 GB/s under the
current expert model, `fit.cpu_rates`), capped at the measured bandwidth when one is saved; a
rate learned under the older model is kept and not used; a probe reading under
`PLAUSIBLE_RAM_GBPS` is never saved. Native Linux, DDR5 and NUMA boxes differ."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.fit.budget import Budget
from makenomistakes.fit.cpu_rates import CPU_BANDWIDTH, CPU_EXPERT_RATE, EXPERT_MODEL, layer_rate
from makenomistakes.fit.device_row import Learned
from makenomistakes.fit.devices import device_for
from makenomistakes.hardware.fingerprint import MachineFingerprint
from makenomistakes.hardware.ram_platforms import RamPlatform, platform_of
from makenomistakes.locking import shared_lock

SCHEMA_VERSION = 2
FILE = "machine.json"
LOCK_FOLDER = "machine"  # the lock's own folder: the state folder's root is nobody's lock
# Below this a probe reading says more about the probe than the RAM: one DDR4 channel reads
# about 15 GB/s, and a reading of 3.0 measures page faults, not the RAM.
PLAUSIBLE_RAM_GBPS = 8.0
ALL_CORES_PROBE = 2  # `hardware.ram_bandwidth.PROBE`: a figure over every physical core
_RAM = ("ram_gbps", "ram_one_thread_gbps", "ram_probe", "ram_measured_at")
_EXPERTS = ("expert_gbps", "expert_readings", "expert_learned_at", "expert_model")
_DECODE = (
    "decode_efficiency",
    "launch_seconds",
    "decode_readings",
    "decode_learned_at",
    "decode_card",
)
_FIELDS = (*_RAM, *_EXPERTS, *_DECODE)


class MachineFactsError(ValueError):
    pass


class MachineFacts(BaseModel):
    """What was measured on one box (schema 1's whole file)."""

    ram_gbps: float | None = Field(default=None, gt=0)  # read + write, `ram_bandwidth`
    # when the probe last ran here, also when its reading was implausible (`ram_gbps` None)
    ram_measured_at: datetime | None = None
    expert_gbps: float | None = Field(default=None, gt=0)  # learned from MoE readings
    expert_readings: int = 0
    expert_learned_at: datetime | None = None
    # The expert speed model the rate was learned under (`fit.cpu_rates.EXPERT_MODEL`);
    # None is the older model (the 0.05 floor), whose rate is not used: `mnm calibration` relearns
    expert_model: int | None = None
    # The probe on one thread, beside `ram_gbps` (every physical core, up to 16)
    ram_one_thread_gbps: float | None = Field(default=None, gt=0)
    ram_probe: int | None = None  # `ram_bandwidth.PROBE` of `ram_gbps`; None: one thread
    # This box's decode efficiency (and launch term, from two model sizes) on its card,
    # learned from readings with every block on the card (`device_learning`)
    decode_efficiency: float | None = Field(default=None, gt=0)
    launch_seconds: float | None = Field(default=None, gt=0)
    decode_readings: int = 0
    decode_learned_at: datetime | None = None
    decode_card: str | None = None  # nvidia-smi's name of the card they were read on

    @property
    def expert_rate(self) -> float:
        """Bytes a second the planner reads RAM-held experts at (`Budget.expert_rate`): the
        learned rate as it is, else the constant capped at the measured bandwidth."""
        if self.expert_gbps and self.expert_model == EXPERT_MODEL:
            return self.expert_gbps * 1e9
        ram = self.ram_gbps if plausible(self.ram_gbps) else None  # an older file's 3.0
        return min(CPU_EXPERT_RATE, ram * 1e9) if ram else CPU_EXPERT_RATE

    @property
    def capped(self) -> bool:
        """The planner's rate is the probe's bandwidth, not the learned one or the constant."""
        return not self.learned_experts and self.expert_rate < CPU_EXPERT_RATE

    @property
    def learned_experts(self) -> bool:
        """A learned rate this mnm prices with (its speed model's)."""
        return bool(self.expert_gbps) and self.expert_model == EXPERT_MODEL

    @property
    def stale_experts(self) -> bool:
        """A rate learned under an older expert model (the 0.05 floor), kept and not used."""
        return bool(self.expert_gbps) and self.expert_model != EXPERT_MODEL

    @property
    def every_core_gbps(self) -> float | None:
        """The probe's figure over every physical core, when plausible; None for a
        one-thread figure or none."""
        every = self.ram_probe is not None and self.ram_probe >= ALL_CORES_PROBE
        return self.ram_gbps if every and plausible(self.ram_gbps) else None

    def priced(self, platform: RamPlatform | None) -> Priced:
        """What a plan reads RAM-held weights at on `platform`: the rate learned from
        this box's readings first, then the every-core probe x the platform's share, then the
        platform's default (a one-thread probe caps it); whole blocks at the
        every-core probe x `LLAMA_RAM_SHARE`, else the platform's figure. No platform (a box
        not read): the constants an RTX 4070 box measured."""
        default = platform.expert_gbps * 1e9 if platform is not None else CPU_EXPERT_RATE
        layer = platform.layer_gbps * 1e9 if platform is not None else float(CPU_BANDWIDTH)
        every = self.every_core_gbps
        if every is not None:
            layer = layer_rate(every)
        if self.learned_experts and self.expert_gbps:
            words = (
                f"priced at {self.expert_gbps:g} GB/s, learned from {self.expert_readings} of "
                "this box's readings (`mnm calibration`)"
            )
            return Priced(self.expert_gbps * 1e9, layer, words)
        if every is not None and platform is not None:
            rate = every * platform.row.share
            words = (
                f"priced at {rate:.1f} GB/s: this box's probe ({every:g} GB/s) x "
                f"{platform.row.share:g}, what llama.cpp reads of it on {platform.row.label}, "
                "until readings teach a rate (`mnm calibration`)"
            )
            return Priced(rate * 1e9, layer, words)
        # a one-thread figure (or any, where no platform was read) caps the default
        capped = every is None or platform is None
        one = self.ram_gbps if plausible(self.ram_gbps) and capped else None
        if one is not None and one * 1e9 < default:  # an older one-thread probe caps it
            words = f"priced at {one:g} GB/s, the RAM's measured bandwidth"
            return Priced(one * 1e9, layer, words)
        if platform is not None:
            return Priced(default, layer, platform.priced_words())
        return Priced(
            default, layer, f"priced at {default / 1e9:g} GB/s, an RTX 4070 machine's figure"
        )

    def learned_device(self) -> Learned | None:
        """The decode figures learned here, for `DeviceRow.with_learned`."""
        if not self.decode_efficiency:
            return None
        at = self.decode_learned_at
        return Learned(
            efficiency=self.decode_efficiency,
            launch_seconds=self.launch_seconds,
            readings=self.decode_readings,
            at=at.date().isoformat() if at is not None else "",
        )


@dataclass(frozen=True)
class Priced:
    """RAM-held experts' and whole blocks' rates (bytes a second) and the experts' in words."""

    expert_rate: float
    layer_rate: float
    words: str


class Taken(BaseModel):
    fingerprint: MachineFingerprint
    taken_at: datetime


class SeenMachine(BaseModel):
    fingerprint: MachineFingerprint
    first_seen: datetime
    last_seen: datetime
    measured: MachineFacts = Field(default_factory=MachineFacts)  # on this box


class Adopted(BaseModel):
    """The last `mnm machine adopt`: the box it re-fitted the plans for, when, and its
    summary. Set as its last step: an adopt that failed before leaves the one before."""

    machine_id: str
    at: datetime
    summary: str


class MachineFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    current: Taken | None = None
    seen: list[SeenMachine] = Field(default_factory=list)
    # schema 1's numbers until the first fingerprint claims them
    unclaimed: MachineFacts | None = None
    # the box a recipe without `planned_on` counts as planned on: the first box mnm saw
    unstamped_on: str | None = None
    # `current` is the box as last read (every command records it); this is the box
    # the plans were last adopted on, which only `mnm machine adopt` sets
    adopted: Adopted | None = None

    def entry(self, machine_id: str) -> SeenMachine | None:
        return next((s for s in self.seen if s.fingerprint.id == machine_id), None)

    @property
    def here(self) -> SeenMachine | None:
        """The current box's entry."""
        return self.entry(self.current.fingerprint.id) if self.current else None

    def facts(self) -> MachineFacts:
        """What plans read: what was measured on the current box (never another box's);
        before any box was read, schema 1's numbers. A box with nothing measured
        reads the constants, said as estimates, until it measures its own."""
        here = self.here
        if here is not None:
            return here.measured
        if self.current is None and self.unclaimed is not None:
            return self.unclaimed
        return MachineFacts()


def plausible(gbps: float | None) -> bool:
    return gbps is not None and gbps >= PLAUSIBLE_RAM_GBPS


def load_file(state_dir: Path) -> MachineFile | None:
    """The whole file (schema 1 brought up); None before anything was taken or measured."""
    path = state_dir / FILE
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise MachineFactsError(f"{path} could not be read: {exc}") from exc
    return parse_file(text, str(path))


def parse_file(text: str, where: str) -> MachineFile:
    """A machine file's text (a backup's too), schema 1 brought up."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise MachineFactsError(f"{where} is not a machine file mnm wrote: {exc}") from exc
    version = data.get("schema_version", 1) if isinstance(data, dict) else None
    if not isinstance(version, int) or isinstance(version, bool):
        raise MachineFactsError(f"{where} is not a machine file mnm wrote: no schema_version")
    if version > SCHEMA_VERSION:
        raise MachineFactsError(f"{where} was written by a newer mnm: run `mnm update`")
    try:
        if version == 1:  # schema 1's two numbers, of no box in particular
            old = {k: v for k, v in data.items() if k in _FIELDS}
            return MachineFile(unclaimed=MachineFacts.model_validate(old))
        return MachineFile.model_validate(data)
    except ValidationError as exc:
        raise MachineFactsError(f"{where} is not a machine file mnm wrote: {exc}") from exc


def load(state_dir: Path) -> MachineFacts | None:
    """The numbers plans read (`MachineFile.facts`); None before the file exists; a newer or
    unreadable file is said."""
    found = load_file(state_dir)
    return found.facts() if found is not None else None


def change_file(state_dir: Path, change: Callable[[MachineFile], MachineFile]) -> MachineFile:
    """Read, change and write it whole under its lock (a doctor and a calibration at once)."""
    with shared_lock(state_dir / LOCK_FOLDER)():
        found = change(load_file(state_dir) or MachineFile())
        path = state_dir / FILE
        tmp = path.with_suffix(".tmp")
        tmp.write_text(found.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return found


def update(state_dir: Path, change: Callable[[MachineFacts], MachineFacts]) -> MachineFacts:
    """A measured number changed: kept under the current box (schema 1's place, `unclaimed`,
    before any box was taken); the numbers plans read now come back."""

    def apply(found: MachineFile) -> MachineFile:
        before = found.facts()
        after = change(before)
        moved = {k: getattr(after, k) for k in _FIELDS if getattr(after, k) != getattr(before, k)}
        here = found.here
        if here is not None:
            here.measured = here.measured.model_copy(update=moved)
            here.fingerprint = with_bandwidth(here.fingerprint, here.measured)
            if found.current is not None:
                found.current.fingerprint = here.fingerprint
        else:
            found.unclaimed = (found.unclaimed or MachineFacts()).model_copy(update=moved)
        return found

    return change_file(state_dir, apply).facts()


def with_ram(
    gbps: float, one_thread: float | None = None, probe: int | None = None
) -> Callable[[MachineFacts], MachineFacts]:
    """The probe's reading saved; one under `PLAUSIBLE_RAM_GBPS` clears what was saved
    instead (an earlier implausible reading included), so nothing caps the constant; its
    time stays, so another box's figure does not stand in for it. `probe` is the probe's
    kind (`ram_bandwidth.PROBE`): only every core's figure prices RAM-held blocks."""
    now = datetime.now(UTC)
    thread = round(one_thread, 1) if one_thread else None
    if not plausible(gbps):
        return lambda f: f.model_copy(
            update={"ram_gbps": None, "ram_measured_at": now, "ram_one_thread_gbps": thread}
        )
    saved = {"ram_gbps": round(gbps, 1), "ram_measured_at": now, "ram_one_thread_gbps": thread}
    return lambda f: f.model_copy(update={**saved, "ram_probe": probe})


def box_priced(budget: Budget, state_dir: Path) -> Budget:
    """`budget` priced with this box's own figures: RAM-held experts and whole
    blocks at what `MachineFacts.priced` says for this box's RAM platform, what the system
    keeps beside the models there, and the decode efficiency (and launch term) learned on
    this card over its row. Nothing saved: the constants."""
    try:
        file = load_file(state_dir)
    except MachineFactsError:
        file = None
    if file is None:
        return budget.model_copy(update={"expert_rate": CPU_EXPERT_RATE})
    found = file.facts()
    platform = platform_here(file)
    priced = found.priced(platform)
    device = budget.device
    if found.decode_card is not None and device_for(found.decode_card).name == device.name:
        device = device.with_learned(found.learned_device())
    update: dict[str, object] = {
        "expert_rate": priced.expert_rate,
        "layer_rate": priced.layer_rate,
        "device": device,
    }
    if platform is not None:
        update["ram_margin"] = platform.margin_bytes
        update["ram_note"] = f"RAM-held experts {priced.words}"
    return budget.model_copy(update=update)


def platform_here(file: MachineFile) -> RamPlatform | None:
    """The RAM platform of the box as last read; None before any box was read."""
    return platform_of(file.current.fingerprint) if file.current is not None else None


def expert_rate(state_dir: Path) -> float:
    """What a plan prices RAM-held experts at; the constant when nothing is saved or the
    file cannot be read (a plan never fails on a hint)."""
    try:
        file = load_file(state_dir)
    except MachineFactsError:
        return CPU_EXPERT_RATE
    if file is None:
        return CPU_EXPERT_RATE
    return file.facts().priced(platform_here(file)).expert_rate


@dataclass(frozen=True)
class Recorded:
    """A fingerprint just taken: the file after it, and the box that was current before."""

    file: MachineFile
    previous: MachineFingerprint | None
    new: bool  # never seen before


def record(state_dir: Path, found: MachineFingerprint, now: datetime | None = None) -> Recorded:
    """`found` as the current box: a box seen before is updated (a part this reading lacked,
    such as the RAM's channels `mnm ls` does not read, keeps what was read before) and moved
    to the end; a new one is added. The first box ever claims schema 1's numbers and the
    recipes planned before fingerprints (`unstamped_on`)."""
    when = now or datetime.now(UTC)
    said: list[tuple[MachineFingerprint | None, bool]] = []

    def apply(file: MachineFile) -> MachineFile:
        known = file.entry(found.id)
        said.append((file.current.fingerprint if file.current else None, known is None))
        if known is None:
            entry = SeenMachine(fingerprint=found, first_seen=when, last_seen=when)
            if not file.seen and file.unclaimed is not None:
                entry.measured, file.unclaimed = file.unclaimed, None
            file.unstamped_on = file.unstamped_on or found.id
        else:
            file.seen.remove(known)
            entry = known.model_copy(
                update={"fingerprint": merged(known.fingerprint, found), "last_seen": when}
            )
        entry.fingerprint = with_bandwidth(entry.fingerprint, entry.measured)
        file.seen.append(entry)
        file.current = Taken(fingerprint=entry.fingerprint, taken_at=when)
        return file

    after = change_file(state_dir, apply)
    previous, new = said[-1]
    return Recorded(file=after, previous=previous, new=new)


def merged(old: MachineFingerprint, new: MachineFingerprint) -> MachineFingerprint:
    """`new`, with each part it could not read taken from `old` (the same box, `id` equal)."""
    parts: dict[str, object] = {}
    for part in ("cpu", "ram", "board"):
        was, now = getattr(old, part), getattr(new, part)
        kept = {k: v for k, v in was.model_dump().items() if getattr(now, k) in (None, [])}
        parts[part] = now.model_copy(update=kept)
    return new.model_copy(update=parts)


def with_bandwidth(found: MachineFingerprint, facts: MachineFacts) -> MachineFingerprint:
    """The bandwidth measured on this box, when it was (the fingerprint carries it)."""
    gbps = found.ram.bandwidth_gbps  # nothing measured here: as the reading had it
    if plausible(facts.ram_gbps):
        gbps = facts.ram_gbps
    elif facts.ram_measured_at is not None:  # measured, and not plausible
        gbps = None
    return found.model_copy(update={"ram": found.ram.model_copy(update={"bandwidth_gbps": gbps})})


def mark_adopted(state_dir: Path, machine_id: str, summary: str) -> Adopted:
    """`mnm machine adopt`'s last step: the box the plans were re-fitted for."""
    done = Adopted(machine_id=machine_id, at=datetime.now(UTC), summary=summary)
    change_file(state_dir, lambda file: file.model_copy(update={"adopted": done}))
    return done


def forget(state_dir: Path, machine_id: str) -> SeenMachine:
    """Drop a box that is not the current one, with what was measured on it."""
    gone: list[SeenMachine] = []

    def apply(file: MachineFile) -> MachineFile:
        if file.current is not None and file.current.fingerprint.id == machine_id:
            raise MachineFactsError(
                f"{machine_id} is this box (the current one): mnm forgets only a box it is not "
                "on now."
            )
        known = file.entry(machine_id)
        if known is None:
            raise MachineFactsError(
                f"No box {machine_id} is kept: `mnm machine history` lists the ones that are."
            )
        file.seen.remove(known)
        gone.append(known)
        return file

    change_file(state_dir, apply)
    return gone[0]
