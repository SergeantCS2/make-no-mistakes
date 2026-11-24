"""A restore keeps every box: `machine.json`'s boxes merged with the ones here, and
every reading of the archive kept, each naming the box it was taken on.

- **The boxes** merge by fingerprint id: first seen the earliest, last seen the latest, each
  measured figure the newer of the two. `current` is never taken from a backup (this box
  decides what it is, at its next read); the first box mnm saw (`unstamped_on`) and schema
  1's numbers stay this state folder's when it has them.
- **The readings** keep their lines; one that names no box (written by an older mnm) is stamped
  with the backup's first box (its `unstamped_on`), so it stays that box's here too. A
  backup without `machine.json` (schema 2) says only its card: a line of another card than
  this one's (its own `gpu`, else the manifest's) is stamped `card:<name>`, a box mnm never
  returns to, so it is kept and never used.

Counted for the user: the archive's readings that belong to another box, kept for when it
returns (`Restored.other_machines`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.machine_facts import (
    MachineFacts,
    MachineFactsError,
    MachineFile,
    SeenMachine,
    change_file,
    parse_file,
)
from makenomistakes.readings_filter import Here

MACHINE_FILE = "machine.json"
OTHER_CARD = "card:"  # a schema 2 backup's reading of another card: its box unknown
_RAM = ("ram_gbps", "ram_one_thread_gbps", "ram_probe", "ram_measured_at")
_EXPERTS = ("expert_gbps", "expert_readings", "expert_learned_at", "expert_model")
_DECODE = (
    "decode_efficiency",
    "launch_seconds",
    "decode_readings",
    "decode_learned_at",
    "decode_card",
)


class MachinesError(ValueError):
    pass


def theirs(data: bytes | None) -> MachineFile | None:
    """The archive's `machine.json`, read; None when it has none."""
    if data is None:
        return None
    try:
        return parse_file(data.decode(errors="replace"), f"the backup's {MACHINE_FILE}")
    except MachineFactsError as exc:
        raise MachinesError(str(exc)) from exc


def merge_into(state_dir: Path, given: MachineFile) -> MachineFile:
    """The archive's boxes merged into this state folder's `machine.json` (written whole under
    its lock)."""
    return change_file(state_dir, lambda here: merged_files(here, given))


def merged_files(here: MachineFile, given: MachineFile) -> MachineFile:
    for entry in given.seen:
        mine = here.entry(entry.fingerprint.id)
        if mine is None:
            here.seen.append(entry)
            continue
        here.seen[here.seen.index(mine)] = _merged_entry(mine, entry)
    current = here.current.fingerprint.id if here.current is not None else None
    # newest last, as `record` keeps it; the current box stays the last
    here.seen.sort(key=lambda s: (s.fingerprint.id == current, s.last_seen))
    if here.unstamped_on is None:
        here.unstamped_on = given.unstamped_on
    if here.unclaimed is None and not here.seen and here.current is None:
        here.unclaimed = given.unclaimed
    return here


def _merged_entry(mine: SeenMachine, given: SeenMachine) -> SeenMachine:
    newer = given if given.last_seen > mine.last_seen else mine
    return mine.model_copy(
        update={
            "fingerprint": newer.fingerprint,
            "first_seen": min(mine.first_seen, given.first_seen),
            "last_seen": max(mine.last_seen, given.last_seen),
            "measured": _newer_facts(mine.measured, given.measured),
        }
    )


def _newer_facts(mine: MachineFacts, given: MachineFacts) -> MachineFacts:
    """Each group (the bandwidth, the expert rate) from whichever measured it later."""
    found = mine.model_dump()
    groups = ((_RAM, "ram_measured_at"), (_EXPERTS, "expert_learned_at"))
    for group, stamp in (*groups, (_DECODE, "decode_learned_at")):
        ours, other = getattr(mine, stamp), getattr(given, stamp)
        if other is not None and (ours is None or other > ours):
            found.update({key: getattr(given, key) for key in group})
    return MachineFacts.model_validate(found)


@dataclass(frozen=True)
class Stamper:
    """What a backup's readings that name no box are stamped with."""

    machine: str | None = None  # the backup's first box (schema 3)
    card: str | None = None  # this machine's card, for a schema 2 backup
    their_card: str | None = None  # the manifest's card (schema 2)

    def lines(self, given: bytes) -> bytes:
        out = []
        for line in given.decode(errors="replace").splitlines():
            out.append(self._line(line))
        return ("\n".join(out) + "\n").encode() if out else b""

    def _line(self, line: str) -> str:
        try:
            data = json.loads(line)
        except ValueError:
            return line
        if not isinstance(data, dict) or data.get("machine") is not None:
            return line
        box = self._box(data.get("gpu"))
        if box is None:
            return line
        data["machine"] = box
        return json.dumps(data, separators=(",", ":"))

    def _box(self, gpu: object) -> str | None:
        if self.machine is not None:
            return self.machine
        taken_on = (gpu if isinstance(gpu, str) else None) or self.their_card
        if self.card is not None and taken_on is not None and taken_on != self.card:
            return f"{OTHER_CARD}{taken_on}"
        return None


def foreign(given: bytes, here: Here) -> list[str | None]:
    """The boxes of the archive's readings that are not this box's (one entry a reading)."""
    found: list[str | None] = []
    for line in given.decode(errors="replace").splitlines():
        try:
            machine = json.loads(line).get("machine")
        except (ValueError, AttributeError):
            continue
        box = machine if isinstance(machine, str) else None
        if not here.owns(box):
            found.append(box)
    return found


def box_words(file: MachineFile | None, machine: str | None) -> str:
    """A box in words for the restore's count: its one line, else what is known of it."""
    if machine is None:
        return "the first box mnm saw"
    if machine.startswith(OTHER_CARD):
        return f"a box with {machine.removeprefix(OTHER_CARD)}"
    entry = file.entry(machine) if file is not None else None
    return entry.fingerprint.words() if entry is not None else f"box {machine}"
