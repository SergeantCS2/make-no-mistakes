"""Readings follow the box: every reading is kept, and every consumer reads this box's
only, through `readings_here`, the one filter (the figure, the corrections, the picture
speeds, the expert rate, the request meter's "enough readings", the self-test's bar, the
nightly's baseline, `mnm calibration`, the web's tab and the export's corrections).

"This box" is `machine.json`'s current fingerprint (`machine_facts`: the router records it at
its start, `mnm ls`, the doctor and `mnm machine show` at every run), the same id a reading
written now is stamped with (`reading_stamp`), so the writer and the reader agree without a
second read of the hardware. A reading without a box (`machine` None: written by an older mnm)
belongs to the first box mnm saw (`MachineFile.unstamped_on`), which is right for a box that
updated from that mnm: every reading it has is its own. Before any box was read, only those count.

Another box's readings stay in the files (and in backups) for when it returns: the same
fingerprint id gets the same figures back. Nothing is borrowed across boxes."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.machine_facts import FILE, MachineFactsError, MachineFile, load_file
from makenomistakes.measurements import Kind, MeasurementStore, Reading

ANOTHER = "from another machine"


@dataclass(frozen=True)
class Here:
    """The box readings are read for: its fingerprint id (None: no box read yet) and whether
    the readings that name no box are its own."""

    id: str | None = None
    claims_unstamped: bool = True
    words: str | None = None  # the box in one line (`MachineFingerprint.words`)

    def owns(self, machine: str | None) -> bool:
        if machine is None:
            return self.claims_unstamped
        return machine == self.id


def here_in(file: MachineFile | None) -> Here:
    """The current box of a machine file; `Here()` before anything was recorded."""
    if file is None:
        return Here()
    current = file.current.fingerprint if file.current is not None else None
    claims = file.unstamped_on is None or (current is not None and file.unstamped_on == current.id)
    if current is None:
        return Here(claims_unstamped=claims)
    return Here(current.id, claims, current.words())


_cache: dict[Path, tuple[tuple[int, int], Here]] = {}
_lock = threading.Lock()


def here_of(state_dir: Path) -> Here:
    """This box as `machine.json` last recorded it, read again only when the file changed;
    `Here()` when it is missing or cannot be read (a reading is a hint, never a failure)."""
    path = state_dir / FILE
    try:
        info = path.stat()
    except OSError:
        return Here()
    stamp = (info.st_mtime_ns, info.st_size)
    with _lock:
        cached = _cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        found = here_in(load_file(state_dir))
    except (MachineFactsError, OSError):
        found = Here()
    with _lock:
        _cache[path] = (stamp, found)
    return found


def readings_here(readings: Iterable[Reading], here: Here) -> list[Reading]:
    """The readings taken on `here`, in their order: the only filter by box."""
    return [r for r in readings if here.owns(r.machine)]


MEMORY_KINDS = (Kind.LOAD, Kind.REQUEST)  # what the memory and speed rows are made of


def memory_readings(readings: Iterable[Reading]) -> list[Reading]:
    """The loads and requests only: a song candidate's score, a prefix check, a
    window of the router's queue and any kind after them are readings too, each shown
    in its own table, never among the memory and speed rows."""
    return [r for r in readings if r.kind in MEMORY_KINDS]


def left_out(readings: Iterable[Reading], here: Here) -> int:
    """How many readings belong to other boxes (kept, not used)."""
    return sum(1 for r in readings if not here.owns(r.machine))


def left_out_words(count: int) -> str | None:
    """`3 readings from another machine kept, not used`; None for none."""
    if not count:
        return None
    return f"{count} reading{'s' * (count != 1)} {ANOTHER} kept, not used"


def store_here(store: MeasurementStore) -> Here:
    return here_of(store.folder.parent)


def instance_readings(store: MeasurementStore, name: str) -> list[Reading]:
    """One instance's readings on this box."""
    return readings_here(store.readings(name), store_here(store))


def every_here(store: MeasurementStore) -> dict[str, list[Reading]]:
    """Every instance's readings on this box (an instance with none here keeps its key)."""
    here = store_here(store)
    return {name: readings_here(found, here) for name, found in store.every().items()}


def per_box(store: MeasurementStore, file: MachineFile | None) -> dict[str | None, int]:
    """How many readings each box holds (`mnm machine show`, `forget`): a reading naming no
    box counts as the first box mnm saw's (None when none was ever recorded)."""
    first = file.unstamped_on if file is not None else None
    found: dict[str | None, int] = {}
    for readings in store.every().values():
        for reading in readings:
            box = reading.machine if reading.machine is not None else first
            found[box] = found.get(box, 0) + 1
    return found


def belongs_to(machine_id: str, file: MachineFile | None) -> Callable[[Reading], bool]:
    """Whether a reading is `machine_id`'s (the unstamped ones too when it is the first box)."""
    first = file.unstamped_on if file is not None else None

    def owned(reading: Reading) -> bool:
        box = reading.machine if reading.machine is not None else first
        return box == machine_id

    return owned
