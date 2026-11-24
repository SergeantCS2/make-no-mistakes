"""This box as mnm knows it: the fingerprint taken now (`read_now`, cheap reads only:
the probe's cards and MemTotal, `/proc/cpuinfo`, sysfs, DMI), recorded in `machine.json`
(`observe`), the box a plan is made on (`planned_here`, for `Recipe.planned_on`), and what
changed since the saved plans were made (`drift`), in one sentence every surface shares:
the doctor's *Machine* line, `mnm ls`'s last line, the web board's banner, the router's
journal and `/__mnm/health`. Nothing here acts on a change (`mnm machine adopt` does):
it is recorded and said.

A recipe saved before mnm kept fingerprints (`planned_on` None) counts as planned on the first
box mnm saw (`MachineFile.unstamped_on`), the best it knows of the box it was planned for."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from makenomistakes.backend import Backend, BackendError, decide
from makenomistakes.backend_now import decide_here, rocm_measured_here
from makenomistakes.config import Settings
from makenomistakes.hardware.box_facts import BoxFacts
from makenomistakes.hardware.fingerprint import MachineFingerprint, differences
from makenomistakes.hardware.machine_reads import Unread, read_fingerprint
from makenomistakes.hardware.probe import Gpu, MemoryInfo, MemorySnapshot, ProbeError
from makenomistakes.local_time import local
from makenomistakes.machine_facts import (
    MachineFactsError,
    MachineFile,
    Recorded,
    load_file,
    record,
)
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import SELFTEST_PREFIX, TUNE_PREFIX
from makenomistakes.store import InstanceStore, StoreError

SHOW = "`mnm machine show`"
ADOPT = "`mnm machine adopt` re-fits them"  # The end of every "the hardware changed"
FORGOTTEN = "a box mnm no longer keeps"
CHANGED = "the hardware changed since"  # every surface's words start so (the nightly reads it)


def read_now(
    cards: list[Gpu] | ProbeError, memory: MemoryInfo | None, box: BoxFacts | None = None
) -> MachineFingerprint | Unread:
    """The box now, from what the caller already probed (one place to replace it)."""
    return read_fingerprint(cards, memory, box)


@dataclass(frozen=True)
class Observed:
    """A fingerprint taken and recorded, or why not."""

    recorded: Recorded | None = None
    unread: str | None = None  # the box could not be read: nothing recorded
    error: str | None = None  # machine.json could not be read or written

    @property
    def file(self) -> MachineFile | None:
        return self.recorded.file if self.recorded is not None else None


def observe(
    settings: Settings,
    cards: list[Gpu] | ProbeError,
    memory: MemoryInfo | None,
    box: BoxFacts | None = None,
) -> Observed:
    """Take the fingerprint and record it as the current box; never raises."""
    found = read_now(cards, memory, box)
    if isinstance(found, Unread):
        return Observed(unread=found.reason)
    found = with_backend(found, settings.backend, settings.gpu, rocm_measured_here(settings, found))
    try:
        return Observed(recorded=record(settings.expanded_state_dir(), found))
    except (MachineFactsError, OSError) as exc:
        return Observed(error=str(exc))


def with_backend(
    found: MachineFingerprint,
    chosen: str | None,
    gpu: str | None = None,
    rocm_measured: bool = False,
) -> MachineFingerprint:
    """The box with the backend it runs (`backend.decide`); None when `MNM_BACKEND`
    names one these cards cannot run (the doctor says why)."""
    try:
        backend: str | None = decide(found, chosen, gpu, rocm_measured).value
    except BackendError:
        backend = None
    return found.model_copy(update={"backend": backend})


@dataclass(frozen=True)
class BoxBackend:
    """The box read now (not recorded) and its backend: `box` None when the box could
    not be read; `backend` None then, or when `MNM_BACKEND` is refused (`refused` says why)."""

    box: MachineFingerprint | None = None
    backend: Backend | None = None
    refused: str | None = None


def backend_read(
    settings: Settings, cards: list[Gpu] | ProbeError, memory: MemoryInfo | None
) -> BoxBackend:
    """The setup's and the doctor's one reading of the box and its backend."""
    found = read_now(cards, memory)
    if isinstance(found, Unread):
        return BoxBackend()
    try:
        return BoxBackend(found, decide_here(settings, found))
    except BackendError as exc:
        return BoxBackend(found, refused=str(exc))


def observe_snapshot(settings: Settings, snapshot: MemorySnapshot) -> Observed:
    return observe(settings, snapshot.box_cards, snapshot.memory)


def planned_here(settings: Settings, snapshot: MemorySnapshot) -> MachineFingerprint | None:
    """The box a plan saved now is made on: read now and recorded; None when it could not be
    read (the recipe then keeps the box it named, or none)."""
    observed = observe_snapshot(settings, snapshot)
    current = observed.file.current if observed.file is not None else None
    return current.fingerprint if current is not None else None


def stamped(recipe: Recipe, box: MachineFingerprint | None) -> Recipe:
    """`recipe` naming `box` as the one it was planned on; unchanged without a box."""
    if box is None:
        return recipe
    return recipe.model_copy(update={"planned_on": box.id, "planned_cards": box.cards_id})


def saved_recipes(settings: Settings) -> list[Recipe]:
    """Every saved plan the box is compared with; none when the store cannot be read (the
    commands that list instances say why)."""
    try:
        return InstanceStore(settings.expanded_state_dir()).list()
    except StoreError:
        return []


@dataclass(frozen=True)
class Drift:
    """The current box against the boxes the saved plans were made on."""

    here: MachineFingerprint
    first_seen: datetime | None
    # the box each instance planned elsewhere was planned on: id -> names
    elsewhere: dict[str, list[str]] = field(default_factory=dict)
    changes: list[str] = field(default_factory=list)  # what differs, card by card

    @property
    def count(self) -> int:
        return sum(len(names) for names in self.elsewhere.values())

    @property
    def names(self) -> list[str]:
        return sorted(n for names in self.elsewhere.values() for n in names)

    def words(self) -> str | None:
        """`the hardware changed since 3 plans were made: the cards (RTX 4070 gone, RTX 3090
        new); `mnm machine adopt` re-fits them`; None when every plan was made on this box."""
        said = self.headline()
        return f"{said[0].lower()}{said[1:]}; {ADOPT}" if said else None

    def planned_box(self, file: MachineFile) -> MachineFingerprint | None:
        """The box most of the plans made elsewhere were made on (one a box no longer kept
        cannot name: None)."""
        if not self.elsewhere:
            return None
        busiest = max(self.elsewhere, key=lambda planned: len(self.elsewhere[planned]))
        entry = file.entry(busiest)
        return entry.fingerprint if entry is not None else None

    def headline(self) -> str | None:
        """`The hardware changed since 3 plans were made: …`, the command left to the page."""
        if not self.count:
            return None
        made = "1 plan was made" if self.count == 1 else f"{self.count} plans were made"
        return f"{CHANGED[0].upper()}{CHANGED[1:]} {made}: {', '.join(self.changes)}"

    def here_words(self) -> str:
        """`this box: RTX 4070 12 GB, …, WSL; first seen <date>`."""
        seen = f"; first seen {local(self.first_seen):%Y-%m-%d}" if self.first_seen else ""
        return f"this box: {self.here.words()}{seen}"


def planned_id(recipe: Recipe, file: MachineFile) -> str | None:
    """The box a recipe was planned on; an older one's is the first box mnm saw."""
    return recipe.planned_on or file.unstamped_on


def planned_words(recipe: Recipe, file: MachineFile | None) -> str | None:
    """`mnm show`'s line: `this box (<id>)`, or the other box and the command; None
    before any box was recorded."""
    if file is None or file.current is None:
        return None
    planned = planned_id(recipe, file)
    if planned is None:
        return "before mnm kept fingerprints"
    if planned == file.current.fingerprint.id:
        return f"this box ({planned})"
    entry = file.entry(planned)
    on = entry.fingerprint.words() if entry is not None else FORGOTTEN
    return f"another box ({planned}: {on}); `mnm machine adopt` re-fits it"


def drift(file: MachineFile, recipes: list[Recipe]) -> Drift | None:
    """None before any box was taken."""
    if file.current is None:
        return None
    here = file.current.fingerprint
    elsewhere: dict[str, list[str]] = {}
    for recipe in recipes:
        if recipe.name.startswith((SELFTEST_PREFIX, TUNE_PREFIX)):
            continue  # a tune's or self-test's copy is gone in minutes
        planned = planned_id(recipe, file)
        if planned is not None and planned != here.id:
            elsewhere.setdefault(planned, []).append(recipe.name)
    changes: list[str] = []
    for planned in elsewhere:
        entry = file.entry(planned)
        found = differences(entry.fingerprint, here) if entry is not None else [FORGOTTEN]
        for change in found or [f"the box ({planned} → {here.id})"]:
            if change not in changes:
                changes.append(change)
    entry = file.entry(here.id)
    return Drift(
        here=here,
        first_seen=entry.first_seen if entry is not None else None,
        elsewhere=elsewhere,
        changes=changes,
    )


def drift_now(settings: Settings, recipes: list[Recipe]) -> Drift | None:
    """From `machine.json` as last recorded (no reading of the box: the web board, health);
    None when nothing was recorded or the file cannot be read."""
    try:
        found = load_file(settings.expanded_state_dir())
    except MachineFactsError:
        return None
    return drift(found, recipes) if found is not None else None


def sentence(found: Drift | None) -> str | None:
    """The one sentence `mnm ls`, the web board and the router say; None when nothing moved."""
    words = found.words() if found is not None else None
    return f"{words[0].upper()}{words[1:]}." if words else None


def start_line(settings: Settings, snapshot: MemorySnapshot, recipes: list[Recipe]) -> str:
    """The router's journal line at its start: what changed since plans were made, or this
    box in one line (`new to mnm` the first time), or why it could not be read."""
    observed = observe_snapshot(settings, snapshot)
    if observed.unread is not None:
        return f"This box could not be read: {observed.unread}."
    if observed.recorded is None:
        return f"This box was not recorded: {observed.error}."
    said = sentence(drift(observed.recorded.file, recipes))
    if said is not None:
        return said
    here = observed.recorded.file.current
    words = here.fingerprint.words() if here is not None else "unknown"
    return f"This box: {words}{' (new to mnm)' if observed.recorded.new else ''}."
