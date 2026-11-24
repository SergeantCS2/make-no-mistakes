"""Master to a reference: the mix in use matched to a reference track's loudness, tone, peak and
stereo width by Matchering in the studio tools image, kept as a child run of the song (role
`master`) whose `master.wav` the export's `mix.*` are made from. The reference is kept with the
master, recorded by its hash, and never leaves the box.

A master stands for the mix it was made from: once the song uses another mix (a render, new
stems, a new pick, take or voice) the master is no longer used and the page says to master
again. The export of a master is the master as made: the reference set its loudness, so the
export's loudness line does not apply (`mnm song master <run> --off` brings it back).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NamedTuple

from makenomistakes.audio_format import Conversion, HouseFormat, HouseFormatError
from makenomistakes.pipeline import song, song_mix, take_audio
from makenomistakes.pipeline.steps import PipelineError
from makenomistakes.run_parts import InputRecord, ModelCredit
from makenomistakes.runs import RunError, RunKind, RunRecord, RunState, RunStore, StepState
from makenomistakes.studio import peaks
from makenomistakes.studio.tools import StudioTools
from makenomistakes.studio.tools_answers import ToolsError

ROLE: Final = "master"
REFERENCE = "reference"  # the input's role: `input-reference-<name>`
MIX_KEY = "mix"  # a master's setting: the mix it was made from (`mix_key`)
MIX_COPY = "mix-source.wav"  # the mix in use beside the master's run for the tools' mount
REFERENCE_COPY = "reference-house.wav"  # the reference in the house format, likewise
MAX_REFERENCE = 200 * 1024 * 1024  # bytes: a 10-minute WAV at 48 kHz 24-bit is 173 MB
TOOL = "Matchering"  # the studio tools' mastering library, credited on each master
LICENCE = "GPL-3.0-or-later (the studio tools' mastering library)"


def mix_key(store: RunStore, run: RunRecord) -> str:
    """The mix the song uses now: the render in use (or the song's own) and its stems."""
    render = song_mix.in_use(store, run)
    return f"{render.id if render else 'own'}:{song_mix.stems_key(store, run)}"


def masters(store: RunStore, run: RunRecord) -> list[RunRecord]:
    """The song's masters, oldest first."""
    return [c for c in store.children(run.id) if c.role == ROLE]


class Why(NamedTuple):
    """Why a master cannot be used: what kind of reason (`mix` when only the mix moved) and
    its words."""

    kind: Literal["gone", "other", "unfinished", "cleaned", "mix"]
    words: str


def refusal(store: RunStore, run: RunRecord, master_id: str) -> Why | None:
    """Why the song cannot use this master (None when it can): the one rule `use`, `in_use`,
    the CLI, the export and the page read, so none calls a cleaned master "used" while the
    export takes the loudness line."""
    try:
        child = store.get(master_id)
    except RunError:
        return Why("gone", f"{master_id} is gone")
    file = store.folder(child) / song.MASTER_FILE
    checks: tuple[tuple[bool, Why], ...] = (
        (child.parent != run.id or child.role != ROLE,
         Why("other", f"{master_id} is no master of {run.id}")),
        (child.state is not RunState.DONE,
         Why("unfinished", f"{master_id} is not a finished master ({child.state.value})")),
        (child.files_removed or not file.is_file(),
         Why("cleaned", f"{master_id} had its files cleaned")),
        (child.settings.get(MIX_KEY) != mix_key(store, run),
         Why("mix", f"{master_id} was made from another mix than {run.id} uses now")),
    )  # fmt: skip
    return next((why for failed, why in checks if failed), None)


def remedy(why: Why, run_id: str) -> str:
    if why.kind == "mix":
        return f"master it again (`mnm song master {run_id} --reference FILE`)"
    return f"`mnm song master {run_id}` lists its masters"


def no_mix(run: RunRecord) -> str | None:
    """Why the song cannot be mastered yet (None when its mix is made): the CLI's and the
    page's one refusal."""
    mixed = run.step(song.MIX)
    if mixed is None or mixed.state is not StepState.DONE:
        return (
            f"{run.id} has no mix yet: a master follows it (`mnm song show {run.id}` says "
            "where the song stands)."
        )
    return None


def in_use(store: RunStore, run: RunRecord) -> RunRecord | None:
    """The master the export uses: the one the song names, while it can be used."""
    named = run.settings.get(song.MASTER_RUN)
    if not named or refusal(store, run, named) is not None:
        return None
    return store.get(named)


def stale(store: RunStore, run: RunRecord) -> str | None:
    """The master the song names when only the mix moved since it was made (the page says
    to master again)."""
    named = run.settings.get(song.MASTER_RUN)
    why = refusal(store, run, named) if named else None
    return named if why is not None and why.kind == "mix" else None


def reference_of(child: RunRecord) -> str:
    """The reference's name as the user gave it."""
    found = next((i for i in child.inputs_recorded if i.role == REFERENCE), None)
    return found.name.removeprefix(f"input-{REFERENCE}-") if found else "a reference"


def make(
    store: RunStore,
    tools: StudioTools,
    house: HouseFormat,
    run: RunRecord,
    name: str,
    data: bytes,
) -> RunRecord:
    """The mix in use mastered to the reference: a new child run, now the master in use."""
    why = no_mix(run)
    if why is not None:
        raise PipelineError(why)
    if not data:
        raise PipelineError("The reference is empty.")
    if len(data) > MAX_REFERENCE:
        raise PipelineError(
            f"The reference is {len(data) / 2**20:.0f} MiB, past the {MAX_REFERENCE >> 20} MiB "
            "a reference may be (a whole song as WAV is under that)."
        )
    tools.installed()  # refused before anything is made: `mnm studio install` builds it
    used = {"tool": "matchering", MIX_KEY: mix_key(store, run)}
    child = store.new(
        RunKind.PIPELINE, song.NAME, f"mastered to {name}", used, parent=run.id, role=ROLE
    )
    started = time.monotonic()
    here = store.folder(child)
    try:
        path = store.add_input(child, name, data, role=REFERENCE)
        digest = hashlib.sha256(data).hexdigest()
        child.inputs_recorded = [
            InputRecord(role=REFERENCE, name=path.name, sha256=digest, stage=ROLE)
        ]
        take_audio.beside(song_mix.mix_source(store, run), here / MIX_COPY)
        house.convert(here, Conversion(source=path.name, target=REFERENCE_COPY))
        answer = tools.master(here, MIX_COPY, REFERENCE_COPY, song.MASTER_FILE)
    except (ToolsError, HouseFormatError, OSError) as exc:
        store.fail(child, str(exc))
        raise PipelineError(f"The master failed: {exc}") from exc
    except BaseException as exc:
        store.fail(child, str(exc) or f"Interrupted ({type(exc).__name__}).")
        raise
    finally:
        for scratch in (MIX_COPY, REFERENCE_COPY):
            (here / scratch).unlink(missing_ok=True)
    took = time.monotonic() - started
    line = f"the mix mastered to {name} ({TOOL} {answer.matchering})"
    child.models = [ModelCredit(tool=f"{TOOL} {answer.matchering}")]
    child = store.finish(child, [here / song.MASTER_FILE], line, took)
    peaks.measure(house, here, [song.MASTER_FILE])
    use(store, run.id, child.id)
    return child


def use(store: RunStore, run_id: str, master_id: str | None) -> RunRecord:
    """A master of this song (None: none, the export's loudness line) as the one in use."""
    with store.locked():
        run = store.get(run_id)
        if master_id is None:
            run.settings.pop(song.MASTER_RUN, None)
        else:
            why = refusal(store, run, master_id)
            if why is not None:
                raise PipelineError(f"{why.words}: {remedy(why, run.id)}.")
            run.settings[song.MASTER_RUN] = master_id
        store.save(run)
    return run


def credits(store: RunStore, run: RunRecord) -> list[ModelCredit]:
    """What made the master in use (Matchering, with its licence in the export); a master made
    before masters were credited is Matchering's too."""
    child = in_use(store, run)
    if child is None:
        return []
    return child.models or [ModelCredit(tool=TOOL)]


def tool_licence(tool: str) -> str:
    """A tool credit's licence words, for the export's licences line."""
    if tool.split(" ", 1)[0] == TOOL:
        return LICENCE
    return "its own licence (its project says)"


def source(store: RunStore, run: RunRecord) -> tuple[Path, RunRecord] | None:
    """The master's file and run when one is in use (the export's mix is made from it)."""
    child = in_use(store, run)
    return None if child is None else (store.folder(child) / song.MASTER_FILE, child)


@dataclass(frozen=True)
class Shown:
    """What the Song page's Master panel shows: the masters with whether each can be used,
    the one in use and one made from another mix."""

    masters: list[RunRecord]
    usable: dict[str, bool]
    in_use: str
    stale: str


def shown(store: RunStore, run: RunRecord) -> Shown:
    current, made = in_use(store, run), masters(store, run)
    return Shown(
        masters=made,
        usable={m.id: refusal(store, run, m.id) is None for m in made},
        in_use=current.id if current else "",
        stale=stale(store, run) or "",
    )
