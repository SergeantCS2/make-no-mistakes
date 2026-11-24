"""`mnm machine adopt`: when the box changed, one command re-fits every saved plan for it
and says what it did. It orders what exists and reimplements nothing:

1. the router idle as the nightly asks it (`nightly.gate`: no request in flight, no tune, no
   start or stop, no ComfyUI job; a minute at most, then "the router is busy: try again"),
   then the box's start lock (`swap.start_turn`); the fingerprint taken (`machine.read_now`)
   and recorded, and the diff said: against the box last read and against every recipe's
   `planned_on`;
2. `MNM_GPU`'s note when cards came, went or moved (`gpu_offer`: printed, never written);
3. the RAM bandwidth probe (`ram_check`), saved under this box;
4. `replan` of every instance (or `--only`'s) on the live budget (`planning_budget`), the
   quant free unless `--keep-quants`, nothing withheld; each saved (`replan.save`, which
   stamps `planned_on`), an instance that cannot be planned here kept as it was ("not for this
   box: why"); then the lock is let go and the missing files are listed with their sizes
   and asked once (`--yes` takes them; a no leaves them for the next start, each named);
5. a tune queued for every llama.cpp instance whose tune is stale or absent (`--no-tune`
   skips), through the router's tuner when it runs, else its queue file;
6. `machine.json`'s `adopted` set last (an adopt that failed before leaves the one before),
   one router journal line, one notice, and the table kept as `<state>/machine/adopt-<date>.txt`
   (the last five).

`--dry-run` says steps 1, 2 and the plans, and writes, probes and downloads nothing."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from makenomistakes import machine
from makenomistakes.adopt_words import AdoptCounts, summary
from makenomistakes.fit import Budget
from makenomistakes.gpu_offer import gpu_note
from makenomistakes.hardware.fingerprint import MachineFingerprint, differences
from makenomistakes.hardware.machine_reads import Unread
from makenomistakes.hardware.ram_bandwidth import RamReading
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.machine_facts import (
    LOCK_FOLDER,
    MachineFile,
    Taken,
    load_file,
    mark_adopted,
    record,
)
from makenomistakes.plan_types import PlanError
from makenomistakes.ram_check import ram_check
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import SELFTEST_PREFIX, TUNE_PREFIX
from makenomistakes.replan import Replanned, planned_as_it_is, replan, save
from makenomistakes.swap import start_turn

KEEP_TABLES = 5
GIB = 2**30
SHOWN = ("quant", "ctx", "gpu_layers", "cpu_moe_layers", "kv_type", "max_model_len", "width")
GPU_MEMORY = "planned GPU memory"


class AdoptError(RuntimeError):
    """Adopt refused before it changed anything (the router busy, the box unread, a name)."""


@dataclass(frozen=True)
class AdoptOptions:
    yes: bool = False  # take every download without asking
    keep_quants: bool = False
    tune: bool = True
    only: tuple[str, ...] = ()
    dry_run: bool = False


@dataclass(frozen=True)
class Fetch:
    path: str
    size: int | None  # None: the plan's facts did not size it


@dataclass
class Row:
    name: str
    plan: str  # before → after, in replan's words
    fetch: list[Fetch] = field(default_factory=list)
    fetched: str = ""  # "fetched", "left for the next start", "failed: …"
    tune: str = ""
    kept: str | None = None  # not for this box: why (the recipe untouched)


@dataclass
class Adoption:
    box: MachineFingerprint
    rows: list[Row] = field(default_factory=list)
    counts: AdoptCounts = field(default_factory=AdoptCounts)
    dry_run: bool = False


@dataclass(frozen=True)
class AdoptDeps:
    """Every outside question, so tests drive the sequence on fakes."""

    service: InstanceService
    wait_idle: Callable[[], str | None]  # None once idle; else why, after a minute
    read_box: Callable[[], MachineFingerprint | Unread]  # cheap reads, nothing recorded
    measure: Callable[[], float | RamReading]  # the RAM bandwidth probe (every core)
    budget: Callable[[], Budget | None]  # the live budget `mnm replan --all` plans on
    ask: Callable[[str], bool]  # the one download question
    download: Callable[[Recipe], None]
    queue_tune: Callable[[Recipe], str]  # words: queued, or why not
    tell_router: Callable[[AdoptCounts], bool]
    notify: Callable[[str], None]
    say: Callable[[str], None]
    gpu: str | None = None  # `MNM_GPU` as set; None unset
    # lets go of the tuner's hold the wait took (else tunes wait out its 45-minute lease)
    release: Callable[[], None] = lambda: None


def adopt(deps: AdoptDeps, options: AdoptOptions) -> Adoption:
    """The box adopted; the tuner's hold the wait for an idle router took is let go at the
    end, whatever happened (the tunes it queued start once the router is idle)."""
    try:
        return _adopt(deps, options)
    finally:
        deps.release()


def _adopt(deps: AdoptDeps, options: AdoptOptions) -> Adoption:
    service, say = deps.service, deps.say
    state = service.settings.expanded_state_dir()
    names = _names(service, options.only)
    if not options.dry_run and (why := deps.wait_idle()) is not None:
        raise AdoptError(f"The router is busy ({why}): try again in a minute.")
    with _locked(service, options.dry_run):
        found = deps.read_box()
        if isinstance(found, Unread):
            raise AdoptError(f"This box could not be read ({found.reason}): nothing adopted.")
        before = load_file(state)
        file = _taken(before, found) if options.dry_run else record(state, found).file
        last = before.current.fingerprint if before is not None and before.current else None
        drift = machine.drift(file, service.all())
        _say_box(say, last, drift, found, saved=bool(names))
        _say_gpu(say, deps.gpu, drift.planned_box(file) if drift else None, last, found)
        if not options.dry_run:
            say(_bandwidth(state, deps.measure))
        done = Adoption(box=found, dry_run=options.dry_run)
        budget = deps.budget()
        for name in names:
            done.rows.append(_replan_one(service, name, budget, options, say))
    if options.dry_run:
        if (asked := _question(done)) is not None:
            say(f"Adopt would ask first: {asked}")
        say("Dry run: nothing was saved, measured or downloaded.")
        return done
    _download(deps, done, options)
    _tunes(deps, done, options)
    done.counts = _counted(done)
    words = summary(done.counts)
    mark_adopted(state, found.id, words)
    deps.tell_router(done.counts)
    deps.notify(f"Adopted {found.words()}: {words}")
    keep_table(state, table_text(done))
    say(f"Adopted this box: {words}.")
    if kept := [r.name for r in done.rows if r.kept is not None]:
        say(
            f"{', '.join(kept)} stay{'s' * (len(kept) == 1)} planned for the box "
            f"{'it was' if len(kept) == 1 else 'they were'} made on, and the doctor keeps saying "
            "so: put that box's cards back and adopt again, or `mnm rm` what is not wanted here."
        )
    return done


def _names(service: InstanceService, only: tuple[str, ...]) -> list[str]:
    every = [r.name for r in service.all() if not r.name.startswith((SELFTEST_PREFIX, TUNE_PREFIX))]
    unknown = [n for n in only if n not in every]
    if unknown:
        raise AdoptError(f"No instance named {', '.join(unknown)}: `mnm ls` lists them.")
    return [n for n in every if n in only] if only else every


@contextlib.contextmanager
def _locked(service: InstanceService, dry_run: bool) -> Iterator[None]:
    """The box's start lock (nothing starts while plans move); a dry run takes none."""
    if dry_run:
        yield
        return
    with start_turn(service.settings, "adopt"):
        yield


def _taken(file: MachineFile | None, found: MachineFingerprint) -> MachineFile:
    """The file as a recorded box would leave it, in memory only (a dry run)."""
    base = file or MachineFile()
    return base.model_copy(update={"current": Taken(fingerprint=found, taken_at=datetime.now(UTC))})


def _say_box(
    say: Callable[[str], None],
    last: MachineFingerprint | None,
    drift: machine.Drift | None,
    found: MachineFingerprint,
    saved: bool = True,
) -> None:
    say(f"This box: {found.words()}.")
    if last is not None and last.id != found.id:
        changed = differences(last, found) or [f"the box ({last.id} → {found.id})"]
        say(f"Since the box mnm last read ({last.id}): {', '.join(changed)}.")
    if drift is not None and drift.count:
        since = ", ".join(drift.changes)
        say(f"Planned on another box: {', '.join(drift.names)}; since then {since}.")
    elif not saved:  # "every saved plan" read oddly with none saved
        say("No instance is saved yet: nothing to plan again (`mnm add` plans for this box).")
    else:
        say("Every saved plan was made on this box; adopt plans them again all the same.")


def _say_gpu(
    say: Callable[[str], None],
    gpu: str | None,
    planned: MachineFingerprint | None,
    last: MachineFingerprint | None,
    found: MachineFingerprint,
) -> None:
    """`MNM_GPU`'s note when the cards differ from the plans' box (else the box last read)."""
    old = planned if planned is not None and planned.cards_id != found.cards_id else last
    if old is None or old.cards_id == found.cards_id:
        return
    for line in gpu_note(gpu, old, found).lines:
        say(f"{line}.")


def _bandwidth(state: Path, measure: Callable[[], float | RamReading]) -> str:
    check = ram_check(state, measure)
    return f"RAM bandwidth: {check.detail}."


def _replan_one(
    service: InstanceService,
    name: str,
    budget: Budget | None,
    options: AdoptOptions,
    say: Callable[[str], None],
) -> Row:
    try:
        done = replan(service, name, budget, keep_quant=options.keep_quants, keep_if_fits=False)
    except InstanceError as exc:
        why = (str(exc.__cause__) if isinstance(exc.__cause__, PlanError) else str(exc)).rstrip(".")
        say(f"{name}: not for this box: {why}; kept as it was.")
        return Row(name, "kept as it was", kept=f"not for this box: {why}")
    if done.tune_note is not None:
        say(done.tune_note)
    if not done.changed:
        say(f"{name} is planned as it is: nothing to change.")
    else:
        say(f"Re-planned {name} for this box:")
        for change in done.changes:
            say(f"  {change}")
    row = Row(name, plan_words(done), [Fetch(p, done.sizes.get(p)) for p in done.new_files])
    if options.dry_run:
        return row
    try:
        save(service, done) if done.changed else planned_as_it_is(service, name)
    except InstanceError as exc:
        say(f"{name}: not saved: {exc}")
        row.kept, row.fetch = f"not saved: {exc}", []
        return row
    if done.changed and service.status(done.new).running:
        say(
            f"{name} runs with its old settings until its next start (`mnm stop {name} --park`: "
            "the router loads it with the new ones at the next request)."
        )
    return row


def plan_words(done: Replanned) -> str:
    """`quant Q4_K_M → Q8_0, ctx 16384 → 32768, placement re-made`; `as it was`."""
    if not done.changed:
        return "as it was"
    shown = []
    for change in done.changes:
        key, _, rest = change.partition(": ")
        if key in SHOWN:
            shown.append(f"{key} {rest.replace(' -> ', ' → ')}")
        elif key == "placement":
            shown.append("placement re-made")
        elif key == GPU_MEMORY:
            shown.append(f"GPU {rest.replace(' -> ', ' → ')}")
    hidden = len(done.changes) - len(shown)
    return ", ".join(shown + ([f"{hidden} more"] if hidden else [])) or "settings re-fitted"


def _question(done: Adoption) -> str | None:
    """`Download 3 files, 14.2 GiB?` for every file the saved plans need; None for none."""
    wanted = [f for row in done.rows for f in row.fetch]
    if not wanted:
        return None
    known = sum(f.size or 0 for f in wanted)
    unknown = sum(1 for f in wanted if f.size is None)
    files = f"{len(wanted)} file{'s' * (len(wanted) != 1)}"
    return f"Download {files}, {gib(known)}{f' and {unknown} of unknown size' if unknown else ''}?"


def _download(deps: AdoptDeps, done: Adoption, options: AdoptOptions) -> None:
    wanted = [(row, f) for row in done.rows for f in row.fetch]
    question = _question(done)
    if question is None:
        return
    for row, f in wanted:
        deps.say(f"  {row.name}: {f.path} ({gib(f.size) if f.size else 'size unknown'})")
    if not (options.yes or deps.ask(question)):
        for row in {id(r): r for r, _f in wanted}.values():
            row.fetched = "left for the next start"
            deps.say(
                f"{row.name}: its new plan is saved; `mnm start {row.name}` (or a request) "
                f"downloads {', '.join(f.path for f in row.fetch)} first."
            )
        return
    for row in {id(r): r for r, _f in wanted}.values():
        try:
            deps.download(deps.service.get(row.name))
            row.fetched = "fetched"
        except Exception as exc:  # one download failing never stops the others
            row.fetched = f"failed: {exc}"
            deps.say(
                f"{row.name}: the download failed ({exc}); `mnm start {row.name}` tries again."
            )


def _tunes(deps: AdoptDeps, done: Adoption, options: AdoptOptions) -> None:
    for row in done.rows:
        if row.kept is not None:
            row.tune = ""
            continue
        if not options.tune:
            row.tune = "skipped (--no-tune)"
            continue
        row.tune = deps.queue_tune(deps.service.get(row.name))


def gib(size: int) -> str:
    return f"{size / GIB:.1f} GiB"


def _counted(done: Adoption) -> AdoptCounts:
    rows = [r for r in done.rows if r.kept is None]
    return AdoptCounts(
        refitted=sum(1 for r in rows if r.plan != "as it was"),
        downloads=sum(len(r.fetch) for r in rows if r.fetched == "fetched"),
        tunes=sum(1 for r in rows if r.tune.startswith("queued")),
        kept=len(done.rows) - len(rows),
        declined=sum(len(r.fetch) for r in rows if r.fetched == "left for the next start"),
    )


def table_text(done: Adoption) -> str:
    """The table as plain text (the report's file, the web job's notes)."""
    lines = [f"mnm machine adopt, {local():%Y-%m-%d %H:%M}: {done.box.words()}"]
    for row in done.rows:
        files = ", ".join(f"{f.path} ({gib(f.size) if f.size else '?'})" for f in row.fetch)
        lines.append(
            f"{row.name}: {row.kept or row.plan}"
            + (f"; files {files} {row.fetched}".rstrip() if files else "")
            + (f"; tune {row.tune}" if row.tune not in ("", "-") else "")
        )
    lines.append(_dry_summary(done) if done.dry_run else summary(done.counts))
    return "\n".join(lines) + "\n"


def _dry_summary(done: Adoption) -> str:
    changing = sum(1 for r in done.rows if r.kept is None and r.plan != "as it was")
    files = sum(len(r.fetch) for r in done.rows if r.kept is None)
    kept = sum(1 for r in done.rows if r.kept is not None)
    said = f"dry run: {changing} plan{'s' * (changing != 1)} would change, {files} to fetch"
    return said + (f", {kept} not for this box" if kept else "")


def keep_table(state: Path, text: str) -> Path:
    """`<state>/machine/adopt-<local time>.txt`, the newest five kept (`mnm report` takes them)."""
    folder = state / LOCK_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"adopt-{local():%Y-%m-%d-%H%M%S-%f}.txt"  # the box's clock
    path.write_text(text, encoding="utf-8")
    for old in sorted(folder.glob("adopt-*.txt"))[:-KEEP_TABLES]:
        old.unlink(missing_ok=True)
    return path


def tables(state: Path) -> list[Path]:
    return sorted((state / LOCK_FOLDER).glob("adopt-*.txt"))


def tune_body() -> bytes:
    return json.dumps({"action": "tune", "now": False}).encode()
