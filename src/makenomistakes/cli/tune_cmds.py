"""`mnm tune`: the tuner on demand, its queue, and the words `mnm add` and
`mnm ls` print for it.

A tune runs in the router's process when the router answers (`POST /__mnm/tune/<name>`;
the trial gives the card back to the first outside request), else in this terminal with a
line per trial."""

from __future__ import annotations

import json
from typing import Annotated, Literal

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.quality_cmds import quality_tune, stopped_words
from makenomistakes.config import Settings
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.quality import FLOOR_MAX, FLOOR_MIN
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.tuned import ScoreMode, TuneKey
from makenomistakes.router_client import router_tune, router_tuning
from makenomistakes.tune.apply import key_or_none, tuned_word
from makenomistakes.tune.cancel import cancel_tune
from makenomistakes.tune.orphans import Refused, closing_now, queue_speed
from makenomistakes.tune.session import ensure_queued, untunable
from makenomistakes.tune.store import (
    DEFAULT_MINUTES,
    Entry,
    Queued,
    TuneStore,
    TuneStoreError,
)
from makenomistakes.tune.wiring import foreground
from makenomistakes.tune.words import capital

QUEUED = (
    "Queued for tuning: runs when the card is idle; `mnm tune {name} --now` runs it now, "
    "`mnm tune --status` lists the queue."
)
NO_ROUTER = (
    "The router is not running: tuning in this terminal (Ctrl-C stops it; nothing else cuts "
    "a trial here, so leave the card alone meanwhile)."
)
QUEUED_NO_ROUTER = (
    "The router is not running: the queue waits for it (`mnm service install`), or "
    "`mnm tune {name} --now` tunes in this terminal."
)

For = Literal["chat", "agents"]


def _store(service: InstanceService) -> TuneStore:
    return TuneStore(service.settings.expanded_state_dir())


def _print(line: str) -> None:
    console.print(escape(line))


def queue_after_add(service: InstanceService, recipe: Recipe) -> None:
    """`mnm add`'s line under the plan: queued, unless the tuner leaves this type alone."""
    if ensure_queued(service, recipe, "added", _print):
        console.print(QUEUED.format(name=recipe.name))


def tune_keys(
    service: InstanceService, recipes: list[Recipe], snapshot: MemorySnapshot
) -> dict[str, TuneKey | None]:
    """Each tuned recipe's key now (`mnm ls`'s word checks it against the saved one)."""
    return {r.name: key_or_none(service, r, snapshot) for r in recipes if r.tuned is not None}


def settings_cell(recipe: Recipe, keys: dict[str, TuneKey | None]) -> str:
    """`mnm ls`'s Settings cell: the settings, the tune's word and the quality tune's."""
    quality = "quality-tuned" if recipe.quality is not None else None
    words = [w for w in (tuned_word(recipe, keys.get(recipe.name)), quality) if w]
    return ", ".join([recipe.settings_summary, *words])


@app.command()
def tune(
    name: Annotated[
        str | None, typer.Argument(help="The instance to tune.", autocompletion=instance_names)
    ] = None,
    now: Annotated[
        bool, typer.Option(help="Skip the idle minute (still parks by the rules).")
    ] = False,
    for_: Annotated[
        For | None,
        typer.Option(
            "--for",
            help="What the score weighs: chat (decode only) or agents (long prompts too); "
            "default: agents for a coding instance, else decode with the 2k prompt speed.",
        ),
    ] = None,
    minutes: Annotated[int, typer.Option(min=1, help="Wall-clock budget.")] = DEFAULT_MINUTES,
    every: Annotated[bool, typer.Option("--all", help="Every instance the tuner tunes.")] = False,
    status: Annotated[
        bool, typer.Option("--status", help="The queue, the tune running, each result.")
    ] = False,
    cancel: Annotated[bool, typer.Option("--cancel", help="Unqueue it and stop its tune.")] = False,
    quality: Annotated[
        bool,
        typer.Option(
            "--quality",
            help="Choose the quant instead: each quant that runs here scored against Q8_0 "
            "and timed, the most faithful one over the speed floor kept.",
        ),
    ] = False,
    floor: Annotated[
        float | None,
        typer.Option(
            min=FLOOR_MIN,
            max=FLOOR_MAX,
            help="With --quality: the decode floor (tok/s) for this tune.",
        ),
    ] = None,
    quants: Annotated[
        str | None,
        typer.Option(help="With --quality: the quants to measure (comma-separated)."),
    ] = None,
    keep: Annotated[
        bool, typer.Option(help="With --quality: keep the quants not chosen on disk.")
    ] = False,
    yes: Annotated[bool, typer.Option(help="With --quality: download without asking.")] = False,
) -> None:
    """Tune an instance's settings on this box: real starts, measured on fresh prompts."""
    service = InstanceService(Settings())
    if status:
        _status(service)
        return
    if name is None and not every:
        fail_with(InstanceError("name an instance, or pass --all (or --status)"))
    if quality and not cancel:
        if every or name is None:
            fail_with(InstanceError("--quality tunes one instance: name it"))
        wanted = [q.strip() for q in (quants or "").split(",") if q.strip()]
        quality_tune(service, name, floor, wanted, keep, yes, now)
        return
    if cancel:
        _cancel_all(service, name, every)
        return
    try:
        recipes = service.all() if every else [service.get(name or "")]
    except InstanceError as exc:
        fail_with(exc)
    mode: ScoreMode | None = for_
    chosen = [r for r in recipes if untunable(r) is None]
    for recipe in recipes:
        if (why := untunable(recipe)) is not None and (not every or "downloaded" in why):
            console.print(f"{escape(recipe.name)}: {escape(why)}.")
    if not chosen:
        if not every:
            raise typer.Exit(2)
        return
    for recipe in chosen:
        _tune_one(service, recipe, mode, minutes, now)


def _tune_one(
    service: InstanceService, recipe: Recipe, mode: ScoreMode | None, minutes: int, now: bool
) -> None:
    name = recipe.name
    body = json.dumps({"action": "tune", "mode": mode, "minutes": minutes, "now": now})
    asked = router_tune(service.settings, name, body.encode())
    if asked is not None:
        code, said = asked
        console.print(escape(said))
        if code >= 400:
            raise typer.Exit(1)
        return
    entry = Queued(name=name, mode=mode, minutes=minutes, now=now, reason="asked")
    store = _store(service)
    queued = queue_speed(service, store, entry, _print)  # a quality plan waits, never replaced
    if isinstance(queued, Refused):
        fail_with(InstanceError(queued.words))
    if not now:
        console.print(escape(QUEUED_NO_ROUTER.format(name=name)))
        return
    console.print(NO_ROUTER)
    tuner = foreground(service, lambda line: console.print(escape(line)))
    try:
        end = tuner.tune(queued)
    except KeyboardInterrupt:
        words = stopped_words(service, queued, f"mnm tune {name} --now", "unqueues it")
        console.print(escape(words))
        raise typer.Exit(130) from None
    if end.status not in {"tuned"}:
        raise typer.Exit(1)


def _cancel_targets(service: InstanceService, name: str | None, every: bool) -> list[str]:
    """The one named, or every instance's name and every queued one's: a removed instance's
    quality plan waits in the queue for a cancel (or the router) to end it."""
    if not every:
        return [name or ""]
    try:
        names = [recipe.name for recipe in service.all()]
    except InstanceError as exc:
        fail_with(exc)
    return [*names, *sorted(_queued_names(service) - set(names))]


def _queued_names(service: InstanceService) -> set[str]:
    """Every queued tune's name for `--cancel --all`; a queue folder the system will not list
    is said, and only a removed instance's queued plan, which nothing else names, is missed
    by it: that one is not cancelled, and its name still cancels it."""
    try:
        return {entry.name for entry in _store(service).waiting()}
    except TuneStoreError as exc:
        missed = (
            "a removed instance's queued plan is not cancelled (`mnm tune <name> --cancel` "
            "cancels it by its name)"
        )
        console.print(f"[yellow]{escape(_unlisted(exc, missed))}[/yellow]")
        return set()


def _unlisted(
    exc: TuneStoreError, missed: str = "a removed instance's queued plan is not shown"
) -> str:
    """The queue folder's problem, and what a listing it cannot make misses (so that
    `--cancel --all` never says "not shown" of a plan it did not cancel)."""
    return f"{capital(str(exc))}: {missed}."


def _cancel_all(service: InstanceService, name: str | None, every: bool) -> None:
    """Each cancel's words; exit 1 when one was refused (a run elsewhere goes on)."""
    refused = False
    for target in _cancel_targets(service, name, every):
        took, lines = _cancel(service, target)
        for line in lines:
            console.print(escape(line))
        refused = refused or not took
    if refused:
        raise typer.Exit(1)


def _cancel(service: InstanceService, name: str) -> tuple[bool, list[str]]:
    """(the cancel took or nothing was queued, its lines, the headline first): the router's
    when it answers, else `tune.cancel.cancel_tune` here (a run in another terminal is marked
    for it to end after the trial or step under way)."""
    body = json.dumps({"action": "cancel"}).encode()
    asked = router_tune(service.settings, name, body)
    if asked is not None:
        return asked[0] < 400, [asked[1]]
    cancelled = cancel_tune(service, _store(service), name)
    return cancelled.ok, list(cancelled.lines)


def _status(service: InstanceService) -> None:
    store = _store(service)
    running = router_tuning(service.settings)
    if running is None and (states := store.running()):
        running = states[0].line
    console.print(f"Tuner: {escape(running)}" if running else "Tuner: no tune is running.")
    table = Table(title="tuning", box=box.SIMPLE_HEAD)
    for column in ("Instance", "Queue", "Progress", "Result"):
        table.add_column(column, overflow="fold")
    try:
        recipes = service.all()
    except InstanceError as exc:
        fail_with(exc)
    try:
        waiting, unlisted = {q.name: q for q in store.waiting()}, None
    except TuneStoreError as exc:  # each instance's own entry is still read by its name
        waiting, unlisted = {}, _unlisted(exc)
    for gone in sorted(set(waiting) - {r.name for r in recipes}):  # its plan waits for its end
        cell = queue_cell(waiting[gone], closing_now(service, store, waiting[gone]))
        table.add_row(escape(gone), escape(cell), "-", "removed")
    for recipe in recipes:
        if untunable(recipe) is None or not recipe.downloaded:
            table.add_row(*(escape(cell) for cell in _status_row(service, store, recipe)))
    console.print(table)
    if unlisted is not None:
        console.print(f"[yellow]{escape(unlisted)}[/yellow]")


def _status_row(service: InstanceService, store: TuneStore, recipe: Recipe) -> list[str]:
    """One instance's row: its name, its queue entry, its progress, its result."""
    try:
        queued = store.queued(recipe.name)
        queue = queue_cell(queued, closing_now(service, store, queued)) if queued else "-"
        state = store.state(recipe.name)
    except TuneStoreError as exc:  # never run while it cannot be read: said, not hidden
        state, queue = None, str(exc)
    progress = "-"
    if state is not None:
        trials = store.trials(recipe.name, state.session)
        done = [t for t in trials if t.counted]
        what = "quality" if state.kind == "quality" else state.mode
        unit = "step" if state.kind == "quality" else "trial"
        progress = f"{len(done)} {unit}{'s' if len(done) != 1 else ''} done ({what})"
        if state.cuts:
            progress += f", paused {state.cuts}x"
        if trials:  # the last one, worded (a rejection says why)
            progress += f"; last: {trials[-1].line}"
    result = recipe.tuned.summary + f" ({recipe.tuned.gain:+.0%})" if recipe.tuned else "-"
    if recipe.quality is not None:  # The quant a quality tune kept
        result = f"quality: {recipe.quality.summary}" + ("" if result == "-" else f"; {result}")
    if (gave_up := store.gave_up(recipe.name)) is not None and recipe.tuned is None:
        result = f"gave up: {gave_up.why} ({local(gave_up.at):%Y-%m-%d %H:%M})"
    return [recipe.name, queue, progress, result]


def queue_cell(queued: Entry, closing: str | None) -> str:
    """`idle (asked, since 14:02)`; a quality plan its approval holds, or a tune its run is
    ending or whose run stopped before it ended it, says so (`closing`: `closing_now`)."""
    when = f"{queued.reason}, since {local(queued.queued_at):%H:%M}"
    if closing is not None:
        return f"{closing} ({when})"
    return f"{'now' if queued.now else 'idle'} ({when})"
