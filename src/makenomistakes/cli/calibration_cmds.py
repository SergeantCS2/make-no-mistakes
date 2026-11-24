"""`mnm calibration`: what this box measured, and how it corrects the planner; another
machine's readings are kept and counted, never used."""

from __future__ import annotations

from pathlib import Path

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.calibration import (
    MIN_READINGS,
    MachineSummary,
    SpeedCorrection,
    corrections,
    every_reading,
    export,
    speed_corrections,
)
from makenomistakes.cli._app import app, console
from makenomistakes.config import Settings
from makenomistakes.device_learning import learn_decode, learned_words
from makenomistakes.device_learning import saved as saved_decode
from makenomistakes.expert_rate import learn_from, saved
from makenomistakes.fit.cpu_rates import CPU_EXPERT_RATE, EXPERT_TOUCH_SECONDS
from makenomistakes.fit.devices import device_for, priced_words
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.local_time import local
from makenomistakes.machine_facts import MachineFactsError, load_file, platform_here, update
from makenomistakes.measurements import MeasurementStore, Reading, SpeedUnit
from makenomistakes.music_speed import music_rates
from makenomistakes.quality.readings import QUALITY_CAPTION, newest_scores
from makenomistakes.queue_summary import QUEUE_CAPTION
from makenomistakes.queue_summary import summaries as queue_summaries
from makenomistakes.reading_stamp import beside_words
from makenomistakes.readings_filter import (
    Here,
    left_out,
    left_out_words,
    memory_readings,
    readings_here,
    store_here,
)
from makenomistakes.request_peak import load_and_peak
from makenomistakes.store import StoreError
from makenomistakes.studio.scores import summaries

GIB = 2**30
calibration_app = typer.Typer(
    help="Readings from this box and the corrections they make to the planner.",
    invoke_without_command=True,
)
app.add_typer(calibration_app, name="calibration")


def _gib(value: int | None) -> str:
    return f"{value / GIB:.2f}" if value else "-"


@calibration_app.callback()
def show(ctx: typer.Context) -> None:
    """This box, each engine's correction, then each instance's newest readings (this box's
    only: another machine's are kept for when it returns)."""
    if ctx.invoked_subcommand is not None:
        return
    store = MeasurementStore(Settings().expanded_state_dir())
    here = store_here(store)
    console.print(escape(_machine_line(here)))
    _tuned()
    kept = store.every()
    by_name = {name: readings_here(found, here) for name, found in kept.items()}
    _device(by_name, here)  # The card's row, and what this box taught it
    foreign = [r for found in kept.values() for r in found if not here.owns(r.machine)]
    if not any(by_name.values()):
        console.print(
            "No readings on this box yet: every start and the requests the router forwards "
            "leave one (`mnm try <name>` makes a request)."
        )
        if (said := left_out_words(len(foreign))) is not None:
            console.print(f"[dim]{escape(said)}.[/dim]")
        return
    every = every_reading(by_name)
    engines = Table(title="corrections", box=box.SIMPLE_HEAD)
    for column in ("Engine", "Type", "Readings", "Correction"):
        engines.add_column(column)
    for found in corrections(every):
        engines.add_row(
            found.engine.value, found.group.value, str(found.readings), f"x{found.ratio:.2f}"
        )
    engines.caption = left_out_words(sum(1 for r in foreign if r.ratio is not None))
    console.print(engines)
    console.print(
        f"[dim]A larger measurement applies at once; a smaller one needs {MIN_READINGS} "
        "readings and lowers an estimate by at most 10%.[/dim]"
    )
    speeds = speed_corrections(every)
    if speeds:
        timed = sum(1 for r in foreign if r.speed_ratio is not None)
        console.print(_speed_table(speeds, left_out_words(timed)))
    _experts(by_name)
    _music(every)
    _scores(by_name)
    _prefixes(by_name)
    _queues(by_name)
    _qualities(by_name)
    for name, readings in by_name.items():
        others = left_out(kept.get(name, []), here)
        plain = memory_readings(readings)  # the scores have their own table
        if plain or others:
            console.print(_readings_table(name, plain[-10:], others))


def _music(every: list[Reading]) -> None:
    """Each music model's seconds of work a second of audio on this box, once it has
    the readings a plan takes it from (`music_speed`)."""
    backends = sorted({r.backend or "" for r in every if r.speed_unit is SpeedUnit.AUDIO})
    rates = [rate for b in backends for rate in music_rates(every, b or None)]
    if not rates:
        return
    table = Table(title="music speed", box=box.SIMPLE_HEAD)
    for column in ("Family", "Model file", "Image", "Readings", "Work a second of audio"):
        table.add_column(column)
    for rate in rates:
        table.add_row(
            escape(rate.family),
            escape(rate.file),
            escape(rate.backend or "?"),
            str(rate.readings),
            f"{rate.rate:.2f} s",
        )
    table.caption = "audio.cpp's own timing; plans and clients' waits take these (this box's)."
    console.print(table)


def _scores(by_name: dict[str, list[Reading]]) -> None:
    """Each song instance's candidates as the studio's scorer heard them, per scorer
    and settings (relative between candidates of one scorer, never a Suno figure)."""
    found = summaries(by_name)
    if not found:
        return
    table = Table(title="scores", box=box.SIMPLE_HEAD)
    for column in ("Instance", "Scorer", "Settings", "Candidates"):
        table.add_column(column)
    for summary in found:
        table.add_row(
            escape(summary.instance),
            escape(summary.scorer),
            escape(summary.settings),
            escape(summary.words()),
        )
    table.caption = "The mean of enjoyment and production quality (1 to 10), this box's only."
    console.print(table)


def _prefixes(by_name: dict[str, list[Reading]]) -> None:
    """Each instance's newest `mnm try --agent` reading, this box's."""
    rows = [
        (name, found)
        for name, readings in by_name.items()
        if (found := next((r for r in reversed(readings) if r.prefix is not None), None))
    ]
    if not rows:
        return
    table = Table(title="prompt cache", box=box.SIMPLE_HEAD)
    for column in ("Instance", "Prefix", "Cold", "Warm", "After a park", "When"):
        table.add_column(column)
    for name, reading in rows:
        figures = reading.prefix
        assert figures is not None  # the row was chosen for it
        after = (
            f"{figures.reload_seconds or 0:.1f} s back + {figures.restored_seconds:.1f} s read"
            if figures.restored_seconds is not None
            else "-"
        )
        table.add_row(
            escape(name),
            f"{figures.tokens:,} tokens",
            f"{figures.cold_seconds:.1f} s",
            f"{figures.warm_seconds:.1f} s",
            after,
            f"{local(reading.at):%Y-%m-%d %H:%M}",
        )
    table.caption = "`mnm try <name> --agent`: the prompt read; after a park, the slot put back."
    console.print(table)


def _queues(by_name: dict[str, list[Reading]]) -> None:
    """What waited for each instance, its windows summed per slot count (this box's)."""
    rows = queue_summaries(by_name)
    if not rows:
        return
    table = Table(title="queue", box=box.SIMPLE_HEAD)
    columns = ("Instance", "Slots", "Requests", "Waited for a slot", "Waited for a load")
    for column in (*columns, "llama.cpp's own wait", "Context held"):
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(
            escape(row.name),
            str(row.figures.slots),
            row.requests_words(),
            row.slot_words(),
            row.load_words(),
            row.engine_words(),
            row.context_words(),
        )
    table.caption = QUEUE_CAPTION
    console.print(table)


def _qualities(by_name: dict[str, list[Reading]]) -> None:
    """What each quality tune measured, each quant's newest scores on this box."""
    rows = [(name, score) for name, found in by_name.items() for score in newest_scores(found)]
    if not rows:
        return
    table = Table(title="quality", box=box.SIMPLE_HEAD)
    for column in ("Instance", "Quant", "Against", "KLD", "Same top token", "PPL", "Decode"):
        table.add_column(column, overflow="fold")
    table.add_column("When")
    for name, score in rows:
        f = score.figures
        kld, same, decode = f.cells
        table.add_row(
            escape(name),
            escape(f.quant),
            escape(f"{f.reference} on {f.text}"),
            kld,
            same,
            f"{f.ppl:.2f}",
            decode,
            f"{local(score.at):%Y-%m-%d %H:%M}",
        )
    table.caption = QUALITY_CAPTION
    console.print(table)


def _machine_line(here: Here) -> str:
    """One sentence for the box the readings are read for."""
    if here.words is None:
        return (
            "This box has not been read yet (`mnm machine show` reads it): the readings that "
            "name no box are used."
        )
    return f"This box: {here.words}; its readings only."


def _tuned() -> None:
    """Each tuned instance's score against mnm's plan as its first trial measured it;
    nothing when none is tuned."""
    try:
        recipes = [r for r in InstanceService(Settings()).all() if r.tuned is not None]
    except InstanceError:
        return
    if not recipes:
        return
    table = Table(title="tuned", box=box.SIMPLE_HEAD)
    for column in ("Instance", "Settings chosen", "Decode tok/s", "Score", "mnm's plan", "Gain"):
        table.add_column(column, overflow="fold")
    for recipe in recipes:
        tuned = recipe.tuned
        if tuned is None:
            continue
        table.add_row(
            escape(recipe.name),
            escape(tuned.summary),
            f"{tuned.score.decode:.1f} (plan {tuned.seed.decode:.1f})",
            f"{tuned.score.value:.1f} ({tuned.score.mode})",
            f"{tuned.seed.value:.1f}",
            f"{tuned.gain:+.0%} over {tuned.trials} trials",
        )
    console.print(table)


def _experts(by_name: dict[str, list[Reading]]) -> None:
    """The rate this box's MoE readings imply for experts kept in RAM, saved for the
    planner in `<state>/machine.json`; nothing without such readings."""
    settings = Settings()
    try:
        learned = learn_from(InstanceService(settings), by_name)
        if learned is None:
            return
        facts = update(settings.expanded_state_dir(), saved(learned))
    except (InstanceError, StoreError, MachineFactsError) as exc:
        console.print(f"[yellow]Experts in RAM: not learned ({escape(str(exc))}).[/yellow]")
        return
    console.print(
        f"Experts kept in RAM decode at about {learned.gbps:g} GB/s on this box "
        f"({learned.readings} reading{'s' * (learned.readings != 1)}); saved, so plans price "
        f"them at {facts.expert_rate / 1e9:.1f} GB/s ({_default_words(settings)}: "
        f"{CPU_EXPERT_RATE / 1e9:g}, each used expert {EXPERT_TOUCH_SECONDS * 1e6:g} us more)."
    )


def _default_words(settings: Settings) -> str:
    """What priced RAM-held experts before this box's readings: its RAM platform's row,
    else the figure measured on an RTX 4070 under WSL."""
    try:
        file = load_file(settings.expanded_state_dir())
    except MachineFactsError:
        file = None
    platform = platform_here(file) if file is not None else None
    if platform is None or platform.kind.value in ("wsl", "unknown"):
        return "an RTX 4070 machine's figure"
    return (
        f"{platform.row.label}'s default {platform.expert_gbps:g} GB/s; "
        "an RTX 4070 machine's figure"
    )


def _device(by_name: dict[str, list[Reading]], here: Here) -> None:
    """The card's row and what this box's readings taught about it: the decode
    efficiency (and the launch term, from two model sizes), saved for the planner."""
    settings = Settings()
    state = settings.expanded_state_dir()
    try:
        file = load_file(state)
        cards = file.current.fingerprint.cards if file and file.current else []
        card = cards[0].name if cards else None
        learned = learn_decode(InstanceService(settings), by_name, here, card)
        facts = (
            update(state, saved_decode(learned)) if learned is not None else file and file.facts()
        )
    except (InstanceError, StoreError, MachineFactsError) as exc:
        console.print(f"[yellow]Decode efficiency: not learned ({escape(str(exc))}).[/yellow]")
        return
    named = learned.card if learned is not None else card
    if named is None:
        return
    row = device_for(named)
    if facts and facts.decode_card is not None and device_for(facts.decode_card) == row:
        row = row.with_learned(facts.learned_device())
    console.print(f"Card: {escape(priced_words(row))}.")
    if learned is not None:
        console.print(
            f"Learned {escape(learned_words(learned))}; saved, so plans on this card use it."
        )


def _speed_table(speeds: list[SpeedCorrection], left: str | None = None) -> Table:
    """Each picture family's time against the planner's: its s/image and s/clip."""
    table = Table(title="picture and clip speed", box=box.SIMPLE_HEAD)
    for column in ("Family", "Readings", "Correction"):
        table.add_column(column)
    for found in speeds:
        table.add_row(escape(found.label), _counted(found), f"x{found.ratio:.2f}")
    table.caption = (
        f"A family is corrected from {MIN_READINGS} readings: warm ones, or first pictures "
        "after a start too while warm ones are fewer (those also load the weights, so the "
        "correction leans slow); `mnm fit` prints each family's s/image and s/clip with it."
        + (f" {left}." if left else "")
    )
    return table


def _counted(found: SpeedCorrection) -> str:
    """`4 (3 first after a start)`, or `1 (+3 first after a start, not counted)`, and the
    readings left out as implausible after it."""
    ignored = f"; {found.ignored_words}" if found.ignored_words else ""
    if found.with_cold:
        return f"{found.readings} ({found.cold} first after a start){ignored}"
    cold = f" (+{found.cold} first after a start, not counted)" if found.cold else ""
    return f"{found.readings}{cold}{ignored}"


def _readings_table(name: str, readings: list[Reading], others: int = 0) -> Table:
    """Narrow enough for an 80-column terminal; the newest build and settings go in the
    caption (a figure is reused only for the same pair), with the readings taken beside
    something outside mnm (marked `*`) and those of another machine."""
    if not readings:
        return Table(title=name, caption=left_out_words(others), box=box.SIMPLE_HEAD)
    newest = readings[-1]
    caption = f"build {newest.build or 'unknown'}, settings {newest.settings}"
    both = load_and_peak(readings)
    if both is not None:  # what a request adds to what the load held
        load, peak = both
        caption += (
            f"; loaded {load / GIB:.2f} GiB, a request's peak {peak / GIB:.2f} GiB "
            f"(+{(peak - load) / GIB:.2f}, `mnm try --measure`)"
        )
    beside = [(r, words) for r in readings if (words := beside_words(r)) is not None]
    for r, words in beside:
        caption += f"; * {local(r.at):%m-%d %H:%M} {words}"
    if (left := left_out_words(others)) is not None:
        caption += f"; {left}"
    table = Table(title=name, caption=caption, box=box.SIMPLE_HEAD)
    for column in ("When", "Kind", "VRAM GiB", "Est. GiB", "Load s", "Speed"):
        table.add_column(column, no_wrap=True)
    marked = {id(r) for r, _words in beside}
    for r in readings:
        speed = f"{r.speed:g} {r.speed_unit.value}" if r.speed and r.speed_unit else "-"
        if r.speed_ratio is not None:  # a picture's or clip's: the planner's figure beside it
            speed += f" (est. {r.estimate_seconds:g})"
        if r.cold and r.speed:
            speed += ", first"
        table.add_row(
            f"{local(r.at):%m-%d %H:%M}{' *' if id(r) in marked else ''}",
            r.kind.value,
            _gib(r.vram_bytes),
            _gib(r.estimate_bytes),
            f"{r.load_seconds:g}" if r.load_seconds is not None else "-",
            escape(speed),
        )
    return table


@calibration_app.command("export")
def export_cmd() -> None:
    """Write every reading and correction to one file in this folder, to paste into a chat."""
    store = MeasurementStore(Settings().expanded_state_dir())
    by_name, here = store.every(), store_here(store)
    gpu = probe_memory().gpu
    written = export(
        by_name,
        gpu.name if gpu else None,
        gpu.vram_total_bytes if gpu else None,
        here={name: readings_here(found, here) for name, found in by_name.items()},
        machine=MachineSummary(id=here.id, words=here.words),
    )
    path = Path.cwd() / f"mnm-calibration-{local():%Y%m%d-%H%M%S}.json"
    path.write_text(written.model_dump_json(indent=2) + "\n", encoding="utf-8")
    count = sum(len(r) for r in by_name.values())
    others = sum(left_out(found, here) for found in by_name.values())
    mine = f" ({others} of another machine, each naming its box)" if others else ""
    console.print(
        f"Wrote {escape(str(path))}: {count} readings from {len(by_name)} instances{mine}."
    )
