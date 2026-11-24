"""`mnm tune <name> --quality`: the plan shown and asked for, the `full` image
pulled and the candidates downloaded here, then the tune queued with the router's tuner (or,
with `--now` and no router, run in this terminal)."""

from __future__ import annotations

from collections.abc import Sequence

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.cli._app import console, fail_with, print_progress
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.quality import QualityPlanError
from makenomistakes.quality.approve import approve
from makenomistakes.quality.prepare import Offer, offer
from makenomistakes.quality.words import plan_rows, plan_words
from makenomistakes.tune.store import Entry, TuneStore, TuneStoreError
from makenomistakes.tune.trial_types import Cut
from makenomistakes.tune.wiring import foreground
from makenomistakes.tune.words import of

QUEUED_NO_ROUTER = (
    "The router is not running: the quality tune waits for it (`mnm service install`), or "
    "`mnm tune {name} --quality --now` runs it in this terminal."
)


def quality_tune(
    service: InstanceService,
    name: str,
    floor: float | None,
    quants: Sequence[str],
    keep: bool,
    yes: bool,
    now: bool,
) -> None:
    try:
        offered = offer(service, name, floor, quants, keep)
    except (QualityPlanError, InstanceError) as exc:
        fail_with(exc)
    show(offered)
    if offered.room_short:
        fail_with(QualityPlanError(offered.short_words))
    if not yes and not typer.confirm("Download what is missing and measure?", default=False):
        raise typer.Exit(1)
    try:
        approved = approve(service, name, offered, now, print_progress)
    except (QualityPlanError, InstanceError) as exc:
        fail_with(exc)
    except KeyboardInterrupt:
        console.print(
            "Stopped: a plan still downloading ended here with what it downloaded; one whose "
            "downloads were done stays queued for the router (`mnm tune --status` shows the "
            f"queue, `mnm tune {name} --cancel` ends it)."
        )
        raise typer.Exit(130) from None
    if approved.replaced:
        console.print(escape(f"The quality tune {approved.replaced}."))
    if approved.answer is not None:
        console.print(escape(approved.answer[1]))
        return
    if approved.waiting is not None:
        console.print(escape(approved.waiting))
        return
    if not now:
        console.print(escape(QUEUED_NO_ROUTER.format(name=name)))
        return
    console.print(
        "The router is not running: measuring in this terminal (Ctrl-C stops it; nothing else "
        "cuts a step here, so leave the card alone meanwhile)."
    )
    tuner = foreground(service, lambda line: console.print(escape(line)))
    try:
        end = tuner.tune(approved.entry, Cut())
    except KeyboardInterrupt:
        resume, ends = f"mnm tune {name} --quality --now", "ends it with what it downloaded"
        console.print(escape(stopped_words(service, approved.entry, resume, ends)))
        raise typer.Exit(130) from None
    if end.status != "tuned":
        raise typer.Exit(1)


def stopped_words(service: InstanceService, entry: Entry, resume: str, ends: str) -> str:
    """Ctrl-C's line in a terminal's run of `entry`: read from that very entry as queued now
    (never whatever the name holds: a quality tune's end queues a speed tune for its new
    quant). One the user cancelled meanwhile is ended by a router's next look or a second
    cancel, never resumed; any other stays queued."""
    name, unit = entry.name, of(entry).unit
    try:
        queued = TuneStore(service.settings.expanded_state_dir()).current(entry)
    except TuneStoreError:
        queued = None
    if queued is None:  # it saved and ended, or its end ran as it stopped
        return f"Stopped. It had finished or ended: `mnm tune --status` shows what {name} kept."
    if queued.cancelled:
        return (
            f"Stopped. It was cancelled: a router running now or started later ends it, as "
            f"does `mnm tune {name} --cancel`."
        )
    if queued.saving:
        return (
            f"Stopped while it saved its result: a router running now or started later "
            f"settles it, as does `{resume}` (`mnm tune --status` shows what {name} kept)."
        )
    return (
        f"Stopped. It stays queued: a router running now or started later resumes it from the "
        f"last whole {unit}, as does `{resume}`; `mnm tune {name} --cancel` {ends}."
    )


def show(offered: Offer) -> None:
    table = Table(title=f"quality tune: {escape(offered.plan.repo_id)}", box=box.SIMPLE_HEAD)
    for column in ("Quant", "Role", "Size", "Here", "Planned decode"):
        table.add_column(column, overflow="fold")
    for row in plan_rows(offered):
        here = "yes" if row.here else "download"
        table.add_row(escape(row.quant), row.role, row.size, here, row.decode)
    console.print(table)
    for line in plan_words(offered):
        console.print(escape(line))
