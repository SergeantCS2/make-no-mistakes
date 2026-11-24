"""Instance lifecycle: `mnm add`, `start`, `stop`, `ls`, `rm`, `logs`, `url`."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.markup import escape
from rich.table import Table

from makenomistakes.agent_profile import AGENT_CTX, FLOORS, agent_context, floor_for, requested_type
from makenomistakes.capacity import FitNow
from makenomistakes.cli._app import (
    GIB,
    GpuOption,
    RamOption,
    VersionOption,
    VramOption,
    app,
    budget_override,
    console,
    fail_with,
    print_block,
    print_changes,
    print_diagnoses,
    print_progress,
    print_raw,
    spinner_progress,
)
from makenomistakes.cli.complete import instance_names
from makenomistakes.cli.disk_cmds import print_disk, print_warning
from makenomistakes.cli.hub_cmds import print_plan
from makenomistakes.cli.large_lines import download_words
from makenomistakes.cli.ls_lines import (
    print_box,
    print_comfy,
    print_machine,
    print_queue,
    print_tuning,
    state_cell,
)
from makenomistakes.cli.notify_cmd import notified
from makenomistakes.cli.tune_cmds import queue_after_add, settings_cell, tune_keys
from makenomistakes.config import Settings
from makenomistakes.consumers import Consumer
from makenomistakes.diagnose import LogReview
from makenomistakes.download import CacheStatus, cache_status
from makenomistakes.draft import drafting_with
from makenomistakes.fit import KvType, ProfileName
from makenomistakes.harness import consumer_urls
from makenomistakes.harness_agents import floor_line
from makenomistakes.held_back import held_back_in
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_types import (
    InstanceError,
    NoLogError,
    Stopped,
    by_type,
)
from makenomistakes.instance_vram import vram_holders
from makenomistakes.instances import InstanceService
from makenomistakes.owner_start import start_for_owner
from makenomistakes.plan import PlanError, make_plan
from makenomistakes.plan_budget import planning_budget
from makenomistakes.progress_words import parked_for_room
from makenomistakes.quality.readings import measured_quants
from makenomistakes.recipes import Recipe
from makenomistakes.router_client import router_now
from makenomistakes.supervisor import ContainerState
from makenomistakes.workflows import WorkflowStore


@app.command()
@notified("add", key="reference")
def add(
    reference: Annotated[
        str,
        typer.Argument(
            help="A Hugging Face, Civitai or LM Studio link, org/name, or org/name/<file|folder>."
        ),
    ],
    name: Annotated[
        str | None, typer.Option(help="Instance name (default: from the repo).")
    ] = None,
    profile: Annotated[
        ProfileName,
        typer.Option(help="quality, balanced or speed (custom: what --quant or --ctx make)."),
    ] = ProfileName.BALANCED,
    quant: Annotated[str | None, typer.Option(help="Explicit quant, e.g. Q5_K_M.")] = None,
    ctx: Annotated[int | None, typer.Option(help="Explicit context length.")] = None,
    kv: Annotated[KvType | None, typer.Option(help="KV cache type override.")] = None,
    version: VersionOption = None,
    model_type: Annotated[
        Modality | None,
        typer.Option("--type", help="Add a chat model as `coding` (or a coding model as `chat`)."),
    ] = None,
    for_agents: Annotated[
        bool,
        typer.Option(
            "--for-agents",
            help=f"The agent profile: type coding, context {AGENT_CTX:,} or the "
            "model's own when smaller, tools on.",
        ),
    ] = False,
    client: Annotated[
        str | None,
        typer.Option(
            "--for",
            help="The agent profile for one client's per-request floor: "
            + ", ".join(FLOORS)
            + "; implies --for-agents.",
        ),
    ] = None,
    download: Annotated[bool, typer.Option(help="Download the files now.")] = True,
    tune: Annotated[
        bool, typer.Option(help="Queue a tune for when the card is idle (`mnm tune`).")
    ] = True,
    upscaler: Annotated[
        bool,
        typer.Option(
            help="Image models: fetch the 18 MB ESRGAN upscaler too, for `/sdcpp/v1/upscale` and "
            "the engine page's upscale step."
        ),
    ] = True,
    vram_gb: VramOption = None,
    ram_gb: RamOption = 14.0,
    gpu: GpuOption = "RTX 4070",
) -> None:
    """Plan an instance for this machine, save its recipe, and download the files."""
    settings = Settings()
    service = InstanceService(settings)
    try:
        if name is not None:
            service.check_name(name)
        for_agents = for_agents or client is not None
        floor_for(client)  # an unknown client is refused before the Hub is asked
        budget = budget_override(vram_gb, ram_gb, gpu) or planning_budget(service)
        wanted = requested_type(model_type, for_agents)
        plan = make_plan(reference, settings, budget, version, wanted, quant=quant)
        if for_agents:
            ctx = agent_context(plan, ctx, client)
        holders = vram_holders(plan)
        shown = list(plan.profiles.values())
        measured = measured_quants(service.measurements, plan.facts.repo_id)
        try:
            chosen = plan.choose(profile, quant, ctx, kv)
        except PlanError:
            print_plan(plan, shown, holders, measured)  # the table says why; the holders, what
            raise
        instance = name or service.check_name(service.default_name(plan.facts))
        print_plan(plan, shown if chosen in shown else [*shown, chosen], holders, measured)
        print_warning(settings)  # the disk policy's line, from its first warning
        recipe = service.add(plan, chosen, instance, upscaler=upscaler)
        console.print(
            f"\n[bold]{instance}[/bold]: {chosen.name.value} profile, {escape(chosen.quant)}, "
            f"{escape(chosen.settings_summary)}, port {recipe.port}, "
            f"{_to_download(chosen.download_bytes, cache_status(recipe, service.models_dir))}"
            f"{download_words(chosen)}."
        )
        if for_agents:
            console.print(escape(floor_line(recipe, floor_for(client))))
        if download:
            service.download(recipe, print_progress)
        if tune:
            queue_after_add(service, recipe)
    except (PlanError, InstanceError) as exc:
        fail_with(exc)
    console.print(f"Saved. Next: [bold]mnm start {instance}[/bold]")


@app.command()
@notified("start", key="name", success=False)
def start(
    name: Annotated[str, typer.Argument(autocompletion=instance_names)],
    stop_others: Annotated[
        bool,
        typer.Option(
            "--stop-others",
            help="First park the running instances whose VRAM it needs (loaded again on request).",
        ),
    ] = False,
    anyway: Annotated[
        bool,
        typer.Option("--anyway", help="Start beside the instances it will not fit with."),
    ] = False,
    swap: Annotated[
        bool,
        typer.Option(
            "--swap",
            hidden=True,
            help="What an older mnm's request shim ran to wake a member; refused now.",
        ),
    ] = False,
) -> None:
    """Start the container, wait until it is ready, and print the URLs to paste."""
    if swap:  # an old request shim forwards to port + 10000, where no engine listens now
        fail_with(
            InstanceError(
                f"{name} is served through mnm's router; this request came through an older "
                "mnm's request shim, which cannot reach the new engine. Point the "
                f"client at `mnm url {name}`; `mnm stop {name}` removes the old shim."
            )
        )
    service = InstanceService(Settings())
    try:
        recipe = service.get(name)
        with console.status(f"Starting {name}…") as spinner:
            done = start_for_owner(service, recipe, spinner_progress(spinner), stop_others, anyway)
    except InstanceError as exc:
        fail_with(exc)
    if done.parked:
        console.print(escape(parked_for_room(done.parked)))
    started = done.started
    recipe = started.recipe
    print_changes(started.fixes)
    console.print(f"[green]{name} is ready.[/green]")
    for note in started.notes:  # after the spinner, so a progress line cannot hide them
        console.print(f"[yellow]Note:[/yellow] {escape(note)}")
    print_urls(recipe)
    try:
        _, review = service.logs(recipe, 0)
    except InstanceError as exc:  # it is up; only its log could not be read
        console.print(f"[yellow]Note:[/yellow] its log could not be read ({escape(str(exc))}).")
        return
    _print_review(review, summarise=name)


@app.command()
def stop(
    names: Annotated[
        list[str] | None, typer.Argument(show_default=False, autocompletion=instance_names)
    ] = None,
    every: Annotated[
        bool, typer.Option("--all", help="Stop every instance that has a container.")
    ] = False,
    park: Annotated[
        bool,
        typer.Option(
            "--park",
            help="Unload but keep it loadable: the router loads it at its next request "
            "(with --all, every instance, a stopped one too).",
        ),
    ] = False,
) -> None:
    """Stop and remove the container; the recipe and files stay. Every name is checked before
    anything stops, and an instance that is not running is not an error. Stopped means off
    (the router never loads it); --park means unloaded and loadable on request."""
    service = InstanceService(Settings())
    if every == bool(names):
        fail_with(InstanceError("Name the instances to stop, or pass --all (not both)."))
    try:
        outcomes = service.stop_many(  # A slot saved first says its line here
            names or None, park=park, told=lambda line: console.print(escape(line))
        )
    except InstanceError as exc:
        fail_with(exc)
    for item in outcomes:
        console.print(f"{escape(item.name)} {_stopped_words(item, park)}.")
    if every and not outcomes:
        console.print("No instances." if park else "Nothing is running.")
    if every:  # nothing runs as a set any more, said after what stopped
        try:
            _sets_stopped(WorkflowStore(service.settings.expanded_state_dir()))
        except InstanceError as exc:
            fail_with(InstanceError(f"The sets could not be marked stopped: {exc}"))


PARKED_WORDS = "parked: loads at its next request"


def _sets_stopped(store: WorkflowStore) -> None:
    for workflow in store.list():
        if workflow.started:
            store.mark_started(workflow.name, False)


def _stopped_words(item: Stopped, park: bool) -> str:
    if park:
        if not item.downloaded:  # the router answers 409 "not downloaded"
            return (
                f"is not downloaded yet, so nothing loads it on request: `mnm start {item.name}` "
                "downloads and starts it"
            )
        if item.was is ContainerState.EXITED:
            return (
                f"had exited; its container is removed and its log kept (`mnm logs "
                f"{item.name}`); {PARKED_WORDS}"
            )
        return PARKED_WORDS
    return {
        ContainerState.RUNNING: "stopped",
        ContainerState.EXITED: "had exited; its container is removed and its log kept "
        f"(`mnm logs {item.name}`)",
        ContainerState.ABSENT: "was not loaded; it stays off now (the router no longer "
        "loads it on request)",
    }[item.was]


@app.command()
def rm(
    name: Annotated[str, typer.Argument(autocompletion=instance_names)],
    files: Annotated[
        bool,
        typer.Option(
            "--files", help="Also delete the downloads no other instance uses (frees the disk)."
        ),
    ] = False,
) -> None:
    """Stop the instance and delete its recipe (downloaded files are kept unless --files)."""
    service = InstanceService(Settings())
    try:
        drafting = drafting_with(service.all(), name)
        cleanup = service.remove(service.get(name), files=files)
    except InstanceError as exc:
        fail_with(exc)
    if drafting is not None:
        console.print(f"[yellow]{escape(drafting)}[/yellow]")
    if cleanup is None:
        console.print(
            f"{name} removed. Its files stay under {service.models_dir}, so adding it again "
            f"downloads nothing (`mnm rm {name} --files` would have deleted them)."
        )
        return
    console.print(
        f"{name} removed; {len(cleanup.deleted)} "
        f"{'file' if len(cleanup.deleted) == 1 else 'files'} deleted, "
        f"{cleanup.freed_bytes / GIB:.1f} GiB freed."
    )
    if cleanup.shared:
        console.print(
            f"[dim]Kept {len(cleanup.shared)} "
            f"{'file' if len(cleanup.shared) == 1 else 'files'} other instances use: "
            f"{escape(', '.join(Path(rel).name for rel in cleanup.shared))}.[/dim]"
        )


VERDICT_STYLE: dict[FitNow, str] = {
    FitNow.RUNNING: "dim",
    FitNow.FITS: "green",
    FitNow.NEAR_LIMIT: "yellow",
    FitNow.NO_FIT: "red",
    FitNow.TAKES_TURNS: "cyan",
    FitNow.ON_REQUEST: "cyan",
    FitNow.UNKNOWN: "dim",
}


@app.command()
def ls(
    disk: Annotated[
        bool,
        typer.Option("--disk", help="Each instance's files on disk and when it was last used."),
    ] = False,
) -> None:
    """List instances with their container and health state."""
    service = InstanceService(Settings())
    try:
        statuses = service.statuses()
    except InstanceError as exc:
        fail_with(exc)
    snapshot = service.memory()
    print_machine(snapshot)
    if not statuses:
        console.print("No instances yet. Try: mnm add unsloth/Qwen3-8B-GGUF")
        print_comfy(service)
        print_box(service, snapshot, [])  # recorded: nothing to compare yet
        return
    verdicts = service.verdicts(
        statuses,
        snapshot,
        held_back_in(WorkflowStore(service.settings.expanded_state_dir()), statuses),
    )
    docker_error = next((s.error for s in statuses if s.error), None)
    if docker_error:  # once, above the table: per row it would push every column out
        console.print(f"[red]Docker could not be asked: {escape(docker_error)}[/red]")
    # Fits a 120-column terminal without cutting a word: no vertical rules, and only the
    # settings cell may wrap. The repo id is the name's link (`mnm show` prints it).
    # At 80 columns no_wrap columns were cut to "7…" ports and "not downloa…":
    # the port and the verdicts keep their width, names and settings fold instead.
    table = Table(title="instances", box=box.SIMPLE_HEAD)
    table.add_column("Name", overflow="fold", min_width=6)
    table.add_column("Type", overflow="fold", min_width=8)  # speech-to-text
    table.add_column("Port", no_wrap=True, min_width=4)
    table.add_column("Quant", overflow="fold", min_width=6)
    table.add_column("Settings", overflow="fold", min_width=8)
    table.add_column("State", overflow="fold", min_width=10)  # "downloaded" whole
    table.add_column("Fits now", overflow="fold", min_width=9)
    keys = tune_keys(service, [s.recipe for s in statuses], snapshot)  # "tuned +18%"
    for index, group in enumerate(by_type(statuses)):  # one section per type
        if index:
            table.add_section()
        for status in group:
            recipe = status.recipe
            verdict = verdicts[recipe.name]
            style = VERDICT_STYLE[verdict.level]
            table.add_row(
                f"[link={recipe.source_url}]{escape(recipe.name)}[/link]",
                recipe.modality.label,
                str(recipe.port),
                escape(recipe.quant),
                escape(settings_cell(recipe, keys)),
                state_cell(status),
                f"[{style}]{escape(verdict.words)}[/{style}]",
            )
    console.print(table)
    for name, verdict in verdicts.items():
        if (
            verdict.level
            in (
                FitNow.NO_FIT,
                FitNow.NEAR_LIMIT,
                FitNow.TAKES_TURNS,
                FitNow.ON_REQUEST,
            )
            or verdict.comfy_freed
        ):  # ComfyUI's card counted free is said
            console.print(f"[dim]{escape(name)}: {escape(verdict.note)}[/dim]")
    now = router_now(service.settings)  # one look at the router: what waits, what it tunes
    print_comfy(service)
    print_queue(now)
    print_disk(service, disk)
    print_tuning(service, now)
    print_box(service, snapshot, [s.recipe for s in statuses])  # the last line


@app.command()
def logs(
    name: Annotated[str, typer.Argument(autocompletion=instance_names)],
    tail: Annotated[int, typer.Option(help="Lines from the end (0 = everything).")] = 400,
    review_only: Annotated[
        bool, typer.Option("--review", help="Only the warnings, errors and diagnosis.")
    ] = False,
) -> None:
    """Show the engine log, then every warning and error in it with what each one means."""
    service = InstanceService(Settings())
    try:
        text, review = service.logs(service.get(name), tail)
    except NoLogError as exc:  # nothing to show is an answer, not a failure
        console.print(escape(str(exc)))
        return
    except InstanceError as exc:
        fail_with(exc)
    if not review_only:
        print_raw(text)
    _print_review(review)


# --- helpers ----------------------------------------------------------------------


def _to_download(planned: int, cache: CacheStatus) -> str:
    """What `add` will fetch: the plan's files minus what the cache already holds."""
    if cache.cached == 0:
        return f"{planned / GIB:.1f} GiB to download"
    here = f"{cache.cached_bytes / GIB:.1f} GiB already here"
    if cache.cached == cache.files:
        return f"nothing to download ({here})"
    return f"{max(planned - cache.cached_bytes, 0) / GIB:.1f} GiB to download ({here})"


def _print_review(review: LogReview, summarise: str | None = None) -> None:
    """Every warning and error the engine logged, so nothing hides in a long log; a clean log
    says so instead of printing nothing. With `summarise` (the instance's name, after a start)
    the known-harmless warnings are one counted line instead."""
    if review.clean:
        console.print("[dim]Engine log: no warnings or errors.[/dim]")
        return
    if not summarise:  # a ready start has no failure to explain (no "Likely cause" on it)
        print_diagnoses(review.diagnoses)
    warnings = review.notable_warnings if summarise else review.warnings
    for label, style, lines in (
        ("error", "red", review.errors),
        ("warning", "yellow", warnings),
    ):
        for line in lines:
            console.print(f"[{style}]engine {label}:[/{style}] {escape(line.text)}")
            if line.note:
                console.print(f"  [dim]→ {escape(line.note)}[/dim]")
    harmless = len(review.harmless_warnings)
    if summarise and harmless:
        noun = "warning" if harmless == 1 else "warnings"
        console.print(
            f"[dim]Engine log: {harmless} known, harmless {noun} "
            f"(`mnm logs {escape(summarise)} --review` lists each with what it means).[/dim]"
        )


def print_urls(recipe: Recipe, consumer: Consumer = Consumer.ODYSSEUS) -> None:
    for item in consumer_urls(recipe, consumer):
        console.print(f"[bold]{escape(item.label)}[/bold]")
        if item.url:  # the stitch line is a command with no address of its own
            print_block(f"  {item.url}")
        console.print(f"  [dim]{escape(item.note)}[/dim]")
        if item.snippet:
            print_block(item.snippet)
