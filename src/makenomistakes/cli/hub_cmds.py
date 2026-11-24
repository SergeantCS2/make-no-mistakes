"""`mnm resolve` and `mnm fit`: look before you download."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from makenomistakes.busy_card import plan_note
from makenomistakes.cli._app import (
    FIT_STYLE,
    GIB,
    GpuOption,
    RamOption,
    VersionOption,
    VramOption,
    app,
    budget_override,
    console,
    fail_with,
)
from makenomistakes.cli.large_lines import head_lines, placement_lines, plan_lines
from makenomistakes.cli.quality_lines import print_measured
from makenomistakes.config import Settings
from makenomistakes.device_words import backend_tail, device_lines, pool_words
from makenomistakes.engine_support import installed_image
from makenomistakes.fit import Budget, FitLevel, Profile
from makenomistakes.fit.cards import ranked
from makenomistakes.fit.estimate import every_expert_in_ram
from makenomistakes.fit.music import ESTIMATE_WORDS, MusicTime, TimeBasis, music_time
from makenomistakes.fit.music import row_for as music_row
from makenomistakes.gguf import GgufHeader
from makenomistakes.hub import ModelFacts
from makenomistakes.hub.converted import kit_of
from makenomistakes.hub.facts import Engine, Format
from makenomistakes.hub.voices import kit_for, package_voices
from makenomistakes.instance_types import InstanceError
from makenomistakes.instance_vram import vram_holders
from makenomistakes.instances import InstanceService
from makenomistakes.music_basis import plan_basis
from makenomistakes.plan import Plan, PlanError, make_plan
from makenomistakes.plan import resolve as resolve_reference
from makenomistakes.plan_budget import planning_budget
from makenomistakes.quality.readings import QuantScore, measured_quants
from makenomistakes.recipes.llamacpp import LLAMACPP_IMAGE


@app.command()
def resolve(
    reference: Annotated[str, typer.Argument(help="HF link, org/name, org/name:QUANT, hf.co/…")],
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Show what a reference resolves to: modality, format, sizes, quants."""
    try:
        facts = resolve_reference(reference, Settings())
    except PlanError as exc:
        fail_with(exc)
    if as_json:
        console.print_json(facts.model_dump_json())
    else:
        print_facts(facts)


@app.command()
def fit(
    reference: Annotated[
        str,
        typer.Argument(
            help="A Hugging Face, Civitai or LM Studio link, org/name, or org/name/<file|folder>."
        ),
    ],
    version: VersionOption = None,
    vram_gb: VramOption = None,
    ram_gb: RamOption = 14.0,
    gpu: GpuOption = "RTX 4070",
) -> None:
    """Read the GGUF header remotely and show Quality / Balanced / Speed profiles."""
    settings = Settings()
    try:
        service = InstanceService(settings)
        budget = budget_override(vram_gb, ram_gb, gpu) or planning_budget(service)
        plan = make_plan(reference, settings, budget, version)
        holders = vram_holders(plan)
    except (PlanError, InstanceError) as exc:
        fail_with(exc)
    measured = measured_quants(service.measurements, plan.facts.repo_id)
    print_plan(plan, list(plan.profiles.values()), holders, measured)


def print_facts(facts: ModelFacts) -> None:
    console.print(
        f"[bold]{escape(facts.display_name)}[/bold]  ({escape(facts.repo_id)} @ {facts.revision})"
    )
    if facts.via:
        console.print(f"[dim]{escape(facts.via)}[/dim]")
    rows = [
        ("Type", facts.modality.label),
        ("Format", facts.format.value),
        ("Architecture", facts.architecture or "?"),
        ("Parameters", f"{facts.params_b:.2f} B" if facts.params_b else "?"),
        ("Native context", f"{facts.context_length:,}" if facts.context_length else "?"),
        ("Chat template", "yes" if facts.has_chat_template else "no"),
        ("Tool calling", "declared in template" if facts.supports_tools else "not declared"),
        ("Thinking", "likely" if facts.reasoning_hint else "no hint"),
        ("Gated", "yes" if facts.gated else "no"),
        ("License", facts.license or "?"),
        ("Base model", facts.base_model or "?"),
        ("All repo files", f"{facts.total_bytes / GIB:.1f} GiB"),
    ]
    for key, value in rows:
        console.print(f"  {key:<15} {escape(value)}")
    if facts.projector_files:
        console.print(f"  {'Vision proj.':<15} {escape(', '.join(facts.projector_files))}")
    for line in mtp_lines(facts) + head_lines(facts.draft_heads):
        console.print(line)
    if facts.gguf_variants:
        whisper = facts.format is Format.GGML
        table = Table(title="Model sizes" if whisper else "GGUF variants", show_lines=False)
        table.add_column("Size label" if whisper else "Quant")
        table.add_column("Size", justify="right")
        table.add_column("Files", justify="right")
        for variant in facts.gguf_variants:
            size = f"{variant.size_bytes / GIB:.2f} GiB"
            table.add_row(escape(variant.quant), size, str(len(variant.files)))
        console.print(table)


def _hybrid_note(header: GgufHeader) -> str:
    """A hybrid's KV in words: how few of its layers keep one."""
    if not header.recurrent_layers:
        return ""
    heads = header.kv_heads_per_layer()
    kv = sum(1 for il, holds in enumerate(header.kv_layers()) if holds and heads[il])
    recurrent = sum(
        1 for il, r in enumerate(header.recurrent_layers) if r and il < header.trunk_layers
    )
    if kv + recurrent == header.trunk_layers:
        return (
            f"; {kv} of {header.trunk_layers} layers keep a KV cache, the rest a fixed "
            "recurrent state (linear attention)"
        )
    # Nemotron-H: attention, Mamba and feed-forward blocks; only the first keep a KV
    return (
        f"; {kv} of {header.trunk_layers} layers keep a KV cache, {recurrent} a fixed recurrent "
        "state (Mamba), the rest neither"
    )


def _quants_word(quants: list[str], total: int) -> str:
    """Which quants a head drafts for, in a few words."""
    if len(quants) == total:
        return "every quant"
    if len(quants) <= 4:
        return ", ".join(quants)
    return f"{len(quants)} of the {total} quants"


def mtp_lines(facts: ModelFacts) -> list[str]:
    """The repo's MTP heads beside the projector: which variants draft with each, and
    the ones none does (a head without a quant label)."""
    heads: dict[str, list[str]] = {}
    for variant in facts.gguf_variants:
        if variant.mtp is not None:
            heads.setdefault(variant.mtp, []).append(variant.quant)
    sizes = {v.mtp: v.mtp_bytes for v in facts.gguf_variants if v.mtp is not None}
    lines = []
    for head, quants in heads.items():
        which = _quants_word(quants, len(facts.gguf_variants))
        lines.append(
            f"  {'MTP head':<15} {escape(head)} ({sizes[head] / GIB:.2f} GiB, drafts for "
            f"{escape(which)}: `mnm set <name> draft=mtp`)"
        )
    lines += [f"  {'MTP head':<15} {escape(h)} (unused)" for h in facts.unused_mtp_heads]
    return lines


def _plan_headline(plan: Plan) -> None:
    """What the plan is of: the GGUF's shape, the image kit, or the audio engine's files."""
    header, facts = plan.header, plan.facts
    if header is not None:
        ctx = f"native ctx {header.context_length:,}" if header.context_length else "ctx unknown"
        console.print(
            f"[bold]{escape(facts.display_name)}[/bold]  {escape(header.architecture)}, "
            f"{header.block_count} layers, {header.kv_heads} KV heads x {header.head_dim}, {ctx}"
            + _hybrid_note(header)
        )
    elif plan.kit is not None:
        console.print(f"[bold]{escape(facts.display_name)}[/bold]  {escape(plan.image_summary)}")
    elif facts.engine is Engine.WHISPERCPP:
        sizes = ", ".join(v.quant for v in facts.gguf_variants)
        console.print(
            f"[bold]{escape(facts.display_name)}[/bold]  {facts.modality.label} "
            f"(whisper.cpp); sizes: {escape(sizes)}"
        )
    elif facts.engine is Engine.VLLM:
        quant = next(iter(plan.profiles.values())).quant if plan.profiles else "?"
        console.print(
            f"[bold]{escape(facts.display_name)}[/bold]  {facts.modality.label} on vLLM "
            f"({escape(quant)} safetensors, {facts.vllm_bytes / GIB:.1f} GiB; no GGUF in the "
            "repo). vLLM takes a fixed share of the card, sized for the context."
        )
    elif facts.engine is Engine.AUDIOCPP and facts.voice_family is not None:
        kit = kit_for(facts.voice_family)
        quants = ", ".join(v.quant for v in facts.gguf_variants)
        console.print(
            f"[bold]{escape(facts.display_name)}[/bold]  {facts.modality.label} "
            f"({escape(kit.label)} on audio.cpp); files: {escape(quants)}; voices: "
            f"{escape(', '.join(package_voices(facts.gguf_variants, facts.voice_family)))}"
        )
    elif facts.engine is Engine.AUDIOCPP and facts.music_family is not None:
        _music_line(facts, plan)
    others = plan.other_versions
    if others and (plan.kit is not None or facts.engine is Engine.VLLM):
        console.print(
            f"Version {escape(facts.version or '')} of {len(facts.versions)}; "
            f"the others: {escape(', '.join(others))}. Plan one with "
            f"[bold]--version <name>[/bold] (part of a name is enough when it is unique)."
        )


def _music_line(facts: ModelFacts, plan: Plan) -> None:
    """A studio package: its kind and licence, and how its figures were come by, for each
    quant the plan's rows hold (never the first variant's file whatever the rows say: a BF16
    row would read "measured" when only Q8_0 was)."""
    assert facts.music_family is not None
    first = facts.gguf_variants[0].files[0]  # a converted checkpoint is its own kit
    kit = kit_of(facts.music_family, first)
    row = music_row(facts.music_family, first)
    quants = ", ".join(v.quant for v in facts.gguf_variants)
    basis = plan_basis(Settings().expanded_state_dir(), facts.music_family)
    budget = plan.budget
    times = {
        p.variant.quant: music_time(
            facts.music_family, budget.device, p.variant.files[0], budget.music_rates
        )
        for p in plan.profiles.values()
    }
    console.print(
        f"[bold]{escape(facts.display_name)}[/bold]  {facts.modality.label} "
        f"({escape(kit.label)} on audio.cpp); files: {escape(quants)}; licence "
        f"{escape(kit.licence)} (commercial use: {kit.commercial})"
    )
    console.print(escape(figures_words(row.source, basis == ESTIMATE_WORDS, basis, times)))


def figures_words(
    source: str, vram_estimated: bool, basis: str, times: dict[str, MusicTime]
) -> str:
    """Where a music plan's VRAM and time came from, in one sentence: both the row's estimate,
    or each its own (a time measured here, or the estimate scaled to this device);
    `times` by quant, each named when they were come by differently."""
    found = {(t.basis, t.words) for t in times.values()}
    if len(found) > 1:  # the quants' times differ in how they were come by
        each = " and ".join(f"{t.words} for {quant}" for quant, t in times.items())
        vram = f"an estimate ({source})" if vram_estimated else basis
        return f"Its VRAM is {vram}; its time {each}."
    time = next(iter(times.values()))
    if vram_estimated and time.basis is TimeBasis.ESTIMATE:
        return f"Its VRAM and time are an estimate ({source})."
    vram = f"an estimate ({source})" if vram_estimated else basis
    if time.basis is TimeBasis.ESTIMATE:
        return f"Its VRAM is {vram}; its time is an estimate ({source})."
    return f"Its VRAM is {vram}; its time {time.words}."


def _cards_line(budget: Budget, profiles: list[Profile]) -> None:
    """Several cards: each one's free VRAM, which the layers are split by; then the
    row the card is priced from."""
    for line in device_lines(budget, profiles):
        console.print(f"[dim]{escape(line)}[/dim]")
    if budget.cards:
        main = ranked(list(budget.cards))[0].index  # The widest link's
        each = "; ".join(
            f"{c.index}: {escape(c.name)}, {c.free_bytes / GIB:.1f} GiB free"
            + (f", x{c.lanes}" if c.lanes else "")
            + (" (main)" if c.index == main else "")
            for c in budget.cards
        )
        console.print(f"Cards: {each} (layers split in proportion; the speed is the split's)")


def _large_lines(plan: Plan, profiles: list[Profile]) -> None:
    """The architecture's llama.cpp build, an Engram table on disk, refused quants."""
    if plan.header is None:
        return
    # The box's own llama.cpp build is the one judged against the merge
    image = installed_image(plan.header.architecture, plan.image or LLAMACPP_IMAGE)
    for line in plan_lines(plan, profiles, image):
        console.print(f"[yellow]{escape(line)}[/yellow]")
    for line in placement_lines(profiles, plan.budget.device):  # Each card's part
        console.print(escape(line))


def print_budget(budget: Budget) -> None:
    """`Budget: 10.7 GiB free VRAM of 12.0 GiB (…), 19.4 GiB free RAM, …`, and WSL's margin."""
    vram = f"{budget.card_free_bytes / GIB:.1f} GiB free VRAM"
    if budget.vram_total_bytes is not None and budget.vram_used_bytes is not None:
        in_use = budget.vram_used_bytes + budget.held_bytes
        vram += (
            f" of {budget.vram_total_bytes / GIB:.1f} GiB ({in_use / GIB:.1f} GiB in use right now"
        )
        vram += f", {budget.held_bytes / GIB:.1f} of it mnm's)" if budget.held_by else ")"
    pool = pool_words(budget)  # A plan made in RAM says so
    said = pool if pool is not None else f"{vram}, {budget.ram_free_bytes / GIB:.1f} GiB free RAM"
    console.print(f"Budget: {said}, {budget.gpu_bandwidth / 1e9:.0f} GB/s{backend_tail(budget)}")
    if budget.spill_words is not None:
        console.print(f"[dim]{escape(budget.spill_words)}[/dim]")


def print_plan(
    plan: Plan,
    profiles: list[Profile],
    holders: list[str],
    measured: Mapping[str, QuantScore] | None = None,
) -> None:
    """The plan table, what `mnm tune --quality` measured of the repo's quants on this box,
    then the running instances to stop when it falls short (may be empty)."""
    header, budget = plan.header, plan.budget
    _plan_headline(plan)
    if plan.facts.via:  # another source's link
        console.print(f"[dim]{escape(plan.facts.via)}[/dim]")
    print_budget(budget)
    _cards_line(budget, profiles)
    note = plan_note(budget)
    if note:
        console.print(f"[yellow]{escape(note)}[/yellow]")
    if budget.correction_note:
        console.print(f"Calibration: {escape(budget.correction_note)}")
    if header is not None and header.latent_kv_unpriced:
        console.print("[yellow]DeepSeek-style attention: KV estimates are pessimistic.[/yellow]")
    _large_lines(plan, profiles)
    if plan.speed_note is not None:
        console.print(f"Calibration: {escape(plan.speed_note)}")
    elif plan.kit is not None:
        console.print("[dim]Image estimates are uncalibrated until the first measured run.[/dim]")
    if plan.facts.engine is Engine.WHISPERCPP:
        console.print(
            "[dim]Speech-to-text buffers are measured on a CPU build; the CUDA total is checked "
            "on the first run.[/dim]"
        )
    table = Table(title="Profiles (estimates)")
    for column in ("Profile", "Quant", "Settings"):
        table.add_column(column, no_wrap=True)
    table.add_column("Memory est.", overflow="fold")  # the card's share, and RAM's
    table.add_column("Fit", min_width=11)  # "comfortable" whole, never "comforta…"
    table.add_column("Speed")
    for p in profiles:
        style = FIT_STYLE[p.fit.level]
        table.add_row(
            p.name.value,
            escape(p.quant),
            p.settings_summary,
            p.vram_words,
            f"[{style}]{fit_text(p, header)}[/{style}]",
            p.speed_summary,
        )
    console.print(table)
    # Beside the table the reasons wrapped to one word a line at 80 columns.
    for p in profiles:
        console.print(f"  [bold]{p.name.value}[/bold]: {escape(p.why)}")
    print_measured(measured or {}, (p.quant for p in profiles))
    if holders:
        names = ", ".join(escape(n) for n in holders)
        one = len(holders) == 1  # sets were called workflows once; the web's note says set too
        console.print(
            f"[yellow]Planned as if {names} were stopped[/yellow] (about "
            f"{plan.budget.held_bytes / GIB:.1f} GiB {'it holds' if one else 'they hold'} counts "
            f"as free): stop {'it' if one else 'them'} before starting this one, or run them "
            "together in a set, which checks the fit."
        )


def fit_text(profile: Profile, header: GgufHeader | None) -> str:
    level = profile.fit.level
    if level is FitLevel.COMFORTABLE:
        return level.value
    if header is not None and every_expert_in_ram(profile.fit, header):
        return (  # `--cpu-moe -ngl N`
            f"{level.value} (every expert in RAM, {profile.fit.gpu_layers}/{header.block_count} "
            "layers' dense part on GPU)"
        )
    if header is not None and profile.fit.cpu_moe_layers:
        return (
            f"{level.value} (experts of {profile.fit.cpu_moe_layers}/{header.block_count} "
            "layers in RAM, the rest on GPU)"
        )
    if header is not None:
        return f"{level.value} ({profile.fit.gpu_layers}/{header.block_count} layers on GPU)"
    return f"{level.value} (weights stream from RAM)" if level is FitLevel.OFFLOAD else level.value
