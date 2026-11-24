"""A style training as a run: `mnm studio train style` checks the name, the songs and the
plan, downloads the ACE-Step checkpoint it trains against, copies the songs into a run of kind
`training` with `kind: style` in its settings, writes the dataset and the job, and leaves it
queued for the router's trainer (`train_runner`), which takes it as it takes a voice. At the
end the adapter joins the style library (`style_library`) with the training's settings, the
songs' hashes and the box that trained it.

As a voice's, a style's run joins the queue only once everything is written (its `train` step
saved last: `train_jobs.waiting`). Nothing is downloaded or copied for a training refused.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from makenomistakes.config import Settings
from makenomistakes.download import Fetcher
from makenomistakes.readings_filter import here_of
from makenomistakes.run_parts import InputRecord
from makenomistakes.runs import RunKind, RunRecord, RunStore, StepState
from makenomistakes.studio import style_data, style_plan, style_weights, train_data
from makenomistakes.studio.style_library import (
    StyleError,
    StyleLibrary,
    StyleSource,
    check_name,
    check_trigger,
)
from makenomistakes.studio.style_trainer import (
    BUILD,
    RESULT,
    STYLE,
    StyleJob,
    StyleTrainer,
    StyleTrainerError,
    read_trained,
)
from makenomistakes.studio.train_jobs import (
    LOG,
    STAGE,
    Queued,
    TrainError,
    enqueue,
    kind_of,
    queued,
    training_of,
)
from makenomistakes.studio.trainer import Variant, variant_of, write_job

DEFAULT_EPOCHS = 400  # ACE-Step's tutorial: 500 for about 100 songs, more for 10 to 20
KEYS = (BUILD, "source", "epochs", "rank", "preset", "max_seconds", "trigger", "caption")
Say = Callable[[str], None]


def queue_style(  # noqa: PLR0913 - the training's parts, each its own flag
    settings: Settings,
    trainer: StyleTrainer,
    name: str,
    folder: Path,
    source: StyleSource,
    *,
    caption: str,
    card_gib: float | None,
    ram_gib: float | None = None,
    trigger: str | None = None,
    variant: Literal["base", "turbo"] = "base",
    epochs: int = DEFAULT_EPOCHS,
    max_seconds: float = style_data.DEFAULT_MAX_SECONDS,
    licence: str = "",
    fetcher: Fetcher | None = None,
    say: Say = lambda _line: None,
) -> Queued:
    """Check, plan, download, copy and queue; refused before anything is written, and the
    checkpoint downloaded before the run is made (a failed download queues nothing)."""
    state = settings.expanded_state_dir()
    store = RunStore.for_settings(settings)
    trigger = (trigger or name).strip()
    _refuse(settings, store, name, trigger, caption, epochs, max_seconds)
    ledger = trainer.installed()
    image_variant = variant_of(ledger.image) or Variant.CPU
    on_card = image_variant is Variant.CUDA
    if not on_card:
        _refuse_ram(ram_gib)
    machine = here_of(state).id
    planned = style_plan.plan(card_gib if on_card else None, max_seconds)
    measured = style_plan.read_probes(state).get(
        style_plan.probe_key(machine, variant, planned.preset.name, max_seconds)
    )
    planned = style_plan.plan(planned.card_gib, max_seconds, measured)
    if (why := style_plan.refusal(planned)) is not None:
        raise TrainError(why)
    found = style_data.survey_songs(folder, max_seconds)
    weights = style_weights.fetch(settings, variant, fetcher, say)
    _refuse(settings, store, name, trigger, caption, epochs, max_seconds)  # again: the download
    preset = planned.preset  # took minutes, and another queue may have come meanwhile
    job = style_plan.fit_job(
        StyleJob(
            name=name,
            variant=variant,
            dit_dir=style_weights.container_path(settings, weights.dit_dir),
            vae_dir=style_weights.container_path(settings, weights.vae_dir),
            text_encoder_dir=style_weights.container_path(settings, weights.text_encoder_dir),
            epochs=epochs,
            rank=preset.rank,
            alpha=preset.alpha,
            grad_accum=preset.grad_accum,
            optimizer=preset.optimizer,
            offload_encoder=preset.offload_encoder,
            max_seconds=max_seconds,
            device="cpu",
        ),
        on_card=on_card,
        card_gib=planned.card_gib,
        resumed=False,
    ).job
    run = store.new(
        RunKind.TRAINING,
        name,
        f"An ACE-Step style from {folder.name}",
        {
            "kind": STYLE,
            BUILD: variant,
            "source": source.value,
            "epochs": str(epochs),
            "rank": str(job.rank),
            "preset": preset.name,
            "max_seconds": f"{max_seconds:.0f}",
            "trigger": trigger,
            "caption": caption,
            "image": ledger.image,
            "variant": image_variant.value,
            **({"licence": licence} if licence else {}),
        },
    )

    def copy(here: Path) -> list[InputRecord]:
        recorded = train_data.copy_in(found, here / style_data.SONGS, "song", "song")
        copied = train_data.copied_names(found, "song")
        style_data.write_dataset(here, style_data.dataset(found, copied, caption, trigger))
        write_job(here, job)
        return recorded

    enqueue(store, run, "songs", copy, list(found.warnings))
    return Queued(run, found, _estimate(planned, epochs, len(found.recordings)))


def _estimate(planned: style_plan.StylePlan, epochs: int, songs: int) -> str:
    hours = planned.hours(epochs, songs)
    timed = (
        f"; about {hours:.1f} hours of a free card by this box's probe" if hours is not None else ""
    )
    return f"{epochs} epochs on {songs} songs, {planned.words}{timed}"


def _refuse(
    settings: Settings,
    store: RunStore,
    name: str,
    trigger: str,
    caption: str,
    epochs: int,
    max_seconds: float,
) -> None:
    library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    try:
        check_name(name)
        check_trigger(trigger)
    except StyleError as exc:
        raise TrainError(str(exc)) from exc
    if library.has(name):
        raise TrainError(
            f"A style named '{name}' is in the library: `mnm studio styles rm {name}` first."
        )
    if any(r.target == name and kind_of(r) == STYLE for r in queued(store)):
        raise TrainError(f"A training of the style '{name}' is queued already.")
    if not caption.strip() or len(caption) > 512:
        raise TrainError(
            "A style needs a caption (what the songs sound like, at most 512 characters): "
            '`--caption "warm lo-fi hip hop, dusty drums, tape hiss"`.'
        )
    if not 1 <= epochs <= 10_000:
        raise TrainError("Epochs are 1 to 10000.")
    if not 30 <= max_seconds <= 600:
        raise TrainError("--max-seconds is 30 to 600 (a song longer is cut there).")


def _refuse_ram(ram_gib: float | None) -> None:
    """A training on the processor holds the whole model in fp32 with its encoders: refused
    on a box whose RAM cannot hold it beside the system (`style_plan.CPU_RAM_GIB`)."""
    if ram_gib is not None and ram_gib < style_plan.CPU_RAM_GIB + style_plan.CPU_RAM_SPARE_GIB:
        raise TrainError(
            f"A style training on the processor needs about {style_plan.CPU_RAM_GIB:.0f} GiB "
            f"of RAM beside the system, and this machine has {ram_gib:.1f} GiB in all."
        )


def finish_style(
    settings: Settings, store: RunStore, run: RunRecord, seconds: float | None
) -> RunRecord:
    """The adapter into the style library (checked as audio.cpp reads it), the run done, the
    work folder's songs, tensors and checkpoints removed. The run is read again first: the
    probe wrote its reading on it while the training ran."""
    run = store.get(run.id)
    here = store.folder(run)
    trained = read_trained(here)
    library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    training = training_of(settings, run, trained, KEYS)
    try:
        row = library.add(
            run.target,
            here / Path(RESULT).parent,
            trained.variant,
            run.settings.get("trigger", run.target),
            StyleSource(run.settings["source"]),
            caption=run.settings.get("caption", ""),
            licence=run.settings.get("licence", ""),
            training=training,
        )
    except StyleError as exc:
        raise StyleTrainerError(
            f"The style trained but could not join the library: {exc} Its files stay in "
            f"{here / Path(RESULT).parent}."
        ) from exc
    drop_work(here)
    log = here / LOG
    step = run.step(STAGE)
    if step is not None:
        step.state = StepState.DONE
        step.seconds = trained.seconds
    words = (
        f"{row.name}: an ACE-Step 1.5 {row.variant} LoRA, rank {row.rank}, "
        f"{trained.epochs} epochs; its trigger '{row.trigger}'"
    )
    numbers = {"epochs": float(trained.epochs)}
    if trained.steps is not None:
        numbers["steps"] = float(trained.steps)
    if trained.loss is not None:
        numbers["loss"] = trained.loss
    return store.finish(run, [log] if log.is_file() else [], words, seconds, numbers)


def drop_work(here: Path) -> None:
    """What only the training needed: the copied songs, the checkpoint links, the tensors,
    the checkpoints, the result (the library holds its own copy)."""
    for name in (style_data.SONGS, "ckpt", "tensors", "out", "result"):
        shutil.rmtree(here / name, ignore_errors=True)
    for name in ("job.json", "progress.json", style_data.DATASET_JSON):
        (here / name).unlink(missing_ok=True)
