"""A voice training as a run: `mnm studio train voice` checks the recordings,
copies them into a run of kind `training` (its folder under the outputs, its record under the
state folder) and leaves it queued; the router's trainer (`train_runner.py`) takes it when the
card is free, cuts it for requests, resumes it, and at the end hands the voice to the library
through the same checks `mnm voices add` makes (`rvc_checkpoint`), with the training's settings,
the recordings' hashes and the box that trained it on the row.

A training joins the queue only once its recordings are copied and its job written (its
`train` step is saved last: `waiting`), so the router never takes one half made. A queued
training has no process (`pid` None: never cleaned); the router writes its own when it takes
it, so a router that stops leaves it interrupted, and the next router resumes it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

from makenomistakes.config import Settings
from makenomistakes.readings_filter import here_of
from makenomistakes.run_parts import InputRecord
from makenomistakes.runs import RunKind, RunRecord, RunState, RunStore, StepRecord, StepState
from makenomistakes.studio import train_data
from makenomistakes.studio.row_library import Training, sha256_of
from makenomistakes.studio.trainer import (
    Job,
    StudioTrainer,
    TrainerError,
    Variant,
    read_progress,
    read_trained,
    variant_of,
    write_job,
)
from makenomistakes.studio.voice_library import (
    Consent,
    VoiceEngine,
    VoiceError,
    VoiceLibrary,
    VoiceSource,
)

RATES = (32_000, 40_000, 48_000)
DEFAULT_RATE = 40_000  # RVC's usual rate: audio.cpp's converter sings at the voice's own
DEFAULT_EPOCHS = 200  # the community's 200 to 400
DEFAULT_BATCH = 8
SAVE_EVERY = {Variant.CUDA: 5, Variant.CPU: 1}  # epochs between checkpoints: what a cut loses
# One epoch over one minute of recordings on four vCPUs took 50 s
CPU_SECONDS_PER_EPOCH_MINUTE = 50.0
STAGE = "train"
VOICE = "voice"  # a training's kind when its settings name none
DATASET = "dataset"
LOG = "train.log"
KEYS = ("engine", "source", "sample_rate", "epochs", "batch_size", "variant", "consent_sha256")
# a style's: what `mnm studio train show` lists beside the voice's
STYLE_KEYS = ("build", "rank", "preset", "max_seconds", "trigger", "caption")


class TrainError(ValueError):
    """A training mnm will not queue or finish, worded for the user."""


@dataclass(frozen=True)
class Queued:
    run: RunRecord
    survey: train_data.Survey
    estimate: str  # how long it may take, in words


def estimate_words(variant: Variant, epochs: int, minutes: float, cores: int) -> str:
    if variant is Variant.CUDA:
        return (
            f"{epochs} epochs on the card: the community's figure is one to four hours for "
            "200 to 400 epochs on 10 to 30 minutes (not measured on this box yet)"
        )
    seconds = CPU_SECONDS_PER_EPOCH_MINUTE * epochs * max(minutes, 0.1) * 4 / max(cores, 1)
    hours = seconds / 3600
    when = f"about {hours:.0f} hours" if hours >= 2 else f"about {seconds / 60:.0f} minutes"
    return (
        f"{epochs} epochs on the CPU: {when} on {cores} cores (estimated from a run on four "
        "CPU cores: 50 s an epoch per minute of recordings)"
    )


def cpu_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - not Linux
        return os.cpu_count() or 1


def queue_voice(  # noqa: PLR0913 - the training's parts, each its own flag
    settings: Settings,
    trainer: StudioTrainer,
    name: str,
    folder: Path,
    source: VoiceSource,
    *,
    consent: Path | None = None,
    sample_rate: int = DEFAULT_RATE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    licence: str = "",
) -> Queued:
    """Check, copy and queue; refused before anything is written."""
    library = VoiceLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    store = RunStore.for_settings(settings)
    _refuse(library, store, name, sample_rate, epochs, batch_size)
    ledger = trainer.installed()
    variant = variant_of(ledger.image) or Variant.CPU  # the image's own build, from its tag
    found = train_data.survey(folder, sample_rate)
    if consent is not None and not consent.is_file():
        raise TrainError(f"{consent} is not a file (the consent document, kept by you).")
    cores = cpu_cores()
    job = Job(
        name=name,
        sample_rate=sample_rate,
        epochs=epochs,
        batch_size=batch_size,
        save_every=SAVE_EVERY[variant],
        cpu_cores=cores,
        gpu="0" if variant is Variant.CUDA else "-",
    )
    run = store.new(
        RunKind.TRAINING,
        name,
        f"An RVC voice from {folder.name}",
        {
            "engine": VoiceEngine.RVC.value,
            "source": source.value,
            "sample_rate": str(sample_rate),
            "epochs": str(epochs),
            "batch_size": str(batch_size),
            "variant": variant.value,
            "image": ledger.image,
            **(
                {"consent_sha256": sha256_of(consent), "consent_date": date.today().isoformat()}
                if consent
                else {}
            ),
            **({"licence": licence} if licence else {}),
        },
    )

    def copy(here: Path) -> list[InputRecord]:
        recorded = train_data.copy_in(found, here / DATASET)
        write_job(here, job)
        return recorded

    run.voices = [name]
    enqueue(store, run, "recordings", copy, list(found.warnings))
    minutes = found.seconds / 60
    return Queued(run, found, estimate_words(variant, epochs, minutes, cores))


def _refuse(
    library: VoiceLibrary,
    store: RunStore,
    name: str,
    rate: int,
    epochs: int,
    batch: int,
) -> None:
    library.check_name(name)
    if library.has(name):  # by its name: a library that cannot be listed still answers
        raise TrainError(f"A voice named '{name}' is in the library: `mnm voices rm {name}` first.")
    if any(r.target == name and kind_of(r) == VOICE for r in queued(store)):
        raise TrainError(f"A training of '{name}' is queued already (`mnm studio train ls`).")
    if rate not in RATES:
        raise TrainError(f"{rate} Hz: a voice trains at 32000, 40000 or 48000.")
    if not 1 <= epochs <= 10_000 or not 1 <= batch <= 64:
        raise TrainError("Epochs are 1 to 10000, the batch 1 to 64.")


def enqueue(
    store: RunStore,
    run: RunRecord,
    what: str,
    copy: Callable[[Path], list[InputRecord]],
    notes: list[str],
) -> None:
    """`copy` fills the new run's folder (the inputs, the job) and gives the inputs' records;
    then the run is queued, its `train` step saved last: from there the router may take it.
    A failure, or the queueing process cut off meanwhile (Ctrl-C), fails the run, so no
    training half made waits for a router that never takes it; one whose process was killed
    outright is `half_made`."""
    here = store.folder(run)
    try:
        run.inputs_recorded = copy(here)
    except OSError as exc:
        store.fail(run, f"The {what} could not be copied in: {exc}")
        raise TrainError(f"The {what} could not be copied into {here}: {exc}") from exc
    except BaseException:
        store.fail(run, f"Cut off while its {what} were copied in: queue it again.")
        raise
    run.pid = None  # queued: the router writes its own when it takes it
    run.pid_started = None
    run.role = "training"
    run.notes = notes
    run.steps = [StepRecord(stage=STAGE)]  # last: from here the router may take it
    store.save(run)


def held_elsewhere(run: RunRecord) -> bool:
    """Another live process trains it (a second router, `mnm router --port …` beside the
    service): never taken over, since its start would remove that router's container."""
    return run.pid is not None and run.pid != os.getpid() and not run.interrupted


def half_made(run: RunRecord) -> bool:
    """A training whose queueing process died before its step was saved (killed while its
    inputs were copied): never taken, and cleanable (`RunRecord.kept_for_owner`)."""
    return run.kind is RunKind.TRAINING and run.interrupted and run.step(STAGE) is None


def kind_of(run: RunRecord) -> str:
    """What a training makes: `voice` (runs made before styles carry no kind) or `style`."""
    return run.settings.get("kind", VOICE)


def waiting(run: RunRecord) -> bool:
    """A training not yet ended and ready to run: its recordings copied and its job written
    (the `train` step is saved after both)."""
    return (
        run.kind is RunKind.TRAINING
        and run.state is RunState.RUNNING
        and run.step(STAGE) is not None
    )


def queued(store: RunStore) -> list[RunRecord]:
    """Trainings not yet ended (queued, training, cut, or left by a router that stopped),
    oldest first: the trainer's queue."""
    return sorted((r for r in store.all() if waiting(r)), key=lambda r: r.created_at)


def trainings(store: RunStore) -> list[RunRecord]:
    return sorted(
        (r for r in store.all() if r.kind is RunKind.TRAINING),
        key=lambda r: r.created_at,
        reverse=True,
    )


# What a cancelled training's run record says (an older mnm's words start the same way).
CANCELLED = "Cancelled on request."


def state_words(store: RunStore, run: RunRecord, training: str | None) -> str:
    """Where a training is, worded: `training` is the router's busy line for the one it runs
    ("training alto (epoch 12 of 200)"), None when it runs none or does not answer."""
    if run.state is RunState.DONE:
        return "done"
    if run.state is RunState.FAILED:
        return "cancelled" if run.error.startswith("Cancelled") else "failed"
    if half_made(run):
        return "cut off while its inputs were copied in: queue it again (`mnm clean runs`)"
    if training is not None and training.startswith(f"training {run.target} "):
        return training.removeprefix(f"training {run.target} ").strip("()")
    progress = read_progress(store.folder(run))
    paused = ""
    if progress is not None and progress.epoch:
        paused = f", paused in epoch {progress.epoch} of {progress.epochs}"
    if run.interrupted:
        return f"waits for the router{paused}"
    return f"queued{paused}"


def finish(settings: Settings, store: RunStore, run: RunRecord, seconds: float | None) -> RunRecord:
    """The trained voice into the library (checked and its `f0` made audio.cpp's), the run
    done, the work folder's copies of the recordings and checkpoints removed."""
    here = store.folder(run)
    trained = read_trained(here)
    out = here / "out"
    library = VoiceLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    index = out / "model.index"
    consent = run.settings.get("consent_sha256")
    given = run.settings.get("consent_date")
    try:
        row = library.add(
            run.target,
            VoiceEngine.RVC,
            VoiceSource(run.settings["source"]),
            model=out / "model.pth",
            index=index if index.is_file() else None,
            licence=run.settings.get("licence", ""),
            note=f"Trained by mnm's trainer (run {run.id}).",
            training=training_of(settings, run, trained),
            consent=(
                Consent(sha256=consent, date=date.fromisoformat(given) if given else date.today())
                if consent
                else None
            ),
        )
    except VoiceError as exc:
        raise TrainerError(
            f"The voice trained but could not join the library: {exc} Its files stay in {out}."
        ) from exc
    _drop_work(here)
    log = here / LOG
    made = [log] if log.is_file() else []
    step = run.step(STAGE)
    if step is not None:
        step.state = StepState.DONE
        step.seconds = trained.seconds
    if trained.index != "ivf":
        run.notes.append(
            "No retrieval index: too few recordings for one audio.cpp reads (an IVF index); "
            "the voice sings without retrieval blending."
        )
    words = f"{row.name}: RVC v2, {trained.sample_rate // 1000} kHz, {trained.epochs} epochs"
    return store.finish(
        run,
        made,
        words,
        seconds,
        {"epochs": float(trained.epochs), "steps": float(trained.steps)},
    )


class Made(Protocol):
    """What a driver's result says of any training: its epochs and steps."""

    @property
    def epochs(self) -> int: ...

    @property
    def steps(self) -> int | None: ...


def training_of(
    settings: Settings, run: RunRecord, trained: Made, keys: tuple[str, ...] = KEYS
) -> Training:
    """What the library's row keeps of a training: the run's settings named by `keys`, the
    steps, when, the inputs by sha256 and the box (a voice's consent is its row's own)."""
    kept = {k: v for k, v in run.settings.items() if k in keys and k != "consent_sha256"}
    return Training(
        settings={
            **kept,
            "steps": str(trained.steps) if trained.steps is not None else "",
            "trained_at": datetime.now(UTC).isoformat(),
        },
        epochs=trained.epochs,
        dataset=[record.sha256 for record in run.inputs_recorded],
        machine=here_of(settings.expanded_state_dir()).id,
    )


def _drop_work(here: Path) -> None:
    """What only the training needed: the copied recordings, Applio's work and checkpoints,
    the result's files (the library holds its own copy)."""
    for name in (DATASET, "applio", "out"):
        shutil.rmtree(here / name, ignore_errors=True)
    for name in ("job.json", "progress.json"):
        (here / name).unlink(missing_ok=True)
