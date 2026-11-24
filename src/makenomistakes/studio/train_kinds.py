"""What the router's trainer does differently for each kind of training: a voice
(Applio) or a style (ACE-Step 1.5). The runner (`train_runner`) keeps the
queue, the waits, the cuts and the container's life; a kind says which image runs the job,
whether the job may start now, how the job is fitted to the build installed now, what else
the container sees, what watches one start of it, and how a finished training joins its
library.

**A start's watch** (`watch`) is one object per container start, held by the runner's
`_waited` and looked at every few seconds: a voice's says nothing; a style's on a card is the
probe (`Probe`). The runner joins its watcher thread before it reads the verdict, so a cut
that lands while the probe judges never loses the verdict.

**The probe**: as a style's container starts on a card, the card's memory in use is read
(what other programs hold, before the training holds any); once the training stage begins
(after preprocessing, which a resumed training skips) the card is sampled every second until
that start's first checkpoint is saved, and the training's own peak (the highest less the
outside use) is judged by `style_plan.judged`, kept for the next plan, and written on the run
(`PROBED`): a training stopped for good, paused until the room is free (its next start waits
for it, `waits`), or going on; once judged it is not probed again.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from makenomistakes.gpu_peak import PeakSampler
from makenomistakes.instances import InstanceService
from makenomistakes.plan_budget import planning_budget
from makenomistakes.readings_filter import here_of
from makenomistakes.recipes.model import HF_CACHE_SUBDIR
from makenomistakes.runs import RunError, RunRecord, RunStore
from makenomistakes.studio import style_jobs, style_plan, train_jobs
from makenomistakes.studio.built_image import BuiltImage
from makenomistakes.studio.style_data import DEFAULT_MAX_SECONDS
from makenomistakes.studio.style_trainer import (
    BUILD,
    RESULT,
    STYLE,
    TRAIN_STAGE,
    StyleJob,
    StyleTrainer,
    StyleTrainerError,
)
from makenomistakes.studio.trainer import (
    Progress,
    Stop,
    StudioTrainer,
    Variant,
    read_job,
    read_progress,
    variant_of,
    write_job,
)

GIB = 2**30
PROBE_EVERY = 1.0  # seconds between samples of the card while a style's first epoch runs
PROBED = "probe_peak_gib"  # on the run once judged: the training's own peak, never again
OUTSIDE = "probe_outside_gib"  # beside it: what other programs held as that start began
EPOCH_SECONDS = "epoch_seconds"


class Watch(Protocol):
    """One start of a training's container, watched while it runs."""

    def check(self) -> Stop | None:
        """Every few seconds: why the training must stop now, else None."""
        ...

    def close(self) -> None: ...


class TrainKind(Protocol):
    result: str  # the driver's result file in the job folder: a finished training has it

    def image(self) -> BuiltImage: ...

    def waits(self, run: RunRecord, image: str) -> str | None:
        """Why this training cannot start on `image` now (the card's room it measured), else
        None."""
        ...

    def prepare(self, store: RunStore, run: RunRecord, image: str) -> list[str]:
        """The job fitted to the image installed now, recorded on the run; the card's docker
        arguments for a card build, else none."""
        ...

    def mounts(self) -> list[str]:
        """What the container sees beyond its job folder (`docker run -v`)."""
        ...

    def watch(self, store: RunStore, run: RunRecord) -> Watch:
        """The container has just started: what watches this start of it."""
        ...

    def finish(self, store: RunStore, run: RunRecord, seconds: float | None) -> RunRecord: ...


class Unwatched:
    """A start nothing but the runner's own watcher looks at."""

    def check(self) -> Stop | None:
        return None

    def close(self) -> None:
        return None


class VoiceKind:
    """An RVC voice."""

    result = "out/result.json"

    def __init__(self, service: InstanceService, image: Callable[[], StudioTrainer],
                 gpu: Callable[[], list[str]]) -> None:  # fmt: skip
        self._service, self._image, self._gpu = service, image, gpu

    def image(self) -> BuiltImage:
        return self._image()

    def waits(self, run: RunRecord, image: str) -> str | None:
        del run, image
        return None

    def prepare(self, store: RunStore, run: RunRecord, image: str) -> list[str]:
        """The card or not, by the image installed now (a reinstall or a rollback may have
        changed its build), and the image recorded on the run."""
        variant = variant_of(image) or Variant.CPU
        job = read_job(store.folder(run))
        gpu = "0" if variant is Variant.CUDA else "-"
        if job.gpu != gpu:
            write_job(store.folder(run), job.model_copy(update={"gpu": gpu}))
        run.settings["image"] = image
        run.settings["variant"] = variant.value
        store.save(run)
        return self._gpu() if variant is Variant.CUDA else []

    def mounts(self) -> list[str]:
        return []

    def watch(self, store: RunStore, run: RunRecord) -> Watch:
        del store, run
        return Unwatched()

    def finish(self, store: RunStore, run: RunRecord, seconds: float | None) -> RunRecord:
        return train_jobs.finish(self._service.settings, store, run, seconds)


class StyleKind:
    """An ACE-Step style: the models folder's Hugging Face cache read-only, and the
    probe on a card."""

    result = RESULT

    def __init__(self, service: InstanceService, image: Callable[[], StyleTrainer],
                 gpu: Callable[[], list[str]]) -> None:  # fmt: skip
        self._service, self._image, self._gpu = service, image, gpu

    def image(self) -> BuiltImage:
        return self._image()

    def waits(self, run: RunRecord, image: str) -> str | None:
        """A training whose own peak a probe measured starts again only once the card has
        that room (what mnm's running models hold counted free: the start parks them)."""
        own = run.settings.get(PROBED)
        if own is None or variant_of(image) is not Variant.CUDA:
            return None
        budget = planning_budget(self._service)
        if budget is None:
            return None  # the card cannot be read: the start judges
        return style_plan.room_words(float(own), budget.vram_free_bytes / GIB)

    def prepare(self, store: RunStore, run: RunRecord, image: str) -> list[str]:
        """The job for the build installed now (`style_plan.fit_job`), recorded on the run."""
        on_card = (variant_of(image) or Variant.CPU) is Variant.CUDA
        folder = store.folder(run)
        try:
            job = StyleJob.model_validate_json((folder / "job.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise StyleTrainerError(f"The style's job.json cannot be read ({exc}).") from exc
        snapshot = self._service.memory().gpu if on_card else None
        card = snapshot.vram_total_bytes / GIB if snapshot is not None else None
        resumed = _saved(store, run) > 0
        fitted = style_plan.fit_job(job, on_card=on_card, card_gib=card, resumed=resumed)
        if fitted.job != job:
            write_job(folder, fitted.job)
        if fitted.preset is not None:
            run.settings.update(rank=str(fitted.preset.rank), preset=fitted.preset.name)
        run.settings["image"] = image
        run.settings["variant"] = (Variant.CUDA if on_card else Variant.CPU).value
        store.save(run)
        return self._gpu() if on_card else []

    def mounts(self) -> list[str]:
        """The Hugging Face cache, where the checkpoint is: its snapshot links point into the
        cache's blob folders, so the cache is mounted whole, read-only, never the rest of the
        models folder."""
        hf = self._service.settings.expanded_models_dir() / HF_CACHE_SUBDIR
        return [f"{hf}:/models/{HF_CACHE_SUBDIR}:ro"]

    def watch(self, store: RunStore, run: RunRecord) -> Watch:
        if run.settings.get("variant") != Variant.CUDA.value or PROBED in run.settings:
            return Unwatched()
        return Probe(self._service, store, run)

    def finish(self, store: RunStore, run: RunRecord, seconds: float | None) -> RunRecord:
        return style_jobs.finish_style(self._service.settings, store, run, seconds)


class Probe:
    """One start's probe (the module's docstring): the outside use read now, the card sampled
    from the training stage on, judged once that start's first checkpoint is saved."""

    def __init__(self, service: InstanceService, store: RunStore, run: RunRecord) -> None:
        self._service, self._store, self._run = service, store, run
        self._outside = service.used_vram()  # before the training holds any of the card
        self._saved_at_start = _saved(store, run)
        self._sampler: PeakSampler | None = None
        self._judged = False

    def check(self) -> Stop | None:
        if self._judged:
            return None
        progress = read_progress(self._store.folder(self._run))
        if progress is None or progress.stage != TRAIN_STAGE:
            return None
        if self._sampler is None:
            self._sampler = PeakSampler(self._service.used_vram, every=PROBE_EVERY).__enter__()
            return None
        if (progress.saved or 0) <= self._saved_at_start:
            return None
        self._judged = True  # one verdict per start
        sampler = self._sampler
        self.close()  # its last sample taken
        return self._judge(sampler.peak, progress)

    def close(self) -> None:
        if self._sampler is not None:
            self._sampler.__exit__(None, None, None)
            self._sampler = None

    def _judge(self, peak: int | None, progress: Progress) -> Stop | None:
        snapshot = self._service.memory().gpu
        if peak is None or self._outside is None or snapshot is None:
            return None  # the card could not be read: nothing is judged, nothing stops
        outside = self._outside / GIB
        own = max(0.0, peak / GIB - outside)
        card = snapshot.vram_total_bytes / GIB
        settings = self._run.settings
        max_seconds = float(settings.get("max_seconds", DEFAULT_MAX_SECONDS))
        state = self._service.settings.expanded_state_dir()
        machine = here_of(state).id
        reading = style_plan.ProbeReading(
            peak_gib=round(own, 2),
            card_gib=round(card, 2),
            outside_gib=round(outside, 2),
            epoch_seconds=progress.epoch_seconds,
            songs=len(self._run.inputs_recorded),
            machine=machine,
        )
        key = style_plan.probe_key(
            machine, settings.get(BUILD, "base"), settings.get("preset", ""), max_seconds
        )
        style_plan.keep_probe(state, key, reading)
        try:
            fresh = self._store.get(self._run.id)
        except RunError:
            return None
        fresh.settings[PROBED] = f"{own:.2f}"
        fresh.settings[OUTSIDE] = f"{outside:.2f}"
        if progress.epoch_seconds is not None:
            fresh.settings[EPOCH_SECONDS] = f"{progress.epoch_seconds:.0f}"
        self._store.save(fresh)
        return style_plan.judged(own, outside, card, max_seconds)


def _saved(store: RunStore, run: RunRecord) -> int:
    """The epoch the training's latest checkpoint was saved after (0: none yet)."""
    progress = read_progress(store.folder(run))
    return progress.saved or 0 if progress is not None else 0


def kinds_for(
    service: InstanceService,
    voice_image: Callable[[], StudioTrainer],
    style_image: Callable[[], StyleTrainer],
    gpu: Callable[[], list[str]],
) -> dict[str, TrainKind]:
    return {
        train_jobs.VOICE: VoiceKind(service, voice_image, gpu),
        STYLE: StyleKind(service, style_image, gpu),
    }
