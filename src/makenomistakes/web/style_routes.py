"""System → Trainer's styles: the style trainer's image (install for this box or the
CPU, rollback, uninstall, each a job), a form that teaches ACE-Step a style from a folder of
songs on this machine (a job: the checkpoint's download takes a while the first time), and the
style library with Remove; what `mnm studio trainer --for styles`, `mnm studio train style`
and `mnm studio styles` do."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Form
from fastapi.responses import RedirectResponse

from makenomistakes.config import Settings
from makenomistakes.hardware.probe import probe_memory
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import BUILD_LINES, JobRegistry
from makenomistakes.router_client import router_train
from makenomistakes.runs import RunError
from makenomistakes.studio import style_data, style_jobs, style_use
from makenomistakes.studio.built_image import ImageLedger
from makenomistakes.studio.style_library import (
    SOURCE_WORDS,
    StyleError,
    StyleLibrary,
    StyleSource,
)
from makenomistakes.studio.style_trainer import (
    CARRIES,
    IMAGE_GIB,
    StyleTrainer,
    StyleTrainerError,
    style_variant,
)
from makenomistakes.studio.train_data import TrainDataError
from makenomistakes.studio.trainer import VARIANT_WORDS, Variant, variant_of
from makenomistakes.web.image_routes import image_job, image_state, register_image_routes

STYLE_JOB = "studio-style-trainer"
NO_ROUTER = "no router answers on this machine: the training waits for it (`mnm service status`)"
GIB = 2**30


def style_page(
    settings: Settings,
    trainer: Callable[[Variant | None], StyleTrainer],
    jobs: JobRegistry,
) -> dict[str, object]:
    """The trainer page's style part: the image, the library, the form's choices."""
    state = image_state(trainer(None))
    ledger = state["ledger"]
    variant = variant_of(ledger.image) if isinstance(ledger, ImageLedger) else None
    library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    return {
        "style_ledger": state["ledger"],
        "style_missing": state["missing"],
        "style_problem": state["problem"],
        "style_current": trainer(variant).image_tag() if variant else None,
        "style_variant_words": VARIANT_WORDS[variant] if variant else None,
        "style_gib": IMAGE_GIB,
        "style_carries": CARRIES,
        "styles": library.all(),
        "style_problems": library.problems(),
        "style_sources": [(s.value, SOURCE_WORDS[s]) for s in StyleSource],
        "style_source_words": SOURCE_WORDS,
        "style_defaults": {
            "epochs": style_jobs.DEFAULT_EPOCHS,
            "max_seconds": int(style_data.DEFAULT_MAX_SECONDS),
        },
        "style_job": jobs.latest_for(STYLE_JOB),
    }


def register_style_routes(
    app: FastAPI,
    trainer: Callable[[Variant | None], StyleTrainer],
    jobs: JobRegistry,
    settings: Settings,
) -> None:
    job = image_job(jobs, STYLE_JOB, "The style trainer's")

    @app.post("/system/trainer/styles/install")
    def install(cpu: Annotated[str, Form()] = "") -> RedirectResponse:
        try:
            variant = style_variant(cpu == "on")
        except StyleTrainerError as exc:
            raise InstanceError(str(exc)) from exc
        image = trainer(variant)
        job(
            f"install the style trainer for {VARIANT_WORDS[variant]}",
            lambda report: f"built {image.install(report).image}",
            BUILD_LINES,
        )
        return RedirectResponse("/system/trainer", status_code=303)

    register_image_routes(
        app,
        "/system/trainer/styles",
        lambda: trainer(None),
        job,
        "the trained styles stay",
        back="/system/trainer",
    )

    @app.post("/system/trainer/style")
    def train(  # noqa: PLR0913, PLR0917 - the form's fields
        name: Annotated[str, Form()],
        folder: Annotated[str, Form()],
        caption: Annotated[str, Form()],
        source: Annotated[str, Form()],
        trigger: Annotated[str, Form()] = "",
        build: Annotated[str, Form()] = "base",
        epochs: Annotated[int, Form()] = style_jobs.DEFAULT_EPOCHS,
        max_seconds: Annotated[float, Form()] = style_data.DEFAULT_MAX_SECONDS,
        licence: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        """A folder of songs on this machine (the page runs here), queued as `mnm studio
        train style` queues it, as a job: the checkpoint's download and the songs' copy are
        said as lines, then where the router has the training."""
        try:
            songs_from = StyleSource(source)
        except ValueError as exc:
            raise InstanceError(f"'{source}' is not whose songs they are: {exc}") from exc
        if build not in ("base", "turbo"):
            raise InstanceError(f"'{build}' is no ACE-Step build: base or turbo.")
        plays: Literal["base", "turbo"] = "base" if build == "base" else "turbo"

        def work(report: Callable[[str], None]) -> str:
            snapshot = probe_memory()
            gpu, memory = snapshot.gpu, snapshot.memory
            try:
                queued = style_jobs.queue_style(
                    settings,
                    trainer(None),
                    name.strip(),
                    Path(folder.strip()).expanduser(),
                    songs_from,
                    caption=caption.strip(),
                    card_gib=gpu.vram_total_bytes / GIB if gpu is not None else None,
                    ram_gib=memory.total_bytes / GIB if memory is not None else None,
                    trigger=trigger.strip() or None,
                    variant=plays,
                    epochs=epochs,
                    max_seconds=max_seconds,
                    licence=licence.strip(),
                    say=report,
                )
            except (ValueError, TrainDataError, StyleError, RunError) as exc:
                raise StyleTrainerError(str(exc)) from exc
            asked = router_train(settings, queued.run.id, b'{"now": false}')
            said = asked[1] if asked is not None else NO_ROUTER
            return f"{queued.survey.words}; {queued.estimate}: {said}"

        job(f"queue the style {name.strip()}", work, 0)
        return RedirectResponse("/system/trainer", status_code=303)

    @app.post("/system/trainer/styles/{name}/rm")
    def remove(name: str) -> RedirectResponse:
        library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
        try:
            style_use.refuse_removal(settings, name)
            library.remove(name)
        except StyleError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse("/system/trainer", status_code=303)
