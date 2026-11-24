"""System → Trainer: the trainer's image (install for this box or the CPU,
rollback, uninstall, each a job the page polls), the trainings with where each is, a form to
train a voice from a folder of recordings on this machine, and Cancel; what `mnm studio
trainer` and `mnm studio train` do."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError
from makenomistakes.jobs import BUILD_LINES, JobRegistry
from makenomistakes.router_client import router_train, router_training
from makenomistakes.runs import RunError, RunKind, RunStore
from makenomistakes.studio import train_jobs
from makenomistakes.studio.built_image import ImageLedger
from makenomistakes.studio.style_trainer import StyleTrainer
from makenomistakes.studio.train_data import TrainDataError
from makenomistakes.studio.trainer import (
    CARRIES,
    IMAGE_GIB,
    VARIANT_WORDS,
    StudioTrainer,
    TrainerError,
    Variant,
    image_tag,
    install_variant,
    variant_of,
)
from makenomistakes.studio.voice_library import SOURCE_WORDS, VoiceError, VoiceSource
from makenomistakes.web.common import Render
from makenomistakes.web.image_routes import image_job, image_state, register_image_routes
from makenomistakes.web.style_routes import register_style_routes, style_page

TRAINER_JOB = "studio-trainer"
NO_ROUTER = "no router answers on this machine: the training waits for it (`mnm service status`)"


def register_train_routes(
    app: FastAPI,
    render: Render,
    trainer: Callable[[Variant | None], StudioTrainer],
    jobs: JobRegistry,
    settings: Settings,
    style_trainer: Callable[[Variant | None], StyleTrainer],
) -> None:
    job = image_job(jobs, TRAINER_JOB, "The trainer's")
    store = RunStore.for_settings(settings)
    register_style_routes(app, style_trainer, jobs, settings)

    @app.get("/system/trainer", response_class=HTMLResponse)
    def page(request: Request) -> HTMLResponse:
        state = image_state(trainer(None))
        ledger = state["ledger"]
        variant = variant_of(ledger.image) if isinstance(ledger, ImageLedger) else None
        training = router_training(settings)
        runs = [
            (run, train_jobs.state_words(store, run, training))
            for run in train_jobs.trainings(store)
        ]
        return render(
            request,
            "trainer.html",
            tab="system",
            **state,
            current=image_tag(variant) if variant else None,
            variant_words=VARIANT_WORDS[variant] if variant else None,
            image_gib=IMAGE_GIB,
            carries=CARRIES,
            runs=runs,
            sources=[(s.value, SOURCE_WORDS[s]) for s in VoiceSource],
            defaults={"rate": train_jobs.DEFAULT_RATE, "epochs": train_jobs.DEFAULT_EPOCHS},
            job=jobs.latest_for(TRAINER_JOB),
            kind_of=train_jobs.kind_of,
            **style_page(settings, style_trainer, jobs),
        )

    @app.post("/system/trainer/install")
    def install(cpu: Annotated[str, Form()] = "") -> RedirectResponse:
        try:
            variant = install_variant(cpu == "on")
        except TrainerError as exc:
            raise InstanceError(str(exc)) from exc
        image = trainer(variant)
        job(
            f"install the trainer for {VARIANT_WORDS[variant]}",
            lambda report: f"built {image.install(report).image}",
            BUILD_LINES,
        )
        return RedirectResponse("/system/trainer", status_code=303)

    register_image_routes(
        app, "/system/trainer", lambda: trainer(None), job, "the trained voices stay"
    )

    @app.post("/system/trainer/train")
    def train(
        name: Annotated[str, Form()],
        folder: Annotated[str, Form()],
        source: Annotated[str, Form()],
        epochs: Annotated[int, Form()] = train_jobs.DEFAULT_EPOCHS,
        rate: Annotated[int, Form()] = train_jobs.DEFAULT_RATE,
        consent: Annotated[str, Form()] = "",
    ) -> RedirectResponse:
        """A folder and a consent document on this machine (the page runs here), read and
        copied as a job (thousands of files may take a while), queued as `mnm studio train
        voice` queues it; the job's lines say what the recordings were and what the router
        said."""
        try:
            voice = VoiceSource(source)
        except ValueError as exc:
            raise InstanceError(f"'{source}' is not whose voice it is: {exc}") from exc

        def work(_report: Callable[[str], None]) -> str:
            try:
                queued = train_jobs.queue_voice(
                    settings,
                    trainer(None),
                    name.strip(),
                    Path(folder.strip()).expanduser(),
                    voice,
                    consent=Path(consent.strip()).expanduser() if consent.strip() else None,
                    sample_rate=rate,
                    epochs=epochs,
                )
            except (ValueError, TrainDataError, VoiceError, RunError) as exc:
                raise TrainerError(str(exc)) from exc
            asked = router_train(settings, queued.run.id, b'{"now": false}')
            said = asked[1] if asked is not None else NO_ROUTER
            return f"{queued.survey.words}: {said}"

        job(f"queue the training of {name.strip()}", work, 0)
        return RedirectResponse("/system/trainer", status_code=303)

    @app.post("/system/trainer/{run_id}/cancel")
    def cancel(run_id: str) -> RedirectResponse:
        try:
            run = store.get(run_id)
        except RunError as exc:
            raise InstanceError(str(exc)) from exc
        if run.kind is not RunKind.TRAINING:
            raise InstanceError(f"{run_id} is not a training.")
        answer = router_train(settings, run.id, b'{"action": "cancel"}')
        if answer is None:
            if run.going and run.pid is not None:
                raise InstanceError(
                    f"The training {run.id} runs in another process: ask the router."
                )
            if run.going or run.interrupted:
                store.fail(run, train_jobs.CANCELLED)
        elif answer[0] >= 400:
            raise InstanceError(answer[1])
        return RedirectResponse("/system/trainer", status_code=303)
