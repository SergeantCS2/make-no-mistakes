"""The Test it panel: `mnm try` from the instance page, as a job.

Uploads (a picture, a mask, a clip) and what comes back (a PNG, a clip, a WAV) live in a
temporary folder per run, made inside the job, never in the checkout or the models folder;
each instance keeps only its newest run. An upload larger than `MAX_UPLOAD` is refused.
"""

from __future__ import annotations

import atexit
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.jobs import JobKind, JobRegistry
from makenomistakes.recipes import Recipe
from makenomistakes.smoke import Picture, TryError
from makenomistakes.try_figures import Figure
from makenomistakes.try_run import TryInputs, TryRun, http_client, run_try, used_bytes
from makenomistakes.web.instance_routes import refuse_if_busy

MAX_UPLOAD = 32 * 2**20  # a picture or a short clip; a longer one is the CLI's job
MEDIA_TYPES = {
    ".png": "image/png",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".avi": "video/x-msvideo",
    ".wav": "audio/wav",
}
_SUFFIX = re.compile(r"^\.[a-z0-9]{1,5}$")


class TryRunner(Protocol):
    def __call__(
        self,
        service: InstanceService,
        recipe: Recipe,
        out_dir: Path,
        inputs: TryInputs,
        measure: bool,
    ) -> TryRun: ...


def default_runner(
    service: InstanceService, recipe: Recipe, out_dir: Path, inputs: TryInputs, measure: bool
) -> TryRun:
    with http_client(service.settings) as http:
        return run_try(service, recipe, http, out_dir, inputs, used_bytes if measure else None)


@dataclass(frozen=True)
class TryShown:
    """One finished run, as the panel shows it."""

    answer: str
    seconds: float
    numbers: dict[Figure, float]
    vram_line: str | None
    media: Path | None
    steps: tuple[str, ...] = ()  # a tool-call test's lines (`--tools`), one per step

    @property
    def media_type(self) -> str | None:
        return MEDIA_TYPES.get(self.media.suffix) if self.media else None

    @property
    def kind(self) -> str | None:
        """image, video or audio: which element the panel plays it in."""
        found = self.media_type
        return found.split("/")[0] if found else None


class TryResults:
    """Each instance's newest finished run; a new run's folder replaces the old one's, a
    removed instance's goes with it, and every folder goes when the UI exits."""

    def __init__(self) -> None:
        self._runs: dict[str, tuple[Path, TryShown]] = {}
        self._lock = threading.Lock()
        atexit.register(self.clear)

    def add(self, name: str, folder: Path, shown: TryShown) -> None:
        with self._lock:
            old = self._runs.get(name)
            self._runs[name] = (folder, shown)
        if old is not None and old[0] != folder:
            shutil.rmtree(old[0], ignore_errors=True)

    def get(self, name: str) -> TryShown | None:
        with self._lock:
            found = self._runs.get(name)
            return found[1] if found else None

    def forget(self, name: str) -> None:
        with self._lock:
            old = self._runs.pop(name, None)
        if old is not None:
            shutil.rmtree(old[0], ignore_errors=True)

    def clear(self) -> None:
        with self._lock:
            folders = [folder for folder, _shown in self._runs.values()]
            self._runs.clear()
        for folder in folders:
            shutil.rmtree(folder, ignore_errors=True)


def upload(file: UploadFile | None, what: str) -> tuple[str, bytes] | None:
    """An upload's name and bytes, refused above `MAX_UPLOAD` (the request as a whole is
    refused before it is read when it says it is larger: `too_large`). Test it and Run both
    take their pictures, masks and clips through here."""
    if file is None or not file.filename:
        return None
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise InstanceError(
            f"The {what} is larger than {MAX_UPLOAD // 2**20} MiB: use a smaller one."
        )
    return (file.filename, data) if data else None


def too_large(path: str, length: str | None) -> bool:
    """A Test it request bigger than its three uploads may be: refused before Starlette
    spools it to disk."""
    if not path.endswith(("/try", "/run")) or length is None or not length.isdigit():
        return False
    return int(length) > 3 * MAX_UPLOAD + 2**20  # three files and the form around them


def register_try_routes(
    app: FastAPI,
    service: InstanceService,
    jobs: JobRegistry,
    results: TryResults,
    runner: TryRunner | None = None,
) -> None:
    run_one = runner or default_runner

    @app.post("/instances/{name}/try")
    def try_it(  # noqa: PLR0913, PLR0917  # a form handler: its fields are the UI
        name: str,
        speed: Annotated[bool, Form()] = False,
        tools: Annotated[bool, Form()] = False,
        measure: Annotated[bool, Form()] = False,
        say: Annotated[str, Form(max_length=2000)] = "",
        image: Annotated[UploadFile | None, File()] = None,
        mask: Annotated[UploadFile | None, File()] = None,
        audio: Annotated[UploadFile | None, File()] = None,
    ) -> RedirectResponse:
        recipe = service.get(name)
        refuse_if_busy(jobs, name, "test it")
        picture = (up := upload(image, "picture")) and up[1]
        masking = (up := upload(mask, "mask")) and up[1]
        clip = (up := upload(audio, "clip")) and up[1]
        if masking is not None and picture is None:
            raise InstanceError("A mask needs the picture it masks: add a picture too.")
        suffix = Path(audio.filename or "").suffix.lower() if audio is not None else ""
        clip_name = f"clip{suffix if _SUFFIX.match(suffix) else '.wav'}"

        def work(report: Callable[[str], None]) -> list[str]:
            report(
                "sending the tool-call test's five requests…"
                if tools
                else "sending the card's request…"
            )
            folder = Path(tempfile.mkdtemp(prefix="mnm-try-"))
            try:
                clip_path = None
                if clip is not None:
                    clip_path = folder / clip_name
                    clip_path.write_bytes(clip)
                inputs = TryInputs(
                    audio=clip_path,
                    speed=speed,
                    say=say.strip() or None,
                    picture=Picture(picture, masking) if picture else None,
                    tools=tools,
                )
                run = run_one(service, recipe, folder, inputs, measure)
            except BaseException as exc:  # nothing of a failed run stays on disk
                shutil.rmtree(folder, ignore_errors=True)
                if isinstance(exc, TryError):  # worded for the user already
                    raise InstanceError(str(exc)) from exc
                raise
            result = run.result
            steps = tuple(step.line for step in run.tools.steps) if run.tools else ()
            shown = TryShown(
                result.answer, result.seconds, result.numbers, run.vram_line, result.saved, steps
            )
            results.add(name, folder, shown)
            return [f"{result.answer} ({result.seconds:.2f} s)", *steps] + [
                line for line in (run.vram_line, run.company, run.verdict) if line
            ]

        jobs.start_unless_active(JobKind.TRY, name, work, label=f"test {name}")
        return RedirectResponse(f"/instances/{name}#test", status_code=303)

    @app.get("/instances/{name}/try/media")
    def media(name: str) -> FileResponse:
        shown = results.get(name)
        if shown is None or shown.media is None or not shown.media.exists():
            raise InstanceError("That result is gone: test it again.")
        return FileResponse(shown.media, media_type=shown.media_type)
