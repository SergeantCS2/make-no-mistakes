"""System → Voices: the voice library as `mnm voices` shows it, a voice added
from an uploaded reference recording or RVC checkpoint, a consent document's hash recorded,
a voice removed. A song's own voice is changed on its page (`/songs/<id>/voice`)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse

from makenomistakes.instance_types import InstanceError
from makenomistakes.studio.voice_library import (
    ENGINE_WORDS,
    NEEDS_CONSENT,
    SOURCE_WORDS,
    VoiceEngine,
    VoiceError,
    VoiceLibrary,
    VoiceSource,
)
from makenomistakes.web.common import Render

# a reference is 30 s at most (a 48 kHz 24-bit stereo WAV is about 8.6 MB); an RVC checkpoint
# is about 55 MB, its index up to a few hundred: a larger one goes through `mnm voices add`
MAX_UPLOAD = 256 * 2**20
CONSENT_UPLOAD = 32 * 2**20


def register_voice_routes(app: FastAPI, render: Render, library: VoiceLibrary) -> None:
    @app.get("/system/voices", response_class=HTMLResponse)
    def voices(request: Request) -> HTMLResponse:
        rows = [(row, library.missing_files(row)) for row in library.all()]
        return render(
            request,
            "voices.html",
            tab="system",
            rows=rows,
            problems=library.problems(),
            engines=[(e.value, ENGINE_WORDS[e]) for e in VoiceEngine],
            sources=[(s.value, SOURCE_WORDS[s]) for s in VoiceSource],
            needs_consent=NEEDS_CONSENT,
            engine_words=ENGINE_WORDS,
            source_words=SOURCE_WORDS,
        )

    @app.post("/system/voices")
    def add(  # noqa: PLR0913, PLR0917 - the form's fields
        name: Annotated[str, Form(max_length=64)],
        engine: Annotated[VoiceEngine, Form()],
        source: Annotated[str, Form(max_length=20)] = "",
        licence: Annotated[str, Form(max_length=400)] = "",
        note: Annotated[str, Form(max_length=400)] = "",
        reference: Annotated[UploadFile | None, File()] = None,
        model: Annotated[UploadFile | None, File()] = None,
        index: Annotated[UploadFile | None, File()] = None,
    ) -> RedirectResponse:
        if source not in {s.value for s in VoiceSource}:
            raise InstanceError(
                "Say whose voice it is (the export's disclosure names it, and someone else's "
                "asks for a consent record)."
            )
        with tempfile.TemporaryDirectory(prefix="mnm-voice-") as scratch:
            given = {
                role: _saved(upload, Path(scratch), role, MAX_UPLOAD)
                for role, upload in (("reference", reference), ("model", model), ("index", index))
            }
            try:
                library.add(
                    name.strip(), engine, VoiceSource(source), reference=given["reference"],
                    model=given["model"], index=given["index"], licence=licence.strip(),
                    note=note.strip(),
                )  # fmt: skip
            except VoiceError as exc:
                raise InstanceError(str(exc)) from exc
        return RedirectResponse("/system/voices", status_code=303)

    @app.post("/system/voices/{name}/consent")
    def consent(
        name: str,
        document: Annotated[UploadFile, File()],
        note: Annotated[str, Form(max_length=400)] = "",
    ) -> RedirectResponse:
        with tempfile.TemporaryDirectory(prefix="mnm-consent-") as scratch:
            path = _saved(document, Path(scratch), "document", CONSENT_UPLOAD)
            if path is None:
                raise InstanceError("Choose the consent document: only its sha256 is kept.")
            try:
                library.consent(name, path, note.strip())
            except VoiceError as exc:
                raise InstanceError(str(exc)) from exc
        return RedirectResponse("/system/voices", status_code=303)

    @app.post("/system/voices/{name}/rm")
    def remove(name: str, keep_files: Annotated[str, Form()] = "") -> RedirectResponse:
        try:
            library.remove(name, files=keep_files != "true")
        except VoiceError as exc:
            raise InstanceError(str(exc)) from exc
        return RedirectResponse("/system/voices", status_code=303)


def voice_upload_too_large(path: str, length: str | None) -> bool:
    """A voice's upload past what the page takes: refused before Starlette spools it."""
    if path != "/system/voices" or length is None or not length.isdigit():
        return False
    return int(length) > 3 * MAX_UPLOAD + 2**20  # three files and the form around them


def _saved(upload: UploadFile | None, folder: Path, role: str, limit: int) -> Path | None:
    """An upload written to the scratch folder under its own name (the library reads the kind
    of file from its suffix and words a refusal with it), refused past `limit`; None when
    nothing was chosen."""
    if upload is None or not upload.filename:
        return None
    name = Path(upload.filename.replace("\\", "/")).name.strip()
    if not name or name.startswith("."):
        name = role
    target = folder / role / name[:120]
    target.parent.mkdir()
    written = 0
    with target.open("wb") as handle:
        while block := upload.file.read(1 << 20):
            written += len(block)
            if written > limit:
                raise InstanceError(
                    f"{upload.filename} is larger than {limit // 2**20} MiB, which this page "
                    "takes: `mnm voices add` takes it from a file on this machine."
                )
            handle.write(block)
    return target
