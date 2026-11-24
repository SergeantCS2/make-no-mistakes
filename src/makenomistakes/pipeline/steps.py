"""What a pipeline step does: ask a chat model for JSON, make a picture, a clip or a
line through the router (`mnm try`'s request with a prompt), and stitch clips and lines into
one film with the ffmpeg inside the whisper.cpp engine image (GUIDE §9.5), so nothing is
installed on the host.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from makenomistakes.consumers import own_origin
from makenomistakes.hardware.probe import CommandRunner
from makenomistakes.instance_types import InstanceError
from makenomistakes.pipeline.short_film import (
    SHOTLIST_SCHEMA,
    FilmInputs,
    ShotList,
    ShotListError,
    parse_shotlist,
    shotlist_prompt,
)
from makenomistakes.recipes import WHISPERCPP_IMAGE, Recipe
from makenomistakes.smoke import Picture, TryError, try_instance

ANSWER_TOKENS = 4096
STITCH_SECONDS = 600.0


class StepError(RuntimeError):
    """A step that did not make its item, worded for the user."""


class PipelineError(InstanceError):
    """A pipeline problem worded for the user (the CLI and the web print it alike)."""


class _Message(BaseModel):
    content: str | None = None


class _Choice(BaseModel):
    message: _Message = Field(default_factory=_Message)


class _Reply(BaseModel):
    choices: list[_Choice] = Field(default_factory=list)


Schema = tuple[str, dict[str, object]]  # a JSON schema's name and body


def chat_json(
    recipe: Recipe,
    http: httpx.Client,
    messages: list[dict[str, str]],
    schema: Schema = ("shot_list", SHOTLIST_SCHEMA),
    origin: str | None = None,
) -> str:
    """One chat answer as JSON in `schema`'s shape, from `origin` (the instance's door
    through the router, or a desk role's: `/upstream/chat`)."""
    name, shape = schema
    body = {
        "model": recipe.served_name,
        "messages": messages,
        "max_tokens": ANSWER_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        # llama-server and vLLM turn the schema into a grammar; the answer is still checked
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": name, "schema": shape},
        },
    }
    try:
        response = http.post((origin or own_origin(recipe)) + "/v1/chat/completions", json=body)
    except httpx.TransportError as exc:
        raise StepError(
            f"{recipe.name} did not answer ({exc.__class__.__name__}): is the router up "
            "(`mnm service status`)?"
        ) from exc
    if response.status_code != httpx.codes.OK:
        raise StepError(
            f"{recipe.name} answered HTTP {response.status_code}: {response.text[:300]}"
        )
    try:
        reply = _Reply.model_validate(response.json())
    except (ValueError, ValidationError) as exc:
        raise StepError(f"{recipe.name} answered something that is not a chat reply.") from exc
    content = reply.choices[0].message.content if reply.choices else None
    if not content or not content.strip():
        raise StepError(f"{recipe.name} answered with empty content.")
    return content


def ask_shotlist(recipe: Recipe, http: httpx.Client, inputs: FilmInputs) -> ShotList:
    """The chat model's shot list; one more try, told what was wrong, before giving up."""
    messages = [{"role": "user", "content": shotlist_prompt(inputs)}]
    answer = chat_json(recipe, http, messages)
    try:
        return parse_shotlist(answer, inputs.shots)
    except ShotListError as exc:
        messages += [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": f"That is not usable: {exc}. Answer with the JSON only."},
        ]
    answer = chat_json(recipe, http, messages)
    try:
        return parse_shotlist(answer, inputs.shots)
    except ShotListError as exc:
        raise StepError(f"{recipe.name}'s shot list is not usable twice over: {exc}.") from exc


def make_media(
    recipe: Recipe,
    http: httpx.Client,
    folder: Path,
    name: str,
    prompt: str,
    picture: bytes | None = None,
) -> Path:
    """One picture, clip or line through the router, saved in the run's folder as `name`
    (its suffix the one the engine's file has)."""
    work = Path(tempfile.mkdtemp(prefix=".step-", dir=folder))
    try:
        result = try_instance(
            recipe, http, work, prompt=prompt, picture=Picture(picture) if picture else None
        )
        if result.saved is None:
            raise StepError(f"{recipe.name} answered without a file: {result.answer}")
        target = folder / (Path(name).stem + result.saved.suffix)
        target.unlink(missing_ok=True)
        result.saved.replace(target)
        return target
    except TryError as exc:
        raise StepError(str(exc)) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


@dataclass(frozen=True)
class Part:
    """One shot of the film: its clip, and its line when it has one (file names in the run's
    folder)."""

    clip: str
    line: str | None


class Stitcher(Protocol):
    def stitch(self, folder: Path, parts: list[Part], film: str) -> Path: ...


class DockerStitcher:
    """ffmpeg in an image already on the box (the whisper.cpp engine's, else the ComfyUI image
    mnm built: both carry it; nothing is pulled), with the run's folder as its only mount and
    no network. Each clip gets its line (padded with silence, cut at the clip's end), then the
    shots are joined without re-encoding the video."""

    def __init__(
        self,
        runner: CommandRunner,
        image: str = WHISPERCPP_IMAGE,
        also: Callable[[], list[str]] | None = None,
    ) -> None:
        self.runner = runner
        self.image = image
        self.also = also or list  # more images with ffmpeg, asked at each stitch

    def stitch(self, folder: Path, parts: list[Part], film: str) -> Path:
        image = self._found_image()  # asked at each stitch: never kept on the object
        joined: list[str] = []
        try:
            for number, part in enumerate(parts, start=1):
                suffix = Path(part.clip).suffix.lower() or ".webm"
                out = f".part-{number:02d}{suffix}"
                audio = (
                    ["-i", part.line]
                    if part.line
                    else ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]
                )
                self._ffmpeg(
                    image,
                    folder,
                    [
                        "-i",
                        part.clip,
                        *audio,
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "libopus" if suffix in (".webm", ".mkv") else "aac",
                        "-af",
                        "apad",
                        "-shortest",
                        out,
                    ],
                )
                joined.append(out)
            (folder / ".parts.txt").write_text("".join(f"file '{p}'\n" for p in joined))
            self._ffmpeg(
                image,
                folder,
                ["-f", "concat", "-safe", "0", "-i", ".parts.txt", "-c", "copy", film],
            )
        finally:
            for name in [*joined, ".parts.txt"]:
                (folder / name).unlink(missing_ok=True)
        return folder / film

    def _found_image(self) -> str:
        """The first image on the box that carries ffmpeg."""
        wanted = [self.image, *(i for i in self.also() if i != self.image)]
        for image in wanted:
            if self.runner.run(["docker", "image", "inspect", image], timeout=30)[0] == 0:
                return image
        raise StepError(
            f"The stitch uses the ffmpeg inside {' or '.join(wanted)}, and none is on this box: "
            f"`docker pull {wanted[0]}` (a few GB; a speech-to-text model's first start pulls "
            "it too), then Resume."
        )

    def _ffmpeg(self, image: str, folder: Path, args: list[str]) -> None:
        argv = [
            "docker", "run", "--rm", "--network", "none",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{folder}:/w", "-w", "/w", "--entrypoint", "ffmpeg", image,
            "-hide_banner", "-loglevel", "error", "-y", *args,
        ]  # fmt: skip
        code, _out, err = self.runner.run(argv, timeout=STITCH_SECONDS)
        if code != 0:
            raise StepError(f"ffmpeg could not join the film: {err.strip()[-400:] or code}")
