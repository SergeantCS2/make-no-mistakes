"""The Short film template: an idea → a shot list → a still per shot → each line
spoken → each still animated → one film.

The stages run in that order on purpose: every still, then every line, then every clip, so
the router loads each model once (the video model, the largest, last). The shot list and the
stills are review points: the user edits the shots, redraws a still, or continues.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes.hub.facts import CHAT_LIKE, Modality

NAME = "short-film"
MAX_SHOTS = 8
SHOTLIST, STILLS, LINES, CLIPS, FILM = "shotlist", "stills", "lines", "clips", "film"
STAGES = (SHOTLIST, STILLS, LINES, CLIPS, FILM)
REVIEWS = (SHOTLIST, STILLS)  # where a run waits for the user, unless it runs straight through
SHOTLIST_FILE = "shotlist.json"
FILM_FILE = "film.webm"  # clips from sd-server; a workflow's mp4 clips make film.mp4
FILM_NAMES = ("film.webm", "film.mp4", "film.mkv", "film.mov")
WORKFLOW = "comfy:"  # a video role that names a ComfyUI workflow: `comfy:wan14b`
CLIP_SUFFIXES = (".webm", ".mp4", ".mkv", ".mov")
# A workflow's clips at the shape sd-server's video instances default to: the Comfy-Org
# template's own defaults made 1280x704x121 clips on an RTX 4070, 11 and 8 minutes each where
# 832x480x33 took 48 s by hand.
WORKFLOW_SIZE = (832, 480)
WORKFLOW_FRAMES = 33
FRAME_FIELDS = ("length", "num_frames", "frames", "frame_count")  # a clip's frame count
# The types each role takes (the instance pickers offer only those).
ROLES: dict[str, frozenset[Modality]] = {
    "chat": CHAT_LIKE,
    "image": frozenset({Modality.IMAGE_GEN}),
    "video": frozenset({Modality.VIDEO}),
    "voice": frozenset({Modality.TEXT_TO_SPEECH}),
}
STAGE_LABELS = {
    SHOTLIST: "Shot list",
    STILLS: "Stills",
    LINES: "Lines",
    CLIPS: "Clips",
    FILM: "Film",
}


class FilmInputs(BaseModel):
    """What the user gives the template; kept in the run's settings as strings."""

    idea: str = Field(min_length=3, max_length=4000)
    shots: int = Field(default=3, ge=1, le=MAX_SHOTS)
    style: str = Field(default="cinematic, soft light", max_length=300)
    chat: str
    image: str
    video: str  # an instance, or `comfy:<workflow>`
    voice: str | None = None  # no voice: a silent film
    auto: bool = False  # run straight through the review points
    # a workflow's clip shape (`--clip-size`, `--clip-frames`); an instance's is its own
    clip_size: str | None = Field(default=None, pattern=r"^[1-9][0-9]{1,4}x[1-9][0-9]{1,4}$")
    clip_frames: int | None = Field(default=None, ge=1, le=1000)

    def clip_shape(self) -> tuple[int, int, int]:
        """A workflow's clips: width, height, frames (832x480, 33 unless asked)."""
        width, height = (
            (int(n) for n in self.clip_size.split("x")) if self.clip_size else WORKFLOW_SIZE
        )
        return width, height, self.clip_frames or WORKFLOW_FRAMES

    def settings(self) -> dict[str, str]:
        values = self.model_dump(exclude_none=True)
        return {
            key: str(value).lower() if isinstance(value, bool) else str(value)
            for key, value in values.items()
        }

    @classmethod
    def from_settings(cls, settings: dict[str, str]) -> FilmInputs:
        values: dict[str, object] = dict(settings)
        values["auto"] = settings.get("auto") == "true"
        return cls.model_validate(values)


class Shot(BaseModel):
    still_prompt: str = Field(min_length=1, max_length=1500)
    motion_prompt: str = Field(min_length=1, max_length=1500)
    line: str = Field(default="", max_length=600)

    @field_validator("still_prompt", "motion_prompt", "line")
    @classmethod
    def _one_line(cls, value: str) -> str:
        return " ".join(value.split())


class ShotList(BaseModel):
    title: str = Field(default="", max_length=200)
    shots: list[Shot] = Field(min_length=1, max_length=MAX_SHOTS)


SHOTLIST_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "still_prompt": {"type": "string"},
                    "motion_prompt": {"type": "string"},
                    "line": {"type": "string"},
                },
                "required": ["still_prompt", "motion_prompt", "line"],
            },
        },
    },
    "required": ["title", "shots"],
}


def shotlist_prompt(inputs: FilmInputs) -> str:
    return (
        f"Write a shot list for a short film of exactly {inputs.shots} shots, each about "
        "2 to 5 seconds long.\n"
        f"The idea: {inputs.idea}\n"
        f"The visual style: {inputs.style}\n\n"
        "Answer with JSON only, in this shape: "
        '{"title": "...", "shots": [{"still_prompt": "...", "motion_prompt": "...", '
        '"line": "..."}]}.\n'
        "- still_prompt: one picture-model prompt for the shot's first frame: subject, "
        "setting, framing, light. Repeat what the viewer must recognise (a character's look, "
        "the place) in every shot, since each picture is drawn on its own.\n"
        "- motion_prompt: what moves during the shot, for a video model, in one sentence.\n"
        "- line: what a narrator says over the shot, at most 15 words, or an empty string."
    )


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class ShotListError(ValueError):
    """The chat model's answer is not a usable shot list (the message says why)."""


def parse_shotlist(text: str, shots: int | None = None) -> ShotList:
    """The shot list from a chat answer: plain JSON, or JSON in a code fence, or the first
    `{…}` in the text; `shots` cuts a longer list to the count asked for."""
    body = _FENCE.sub("", text.strip())
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ShotListError("the answer holds no JSON object")
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ShotListError(f"the JSON does not parse ({exc.msg} at character {exc.pos})") from None
    try:
        found = ShotList.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ShotListError(f"{where}: {first['msg']}") from None
    if shots is not None and len(found.shots) > shots:
        found = found.model_copy(update={"shots": found.shots[:shots]})
    return found


def still_prompt(inputs: FilmInputs, shot: Shot, width: int, height: int) -> str:
    """The picture prompt, at the video's own size: sd-server takes the size from the
    prompt's extra arguments (the router drops a request's `size`)."""
    size = json.dumps({"width": width, "height": height}, separators=(",", ":"))
    return f"{shot.still_prompt}, {inputs.style}<sd_cpp_extra_args>{size}</sd_cpp_extra_args>"


def film_file(clip: str) -> str:
    """The film's name follows its clips' container (`-c:v copy` keeps their codec)."""
    suffix = "." + clip.rsplit(".", 1)[-1].lower() if "." in clip else ".webm"
    return f"film{suffix}" if f"film{suffix}" in FILM_NAMES else FILM_FILE


def workflow_of(video: str) -> str | None:
    """The workflow a video role names, or None for an instance."""
    return video.removeprefix(WORKFLOW) if video.startswith(WORKFLOW) else None


def item_file(stage: str, index: int, suffix: str) -> str:
    """`still-01.png`, `line-02.wav`, `clip-03.webm`."""
    return f"{stage.rstrip('s')}-{index + 1:02d}{suffix}"
