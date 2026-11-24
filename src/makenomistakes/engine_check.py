"""`mnm engines check`: whether the registry holds a newer build of each
engine image this box has, without pulling it, and whether that build runs a saved instance
the installed one is known to lack (`engine_support.MERGES`, what `Cause.IMAGE_TOO_OLD`
means).

The registry is asked through the supervisor's runner (`DockerSupervisor.remote`: `docker
buildx imagetools inspect`, no token: ghcr.io and Docker Hub both answer it) and its answer
typed here. Newer means a different manifest than any pull of the local build came from
(`RepoDigests`) and a later `created`. Nothing is pulled or updated: `mnm engines update` does
that, and tests itself and rolls back. The last check is kept in
`<state>/engine-check.json` for the doctor's Engines line, the web's System → Engines and the
nightly's page; it is never carried by a backup (it describes the registry on one day)."""

from __future__ import annotations

import contextlib
import json
import os
import platform
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.doctor_types import Check, Status
from makenomistakes.engine_support import MERGES
from makenomistakes.engines import box_images
from makenomistakes.hub.facts import Engine
from makenomistakes.local_time import local
from makenomistakes.recipes import Recipe
from makenomistakes.supervisor import DockerSupervisor, LocalImage, SupervisorError

SCHEMA_VERSION = 1
FILE = "engine-check.json"
ENGINE_NAMES = {
    Engine.LLAMACPP: "llama.cpp",
    Engine.SDCPP: "stable-diffusion.cpp",
    Engine.WHISPERCPP: "whisper.cpp",
    Engine.AUDIOCPP: "audio.cpp",
    Engine.VLLM: "vLLM",
}
_ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}


class Upstream(BaseModel):
    """The registry's build of a tag: its manifest's digest and this platform's `created`."""

    digest: str
    created: str  # RFC 3339, the platform's image config

    @property
    def day(self) -> str:
        return self.created[:10]


def host_platform() -> str:
    """`linux/amd64` on the box (the platform a pull would fetch)."""
    return f"linux/{_ARCH.get(platform.machine().lower(), 'amd64')}"


def parse_remote(text: str, where: str | None = None) -> Upstream:
    """`docker buildx imagetools inspect --format '{{json .}}'`: `manifest.digest`, and the
    image config's `created` (one config for a single-platform tag, else one per platform:
    `where`'s, `linux/amd64` by default). ValueError when the answer has neither."""
    try:
        found = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the registry's answer is not JSON ({exc.msg})") from exc
    if not isinstance(found, dict):
        raise ValueError("the registry's answer is not an object")
    manifest = found.get("manifest")
    digest = manifest.get("digest") if isinstance(manifest, dict) else None
    image = found.get("image")
    if isinstance(image, dict) and "created" not in image:
        image = image.get(where or host_platform())
    created = image.get("created") if isinstance(image, dict) else None
    if not isinstance(digest, str) or not isinstance(created, str) or not created:
        raise ValueError("the registry's answer names no digest or build time for this platform")
    return Upstream(digest=digest, created=created)


class State(StrEnum):
    NEWER = "newer"  # the registry holds a later build than the box's
    CURRENT = "current"  # the same build, or nothing later
    NOT_PULLED = "not pulled"  # nothing on the box to compare (its first start pulls)
    UNKNOWN = "unknown"  # the registry or Docker could not say: `why`


class EngineCheck(BaseModel):
    engine: Engine
    image: str
    state: State
    installed_day: str | None = None  # the local image's `Created` day
    installed_build: str | None = None  # its short id
    upstream_day: str | None = None
    upstream_digest: str | None = None
    why: str | None = None  # UNKNOWN: what failed

    @property
    def newer(self) -> bool:
        return self.state is State.NEWER

    @property
    def words(self) -> str:
        """The row in a sentence."""
        name = ENGINE_NAMES[self.engine]
        match self.state:
            case State.NEWER:
                return (
                    f"a newer {name} build exists (built {self.upstream_day}; this box's was "
                    f"built {self.installed_day})"
                )
            case State.CURRENT:
                return f"{name} is the registry's newest build (built {self.installed_day})"
            case State.NOT_PULLED:
                return f"{name} is not pulled: its first start pulls the registry's build"
            case _:
                return f"{name} could not be checked: {self.why}"


class ArchLine(BaseModel):
    """A saved llama.cpp instance whose architecture the installed build is known to lack."""

    instance: str
    architecture: str
    line: str
    runs: bool  # the registry's build is from after the merge that brought it


class EngineReport(BaseModel):
    schema_version: int = SCHEMA_VERSION
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rows: list[EngineCheck] = Field(default_factory=list)
    instances: list[ArchLine] = Field(default_factory=list)

    @property
    def newer(self) -> list[EngineCheck]:
        return [row for row in self.rows if row.newer]

    @property
    def when(self) -> str:
        return f"{local(self.checked_at):%Y-%m-%d %H:%M}"


def update_command(engine: Engine) -> str:
    return f"mnm engines update {engine.value}"


UPDATE_WORDS = (
    "`mnm engines update <engine>` pulls it (a few GB), tests it on a copy of the smallest "
    "instance against the build now, and rolls back by itself when the test fails; nothing "
    "updates by itself"
)


def check_engines(docker: DockerSupervisor, recipes: list[Recipe]) -> EngineReport:
    """Every engine image this box has or an instance uses (one registry question each),
    then the saved llama.cpp instances the installed build is known to lack. A registry
    failure for one image is that row's `unknown`, never the end of the check; Docker that
    cannot be asked at all raises SupervisorError (once, not once per image)."""
    images: dict[str, Engine] = {}
    for engine, image in box_images().items():  # this box's backend's tags
        images[image] = engine
    for recipe in recipes:
        images.setdefault(recipe.image, recipe.engine)
    checked = (
        _check_one(docker, engine, image, any(r.image == image for r in recipes))
        for image, engine in images.items()
    )
    rows = [row for row in checked if row is not None]
    return EngineReport(rows=rows, instances=arch_lines(rows, recipes))


def _check_one(
    docker: DockerSupervisor, engine: Engine, image: str, used: bool
) -> EngineCheck | None:
    here = docker.image(image)  # Docker that cannot be asked ends the check (said once)
    if here is None:
        return EngineCheck(engine=engine, image=image, state=State.NOT_PULLED) if used else None
    base = EngineCheck(
        engine=engine,
        image=image,
        state=State.UNKNOWN,
        installed_day=here.created_day or None,
        installed_build=here.short_id,
    )
    try:
        upstream = parse_remote(docker.remote(image))
    except (SupervisorError, ValueError) as exc:
        return base.model_copy(update={"why": _why(str(exc))})
    return base.model_copy(
        update={
            "state": compare(here, upstream),
            "upstream_day": upstream.day,
            "upstream_digest": upstream.digest,
        }
    )


def compare(here: LocalImage, upstream: Upstream) -> State:
    """The same manifest as a pull of this build came from: current. Else a later `created`:
    newer. A local build with no time (Docker gave none) cannot be judged."""
    if any(pulled.endswith(f"@{upstream.digest}") for pulled in here.repo_digests):
        return State.CURRENT
    if not here.created:
        return State.UNKNOWN
    return State.NEWER if _instant(upstream.created) > _instant(here.created) else State.CURRENT


def _instant(text: str) -> datetime:
    """RFC 3339 with nanoseconds (Docker's), to the microsecond."""
    head, _, rest = text.partition(".")
    if rest:
        digits = "".join(c for c in rest if c.isdigit())
        zone = rest[len(digits) :]
        text = f"{head}.{digits[:6]}{zone}"
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _why(text: str) -> str:
    """The registry's or Docker's complaint, with the remedy where one is known."""
    lowered = text.lower()
    if "'buildx' is not a docker command" in lowered or "unknown command" in lowered:
        return (
            "`docker buildx` is not installed (the `docker-buildx-plugin` package; `mnm setup` "
            "installs Docker's own)"
        )
    if "429" in lowered or "too many requests" in lowered:
        return "the registry answered 429 (too many requests): try again later"
    return text


def arch_lines(rows: list[EngineCheck], recipes: list[Recipe]) -> list[ArchLine]:
    """The saved llama.cpp instances whose architecture came with a merge after the
    installed build (`MERGES`), and whether the registry's build is from after it."""
    by_image = {row.image: row for row in rows}
    found = []
    for recipe in recipes:
        merge = MERGES.get(recipe.architecture or "")
        row = by_image.get(recipe.image)
        if recipe.engine is not Engine.LLAMACPP or merge is None or row is None:
            continue
        if row.installed_day is None or row.installed_day >= merge.day:
            continue  # not known to lack it (a start tells: `architecture_line`)
        arch = recipe.architecture or ""
        if row.upstream_day is not None and row.upstream_day >= merge.day:
            line = (
                f"a newer llama.cpp (built {row.upstream_day}) runs {arch}, which "
                f"{recipe.name} needs (it came with {merge.commit} of {merge.day}): "
                f"`{update_command(Engine.LLAMACPP)}`"
            )
            found.append(ArchLine(instance=recipe.name, architecture=arch, line=line, runs=True))
            continue
        built = f"built {row.upstream_day}" if row.upstream_day else "unknown"
        line = (
            f"{recipe.name} needs {merge.words}; the installed llama.cpp was built "
            f"{row.installed_day} and the registry's ({built}) is not after it yet: check "
            "again in a day or two"
        )
        found.append(ArchLine(instance=recipe.name, architecture=arch, line=line, runs=False))
    return found


def report_path(state_dir: Path) -> Path:
    return state_dir / FILE


def save_report(state_dir: Path, report: EngineReport) -> None:
    """Written whole (a temp file moved into place); a box that cannot write it keeps the
    check's answer on the screen only."""
    path = report_path(state_dir)
    with contextlib.suppress(OSError):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{FILE}.{os.getpid()}.tmp")
        tmp.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


def last_report(state_dir: Path) -> EngineReport | None:
    """The last check; None when there is none, it cannot be read, or a newer mnm wrote it."""
    try:
        found = EngineReport.model_validate_json(report_path(state_dir).read_text("utf-8"))
    except (OSError, ValidationError, ValueError):
        return None
    return found if found.schema_version <= SCHEMA_VERSION else None


DOCTOR_NAME = "Engines"


def doctor_line(state_dir: Path) -> Check:
    """The doctor's Engines line from the last check (the doctor never asks the registry: a
    few seconds an image). A newer build is a plain note, never a WARN: the box runs as it
    did, and an update is the user's."""
    report = last_report(state_dir)
    if report is None:
        return Check(
            name=DOCTOR_NAME,
            status=Status.OK,
            detail="not checked against the registry yet",
            remedy="`mnm engines check` asks whether a newer build exists (nothing is pulled)",
        )
    newer = report.newer
    if not newer:
        unknown = [ENGINE_NAMES[r.engine] for r in report.rows if r.state is State.UNKNOWN]
        detail = f"no newer build on the registry (checked {report.when})"
        if unknown:
            detail += f"; not checked: {', '.join(unknown)}"
        return Check(name=DOCTOR_NAME, status=Status.OK, detail=detail)
    names = ", ".join(f"{ENGINE_NAMES[r.engine]} (built {r.upstream_day})" for r in newer)
    lead = "a newer build exists" if len(newer) == 1 else "newer builds exist"
    return Check(
        name=DOCTOR_NAME,
        status=Status.OK,
        detail=f"{lead}: {names} (checked {report.when})",
        remedy=f"`{update_command(newer[0].engine) if len(newer) == 1 else 'mnm engines update'}` "
        "when you choose: each new build tests itself and rolls back when it fails",
    )
