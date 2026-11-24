"""What the studio's two built images share (the tools, the trainer): an image mnm builds on the
box as ComfyUI's is (an official base, every package pinned by sha256, weights checked at
build), named by a hash of what it is built from, recorded in a ledger under `<state>/studio/`
with the image before it kept for `rollback`, and removed by its own `uninstall` (fails closed:
the ledger stays while Docker cannot say what is gone).

Nothing is pulled and nothing runs until the user installs it; `mnm engines update` never
touches either (they are built, not pulled), and a new pin in mnm is a new tag that the next
install builds.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from makenomistakes.comfy import build_stream
from makenomistakes.config import Settings
from makenomistakes.docker_disk import FREE_SOME, image_room
from makenomistakes.hardware.probe import LineRunner
from makenomistakes.locking import shared_lock

LEDGER_SCHEMA = 1
Progress = Callable[[str], None]


def quiet(_message: str) -> None:
    """No one listening."""


class ImageLedger(BaseModel):
    """`<state>/studio/<image>.json`: the image in use and the one before it."""

    schema_version: int = LEDGER_SCHEMA
    image: str
    previous: str | None = None
    built_at: datetime
    arch: str
    variant: str | None = None  # the trainer's torch build (cpu, cu130); the tools have one
    versions: dict[str, str] = Field(default_factory=dict)  # what the build resolved
    size_bytes: int | None = None


class ImageError(RuntimeError):
    """A built image that could not be built, found or used, worded for the user."""


class ImageMissingError(ImageError):
    """The image is not on this box (its install command builds it)."""


@dataclass(frozen=True)
class ImageSpec:
    """One built image as the user reads about it, and where its record lives."""

    repo: str  # `mnm/studio-tools`
    ledger: str  # its file under `<state>/studio/`
    lock: str
    label: str  # "studio tools", "trainer": inside a sentence
    possessive: str  # "The studio tools'", "The trainer's": a sentence's start
    install: str  # the command that builds it
    uninstall: str
    not_installed: str  # what every use says before an install
    gib: float  # about how large, said before a build
    minutes: str  # about how long a build takes ("10-20")
    build_seconds: float
    error: type[ImageError]
    missing: type[ImageMissingError]


def docker_arch() -> str:
    """Docker's name for this machine's architecture (the requirements file it builds)."""
    machine = platform.machine().lower()
    return "arm64" if machine in ("aarch64", "arm64") else "amd64"


def rotate(old: ImageLedger | None, tag: str) -> tuple[str | None, list[str]]:
    """The image before `tag` and the images no longer kept, after a build of `tag`: the
    new one and the one before it are never among them (after a rollback the build is the
    image the ledger names as before; deleting it would leave a ledger naming nothing)."""
    if old is None:
        return None, []
    previous = old.previous if old.image == tag else old.image
    gone = {old.image, old.previous} - {tag, previous, None}
    return previous, sorted(str(g) for g in gone)


class Versions(BaseModel):
    """An image's `versions` line: what its build resolved."""

    tool: Literal["versions"]
    versions: dict[str, str]


# a job's exit, as the user reads it (the tools' jobs, a training's container)
EXIT_WORDS = {
    137: "killed: out of memory, or `docker kill`",
    139: "crashed (exit 139)",
    125: "Docker could not start it (exit 125)",
}


def sandbox(tmp_size: str) -> list[str]:
    """What every container of these images runs under (a tools job, a training, the build's
    own `versions`): no network, a read-only root with a scratch /tmp, no capabilities, no
    new privileges, a bound on processes, the user's uid."""
    return [
        "--network", "none", "--read-only", "--tmpfs", f"/tmp:rw,size={tmp_size}",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "512",
        "--user", f"{os.getuid()}:{os.getgid()}",
    ]  # fmt: skip


_JSON: TypeAdapter[object] = TypeAdapter(object)


def json_line(line: str) -> object | None:
    """One line a container printed, as JSON; None when it is none. Read by pydantic, whose
    reader bounds nesting: Python's `json` recurses into a deeply nested line until a
    RecursionError, which is no ValueError."""
    try:
        return _JSON.validate_json(line)
    except ValidationError:
        return None


def tool_error(said: str) -> str | None:
    """A container's own worded error (`{"tool": "error", "error": …}`), when its last JSON
    line is one: a tools job's stdout, a training's last log lines."""
    lines = [line for line in said.splitlines() if line.strip().startswith("{")]
    data = json_line(lines[-1]) if lines else None
    if isinstance(data, dict) and data.get("tool") == "error":
        return str(data.get("error") or "it gave no reason")
    return None


def versions_of(stdout: str) -> dict[str, str]:
    """The image's `versions` line (its last), typed; nothing when it is not one."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    try:
        return Versions.model_validate_json(lines[-1]).versions if lines else {}
    except ValidationError:
        return {}


class BuiltImage:
    """An image's ledger, its build, rollback and removal; one lock over the ledger's
    changes. A subclass names its spec, its tag and its build command."""

    spec: ImageSpec

    def __init__(
        self,
        settings: Settings,
        runner: LineRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        from makenomistakes.hardware.runner import SubprocessRunner  # noqa: PLC0415

        self.settings = settings
        self.base = settings.expanded_state_dir() / "studio"
        self.runner: LineRunner = runner or SubprocessRunner()
        self.clock = clock
        self._lock = shared_lock(self.base, self.spec.lock)

    # -- what a subclass says --------------------------------------------------------------

    def image_tag(self) -> str:
        raise NotImplementedError

    def build_argv(self, tag: str, ca: Path | None = None) -> list[str]:
        raise NotImplementedError

    def variant(self) -> str | None:
        """The build's variant, recorded in the ledger (None: the image has one)."""
        return None

    def variant_of(self, image: str) -> str | None:
        """The variant an image of this repository is (from its tag); None: no variants."""
        return None

    def gib(self) -> float:
        """About how large this build's image is (a variant may differ from the spec's)."""
        return self.spec.gib

    # -- the ledger ------------------------------------------------------------------------

    def locked(self) -> AbstractContextManager[None]:
        return self._lock()

    @property
    def ledger_path(self) -> Path:
        return self.base / self.spec.ledger

    def ledger(self) -> ImageLedger | None:
        """What the install built; None before. Unreadable or newer: the spec's error naming
        the file (never "not installed": the next install would lose the rollback)."""
        try:
            text = self.ledger_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self.spec.error(f"{self.ledger_path} cannot be read: {exc}") from exc
        try:
            ledger = ImageLedger.model_validate_json(text)
        except ValidationError as exc:
            raise self.spec.error(
                f"{self.ledger_path} cannot be read ({exc.errors()[0]['msg']}): "
                f"`{self.spec.install}` builds the image again and writes it anew."
            ) from None
        if ledger.schema_version > LEDGER_SCHEMA:
            raise self.spec.error(f"{self.ledger_path} is from a newer mnm: run `mnm update`.")
        return ledger

    def _save(self, ledger: ImageLedger) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(ledger.model_dump_json(indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.ledger_path)

    def _docker(self, args: list[str], timeout: float = 60.0) -> tuple[int, str, str]:
        return self.runner.run(["docker", *args], timeout=timeout)

    def installed(self) -> ImageLedger:
        """The ledger, when its image is on the box; else the spec's missing error."""
        ledger = self.ledger()
        if ledger is None:
            raise self.spec.missing(self.spec.not_installed)
        code, _out, err = self._docker(["image", "inspect", ledger.image], timeout=30)
        if code != 0:
            raise self.spec.missing(
                f"{self.spec.possessive} image {ledger.image} is gone from Docker "
                f"({err.strip()[:200]}): `{self.spec.install}` builds it again."
            )
        return ledger

    # -- building --------------------------------------------------------------------------

    def install(self, progress: Progress = quiet, ca: Path | None = None) -> ImageLedger:
        """Build the image (minutes; the one before stays for `rollback`) and record it."""
        tag = self.image_tag()
        room = image_room(self.runner)
        if (need := room.short_of(self.gib())) is not None:
            raise self.spec.error(
                f"{room.free_gib:.1f} GiB is free in {room.root}, where Docker keeps its images; "
                f"the build needs about {need:.0f}. {FREE_SOME}, then `{self.spec.install}` again."
            )
        progress(f"Building {tag} (about {self.gib():.0f} GB, {self.spec.minutes} minutes)")
        code, tail = build_stream.build(
            self.runner, self.build_argv(tag, ca), progress, self.clock, self.spec.build_seconds
        )
        if code != 0:
            raise self.spec.error(f"{self.spec.possessive} image did not build:\n{tail}")
        versions = self._versions(tag)
        size = self._size(tag)
        with self.locked():
            try:
                old = self.ledger()
            except self.spec.error:
                old = None  # an unreadable ledger: the new build replaces it
            previous, gone = rotate(old, tag)
            ledger = ImageLedger(
                image=tag,
                previous=previous,
                built_at=datetime.now(UTC),
                arch=docker_arch(),
                variant=self.variant(),
                versions=versions,
                size_bytes=size,
            )
            self._save(ledger)
        for image in gone:  # two kept: this one and the one before it
            self._docker(["image", "rm", image])
        return ledger

    def _versions(self, tag: str) -> dict[str, str]:
        code, out, _err = self._docker(  # in the sandbox, as every later container
            ["run", "--rm", *sandbox("1g"), tag, "versions"], timeout=120
        )
        return versions_of(out) if code == 0 else {}

    def _size(self, tag: str) -> int | None:
        code, out, _err = self._docker(["image", "inspect", "--format", "{{.Size}}", tag])
        return int(out.strip()) if code == 0 and out.strip().isdigit() else None

    def rollback(self) -> ImageLedger:
        with self.locked():
            old = self.ledger()
            if old is None or old.previous is None:
                raise self.spec.error(f"There is no earlier {self.spec.label} image to go back to.")
            variant = self.variant_of(old.previous) or old.variant  # the image's own build
            ledger = old.model_copy(
                update={
                    "image": old.previous,
                    "previous": old.image,
                    "variant": variant,
                    "versions": {},
                }
            )
            self._save(ledger)
            return ledger

    def uninstall(self) -> list[str]:
        """Every image mnm built under its name, then the ledger; what it made stays.
        Fails closed: when Docker cannot list or remove one, the ledger stays (it is what
        names the image to remove next time) and the error says so."""
        removed: list[str] = []
        repo = self.spec.repo
        with self.locked():
            code, out, err = self._docker(
                ["image", "ls", "--format", "{{.Repository}}:{{.Tag}}", repo]
            )
            if code != 0:
                raise self.spec.error(
                    f"Docker could not list the {self.spec.label} images "
                    f"({(err or out).strip()[:200]}): nothing was removed; run "
                    f"`{self.spec.uninstall}` again once it answers."
                )
            for image in (i for i in out.split() if i.startswith(f"{repo}:")):
                code, out, err = self._docker(["image", "rm", image])
                if code != 0:
                    raise self.spec.error(
                        f"Docker could not remove {image} ({(err or out).strip()[:200]}): "
                        f"removed so far: {', '.join(removed) or 'nothing'}; the record stays."
                    )
                removed.append(image)
            self.ledger_path.unlink(missing_ok=True)
        return removed
