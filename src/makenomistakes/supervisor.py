"""Start, stop and inspect engine containers through the Docker CLI."""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator

from makenomistakes.hardware.probe import CommandRunner, SubprocessRunner
from makenomistakes.recipes import docker_run_argv, health_path
from makenomistakes.recipes.model import Recipe, container_name
from makenomistakes.routing import PROBE_HEADER

HEALTH_TIMEOUT_SECONDS = 600.0
HEALTH_POLL_SECONDS = 2.0


PULL_TIMEOUT_SECONDS = 3600.0
REMOTE_TIMEOUT_SECONDS = 60.0  # one registry answer
_NO_SUCH_IMAGE = "no such image"  # Docker's wording when a tag or id is not on this box
# `docker inspect` on a missing container: "Error: No such object: …" (older daemons say
# "No such container"). Anything else (permission denied, daemon down) is an error, never
# "not running".
_NO_SUCH_CONTAINER = ("no such object", "no such container")


class SupervisorError(RuntimeError):
    pass


class NoContainerError(SupervisorError):
    """Docker answered that the container does not exist (not that it could not be asked)."""


class AlreadyRunningError(SupervisorError):
    """`start` of an instance whose container already runs."""


class LocalImage(BaseModel):
    """The slice of `docker image inspect --format '{{json .}}'` the runner reads."""

    id: str = Field(alias="Id")
    created: str = Field(default="", alias="Created")
    # `<repo>@sha256:…`: the registry's manifest each pull of this build came from (a
    # registry answering the same digest has nothing newer)
    repo_digests: list[str] = Field(default_factory=list, alias="RepoDigests")

    @field_validator("repo_digests", mode="before")
    @classmethod
    def _none_is_empty(cls, value: object) -> object:
        return [] if value is None else value

    @property
    def short_id(self) -> str:
        return self.id.removeprefix("sha256:")[:12]

    @property
    def created_day(self) -> str:
        return self.created[:10]


class ContainerState(StrEnum):
    RUNNING = "running"
    EXITED = "exited"
    ABSENT = "absent"


class _InspectState(BaseModel):
    """The slice of a container's `.State` we read."""

    running: bool = Field(default=False, alias="Running")
    exit_code: int | None = Field(default=None, alias="ExitCode")
    started_at: str | None = Field(default=None, alias="StartedAt")
    pid: int | None = Field(default=None, alias="Pid")  # its main process on the host
    oom_killed: bool = Field(default=False, alias="OOMKilled")  # the kernel's OOM killer


class _Inspect(BaseModel):
    """`INSPECT_FORMAT`: the state, and the image the container runs (a reading names
    the build that ran, which is not the tag's once `mnm engines update` moved it); `name`
    only in `MANY_FORMAT`'s lines (Docker's `/mnm-<name>`)."""

    state: _InspectState = Field(alias="State")
    image: str | None = Field(default=None, alias="Image")
    name: str = Field(default="", alias="Name")

    def status(self) -> ContainerStatus:
        state = self.state
        return ContainerStatus(
            state=ContainerState.RUNNING if state.running else ContainerState.EXITED,
            exit_code=None if state.running else state.exit_code,
            started_at=state.started_at,
            image_id=self.image,
            pid=state.pid or None,
            oom_killed=state.oom_killed,
        )


INSPECT_FORMAT = '{"State":{{json .State}},"Image":{{json .Image}}}'
# Several containers in one `docker inspect`, one line each with its name
MANY_FORMAT = '{"Name":{{json .Name}},"State":{{json .State}},"Image":{{json .Image}}}'


class ContainerStatus(BaseModel):
    state: ContainerState
    exit_code: int | None = None
    started_at: str | None = None
    image_id: str | None = None  # the build it runs (`sha256:…`), whatever the tag says now
    pid: int | None = None  # the engine's process on the host, while it runs (0: none)
    oom_killed: bool = False  # an exited container the kernel killed out of memory

    @property
    def build(self) -> str | None:
        """The short id readings carry (`LocalImage.short_id`)."""
        return self.image_id.removeprefix("sha256:")[:12] if self.image_id else None


class Health(StrEnum):
    READY = "ready"
    LOADING = "loading"
    UNREACHABLE = "unreachable"


HealthProbe = Callable[[int, str], Health]


def http_health(port: int, path: str = "/health") -> Health:
    """200 is ready; 503 is loading (llama-server); no answer is unreachable."""
    try:
        # Never through a proxy: WSL can copy Windows' HTTP_PROXY into the distro, and
        # NO_PROXY=localhost does not cover 127.0.0.1.
        response = httpx.get(
            f"http://127.0.0.1:{port}{path}",
            timeout=2.0,
            trust_env=False,
            headers={PROBE_HEADER: "1"},  # never a reason for the router to load anything
        )
    except httpx.HTTPError:
        return Health.UNREACHABLE
    if response.status_code == httpx.codes.OK:
        return Health.READY
    return (
        Health.LOADING
        if response.status_code == httpx.codes.SERVICE_UNAVAILABLE
        else Health.UNREACHABLE
    )


class DockerSupervisor:
    def __init__(
        self,
        runner: CommandRunner | None = None,
        probe: HealthProbe = http_health,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._runner = runner or SubprocessRunner()
        self._probe = probe
        self._clock = clock
        self._sleep = sleep

    def ensure_image(self, image: str, progress: Callable[[str], None] | None = None) -> bool:
        """Pull the engine image if it is not present. Returns True when a pull happened."""
        if self.image(image) is not None:
            return False
        if progress:
            progress(f"Pulling engine image {image} (a few GB, once)")
        self.pull(image)
        return True

    def image(self, ref: str) -> LocalImage | None:
        """The local image a tag or id names; None only when Docker says it is not here (a
        daemon that cannot be reached is an error, never "not pulled")."""
        code, out, err = self._run(["docker", "image", "inspect", "--format", "{{json .}}", ref])
        if code != 0:
            detail = (err or out).strip()
            if _NO_SUCH_IMAGE in detail.lower():
                return None
            raise SupervisorError(f"docker image inspect {ref} failed: {detail}")
        try:
            return LocalImage.model_validate_json(out)
        except ValidationError as exc:
            raise SupervisorError(f"unexpected `docker image inspect` output for {ref}") from exc

    def remote(self, ref: str) -> str:
        """What the registry holds for a tag, never pulled (`mnm engines check`): `docker
        buildx imagetools inspect`'s JSON (the manifest's digest, each platform's `created`;
        `docker manifest inspect` prints neither the index's own digest nor a build time)."""
        argv = ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .}}", ref]
        code, out, err = self._run(argv, timeout=REMOTE_TIMEOUT_SECONDS)
        if code != 0:
            raise SupervisorError(" ".join((err or out).split()) or f"exit {code}")
        return out

    def pull(self, ref: str) -> None:
        code, out, err = self._run(["docker", "pull", ref], timeout=PULL_TIMEOUT_SECONDS)
        if code != 0:
            raise SupervisorError(f"docker pull {ref} failed: {(err or out).strip()}")

    def tag(self, image_id: str, ref: str) -> None:
        code, out, err = self._run(["docker", "tag", image_id, ref])
        if code != 0:
            raise SupervisorError(f"docker tag failed: {(err or out).strip()}")

    def start(self, recipe: Recipe, models_dir: Path) -> str:
        status = self.status(recipe.name)
        if status.state is ContainerState.RUNNING:
            raise AlreadyRunningError(f"{recipe.name} is already running")
        if status.state is ContainerState.EXITED:
            self._remove(recipe.name)
        try:
            argv = docker_run_argv(recipe, models_dir)
        except ValueError as exc:  # not downloaded, or MNM_GPU not a card number
            raise SupervisorError(str(exc)) from exc
        code, out, err = self._run(argv, timeout=300.0)
        if code != 0:
            raise SupervisorError(f"docker run failed: {(err or out).strip()}")
        return out.strip()

    def stop(self, name: str) -> None:
        status = self.status(name)
        if status.state is ContainerState.ABSENT:
            return
        if status.state is ContainerState.RUNNING:
            self._docker_stop(name)
        self._remove(name)

    def _remove(self, name: str) -> None:
        """`docker rm`, its exit code read: gone already is fine, anything
        else (a removal in progress, the daemon) is the error, never a stop that "worked"."""
        code, out, err = self._run(["docker", "rm", container_name(name)])
        detail = (err or out).strip()
        if code != 0 and not any(marker in detail.lower() for marker in _NO_SUCH_CONTAINER):
            raise SupervisorError(f"docker rm failed: {detail or f'exit {code}'}")

    def names_with(self, prefix: str) -> list[str]:
        """The instance names of every container, running or exited, whose name starts
        `mnm-<prefix>` (a tune's copies, `mnm-tune-*`); SupervisorError when Docker cannot
        be asked."""
        code, out, err = self._run(
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"name=^{container_name(prefix)}",
                "--format",
                "{{.Names}}",
            ]
        )
        if code != 0:
            raise SupervisorError(f"docker ps failed: {(err or out).strip() or f'exit {code}'}")
        own = container_name("")
        return [line.strip().removeprefix(own) for line in out.splitlines() if line.strip()]

    def halt(self, name: str) -> None:
        """`docker stop` without the `rm`: the container and its log stay for `mnm logs`, and
        the next `start` removes it. Nothing to do unless it runs."""
        if self.status(name).state is ContainerState.RUNNING:
            self._docker_stop(name)

    def _docker_stop(self, name: str) -> None:
        code, out, err = self._run(["docker", "stop", "--time", "20", container_name(name)], 60.0)
        if code != 0:
            raise SupervisorError(f"docker stop failed: {(err or out).strip()}")

    def status(self, name: str) -> ContainerStatus:
        code, out, err = self._run(
            ["docker", "inspect", "--format", INSPECT_FORMAT, container_name(name)]
        )
        if code != 0:
            detail = (err or out).strip()
            if any(marker in detail.lower() for marker in _NO_SUCH_CONTAINER):
                return ContainerStatus(state=ContainerState.ABSENT)
            raise SupervisorError(f"docker inspect failed: {detail or f'exit {code}'}")
        try:
            found = _Inspect.model_validate_json(out)
        except ValidationError:
            try:  # `.State` alone: what the fakes, and an older mnm's format, print
                found = _Inspect(State=_InspectState.model_validate_json(out))
            except ValidationError as exc:
                raise SupervisorError(f"unexpected docker inspect output: {exc}") from exc
        return found.status()

    def statuses(self, names: list[str]) -> dict[str, ContainerStatus]:
        """Every named instance's container from one `docker inspect` (the router's
        sweep asks once, never per instance): ABSENT for each Docker says it has not; any
        other complaint (permission denied, the daemon down) raises, as `status` does."""
        if not names:
            return {}
        argv = ["docker", "inspect", "--format", MANY_FORMAT, *map(container_name, names)]
        code, out, err = self._run(argv)
        complaints = [line for line in err.splitlines() if line.strip()]
        if code != 0 and (
            not complaints
            or not all(any(m in line.lower() for m in _NO_SUCH_CONTAINER) for line in complaints)
        ):
            raise SupervisorError(f"docker inspect failed: {err.strip() or f'exit {code}'}")
        found: dict[str, ContainerStatus] = {}
        for line in out.splitlines():
            if not line.strip():
                continue
            try:
                one = _Inspect.model_validate_json(line)
            except ValidationError as exc:
                raise SupervisorError(f"unexpected docker inspect output: {exc}") from exc
            found[one.name.lstrip("/")] = one.status()
        absent = ContainerStatus(state=ContainerState.ABSENT)
        return {name: found.get(container_name(name), absent) for name in names}

    def logs(self, name: str, tail: int = 400) -> str:
        """Last `tail` lines of the container log; 0 means the whole log."""
        argv = ["docker", "logs", "--tail", str(tail) if tail > 0 else "all", container_name(name)]
        code, out, err = self._run(argv)
        if code != 0:  # a Docker error is never engine output (it would read as a clean log)
            why = err.strip() or f"exit {code}"
            gone = any(word in why.lower() for word in _NO_SUCH_CONTAINER)
            raise (NoContainerError if gone else SupervisorError)(
                f"could not read the container log: {why}"
            )
        return out + err

    def health(self, recipe: Recipe) -> Health:
        return self._probe(recipe.port, health_path(recipe))

    def wait_ready(self, recipe: Recipe, timeout: float = HEALTH_TIMEOUT_SECONDS) -> Health:
        """Poll until the engine answers ready, the container dies, or the timeout passes.

        An engine whose port opens late (sd-server plans its model files first) reads as
        UNREACHABLE while the container is running; that is loading, not failure, so polling
        continues.
        """
        deadline = self._clock() + timeout
        while True:
            health = self._probe(recipe.port, health_path(recipe))
            if health is Health.READY:
                return health
            if self.status(recipe.name).state is not ContainerState.RUNNING:
                try:
                    tail = self.logs(recipe.name, 40)
                except SupervisorError as exc:  # quoted, never read as a clean log
                    tail = f"({exc})\n"
                raise SupervisorError(
                    f"{recipe.name} exited while loading; last log lines:\n{tail}"
                )
            if self._clock() >= deadline:
                return health
            self._sleep(HEALTH_POLL_SECONDS)

    def _run(self, argv: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
        return self._runner.run(argv, timeout=timeout)
