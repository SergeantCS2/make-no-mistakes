"""What ComfyUI's container is doing, as Docker says: not installed, stopped or
running from one `docker inspect`; a Docker that cannot be asked is `UNKNOWN`, never stopped
(only "No such object" means absent); a dead container's address cleared from the ledger.
`ComfyEngine` takes it in."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ValidationError

from makenomistakes.comfy.image import ComfyError, Ledger
from makenomistakes.comfy.sandbox import CONTAINER
from makenomistakes.hardware.probe import LineRunner
from makenomistakes.locking import FolderLock


class State(StrEnum):
    NOT_INSTALLED = "not installed"
    STOPPED = "stopped"
    RUNNING = "running"
    UNKNOWN = "unknown"  # Docker could not be asked: running or not cannot be told


class InspectState(BaseModel):
    Running: bool = False
    ExitCode: int = 0
    OOMKilled: bool = False


class ComfyDockerError(ComfyError):
    """Docker could not be asked about ComfyUI's container: its state is unknown."""


NO_SUCH_CONTAINER = ("no such object", "no such container")  # Docker's "absent", nothing else


class Inspect(BaseModel):
    State: InspectState
    IPAddress: str = ""


class Observed:
    """The container's side of `ComfyEngine`: what it reads from Docker and the ledger."""

    runner: LineRunner
    _lock: FolderLock

    def ledger(self) -> Ledger | None:
        raise NotImplementedError  # ComfyEngine's

    def _ledger_or_none(self) -> Ledger | None:
        raise NotImplementedError  # ComfyEngine's

    def save(self, ledger: Ledger) -> None:
        raise NotImplementedError  # ComfyEngine's

    def checked_state(self) -> State:
        """`state()`, except that an unreadable ledger raises its ComfyError instead of
        reading as not installed: for what is said to the user."""
        return State.NOT_INSTALLED if self.ledger() is None else self.state()

    def state(self) -> State:
        """Not installed, stopped, running, or unknown when Docker does not answer (`observe`
        says Docker's own error): never stopped on a guess. Not a pure read: a stopped
        container's address is cleared from the ledger (`observe`), so the status bar and a
        run's start never wait on an address nothing answers at."""
        try:
            return self.observe()
        except ComfyDockerError:
            return State.UNKNOWN

    def observe(self) -> State:
        """The state from one `docker inspect`; ComfyDockerError when Docker cannot be asked
        (only "No such object" means no container). A stopped container's address (a crash,
        a reboot) is cleared, so nothing waits on it again (the status bar's quick look)."""
        ledger = self._ledger_or_none()
        if ledger is None:
            return State.NOT_INSTALLED
        found = self.container()
        if found is not None and found.State.Running:
            return State.RUNNING
        if ledger.address:
            self._forget_address()
        return State.STOPPED

    def _forget_address(self) -> None:
        with self._lock.attempt() as mine:  # a start or stop under way sets it itself
            ledger = self._ledger_or_none()
            if not mine or ledger is None or ledger.address is None:
                return
            try:  # asked again under the lock: a start may have ended since
                found = self.container()
            except ComfyDockerError:
                return
            if found is None or not found.State.Running:
                self.save(ledger.model_copy(update={"address": None}))

    def address(self, state: State | None = None) -> str | None:
        """Where the container answers, known to run; None when it does not run, when Docker
        cannot say (`asked_address` asks ComfyUI then) or its ledger cannot be read (the doctor
        says why). `state` when the caller has it already."""
        ledger = self._ledger_or_none()
        if ledger is None or ledger.address is None:
            return None
        running = (state or self.state()) is State.RUNNING
        return ledger.address if running else None

    def asked_address(self) -> str | None:
        """Where to ask ComfyUI itself about its work: its address unless Docker says it does
        not run. A Docker that cannot be asked leaves the answer to ComfyUI (unknown is never
        stopped); a box without it, or one mnm stopped, costs one file read."""
        ledger = self._ledger_or_none()
        if ledger is None or ledger.address is None:
            return None
        return ledger.address if self.state() in (State.RUNNING, State.UNKNOWN) else None

    def container_state(self) -> InspectState | None:
        """Docker's word on the container (exit code, killed out of memory); None if gone."""
        found = self._inspect()
        return found.State if found is not None else None

    def _inspect(self) -> Inspect | None:
        try:
            return self.container()
        except ComfyDockerError:
            return None

    def container(self) -> Inspect | None:
        """Docker's word on the container; None when there is none, ComfyDockerError when
        Docker cannot be asked (permission denied, the daemon down)."""
        fmt = "{{json .State}}|{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"
        code, out, err = self._docker(["inspect", "--format", fmt, CONTAINER])
        if code != 0:
            why = (err or out).strip()
            if any(word in why.lower() for word in NO_SUCH_CONTAINER):
                return None
            raise ComfyDockerError(why or f"docker inspect exited {code}")
        state, _, ip = out.strip().partition("|")
        try:
            return Inspect(State=InspectState.model_validate_json(state), IPAddress=ip)
        except ValidationError:
            return None

    def _docker(self, args: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
        return self.runner.run(["docker", *args], timeout=timeout)
