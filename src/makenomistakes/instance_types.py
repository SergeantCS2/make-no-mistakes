"""What `InstanceService` returns and raises: typed results the CLI, the web and workflows
read, kept apart from the service so the read and write sides can import them alone."""

from __future__ import annotations

import socket
from collections.abc import Callable
from dataclasses import dataclass, field

from makenomistakes.diagnose import Diagnosis
from makenomistakes.hub.facts import TYPE_ORDER, Modality
from makenomistakes.recipes import Recipe
from makenomistakes.supervisor import ContainerState, Health

Progress = Callable[[str], None]
CHANGED = "Changed and saved"  # the label every place that lists auto-fix's changes uses


class InstanceError(RuntimeError):
    """A user-facing failure; `diagnoses` carries log-derived causes when there are any,
    `fixes` what auto-fix changed and saved before giving up."""

    def __init__(
        self,
        message: str,
        diagnoses: list[Diagnosis] | None = None,
        fixes: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnoses = diagnoses or []
        self.fixes = fixes or []


class NoLogError(InstanceError):
    """No container and no kept log: an answer, not a failure to read one. `mnm logs`
    prints it and exits 0, so `mnm report` does not count it an error."""


class UnreadableStateError(InstanceError):
    """mnm's own files (a recipe) cannot be read: what it says is the store's reason, and
    nothing about any one instance is known to be wrong, so the router marks nothing stopped
    (else one corrupt recipe file would mark the instance being asked for `off`).
    `InstanceService` raises it wherever a `StoreError` other than a missing name surfaces."""


class DockerError(InstanceError):
    """Docker itself failed during a start (an image pull, the daemon going away): nothing
    is known to be wrong with the instance, so the router does not mark it stopped."""


def port_listening(port: int) -> bool:
    """Something already holds the port on every interface (where instances publish)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("0.0.0.0", port))  # the address instances publish on
        except OSError:
            return True
    return False


@dataclass(frozen=True)
class Stopped:
    """One `stop_many` result: what the container was before (ABSENT: nothing to stop)."""

    name: str
    was: ContainerState
    downloaded: bool  # parked or not, the router loads only what is on disk


@dataclass(frozen=True)
class InstanceStatus:
    recipe: Recipe
    state: ContainerState
    exit_code: int | None = None
    health: Health | None = None  # only probed while running
    error: str | None = None  # Docker could not be asked
    files_present: bool = True  # every recorded file is on disk (recorded is not present)
    started_at: str | None = None  # Docker's, while it runs: a new container is a new start
    build: str | None = None  # the engine build its container runs, while it runs

    @property
    def running(self) -> bool:
        return self.state is ContainerState.RUNNING

    @property
    def evicted(self) -> bool:
        """The disk policy deleted its files: its next start downloads them first."""
        return self.recipe.evicted is not None and not self.files_present

    @property
    def parked(self) -> bool:
        """Not loaded, yet the router loads it at the first request that asks for work: not
        stopped by the user (`off`), its files on disk or evicted by the disk policy (the
        load downloads them first), Docker answering (`router.ready`)."""
        on_disk = self.files_present or self.evicted
        return not self.running and not self.error and not self.recipe.off and on_disk


@dataclass(frozen=True)
class Started:
    recipe: Recipe  # the settings that ended up running (saved)
    fixes: list[str] = field(default_factory=list)  # what auto-fix changed, in order
    notes: list[str] = field(default_factory=list)  # what else the user should know


def by_type(statuses: list[InstanceStatus]) -> list[list[InstanceStatus]]:
    """The statuses grouped by model type, types in `TYPE_ORDER`, names in their saved order
    (creation) within a type; empty types are left out. `mnm ls` and the web list share it."""
    groups: dict[Modality, list[InstanceStatus]] = {}
    for status in statuses:
        groups.setdefault(status.recipe.modality, []).append(status)
    return [groups[m] for m in TYPE_ORDER if m in groups]
