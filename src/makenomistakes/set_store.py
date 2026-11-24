"""The sets' files (`mnm sets`): one JSON file per set under `<state>/workflows`
(`JsonFolder`: written whole, one lock across threads and processes) and an empty marker per
set started as a set under `.in-use/` (`workflows` re-exports it).

Every way into the store fails in its own words: a lock the system will not take, a set file
it will not give back or a marker it will not change is a `WorkflowError` naming the file,
never a raw `OSError`. The sets are read by the router's tick (`held_back_in`, the card jobs'
`held`), which turns a `WorkflowError` into "the sets cannot be read"; a raw error there would
make every tick raise."""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.desk import DeskStore
from makenomistakes.json_folder import BadNameError, JsonFolder, cause, sentence, unlisted
from makenomistakes.roles import ROLE_ORDER
from makenomistakes.set_members import WorkflowError

# A file from a newer runner is refused, never read with fields dropped; 3 is `persistent`,
# stamped only on a set that uses it (`WorkflowStore.save`).
SCHEMA_VERSION = 3
IN_USE = ".in-use"  # the store's markers of the sets started as sets (a dot: no set's name)


class Workflow(BaseModel):
    name: str
    members: dict[str, str]  # role -> instance name, in the order given
    # The members take turns on the GPU: only the first starts with the workflow,
    # the router loads the others on demand (`swap.py`); every member's recipe carries
    # `swap_group = name`.
    swap: bool = False
    # its running members are never unloaded, for room (`held_back`) or idleness
    persistent: bool = False
    schema_version: int = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # Started as a set (`mnm sets start`) and not stopped since: the store's own marker
    # (`<state>/workflows/.in-use/<name>`), never written into the set's file.
    started: bool = Field(default=False, exclude=True)

    def ordered(self) -> list[tuple[str, str]]:
        """Members in `ROLE_ORDER` (language models first), unknown roles last."""
        rank = {role: i for i, role in enumerate(ROLE_ORDER)}
        return sorted(self.members.items(), key=lambda m: (rank.get(m[0], len(rank)), m[0]))


class WorkflowStore:
    """One JSON file per workflow under `<state>/workflows` (`JsonFolder`: written whole,
    one lock across threads and processes)."""

    def __init__(self, state_dir: Path) -> None:
        self._files = JsonFolder(state_dir / "workflows")
        self._started = state_dir / "workflows" / IN_USE  # one empty file per started set
        self.desk = DeskStore(state_dir)  # which set's roles are model ids

    def _locked(self) -> AbstractContextManager[None]:
        """The folder's lock through its door (`JsonFolder.held`), as the recipes store's."""
        folder = self._files.folder
        return self._files.held(
            lambda exc: WorkflowError(
                f"The sets folder {folder} could not be locked ({cause(exc)})."
            )
        )

    def path(self, name: str) -> Path:
        try:
            return self._files.path(name)
        except BadNameError:
            raise WorkflowError(f"no set named '{name}' (`mnm sets ls` lists them)") from None

    def exists(self, name: str) -> bool:
        """Whether the set's file is there. A sets folder mnm cannot enter is this store's
        error naming the file (`Path.exists` raises since Python 3.12), never a raw OSError
        past `desk.read_aliases`, which every model-bound route asks."""
        try:
            return self._files.exists(name)
        except OSError as exc:
            path = self._files.folder / f"{name}.json"
            raise WorkflowError(f"The set file {path} could not be read ({cause(exc)}).") from exc

    def save(self, workflow: Workflow) -> None:
        with self._locked():
            # A set that uses `persistent` needs 3; any other keeps its own (at most 2), so an
            # older runner goes on reading a plain set.
            needed = SCHEMA_VERSION if workflow.persistent else min(workflow.schema_version, 2)
            current = workflow.model_copy(update={"schema_version": needed})
            try:
                self._files.write(workflow.name, current.model_dump_json(indent=2))
            except OSError as exc:
                raise WorkflowError(
                    f"The set file {self.path(workflow.name)} could not be written ({cause(exc)})."
                ) from exc

    def load(self, name: str) -> Workflow:
        with self._locked():
            if not self.exists(name):
                raise WorkflowError(f"no set named '{name}' (`mnm sets ls` lists them)")
            try:
                text = self._files.read(name)
            except OSError as exc:  # never "this version": the file may be fine
                raise WorkflowError(
                    f"The set file {self.path(name)} could not be read ({cause(exc)})."
                ) from exc
            started = self._marked(name)
            try:
                workflow = Workflow.model_validate_json(text)
            except ValidationError as exc:
                raise WorkflowError(
                    f"The set file {self.path(name)} cannot be read "
                    f"({exc.errors()[0]['msg']}). Delete it and create the set again; the "
                    "instances are untouched."
                ) from exc
            if workflow.schema_version > SCHEMA_VERSION:
                raise WorkflowError(
                    f"The set file {self.path(name)} was written by a newer version of "
                    "makenomistakes. Run `mnm update`, then try again."
                )
            return workflow.model_copy(update={"started": started}) if started else workflow

    def _marked(self, name: str) -> bool:
        """Whether the set was started as a set: its in-use mark, a failure to look at which
        is the mark's own error, never the set file's."""
        marker = self._started / name
        try:
            return marker.exists()
        except OSError as exc:
            raise WorkflowError(
                f"The set {name}'s in-use mark ({marker}) could not be read ({cause(exc)})."
            ) from exc

    def mark_started(self, name: str, started: bool) -> None:
        """`mnm sets start` (True), `mnm sets stop`, `rm` and `mnm stop --all` (False)."""
        with self._locked():
            self._files.path(name)  # BadNameError for a name that would leave the folder
            marker = self._started / name
            try:
                if started:
                    self._started.mkdir(parents=True, exist_ok=True)
                    marker.touch()
                else:
                    marker.unlink(missing_ok=True)
            except OSError as exc:
                raise WorkflowError(
                    f"The set {name}'s in-use mark ({marker}) could not be changed ({cause(exc)})."
                ) from exc

    def list(self) -> list[Workflow]:
        """Every set; a folder the system will not list is a WorkflowError naming it, as an
        unreadable set file is, never no sets (which would hold no member back for room)."""
        with self._locked():
            try:
                names = self._files.names()
            except OSError as exc:
                folder = self._files.folder
                raise WorkflowError(sentence(unlisted("sets folder", folder, exc))) from exc
            return sorted((self.load(n) for n in names), key=lambda w: w.created_at)

    def delete(self, name: str) -> None:
        with self._locked():
            if not self.exists(name):
                raise WorkflowError(f"no set named '{name}'")
            marker = self._started / name
            try:  # the mark first: a set made again under this name never reads as started
                marker.unlink(missing_ok=True)
            except NotADirectoryError:  # no folder of marks (a file in its place): no mark
                pass
            except OSError as exc:
                raise WorkflowError(
                    f"The set {name}'s in-use mark ({marker}) could not be removed "
                    f"({cause(exc)}); the set is kept."
                ) from exc
            try:
                self._files.unlink(name)
            except OSError as exc:
                raise WorkflowError(
                    f"The set file {self.path(name)} could not be removed ({cause(exc)})."
                ) from exc
