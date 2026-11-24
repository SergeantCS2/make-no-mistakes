"""The desk: the one set whose roles are model ids (`mnm sets default <set>`).

An app's config names `coder`, `chat`, `memory` … instead of a model; the router resolves the
role to the desk's member (`routing.resolve`), lists it in `/v1/models` with `meta.alias_of`,
and `mnm sets edit desk coder=<another>` swaps the model behind it without touching the app.
A role of no default set is no alias. An instance's own name always wins: a role word is
refused as a new name (`InstanceService.check_name`), `sets default` and `sets edit` refuse a
desk whose role an existing instance is named after, and an alias that still meets one (a
restored backup) is left out (`Aliases.shadowed`), never resolved over the instance.

The choice is one small file, `<state>/desk/desk.json`, beside the sets rather than a field of
one: it is unique by construction, and an older runner ignores it (it serves no aliases)
instead of refusing a set file.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.aliases import Aliases
from makenomistakes.json_folder import JsonFolder, cause
from makenomistakes.set_members import WorkflowError

if TYPE_CHECKING:
    from makenomistakes.workflows import Workflow, WorkflowStore

FOLDER = "desk"
ENTRY = "desk"  # <state>/desk/desk.json
SCHEMA_VERSION = 1


class DeskFile(BaseModel):
    set: str
    schema_version: int = SCHEMA_VERSION
    chosen_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DeskStore:
    """Which set is the desk: one JSON file written whole behind the folder's lock."""

    def __init__(self, state_dir: Path) -> None:
        self._files = JsonFolder(state_dir / FOLDER)

    def _locked(self) -> AbstractContextManager[None]:
        """The folder's lock through its door (`JsonFolder.held`), as the sets store's: the
        router reads the desk at every request (`read_aliases`), which words only the store's
        errors (never a raw `IsADirectoryError`)."""
        folder = self._files.folder
        return self._files.held(
            lambda exc: WorkflowError(
                f"The desk folder {folder} could not be locked ({cause(exc)})."
            )
        )

    @property
    def path(self) -> Path:
        return self._files.path(ENTRY)

    def name(self) -> str | None:
        """The desk's set name; None when none is chosen. WorkflowError for a file that
        cannot be read or comes from a newer mnm (never read with fields dropped)."""
        with self._locked():
            if not self._files.exists(ENTRY):
                return None
            try:
                chosen = DeskFile.model_validate_json(self._files.read(ENTRY))
            except (ValidationError, OSError) as exc:
                raise WorkflowError(
                    f"The desk file {self.path} cannot be read ({exc}). `mnm sets default "
                    "<set>` writes it again; the sets are untouched."
                ) from exc
            if chosen.schema_version > SCHEMA_VERSION:
                raise WorkflowError(
                    f"The desk file {self.path} was written by a newer version of "
                    "makenomistakes. Run `mnm update`, then try again."
                )
            return chosen.set

    def choose(self, name: str) -> None:
        with self._locked():
            try:
                self._files.write(ENTRY, DeskFile(set=name).model_dump_json(indent=2))
            except OSError as exc:
                raise WorkflowError(
                    f"The desk file {self.path} could not be written ({cause(exc)})."
                ) from exc

    def clear(self) -> bool:
        """No desk any more; True when there was one."""
        with self._locked():
            if not self._files.exists(ENTRY):
                return False
            self._files.unlink(ENTRY)
            return True


def read_aliases(store: WorkflowStore, instance_names: Iterable[str]) -> Aliases:
    """The desk's aliases now. A desk file or set that cannot be read gives none, with the
    reason (`problem`): an alias must never resolve to a guess."""
    try:
        desk = store.desk.name()
        if desk is None:
            return Aliases()
        if not store.exists(desk):
            return Aliases(desk, problem=f"the desk set {desk} no longer exists")
        workflow = store.load(desk)
    except WorkflowError as exc:
        return Aliases(problem=str(exc))
    return aliases_of(workflow, instance_names)


def aliases_of(workflow: Workflow, instance_names: Iterable[str]) -> Aliases:
    """A set's roles as the desk's aliases: each role whose word no instance carries, and the
    ones an instance does carry (`shadowed`: the instance's name wins, the role is no id).
    `sets default` refuses a set with any shadowed role; one that meets a restored backup is
    reported (the manifest, the journal, `sets show`)."""
    names = set(instance_names)
    shadowed = tuple(role for role, _ in workflow.ordered() if role in names)
    members = {role: name for role, name in workflow.ordered() if role not in names}
    return Aliases(workflow.name, members, shadowed)


def shadowed_note(aliases: Aliases) -> str | None:
    """What a shadowed role means, in one line; None when none is."""
    if not aliases.shadowed:
        return None
    roles = ", ".join(aliases.shadowed)
    return (
        f"The desk's {roles} {'is' if len(aliases.shadowed) == 1 else 'are'} no model id: an "
        f"instance carries that name, which wins ({clash_remedy(aliases.shadowed[0])})."
    )


def clash_remedy(role: str) -> str:
    return (
        f"`mnm rm {role}`, its downloads stay; then `mnm add <its link> --name {role}-1`, and "
        "point the set's role at it"
    )


def clash_refusal(aliases: Aliases) -> str:
    """Why the set cannot be (or stay) the desk while an instance carries a role's name."""
    listed = ", ".join(aliases.shadowed)
    return (
        f"{aliases.desk} cannot be the desk: an instance is named {listed}, and the desk's "
        f"roles become model ids, which must never mean two things. Give that instance "
        f"another name ({clash_remedy(aliases.shadowed[0])}), then try again."
    )
