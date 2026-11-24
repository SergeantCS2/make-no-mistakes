"""`mnm upgrade <name> --yes` and `--replace` (added beside, the old kept as `<name>-old`
until `--replace`).

`--yes`: the old instance is parked after its replies (as `mnm stop --park` does) when it
has a container, renamed `<name>-old` (`instance_rename`), and the new one saved under the
name, then downloaded. The desk and the sets name instances, so they now reach the new model
under the old name: said, never rewritten. `--replace` removes `<name>-old` through `mnm rm
--files`' rules (nothing another recipe, a draft or a ComfyUI workflow uses)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from makenomistakes.instance_rename import rename
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.model_cleanup import Cleanup
from makenomistakes.recipes.model import Recipe, validate_instance_name
from makenomistakes.supervisor import ContainerState, SupervisorError
from makenomistakes.swap import start_turn
from makenomistakes.tune.store import QualityQueued, TuneStore, TuneStoreError
from makenomistakes.upgrade import Upgrade
from makenomistakes.workflows import WorkflowError, WorkflowStore

OLD = "-old"
Progress = Callable[[str], None]
Park = Callable[[Recipe], None]  # `SwapService.park`: after its replies, loadable again


def old_name(name: str) -> str:
    return f"{name}{OLD}"


def refusal(service: InstanceService, upgrade: Upgrade) -> str | None:
    """Why `--yes` cannot add it beside the old one, in words; None when it can."""
    old = upgrade.old
    target = old_name(old.name)
    if not old.downloaded or old.evicted is not None:
        gone = "its files were evicted by the disk policy" if old.evicted else "not downloaded"
        return (
            f"{old.name} is {gone}: there is nothing to keep beside the new model. `mnm rm "
            f"{old.name}`, then `mnm add {upgrade.successor.repo} --name {old.name}`, does the "
            "same with nothing to undo."
        )
    try:
        validate_instance_name(target)
    except ValueError:
        return (
            f"{old.name} is too long to keep as {target} (64 characters at most): `mnm rm` it "
            "and `mnm add` the new model instead."
        )
    if service.store.exists(target):
        return (
            f"{target} is already there (an earlier upgrade's): `mnm upgrade {old.name} "
            f"--replace` removes it first, or `mnm rm {target}`."
        )
    return _tune_refusal(service, old)


def _tune_refusal(service: InstanceService, old: Recipe) -> str | None:
    """A tune of the old name running or a quality plan of it queued holds the upgrade."""
    store = TuneStore(_state(service))
    try:  # both by the old name: a folder that cannot be listed hides neither
        queued, state = store.queued(old.name), store.state(old.name)
    except TuneStoreError as exc:
        return f"{old.name}'s tuning files cannot be read ({exc}): nothing was changed."
    if state is not None and state.run_beside(None) is not None:
        return f"a tune of {old.name} is running: `mnm tune {old.name} --cancel` first, or wait."
    if isinstance(queued, QualityQueued):  # Its plan measures the old one's quants
        return (
            f"{old.name} has a quality tune queued, planned on what it runs now: `mnm tune "
            f"{old.name} --cancel` first (its downloads go with it), then upgrade."
        )
    return None


def _state(service: InstanceService) -> Path:
    return service.settings.expanded_state_dir()


@dataclass(frozen=True)
class Added:
    new: Recipe
    old: Recipe  # under its new name
    parked: bool  # it was loaded, and was parked after its replies
    sets: list[str] = field(default_factory=list)  # what names the instance, in words
    notes: list[str] = field(default_factory=list)


def add_beside(service: InstanceService, upgrade: Upgrade, park: Park) -> Added:
    """The old one parked and renamed, the new one saved under the name. Not downloaded:
    `download` does that (a long step the CLI shows)."""
    name = upgrade.old.name
    current = service.get(name)
    if current.model_dump(exclude={"local_paths"}) != upgrade.old.model_dump(
        exclude={"local_paths"}
    ):
        raise InstanceError(f"{name} changed while it was planned: run `mnm upgrade {name}` again.")
    refused = refusal(service, dataclasses.replace(upgrade, old=current))  # its files now
    if refused is not None:
        raise InstanceError(refused)
    try:
        state = service.supervisor.status(name).state
    except SupervisorError as exc:
        raise InstanceError(
            f"{name} was left as it is: Docker could not be asked ({exc})."
        ) from exc
    if state is not ContainerState.ABSENT:
        park(current)
    target = old_name(name)
    with start_turn(service.settings, name):  # no start of either name meanwhile
        renamed = rename(service, service.get(name), target)
        try:
            new = service.create(
                upgrade.new.modality, lambda port: upgrade.new.model_copy(update={"port": port})
            )
        except InstanceError:
            rename(service, renamed, name)  # put it back: both names, never neither
            raise
        if renamed.swap_group is not None:  # the set's member is the name: the new one now
            renamed = service.change(target, lambda r: r.model_copy(update={"swap_group": None}))
    return Added(
        new=new,
        old=renamed,
        parked=state is ContainerState.RUNNING,
        sets=set_words(service, name, upgrade.successor.label, target),
        notes=_draft_notes(service, name, target),
    )


def set_words(
    service: InstanceService, name: str, label: str, target: str, done: bool = True
) -> list[str]:
    """Each set (and the desk) that names the instance, in a line: it reaches the new model
    now. A set file that cannot be read is said, never fatal (the upgrade is done)."""
    store = WorkflowStore(_state(service))
    try:
        sets = store.list()
        desk = store.desk.name()
    except WorkflowError as exc:
        reach = "reach" if done else "would reach"
        return [f"The sets could not be read ({exc}); any that name {name} {reach} {label}."]
    lines = []
    for chosen in sets:
        roles = [role for role, member in chosen.members.items() if member == name]
        if not roles:
            continue
        which = f"The desk ('{desk}')" if chosen.name == desk else f"The set '{chosen.name}'"
        now = "now" if done else "would be"
        lines.append(f"{which}: {', '.join(roles)} is {name}, {now} {label} ({target} in no set).")
    return lines


def _draft_notes(service: InstanceService, name: str, target: str) -> list[str]:
    """Other instances drafting with the old model's file keep that file (now `target`'s)."""
    users = sorted(
        r.name
        for r in service.all()
        if r.draft is not None and not r.draft.is_mtp and r.draft.instance == name
    )
    if not users:
        return []
    return [
        f"{', '.join(users)} drafts with the old model's file, which stays as {target}'s "
        f"(`mnm set {users[0]} draft={name}` drafts with the new one, if its vocabulary agrees)."
    ]


def download(service: InstanceService, added: Added, progress: Progress | None) -> Recipe:
    """The new model's files; a download cut short keeps what came (`mnm start` resumes)."""
    try:
        return service.download(added.new, progress)
    except InstanceError as exc:
        raise InstanceError(
            f"{added.new.name} is saved as the new model, and its download stopped: {exc} "
            f"`mnm start {added.new.name}` fetches the rest; {added.old.name} is untouched."
        ) from exc


def replace(service: InstanceService, name: str, park: Park) -> tuple[Recipe, Cleanup | None]:
    """`<name>-old` removed with its files (`mnm rm --files`' rules); refused with why when
    there is none."""
    target = old_name(name)
    old = service.find(target)
    if old is None:
        raise InstanceError(
            f"There is no {target} to remove: `mnm upgrade {name}` keeps the old model as "
            f"{target}, and `--replace` removes it once the new one has proved itself."
        )
    try:
        state = service.supervisor.status(target).state
    except SupervisorError as exc:
        raise InstanceError(f"{target} was kept: Docker could not be asked ({exc}).") from exc
    if state is ContainerState.RUNNING:
        park(old)
    return old, service.remove(service.get(target), files=True)
