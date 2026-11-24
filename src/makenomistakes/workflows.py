"""Workflows: a named set of instances with a role each (docs/GUIDE.md §16).

A *studio* is a chat model, an image model and a voice; a *coding desk* a coder and a memory;
the *Odysseus full set* one of everything Odysseus can use. A workflow starts and stops its
members together, is planned to fit together on this card (the members' estimates against
the free VRAM, the ones already running counted as placed), and has one card per harness
listing every member under its role. Workflows are mnm's own concept: Odysseus, the agents
and the editors see the instances.
"""

from __future__ import annotations

from collections.abc import Callable

from makenomistakes.capacity import FitNow, Verdict
from makenomistakes.consumers import Consumer, HarnessUrl
from makenomistakes.desk import aliases_of, clash_refusal
from makenomistakes.desk_cards import desk_agent_cards
from makenomistakes.hardware.probe import MemorySnapshot
from makenomistakes.harness import consumer_urls
from makenomistakes.harness_agents import AGENT_CLIENTS, AGENT_MODALITIES
from makenomistakes.held_back import held_back_in
from makenomistakes.instance_types import InstanceError, Started
from makenomistakes.instances import InstanceService
from makenomistakes.recipes.model import Recipe, validate_instance_name
from makenomistakes.roles import ROLE_ORDER, ROLES
from makenomistakes.room_verdicts import ComfyRoom, together
from makenomistakes.set_members import (
    GONE,
    Member,
    WorkflowError,
    _taking_turns_together,
    _why_unknown,
    docker_error,
    start_order,
)
from makenomistakes.set_room import ComfyHooks, SetStarted, make_room, release_comfy
from makenomistakes.set_store import IN_USE, SCHEMA_VERSION, Workflow, WorkflowStore
from makenomistakes.supervisor import ContainerState
from makenomistakes.swap import SwapService, start_turn

__all__ = [  # `set_members`', `roles`' and `set_store`'s names, found here too
    "GONE",
    "IN_USE",
    "ROLES",
    "ROLE_ORDER",
    "SCHEMA_VERSION",
    "Member",
    "Workflow",
    "WorkflowError",
    "WorkflowStore",
    "docker_error",
    "start_order",
]

GIB = 2**30
Progress = Callable[[str], None]


def _persistent_ok(swap: bool, persistent: bool) -> None:
    if swap and persistent:
        raise WorkflowError(
            "A swap set's members take turns, parking one another: it cannot be persistent too."
        )


class WorkflowService:
    def __init__(
        self,
        instances: InstanceService,
        store: WorkflowStore | None = None,
        comfy: ComfyHooks | None = None,
    ) -> None:
        self.instances = instances
        self.store = store or WorkflowStore(instances.settings.expanded_state_dir())
        self.comfy = comfy or ComfyHooks.of(instances.settings)  # its card, freed first

    # --- read --------------------------------------------------------------------------

    def get(self, name: str) -> Workflow:
        return self.store.load(name)

    def all(self) -> list[Workflow]:
        return self.store.list()

    def members(self, workflow: Workflow, snapshot: MemorySnapshot | None = None) -> list[Member]:
        """Each member with its container state, estimate and fit verdict (one Docker round
        and one read of its files per member); a removed instance reads as gone."""
        snapshot = snapshot or self.instances.memory()
        # every instance's status, so a verdict names what holds the card (a group-mate makes a
        # parked member "takes turns", as in `mnm ls`)
        statuses = {s.recipe.name: s for s in self.instances.statuses()}
        every = list(statuses.values())
        held = held_back_in(self.store, every)
        verdicts = self.instances.verdicts(every, snapshot, held, None, self.comfy.room())
        members: list[Member] = []
        for role, name in workflow.ordered():
            status = statuses.get(name)
            if status is None:
                members.append(Member(role, name, None, GONE))
                continue
            verdict = verdicts[name]
            estimate = self.instances.held_estimate(status)
            members.append(Member(role, name, status, verdict, estimate))
        return members

    def fit_together(
        self,
        members: list[Member],
        snapshot: MemorySnapshot,
        comfy: ComfyRoom | None = None,
        owner: str = "the set",
    ) -> Verdict:
        """Whether the members that are not running fit the free VRAM beside the ones that
        are (`capacity.fit_together`, the same rules `mnm ls` applies to each one alone),
        what ComfyUI gives back at a start counted free (`comfy`, read now when not given);
        unknown when no sum can be trusted, running when nothing waits. `owner` names the
        set in the wording ("beside the desk's running members")."""
        unknown = _why_unknown(members)
        if unknown is not None:
            return unknown
        waiting = [(m.name, m.estimate) for m in members if not m.running and m.estimate]
        if not waiting:
            return Verdict(level=FitNow.RUNNING, note="Every member is running.")
        if any(m.status and m.status.recipe.swap_group for m in members):
            return _taking_turns_together(members)
        beside = tuple(m.name for m in members if m.running)
        room = self.comfy.room() if comfy is None else comfy
        return together(waiting, snapshot, room, beside, owner)

    def owner(self, workflow: Workflow) -> str:
        """How the wording names a set: "the desk" for the desk, else its name."""
        return "the desk" if self.store.desk.name() == workflow.name else workflow.name

    def listed(
        self, workflow: Workflow, members: list[Member], snapshot: MemorySnapshot
    ) -> Verdict:
        """`sets ls`' Fit together: `fit_together`, except the desk whose members fit one at a
        time though not together reads *on request* (the router loads each role at its
        request, parking the others; `mnm sets start` still refuses it)."""
        verdict = self.fit_together(members, snapshot, owner=self.owner(workflow))
        alone = all(m.verdict.level is not FitNow.NO_FIT for m in members if not m.running)
        if (
            verdict.level is not FitNow.NO_FIT
            or not alone
            or self.store.desk.name() != workflow.name
        ):
            return verdict
        return Verdict(
            level=FitNow.ON_REQUEST,
            note=f"{verdict.note} Members load one at a time on request: the router parks one "
            "role's model to load another's.",
        )

    def cards(
        self, workflow: Workflow, consumer: Consumer
    ) -> list[tuple[str, Recipe, list[HarnessUrl]]]:
        """One card set per member that exists, under its role: every URL in one place. A
        member a coding agent cannot talk to (the memory, the image model) gets its curl
        card, not the agent's "needs a chat model"."""
        if consumer in AGENT_CLIENTS and self._is_desk(workflow):
            return desk_agent_cards(workflow.members, consumer, self.instances.find)
        out: list[tuple[str, Recipe, list[HarnessUrl]]] = []
        for role, name in workflow.ordered():
            recipe = self.instances.find(name)
            if recipe is None:
                continue  # the members table says it is gone
            for_member = consumer
            if consumer in AGENT_CLIENTS and recipe.modality not in AGENT_MODALITIES:
                for_member = Consumer.CURL
            out.append((role, recipe, consumer_urls(recipe, for_member)))
        return out

    def _is_desk(self, workflow: Workflow) -> bool:
        try:
            return self.store.desk.name() == workflow.name
        except WorkflowError:
            return False

    # --- write -------------------------------------------------------------------------

    def create(
        self, name: str, members: dict[str, str], swap: bool = False, persistent: bool = False
    ) -> Workflow:
        """A new workflow: a valid name, known roles, existing instances of the role's type,
        no instance twice; with `swap`, none already taking turns in another workflow.
        Nothing is started."""
        _persistent_ok(swap, persistent)
        try:
            validate_instance_name(name)
        except ValueError as exc:
            raise WorkflowError(str(exc).replace("instance name", "set name")) from exc
        if self.store.exists(name):
            raise WorkflowError(f"A set named '{name}' already exists.")
        seen = self._checked(name, members, swap)
        workflow = Workflow(name=name, members=dict(members), swap=swap, persistent=persistent)
        marked: list[str] = []
        try:  # the members first: a workflow file is saved only once every one is marked
            for instance in seen if swap else ():
                self.instances.change(instance, _in_group(name))
                marked.append(instance)
            self.store.save(workflow)
        except Exception:  # a store error too: no member stays marked for a missing workflow
            for instance in marked:
                self.instances.change(instance, _in_group(None))
            raise
        return workflow

    def edit(
        self, name: str, given: dict[str, str], drop: list[str], persistent: bool | None = None
    ) -> Workflow:
        """Change a set's members: `given` roles set to these instances (added or
        replaced), `drop` roles taken out; the same checks as a new set. In a swap set a member
        that leaves is stopped first and leaves the group; one that joins must not be running
        (it would hold the card outside the turns)."""
        workflow = self.store.load(name)
        if persistent is not None:  # `--persistent` / `--no-persistent`
            _persistent_ok(workflow.swap, persistent)
            workflow = workflow.model_copy(update={"persistent": persistent})
        unknown = [role for role in drop if role not in workflow.members]
        if unknown:
            raise WorkflowError(f"{name} has no {', '.join(unknown)} to drop.")
        members = {r: i for r, i in workflow.members.items() if r not in drop} | given
        self._checked(name, members, workflow.swap)
        if self.store.desk.name() == name:  # a new role becomes a model id: never a name's
            self._desk_ok(workflow.model_copy(update={"members": members}))
        before, after = set(workflow.members.values()), set(members.values())
        joining = sorted(after - before) if workflow.swap else []
        running = [i for i in joining if self.instances.status(self.instances.get(i)).running]
        if running:
            raise WorkflowError(
                f"{', '.join(running)} is running: stop it before it joins a set whose "
                "members take turns."
            )
        edited = workflow.model_copy(update={"members": members})
        marked: list[str] = []
        try:  # as `create`: joiners marked, then the file; a failure takes the marks back
            for instance in joining:
                self.instances.change(instance, _in_group(name))
                marked.append(instance)
            self.store.save(edited)
        except Exception:
            for instance in marked:
                self.instances.change(instance, _in_group(None))
            raise
        # The set no longer lists the leavers: out of the group first, then stopped (one that
        # fails to stop runs as any instance does, not inside the turns)
        if workflow.swap:
            for instance in sorted(before - after):
                self.instances.change(instance, _in_group(None))
                self.instances.stop(self.instances.get(instance))
        return edited

    def _checked(self, name: str, members: dict[str, str], swap: bool) -> dict[str, str]:
        """Known roles, existing instances of the role's type, no instance twice; with `swap`,
        none already taking turns in another set. Each instance by its role."""
        if not members:
            raise WorkflowError("A set needs at least one member: `role=instance`.")
        seen: dict[str, str] = {}
        for role, instance in members.items():
            if role not in ROLES:
                raise WorkflowError(
                    f"'{role}' is not a role; the roles are: {', '.join(ROLE_ORDER)}."
                )
            recipe = self.instances.get(instance)  # InstanceError names the instances there are
            if recipe.modality not in ROLES[role]:
                fits = ", ".join(m.label for m in sorted(ROLES[role]))
                raise WorkflowError(
                    f"{instance} is {recipe.modality.a_label} model and cannot be the {role} "
                    f"(that role takes: {fits})."
                )
            if instance in seen:
                raise WorkflowError(
                    f"{instance} is listed twice ({seen[instance]} and {role}); an instance plays "
                    "one role."
                )
            if swap and recipe.swap_group not in (None, name):
                raise WorkflowError(
                    f"{instance} already takes turns in the set {recipe.swap_group}; an "
                    "instance swaps in one set (`mnm sets rm` that one first)."
                )
            seen[instance] = role
        return seen

    def remove(self, name: str) -> None:
        """The instances stay: a workflow is only the grouping. A swap workflow's members are
        stopped and leave their group before the file goes, so a failure part-way leaves the
        workflow there to remove again."""
        workflow = self.store.load(name)
        for member in self.members(workflow) if workflow.swap else []:
            if member.missing or member.recipe.swap_group != name:
                continue
            self.instances.stop(member.recipe)
            self.instances.change(member.name, _in_group(None))
        self.store.delete(name)
        if self.store.desk.name() == name:  # its roles name nothing now
            self.store.desk.clear()

    def choose_desk(self, name: str) -> Workflow:
        """`mnm sets default <name>`: its roles become model ids (`desk.Aliases`); refused
        while an instance is named after one of them."""
        workflow = self.store.load(name)
        self._desk_ok(workflow)
        self.store.desk.choose(name)
        return workflow

    def _desk_ok(self, workflow: Workflow) -> None:
        found = aliases_of(workflow, (r.name for r in self.instances.all()))
        if found.shadowed:
            raise WorkflowError(clash_refusal(found))

    def start(
        self,
        workflow: Workflow,
        progress: Progress | None = None,
        anyway: bool = False,
        room: bool = True,
    ) -> SetStarted:
        """Start every member that is not running, the largest estimate first (what cannot fit
        fails before the small ones take their room). A gone member refuses before anything
        starts. When the waiting members do not fit, mnm's running instances outside the set
        are parked first as the router would park them (`set_room`; not with `room` False);
        then a won't-fit verdict refuses (unless `anyway`). A failure mid-way stops the walk
        and names what started, and the members already up stay up."""
        members = self.members(workflow, self.instances.memory())
        gone = [m for m in members if m.missing]
        if gone:
            names = ", ".join(f"{m.name} ({m.role})" for m in gone)
            raise WorkflowError(
                f"{workflow.name}: {names} no longer exist. Remove the set or `mnm add` them."
            )
        if workflow.swap:
            return SetStarted(self._start_swapping(workflow, members, progress))
        with start_turn(self.instances.settings, workflow.name, progress):  # one at a time
            if room and release_comfy(self.comfy, workflow.name, progress, members):
                members = self.members(workflow, self.instances.memory())
            parked = make_room(self, workflow, members, progress, anyway, room)
            if parked:  # what the card holds now
                members = self.members(workflow, self.instances.memory())
            self.store.mark_started(workflow.name, True)  # its members hold one another now
            return SetStarted(self._start_together(workflow, members, progress), parked)

    def _start_together(
        self, workflow: Workflow, members: list[Member], progress: Progress | None
    ) -> list[Started]:
        """A plain workflow's members, the largest estimate first, under the box's lock. Each
        one's `off` is cleared by its own start once it is up (`instance_start`): a member
        whose start fails, or one never reached, keeps the flag it had."""
        order = start_order(members)
        started: list[Started] = []
        for member in order:
            if progress:
                progress(f"Starting {member.name} ({member.role})…")
            try:
                started.append(self.instances.start(member.recipe, progress))
            except InstanceError as exc:
                up = ", ".join(s.recipe.name for s in started) or "nothing new"
                raise WorkflowError(
                    f"{workflow.name}: {member.name} ({member.role}) did not start: {exc} "
                    f"Started before it: {up}; they stay up.",
                    exc.diagnoses,
                    exc.fixes,
                ) from exc
        return started

    def _clear_off(self, members: list[Member]) -> None:
        """Started as a swap workflow, the members it leaves parked are no longer stopped by
        the user: they load on request. Called once the first one is up."""
        for member in members:
            self.instances.mark_off(member.recipe, False)

    def _start_swapping(
        self, workflow: Workflow, members: list[Member], progress: Progress | None
    ) -> list[Started]:
        """The first member (the language model, in role order) loads now; the router loads
        the others at their first request."""
        swap = SwapService(self.instances)
        first, rest = members[0], members[1:]
        if first.running:
            self._clear_off(members)  # the parked members load on request from now on
            return []
        if progress:
            progress(f"Starting {first.name} ({first.role}); the others start on demand…")
        try:
            started = swap.start(first.recipe, progress).started  # clears its own `off`
        except InstanceError as exc:  # nothing is made loadable by a start that failed
            raise WorkflowError(
                f"{workflow.name}: {first.name} ({first.role}) did not start: {exc}",
                exc.diagnoses,
                exc.fixes,
            ) from exc
        self._clear_off(rest)  # the parked members load on request from now on
        return [started]

    def stop(self, workflow: Workflow) -> list[str]:
        """Stop every member that exists and has a container; the names stopped. A gone
        member is skipped (there is nothing of it to stop)."""
        self.store.mark_started(workflow.name, False)  # no longer held as a set
        present = [m.name for m in self.members(workflow) if not m.missing]
        stopped = self.instances.stop_many(present) if present else []
        return [s.name for s in stopped if s.was is not ContainerState.ABSENT]


def _in_group(group: str | None) -> Callable[[Recipe], Recipe]:
    return lambda recipe: recipe.model_copy(update={"swap_group": group})
