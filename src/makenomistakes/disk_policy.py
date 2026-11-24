"""The disk policy: on by default at 85% of the models folder's disk,
`mnm disk keep --models 85%|400G|off` in `<state>/disk/disk.json`.

`check()` is the one evaluation: at every `mnm add` and download end, every start, the doctor
and the nightly. From 15 points under the limit the doctor's Model storage line, `mnm
add` and `mnm ls` WARN, and one notification goes at the first crossing of that line and of 5
points under (again only after a drop below). At the limit the least recently used instances'
own files go (`model_cleanup`'s rules: never a file another recipe, LoRA, draft or workflow
lists, only in the download areas), one instance at a time until the gauge is 5 points under
the limit; never a running or pinned instance's, a member of a set in use or persistent, one
used in the last 24 h (`disk_last_used`), or the instance a start or download is for. An
evicted instance stays saved with `local_paths` cleared and `Recipe.evicted` set; its next
start downloads first. When what may go would not bring the gauge under the limit, nothing
goes and the outcome says why: the policy never deletes for nothing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.comfy.models import ModelError, Models
from makenomistakes.disk_last_used import LastUse, last_use
from makenomistakes.disk_settings import (
    DiskSettings,
    DiskSettingsError,
    Usage,
    gauge_for,
    limit_words,
    warning_words,
    when_words,
)
from makenomistakes.disk_types import (
    KEEP_EVICTIONS,
    DiskFile,
    EvictionRecord,
    Gauge,
    Keep,
    Level,
    PolicyState,
    gb,
)
from makenomistakes.held_back import held_back_in
from makenomistakes.instance_types import InstanceError, InstanceStatus
from makenomistakes.journal import to_stderr
from makenomistakes.locking import shared_lock
from makenomistakes.model_cleanup import delete_own_files
from makenomistakes.notify import Notice, notifier
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import Eviction
from makenomistakes.slot_folder import drop_oldest
from makenomistakes.store import StoreError
from makenomistakes.supervisor import ContainerState, ContainerStatus, SupervisorError

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


@dataclass(frozen=True)
class Candidate:
    recipe: Recipe
    used: LastUse
    freeable: int  # its own files' bytes: what evicting it gives back
    files: int


@dataclass
class Judged:
    candidates: list[Candidate] = field(default_factory=list)  # least recently used first
    held: list[str] = field(default_factory=list)  # "qwen3-8b: running", in words


@dataclass
class Outcome:
    """One evaluation: what the gauge read, the warning, what went, why nothing could."""

    keep: Keep
    gauge: Gauge | None
    warning: str | None = None
    evicted: list[EvictionRecord] = field(default_factory=list)
    refused: str | None = None
    lines: list[str] = field(default_factory=list)  # every journal line it wrote
    dropped_bytes: int = 0  # saved prompts, gone before any instance's files

    @property
    def level(self) -> Level:
        return self.gauge.level if self.gauge is not None else Level.OK


class Refused(Exception):  # noqa: N818 - a reason in words, raised inside `check` only
    pass


class DiskPolicy:
    def __init__(
        self,
        service: InstanceService,
        notify: Callable[[Notice], object] | None = None,
        say: Callable[[str], None] = to_stderr,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        usage: Usage | None = None,
    ) -> None:
        self.service = service
        self.settings = DiskSettings(service.settings.expanded_state_dir())
        self._notify = notify or notifier(service.settings).send
        self._say = say
        self._now = now
        self._usage = usage

    @property
    def models_dir(self) -> Path:
        return self.service.models_dir

    def gauge(self, keep: Keep | None = None) -> Gauge | None:
        keep = keep if keep is not None else self.settings.read().keep
        return gauge_for(keep, self.models_dir, self._usage)

    def state(self, keep_out: frozenset[str] = frozenset()) -> PolicyState:
        """What `mnm disk` shows: the limit, the gauge, who would go next and who stays."""
        saved = self.settings.read()
        gauge = gauge_for(saved.keep, self.models_dir, self._usage)
        level = gauge.level if gauge is not None else Level.OK
        found = PolicyState(keep=saved.keep, gauge=gauge, level=level, evictions=saved.evictions)
        if not saved.keep.on:
            return found
        try:
            judged = self._judge(keep_out)
        except Refused as exc:
            found.held = [str(exc)]
            return found
        found.next = [c.recipe.name for c in judged.candidates]
        found.held = judged.held
        return found

    def check(self, keep_out: frozenset[str] = frozenset()) -> Outcome:
        """Measure, warn, notify once per crossing, and at the limit evict. Never raises for
        the start or download it runs beside: what went wrong is the outcome's `refused`."""
        try:
            saved = self.settings.read()
        except DiskSettingsError as exc:
            return Outcome(Keep(), None, refused=f"The disk policy is paused: {exc}.")
        outcome = Outcome(saved.keep, gauge_for(saved.keep, self.models_dir, self._usage))
        gauge = outcome.gauge
        if gauge is None:
            return outcome
        if gauge.level is not Level.OK:
            outcome.warning = warning_words(gauge)
        self._tell_crossings(gauge)
        if gauge.level is Level.OVER:
            self._evict(gauge, keep_out, outcome)
        if outcome.evicted or outcome.dropped_bytes:  # what the gauge reads now, by what went
            freed = sum(record.freed_bytes for record in outcome.evicted) + outcome.dropped_bytes
            after = gauge.after(freed)
            outcome.gauge = after
            outcome.warning = warning_words(after) if after.level is not Level.OK else None
        return outcome

    # --- warnings ----------------------------------------------------------------------

    def _tell_crossings(self, gauge: Gauge) -> None:
        """One notice per warning line crossed upwards; a line fallen under is forgotten, so
        the next crossing tells again."""
        lines = gauge.warnings
        past = [line for line in lines if gauge.percent >= line]
        told: list[float] = []

        def remember(saved: DiskFile) -> DiskFile:
            before = [n for n in saved.notified if n in lines or n == gauge.limit]
            told.extend(line for line in past if line not in before)
            kept = [n for n in before if gauge.percent >= n]
            return saved.model_copy(update={"notified": sorted({*kept, *past})})

        try:
            self.settings.update(remember)
        except (DiskSettingsError, OSError):
            return
        if told:
            self._notify(Notice("mnm disk: the models folder is filling", warning_words(gauge)))

    # --- eviction ----------------------------------------------------------------------

    def _evict(self, gauge: Gauge, keep_out: frozenset[str], outcome: Outcome) -> None:
        state_dir = self.service.settings.expanded_state_dir()
        with shared_lock(state_dir / "swap").attempt() as ours:  # no start meanwhile
            if not ours:
                outcome.refused = (
                    "A start is under way (it holds the box's start lock): the disk policy "
                    "evicts nothing until the next check."
                )
                return
            gauge = self._drop_saved_prompts(gauge, outcome)
            if gauge.level is not Level.OVER:
                return
            try:
                judged = self._judge(keep_out)
                chosen = self._choose(gauge, judged)
                linked = self._linked()
            except Refused as exc:
                outcome.refused = str(exc)
                self._tell_refusal(gauge, str(exc))
                return
            for candidate in chosen:
                record = self._evict_one(candidate, gauge, linked)
                if record is None:
                    continue
                gauge = gauge.after(record.freed_bytes)
                outcome.evicted.append(record)
                line = (
                    f"Evicted {record.name}'s files ({gb(record.freed_bytes)}, last used "
                    f"{when_words(record.last_used)}): the disk policy's limit is "
                    f"{outcome.keep.words}; it downloads them again at its next start."
                )
                outcome.lines.append(line)
                self._say(f"mnm disk: {line}")
                self._notify(Notice(f"mnm disk: {record.name}'s files went", line))

    def _drop_saved_prompts(self, gauge: Gauge, outcome: Outcome) -> Gauge:
        """The saved prompts go first, oldest first, toward the target: a cache, never
        a download the next start waits for."""
        dropped, freed = drop_oldest(self.models_dir, gauge.bytes_over(gauge.target))
        if not dropped:
            return gauge
        outcome.dropped_bytes += freed
        line = (
            f"Dropped {dropped} saved prompt{'s' if dropped != 1 else ''} ({gb(freed)}): the "
            f"disk policy's limit is {outcome.keep.words}; the next stop saves again."
        )
        outcome.lines.append(line)
        self._say(f"mnm disk: {line}")
        return gauge.after(freed)

    def _choose(self, gauge: Gauge, judged: Judged) -> list[Candidate]:
        """The least recently used candidates whose files bring the gauge under the target;
        Refused when all of them would not even bring it under the limit."""
        need_limit = gauge.bytes_over(gauge.limit)
        need_target = gauge.bytes_over(gauge.target)
        if sum(c.freeable for c in judged.candidates) < need_limit:
            why = "; ".join(judged.held) or "no instance has files of its own to give back"
            said = gauge.subject[0].upper() + gauge.subject[1:]
            raise Refused(
                f"{said}, past the disk policy's {limit_words(gauge)}, and it deleted nothing: "
                f"the files it may delete would not bring it under the limit ({why}). What else "
                "fills the disk is listed by `mnm disk`."
            )
        chosen, freed = [], 0
        for candidate in judged.candidates:
            if freed >= need_target:
                break
            chosen.append(candidate)
            freed += candidate.freeable
        return chosen

    def _tell_refusal(self, gauge: Gauge, why: str) -> None:
        """A refusal is said once per crossing of the limit, like the warnings."""
        first = False

        def remember(saved: DiskFile) -> DiskFile:
            nonlocal first
            first = gauge.limit not in saved.notified
            return saved.model_copy(update={"notified": sorted({*saved.notified, gauge.limit})})

        try:
            self.settings.update(remember)
        except (DiskSettingsError, OSError):
            return
        self._say(f"mnm disk: {why}")
        if first:
            self._notify(Notice("mnm disk: over the limit, nothing deleted", why, failed=True))

    def _judge(self, keep_out: frozenset[str]) -> Judged:
        """Every saved instance, as a candidate in least-recently-used order or held with why.
        Refused when what runs or what the sets hold cannot be known."""
        service = self.service
        try:
            recipes = service.all()
        except InstanceError as exc:
            raise Refused(f"The saved instances cannot be read ({exc}).") from exc
        containers: dict[str, ContainerStatus] = {}
        try:
            for recipe in recipes:
                containers[recipe.name] = service.supervisor.status(recipe.name)
        except SupervisorError as exc:
            raise Refused(
                f"Docker could not be asked which instances run ({exc}), so none is evicted."
            ) from exc
        statuses = [
            InstanceStatus(r, containers[r.name].state, files_present=bool(r.local_paths))
            for r in recipes
        ]
        from makenomistakes.workflows import WorkflowStore  # noqa: PLC0415 - imports instances

        kept = held_back_in(WorkflowStore(service.settings.expanded_state_dir()), statuses)
        if kept is None:
            raise Refused("The sets cannot be read, so which members are in use is unknown.")
        linked = self._linked()
        now = self._now()
        judged = Judged()
        for recipe in recipes:
            container = containers[recipe.name]
            used = last_use(
                recipe,
                service.measurements.readings(recipe.name),
                service.store.path(recipe.name),
                container,
                now,
            )
            why = self._held(recipe, container, kept, keep_out, used, now)
            if why is not None:
                judged.held.append(f"{recipe.name}: {why}")
                continue
            others = [r for r in recipes if r.name != recipe.name]
            own = delete_own_files(recipe, others, self.models_dir, linked, dry_run=True)
            if own.freed_bytes == 0:
                shared = "its files are shared" if own.shared else "nothing of its own on disk"
                judged.held.append(f"{recipe.name}: {shared}")
                continue
            judged.candidates.append(Candidate(recipe, used, own.freed_bytes, len(own.deleted)))
        judged.candidates.sort(key=lambda c: c.used.at or datetime.min.replace(tzinfo=UTC))
        return judged

    @staticmethod
    def _held(
        recipe: Recipe,
        container: ContainerStatus,
        kept: frozenset[str],
        keep_out: frozenset[str],
        used: LastUse,
        now: datetime,
    ) -> str | None:
        """Why the policy leaves this instance's files, or None: the first rule that holds."""
        evicted = recipe.evicted
        rules = (
            (recipe.name in keep_out, "starting or downloading now"),
            (container.state is ContainerState.RUNNING, "running"),
            (recipe.pinned, "pinned (`mnm unpin` lets it go)"),
            (recipe.name in kept, "a member of a set in use or persistent"),
            (
                evicted is not None and not recipe.local_paths,
                f"evicted already on {when_words(evicted.at if evicted else None)}",
            ),
            (used.recent(now), f"used in the last 24 h ({used.source}, {when_words(used.at)})"),
        )
        return next((why for holds, why in rules if holds), None)

    def _linked(self) -> frozenset[str]:
        """The files ComfyUI workflows use; Refused when the ledger cannot be read (as `mnm
        rm --files` keeps every file then)."""
        state_dir = self.service.settings.expanded_state_dir()
        try:
            ledger = Models(self.models_dir, state_dir).ledger()
        except ModelError as exc:
            raise Refused(f"Which files ComfyUI workflows use is unknown ({exc}).") from exc
        return frozenset(item.target for item in ledger.links)

    def _evict_one(
        self, candidate: Candidate, gauge: Gauge, linked: frozenset[str]
    ) -> EvictionRecord | None:
        """Mark the recipe evicted, then delete what only it lists (a death between the two
        leaves files a start finds again, never a recipe that claims files it lost)."""
        service = self.service
        name = candidate.recipe.name
        now = self._now()
        try:
            before = service.store.load(name)
            others = [r for r in service.all() if r.name != name]
            service.store.update(
                name,
                lambda r: r.model_copy(
                    update={
                        "local_paths": {},
                        "evicted": Eviction(
                            at=now, freed_bytes=candidate.freeable, last_used=candidate.used.at
                        ),
                    }
                ),
            )
            done = delete_own_files(before, others, self.models_dir, linked)
        except (StoreError, InstanceError, OSError) as exc:
            self._say(f"mnm disk: {name}'s files were not evicted: {exc}")
            return None
        record = EvictionRecord(
            name=name,
            at=now,
            freed_bytes=done.freed_bytes,
            files=len(done.deleted),
            last_used=candidate.used.at,
            usage_before=round(gauge.percent, 1),
            limit=gauge.limit,
            mode=gauge.mode,
            limit_words=limit_words(gauge),
        )
        try:  # the files are gone: the eviction stands in the outcome whatever these say
            if done.freed_bytes != candidate.freeable:
                service.store.update(
                    name,
                    lambda r: r.model_copy(
                        update={
                            "evicted": Eviction(
                                at=now, freed_bytes=done.freed_bytes, last_used=candidate.used.at
                            )
                        }
                    ),
                )
            self.settings.update(
                lambda saved: saved.model_copy(
                    update={"evictions": [*saved.evictions, record][-KEEP_EVICTIONS:]}
                )
            )
        except (StoreError, DiskSettingsError, OSError) as exc:
            self._say(
                f"mnm disk: {name}'s files were evicted, but the record of it was not saved: {exc}"
            )
        return record


def check(
    service: InstanceService, keep_out: frozenset[str] = frozenset(), **seams: object
) -> Outcome:
    """The policy's one evaluation, for every caller (the nightly included); never
    raises: a policy that cannot judge deletes nothing and says why."""
    try:
        return DiskPolicy(service, **seams).check(keep_out)  # type: ignore[arg-type]
    except Exception as exc:  # the start or download beside it must go on
        return Outcome(Keep(), None, refused=f"The disk policy could not run: {exc}")


def fetched(service: InstanceService, name: str) -> Recipe:
    """After a download ends: the evicted mark goes once every file is back, then the policy
    runs (the instance itself held: it was just fetched)."""
    recipe = service.get(name)
    if recipe.evicted is not None and service.files_present(recipe):
        recipe = service.change(name, lambda saved: saved.model_copy(update={"evicted": None}))
    check(service, frozenset({name}))
    return recipe
