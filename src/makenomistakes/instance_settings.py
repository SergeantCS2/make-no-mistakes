"""`mnm set` for an instance: the settings saved through the store (`apply_settings`), with
the two that need more than the recipe: a draft model (another instance) and a MoE's
experts across cards (`experts=`, `card-experts=`), which are placed again from the
model's own files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.card_settings import CARD_EXPERTS_CONTROL, EXPERTS_CONTROL
from makenomistakes.controls import SettingsError, apply_settings
from makenomistakes.draft import DraftError, with_draft
from makenomistakes.draft_room import checked
from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.fit.estimate import DEFAULT_UBATCH, estimate_memory
from makenomistakes.fit.placement import Mode, refill
from makenomistakes.fit.ram_floor import OS_RAM_MARGIN
from makenomistakes.hardware.numa_nodes import read_nodes
from makenomistakes.hub.facts import EMBEDDING_MODALITIES
from makenomistakes.instance_types import InstanceError
from makenomistakes.machine_facts import MachineFactsError, expert_rate, load_file, platform_here
from makenomistakes.model_files import read_model_files
from makenomistakes.numa_setting import NumaFacts
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.llamacpp_cache import cache_folder, keeps_slots, remove_cache_folder
from makenomistakes.setting_room import checked_settings
from makenomistakes.store import StoreError
from makenomistakes.studio import style_use

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService

MODES: dict[str, Mode] = {"auto": "auto", "ram": "ram", "cards": "cards"}


class _MovedError(Exception):
    """The saved recipe changed while its new settings were worked out."""


TRIES = 3  # attempts outside the lock before the change is worked out under it


def update_settings(service: InstanceService, recipe: Recipe, updates: dict[str, str]) -> Recipe:
    """Change and save `recipe`'s settings; InstanceError says why not. The change is worked
    out outside the store's lock (it asks Docker and nvidia-smi: every other store write, a
    download's checkpoint or `mnm set` in another terminal, waited on them) and
    saved only if the file did not move meanwhile; else worked out again."""

    def draft(current: Recipe, raw: str) -> Recipe:
        return checked(service, current, with_draft(current, raw, service.find, service.models_dir))

    def change(current: Recipe) -> Recipe:
        applied = apply_settings(
            current,
            updates,
            draft,
            lambda: service.memory().listed(),
            lambda now, wanted: placed_again(service, now, wanted),
            lambda now: numa_facts(service, now),
        )
        style_use.check(service.settings, current, applied)  # A style it can play
        return checked_settings(service, current, applied)  # a raised context, the MoE cache

    try:
        for _ in range(TRIES):
            seen = service.store.load(recipe.name)
            updated = change(seen)

            def unless_moved(
                current: Recipe, seen: Recipe = seen, done: Recipe = updated
            ) -> Recipe:
                if current != seen:
                    raise _MovedError
                return done

            try:
                return _cache_off(service, seen, service.store.update(recipe.name, unless_moved))
            except _MovedError:
                continue
        before = service.store.load(recipe.name)
        return _cache_off(service, before, service.store.update(recipe.name, change))
    except (SettingsError, DraftError, StoreError) as exc:
        raise InstanceError(str(exc)) from exc


def _cache_off(service: InstanceService, before: Recipe, after: Recipe) -> Recipe:
    """`cache=off`: the saved slots go with it, a conversation's prompt kept for
    nothing; the running engine keeps its own RAM cache until its next start."""
    if keeps_slots(before) and not keeps_slots(after):
        try:
            remove_cache_folder(after, service.models_dir)
        except OSError as exc:  # the setting is saved; the folder is said, never left quietly
            raise SettingsError(
                f"`cache=off` is saved, but {after.name}'s saved prompts could not be deleted "
                f"({exc}); `rm -r {cache_folder(service.models_dir, after.name)}` deletes them."
            ) from exc
    return after


def placed_again(service: InstanceService, recipe: Recipe, wanted: dict[str, str]) -> Recipe:
    """`experts=` and `card-experts=`: the saved placement placed again from the
    model's header, against each card's free VRAM when it was planned; SettingsError, worded,
    when the instance has nothing placed per card or the files cannot be read."""
    args = recipe.args
    placement = args.placement if isinstance(args, LlamaCppArgs) else None
    if not isinstance(args, LlamaCppArgs) or placement is None:
        raise SettingsError(
            f"{', '.join(sorted(wanted))} place a mixture of experts across cards, and "
            f"{recipe.name} has nothing placed per card: it runs on one card, fits whole, or "
            f"was planned before mnm placed experts per card (`mnm replan {recipe.name}`)."
        )
    mode = _mode(wanted.get(EXPERTS_CONTROL.key), args)
    counts = _counts(wanted.get(CARD_EXPERTS_CONTROL.key))
    files = read_model_files(recipe, service.models_dir)
    if files is None or files.header is None:
        raise SettingsError(f"{recipe.name}'s model files cannot be read: download it first.")
    estimate = estimate_memory(
        files.header,
        files.weights_bytes,
        args.ctx,
        args.kv_type,
        files.projector_bytes,
        args.ubatch or DEFAULT_UBATCH,
        recipe.modality in EMBEDDING_MODALITIES,
    )
    rate = expert_rate(service.settings.expanded_state_dir())
    try:
        placed = refill(files.header, estimate, placement, rate, mode, counts)
    except ValueError as exc:
        raise SettingsError(f"{recipe.name}: {exc}") from exc
    chosen = None if mode == "auto" else mode
    return recipe.with_args(placement=placed.placement, experts=chosen)


def _mode(raw: str | None, args: LlamaCppArgs) -> Mode:
    if raw is None:
        return args.experts or "auto"
    text = raw.strip().lower() or "auto"
    if text not in MODES:
        raise SettingsError(f"experts must be one of {', '.join(MODES)}, not '{raw}'")
    return MODES[text]


def _counts(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    parts = [p.strip() for p in raw.split(",")]
    if not all(p.isdigit() for p in parts):
        raise SettingsError(
            f"card-experts takes a whole number per card, separated by commas (3,4), not '{raw}'"
        )
    return tuple(int(p) for p in parts)


def numa_facts(service: InstanceService, recipe: Recipe) -> NumaFacts:
    """What `numa=auto|node:N` reads: this box's NUMA nodes, what the instance keeps in
    RAM (its weights less what the plan puts on the cards; None before the download) and what
    its RAM platform keeps for the system."""
    files = read_model_files(recipe, service.models_dir)
    part = None
    if files is not None:
        part = max(files.weights_bytes - (recipe.vram_estimate_bytes or 0), 0)
    margin = OS_RAM_MARGIN
    try:
        file = load_file(service.settings.expanded_state_dir())
    except MachineFactsError:
        file = None
    platform = platform_here(file) if file is not None else None
    if platform is not None:
        margin = platform.margin_bytes
    return NumaFacts(nodes=read_nodes(), ram_part_bytes=part, margin_bytes=margin)
