"""LoRAs on image instances: find one on the Hub, check it fits the checkpoint, fetch it.

A LoRA is a small add-on trained for one style, character or concept; sd.cpp applies it on
top of the checkpoint at a weight (1.0 is full strength). It only works on the base family
it was trained on, and its header says which: the same cross-attention width that names a
checkpoint's family (`safetensors.sd.lora_info`). An added LoRA is linked into the
instance's LoRA folder, which sd-server rescans on every request, so it shows up on the
engine page at once; an always-on one is part of the startup command (next start).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from makenomistakes.hub import HubClient, HubError, LocalPath, ModelRef, parse_reference
from makenomistakes.hub.civitai import (
    LORA_TYPES,
    CivitaiGoneError,
    CivitaiRef,
    SourcesHub,
    parse_civitai,
)
from makenomistakes.hub.client import TreeEntry
from makenomistakes.hub.facts import Engine
from makenomistakes.hub.image_files import CHECKPOINT_MIN_BYTES
from makenomistakes.hub.kits import FAMILY_LABELS, ImageFamily
from makenomistakes.instance_types import InstanceError, Progress
from makenomistakes.instances import InstanceService
from makenomistakes.recipes.model import LoraFile, Recipe
from makenomistakes.recipes.sdcpp import link_image_files
from makenomistakes.safetensors.reader import SafetensorsError, read_safetensors_header
from makenomistakes.safetensors.sd import LoraInfo, lora_info
from makenomistakes.validation_words import clause

_GENERIC_STEMS = frozenset({"pytorch_lora_weights", "lora", "adapter_model", "model"})


@dataclass(frozen=True)
class LoraPlan:
    lora: LoraFile
    info: LoraInfo
    note: str = ""  # a warning to show before downloading (unknown base, older recipe)


def plan_lora(
    hub: HubClient,
    recipe: Recipe,
    reference: str,
    name: str | None = None,
    weight: float = 1.0,
    always_on: bool = False,
) -> LoraPlan:
    """What adding `reference` to `recipe` would add; raises InstanceError with the reason
    it cannot (not an image instance, no LoRA file, another base family)."""
    if recipe.engine is not Engine.SDCPP:
        raise InstanceError(f"LoRAs apply to image instances; {recipe.name} is not one.")
    source = _lora_source(hub, reference)
    entry, ref = source.entry, source.ref
    try:
        header = read_safetensors_header(hub.file_source(ref.repo_id, ref.revision, entry.path))
    except (HubError, SafetensorsError) as exc:
        raise InstanceError(f"Could not read {source.label}: {exc}") from exc
    info = lora_info(header)
    if info is None:
        raise InstanceError(
            f"{entry.path} holds no LoRA weights (a checkpoint is added with `mnm add`; "
            "embeddings and other add-ons are not supported)."
        )
    note = _compatibility(recipe, ref.repo_id, info)
    if source.trigger_words:
        words = ", ".join(source.trigger_words[:8])
        note = f"{note} Its trigger words (put them in the prompt): {words}.".strip()
    lora = _lora_file(
        name=name or source.name or _name_for(ref.repo_id, entry.path),
        repo_id=ref.repo_id,
        path=entry.path,
        revision=ref.revision,
        size_bytes=entry.size,
        family=_recorded_family(recipe, info),
        weight=weight,
        always_on=always_on,
    )
    if any(existing.key == lora.key for existing in recipe.loras):
        raise InstanceError(f"{recipe.name} already has {ref.repo_id}/{entry.path}.")
    if any(existing.name == lora.name for existing in recipe.loras):
        raise InstanceError(
            f"{recipe.name} already has a LoRA named {lora.name}; give this one another name."
        )
    return LoraPlan(lora, info, note)


@dataclass(frozen=True)
class _Source:
    """Where a LoRA's one file lives: a Hub repo file, or a Civitai version's file."""

    ref: ModelRef
    entry: TreeEntry
    label: str
    name: str | None = None  # the engine page's name, when the source names it better
    trigger_words: tuple[str, ...] = ()


def _lora_source(hub: HubClient, reference: str) -> _Source:
    try:
        civitai = parse_civitai(reference)
        if civitai is None:
            ref = parse_reference(reference)
        elif not isinstance(hub, SourcesHub):
            raise InstanceError("Civitai links need the Civitai client (internal error).")
        else:
            return _civitai_lora(hub, civitai)
    except ValueError as exc:  # refs raise ValueError
        raise InstanceError(str(exc)) from exc
    if isinstance(ref, LocalPath):
        raise InstanceError("Paste a Hugging Face or Civitai link for the LoRA.")
    return _hub_lora(hub, ref)


def _hub_lora(hub: HubClient, ref: ModelRef) -> _Source:
    try:
        tree = hub.tree(ref.repo_id, ref.revision)
    except HubError as exc:
        raise InstanceError(f"Could not read {ref.repo_id}: {exc}") from exc
    files = [
        e
        for e in tree
        if e.is_file
        and e.path.endswith(".safetensors")
        and (ref.path is None or e.path == ref.path or e.path.startswith(ref.path + "/"))
    ]
    if not files:
        raise InstanceError(f"{ref.repo_id} has no .safetensors file to use as a LoRA.")
    # A checkpoint repo that also ships a LoRA (odd layouts): the checkpoints are no LoRA.
    small = [e for e in files if e.size < CHECKPOINT_MIN_BYTES]
    files = small if len(small) == 1 else files
    if len(files) > 1:
        listed = ", ".join(f"{ref.repo_id}/{e.path}" for e in files[:8])
        raise InstanceError(
            f"{ref.repo_id} holds several .safetensors files; paste the link of the one "
            f"you want: {listed}"
        )
    return _Source(ref, files[0], ref.repo_id)


def _civitai_lora(hub: SourcesHub, civitai: CivitaiRef) -> _Source:
    """A Civitai version's file, or, when Civitai no longer has it, the Civitai Archive's
    checked copy on the Hub."""
    try:
        try:
            resolved = hub.civitai.resolve(civitai)
        except CivitaiGoneError as gone:
            found = hub.archived(civitai, gone)
            _need_lora(found.copy.model, found.copy.type)
            return _Source(found.ref, found.entry, found.copy.model)
        _need_lora(resolved.name, resolved.type)
        weights = resolved.version.weights_file()
    except HubError as exc:  # CivitaiError and ArchiveError are HubErrors
        raise InstanceError(str(exc)) from exc
    entry = TreeEntry(type="file", path=weights.name, size=weights.size_bytes)
    return _Source(resolved.ref, entry, resolved.name, None, tuple(resolved.version.trained_words))


def _need_lora(name: str, kind: str) -> None:
    if kind not in LORA_TYPES:
        raise InstanceError(
            f"{name} is a Civitai {kind}, not a LoRA; a checkpoint is added with `mnm add`."
        )


class LoraService:
    """Add, change and remove LoRAs through the instance service (the store's one writer,
    which turns store and validation failures into InstanceError)."""

    def __init__(
        self, instances: InstanceService, hub: Callable[[], HubClient] | None = None
    ) -> None:
        self.instances = instances
        settings = instances.settings
        self._hub = hub or (
            lambda: SourcesHub(settings.hf_token_value(), settings.civitai_token_value())
        )

    def plan(
        self,
        name: str,
        reference: str,
        lora_name: str | None = None,
        weight: float = 1.0,
        always_on: bool = False,
    ) -> LoraPlan:
        recipe = self.instances.get(name)
        with self._hub() as hub:
            return plan_lora(hub, recipe, reference, lora_name, weight, always_on)

    def add(self, name: str, planned: LoraPlan, progress: Progress | None = None) -> Recipe:
        """Fetch it, then save it, then link it: a failed download leaves the instance as
        it was. The engine page lists it without a restart."""
        current = self.instances.get(name)
        self.instances.download(
            current.model_copy(update={"loras": [*current.loras, planned.lora]}), progress
        )  # records the file's path; the saved recipe gains the LoRA only below

        def with_lora(recipe: Recipe) -> Recipe:
            if any(item.name == planned.lora.name for item in recipe.loras):
                raise InstanceError(f"{name} already has a LoRA named {planned.lora.name}.")
            return recipe.model_copy(update={"loras": [*recipe.loras, planned.lora]})

        self._linked(with_lora(self.instances.get(name)))  # a link it cannot make: not saved
        return self._linked(self.instances.change(name, with_lora))

    def update(
        self, name: str, lora: str, weight: float | None = None, always_on: bool | None = None
    ) -> Recipe:
        def change(recipe: Recipe) -> Recipe:
            self._find(recipe, lora)
            changed = [
                _lora_file(
                    **{
                        **item.model_dump(),
                        "weight": item.weight if weight is None else weight,
                        "always_on": item.always_on if always_on is None else always_on,
                    }
                )
                if item.name == lora
                else item
                for item in recipe.loras
            ]
            return recipe.model_copy(update={"loras": changed})

        return self.instances.change(name, change)

    def remove(self, name: str, lora: str) -> Recipe:
        """The file stays in the download cache; only the instance stops offering it."""

        def change(recipe: Recipe) -> Recipe:
            self._find(recipe, lora)
            return recipe.model_copy(update={"loras": [i for i in recipe.loras if i.name != lora]})

        return self._linked(self.instances.change(name, change))

    def _linked(self, recipe: Recipe) -> Recipe:
        if recipe.downloaded:
            try:
                link_image_files(recipe, self.instances.models_dir)
            except (OSError, ValueError) as exc:
                raise InstanceError(f"Could not link {recipe.name}'s LoRAs: {exc}") from exc
        return recipe

    @staticmethod
    def _find(recipe: Recipe, lora: str) -> LoraFile:
        for item in recipe.loras:
            if item.name == lora:
                return item
        names = ", ".join(i.name for i in recipe.loras) or "none yet"
        raise InstanceError(f"{recipe.name} has no LoRA named {lora} (it has: {names}).")


def _lora_file(**fields: object) -> LoraFile:
    """A validated LoRA entry, its limits said in words (weight -4..4, a plain name)."""
    try:
        return LoraFile.model_validate(fields)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            field = str(error["loc"][0]) if error["loc"] else ""
            if field == "weight":
                problems.append(f"the weight must be between -4 and 4 (got {fields['weight']})")
            elif field == "name":
                problems.append(
                    f"a LoRA name is letters, digits, '.', '_' and '-', starting with a letter "
                    f"or digit (got {fields['name']!r}; pass --name)"
                )
            else:
                problems.append(clause(error))
        raise InstanceError("Cannot add this LoRA: " + "; ".join(problems) + ".") from exc


def _compatibility(recipe: Recipe, repo_id: str, info: LoraInfo) -> str:
    """Raise for a LoRA of another family; a note when the match cannot be checked."""
    if not info.families:
        return (
            f"Could not tell which base model {repo_id} was trained on. If images come out "
            "broken, remove it."
        )
    if recipe.image_family is ImageFamily.UNKNOWN:
        return (
            f"{repo_id} is a LoRA for {info.label}; this instance does not record its base "
            "family (added before LoRA support), so the match is not checked."
        )
    if recipe.image_family not in info.families:
        raise InstanceError(
            f"{repo_id} is a LoRA for {info.label}, and {recipe.name} runs "
            f"{FAMILY_LABELS[recipe.image_family]}: a LoRA only works on the base family it "
            "was trained on. Add it to a matching instance."
        )
    if len(info.families) > 1:
        bases = " or ".join(sorted(FAMILY_LABELS[f] for f in info.families))
        return (
            f"{repo_id} is a LoRA for {info.label}; its header does not say whether it was "
            f"trained on {bases}. If images come out broken, remove it."
        )
    return ""


def _recorded_family(recipe: Recipe, info: LoraInfo) -> ImageFamily:
    """The family saved with the LoRA: the instance's when it matches, else the one the
    header names (UNKNOWN when it names none or several)."""
    if recipe.image_family in info.families:
        return recipe.image_family
    return next(iter(info.families)) if len(info.families) == 1 else ImageFamily.UNKNOWN


def _name_for(repo_id: str, path: str) -> str:
    """The file's stem, or the repo's name for the generic `pytorch_lora_weights`; cut to
    what the engine page and the `<lora:name:w>` tag accept."""
    stem = path.rsplit("/", 1)[-1].removesuffix(".safetensors")
    base = repo_id.rsplit("/", 1)[-1] if stem.lower() in _GENERIC_STEMS else stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", base)
    return re.sub(r"^[^A-Za-z0-9]+", "", cleaned).rstrip("-._")[:80] or "lora"
