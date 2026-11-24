"""A style on an ACE-Step instance: `mnm set <instance> style=<name> style-scale=0.8`
names a row of the style library, which audio.cpp merges into the DiT when the instance
loads (`ace_step.lora`, `recipes/audiocpp.music_model`). A style plays on the build it was
trained against only (base or turbo: their noise schedules differ, and upstream advises
against mixing them), so the setting is refused for another build, for a style the library
does not hold, and for one whose files are gone (a restore brings rows, not files).

A song made by an instance with a style carries the style's trigger word at the front of its
style line (`styled`), the word the training put in front of every caption.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from makenomistakes.config import Settings
from makenomistakes.controls import SettingsError
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import AceStepOptions
from makenomistakes.recipes.audiocpp import LORA_OPTION, config_path
from makenomistakes.recipes.model import Recipe
from makenomistakes.run_parts import ModelCredit
from makenomistakes.store import InstanceStore, StoreError
from makenomistakes.studio.style_library import (
    ADAPTER,
    DISCLOSED_WORDS,
    StyleError,
    StyleLibrary,
    StyleRow,
)

if TYPE_CHECKING:
    from makenomistakes.instances import InstanceService


def ace_build(recipe: Recipe) -> Literal["base", "turbo"] | None:
    """The ACE-Step build an instance plays, from its GGUF's name
    (`ace-step-1.5-base-q8_0.gguf`); None when the name does not say."""
    for file in recipe.files:
        name = Path(file).name.lower()
        if "turbo" in name:
            return "turbo"
        if "base" in name:
            return "base"
    return None


def options_of(recipe: Recipe) -> AceStepOptions | None:
    args = recipe.args
    if isinstance(args, AudioCppArgs) and isinstance(args.options, AceStepOptions):
        return args.options
    return None


def check(settings: Settings, before: Recipe, after: Recipe) -> None:
    """Refuse a style the instance cannot play; nothing to say when none is newly named."""
    options = options_of(after)
    if options is None or options.style is None:
        return
    earlier = options_of(before)
    if earlier is not None and earlier.style == options.style:
        return
    library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    try:
        row = library.get(options.style)
    except StyleError as exc:
        raise SettingsError(str(exc)) from exc
    build = ace_build(after)
    if build is not None and row.variant != build:
        raise SettingsError(
            f"'{row.name}' was trained against ACE-Step 1.5 {row.variant}, and {after.name} "
            f"plays {build}: a style plays on its own build only. Add the {row.variant} build "
            f"(`mnm add audio-cpp/audio.cpp-gguf/ACE-Step1.5-GGUF/{row.variant}`) or train the "
            f"style again with `--variant {build}`."
        )
    try:
        library.check_files(row)
    except StyleError as exc:
        raise SettingsError(str(exc)) from exc


def playing(service: InstanceService, recipe: Recipe) -> StyleRow | None:
    """The style the instance's engine plays, as the library holds it: what the config its
    running container started with names (a `mnm set style=` since applies at the next start),
    else the recipe's (a parked instance loads at the request, from the recipe); None for none,
    or one the library no longer holds."""
    settings = service.settings
    name = (
        rendered_style(settings.expanded_models_dir(), recipe)
        if service.status(recipe).running
        else (options.style if (options := options_of(recipe)) is not None else None)
    )
    if name is None:
        return None
    library = StyleLibrary(settings.expanded_state_dir(), settings.expanded_models_dir())
    try:
        row = library.get(name)
    except StyleError:
        return None
    library.check_files(row)  # its credit names these bytes: refused (StyleError) if others
    return row


def rendered_style(models_dir: Path, recipe: Recipe) -> str | None:
    """The style the instance's config names (`ace_step.lora`, written at each start)."""
    try:
        config = json.loads((models_dir / config_path(recipe)).read_text(encoding="utf-8"))
        models = config["models"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    for model in models if isinstance(models, list) else []:
        session = model.get("session_options") if isinstance(model, dict) else None
        lora = session.get(LORA_OPTION) if isinstance(session, dict) else None
        if isinstance(lora, str):
            return PurePosixPath(lora).name
    return None


def styled(tags: str, row: StyleRow | None) -> str:
    """A song's style line with the style's trigger word in front, once (as a whole word:
    `lo` is not in `slow`)."""
    if row is None or re.search(rf"(?<!\w){re.escape(row.trigger)}(?!\w)", tags, re.I):
        return tags
    return f"{row.trigger}, {tags}" if tags.strip() else row.trigger


def users(settings: Settings, name: str) -> list[str]:
    """The instances whose recipe names the style."""
    store = InstanceStore(settings.expanded_state_dir())
    return sorted(
        recipe.name
        for recipe in store.list()
        if (options := options_of(recipe)) is not None and options.style == name
    )


def refuse_removal(settings: Settings, name: str) -> None:
    """A style an instance names stays until the instance stops naming it: its next start
    would load a folder that is gone. When the recipes cannot be read, which instances name
    it is unknown and nothing is removed (a refusal in words, never a traceback in `mnm studio
    styles rm` or a 500 from the web's remove)."""
    try:
        found = users(settings, name)
    except StoreError as exc:
        why = str(exc).rstrip(".")
        raise StyleError(
            f"Which instances play '{name}' cannot be read now ({why}), so nothing was removed."
        ) from exc
    if found:
        named = ", ".join(found)
        raise StyleError(
            f"'{name}' is the style of {named}: `mnm set {found[0]} style=` first"
            f"{' (and the others)' if len(found) > 1 else ''}, then remove it."
        )


def credit(row: StyleRow) -> ModelCredit:
    """The style a candidate was sung with, as the export's disclosure names it."""
    adapter = next((f.sha256 for f in row.files if f.name == ADAPTER), None)
    songs = len(row.training.dataset) if row.training is not None else None
    learned = DISCLOSED_WORDS[row.source]
    trained_on = (
        f"a LoRA trained by mnm on {songs} of {learned}"
        if songs
        else f"a LoRA of {learned}, added to the library"
    )
    return ModelCredit(style=row.name, sha256=adapter, trained_on=trained_on)
