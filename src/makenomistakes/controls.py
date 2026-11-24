"""Per-modality control panels: which recipe settings a user may change, with help text.

The CLI (`mnm set`) and the web UI render the same list, and both apply changes through
`apply_settings`, so the typed recipe stays the only source of truth.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from enum import Enum

from pydantic import ValidationError

from makenomistakes.backend_setting import BACKEND_CONTROL, BackendSettingError, with_backend
from makenomistakes.backend_setting import current as backend_current
from makenomistakes.backend_setting import not_here as backend_not_here
from makenomistakes.backend_setting import offered as backend_offered
from makenomistakes.card_placement import Placement
from makenomistakes.card_settings import (
    CARD_CONTROLS,
    CARD_EXPERTS_CONTROL,
    EXPERTS_CONTROL,
    GPUS_CONTROL,
    PLACEMENT_KEYS,
    SPLIT_CONTROL,
    Cards,
    CardSettingError,
    check_split,
    several_cards,
    with_gpus,
)
from makenomistakes.controls_audio import AUDIO_CONTROLS, VOICE_CONTROLS
from makenomistakes.controls_catalog import Control, Kind
from makenomistakes.controls_llamacpp import (
    DRAFT_CONTROL,
    DRAFT_MAX_CONTROL,
    DRAFT_P_MIN_CONTROL,
    LLM_CONTROLS,
)
from makenomistakes.controls_music import MUSIC_CONTROLS
from makenomistakes.controls_router import ROUTER_CONTROLS
from makenomistakes.controls_sdcpp import IMAGE_CONTROLS
from makenomistakes.controls_vllm import VLLM_CONTROLS
from makenomistakes.engine_args import AudioCppArgs, KvType, LlamaCppArgs, SdCppArgs
from makenomistakes.hub.facts import EMBEDDING_MODALITIES, Engine, Modality
from makenomistakes.hub.kits import kit_for
from makenomistakes.numa_setting import Facts as NumaFacts
from makenomistakes.numa_setting import NumaSettingError, with_numa
from makenomistakes.recipes.audiocpp import known_voices, voices_are_files
from makenomistakes.recipes.model import DraftModel, Recipe
from makenomistakes.tune.override import untuned_if_changed
from makenomistakes.validation_words import worded

__all__ = [
    "Control",
    "Kind",
    "SettingsError",
    "apply_settings",
    "controls_for",
    "current_value",
    "model_note",
    "parse_assignments",
    "settings_rows",
]


class SettingsError(ValueError):
    pass


def controls_for(modality: Modality, engine: Engine | None = None) -> tuple[Control, ...]:
    """The settings an instance of this type offers; a vLLM chat model has vLLM's."""
    catalog = (
        VLLM_CONTROLS
        if engine is Engine.VLLM
        else (
            *LLM_CONTROLS,
            DRAFT_CONTROL,
            DRAFT_MAX_CONTROL,
            DRAFT_P_MIN_CONTROL,
            *IMAGE_CONTROLS,
            *AUDIO_CONTROLS,
            *VOICE_CONTROLS,
            *MUSIC_CONTROLS,
        )
    )
    cards: tuple[Control, ...] = ()
    if several_cards():  # one card: nothing to choose
        cards = (
            (GPUS_CONTROL,)
            if engine is Engine.VLLM
            else (SPLIT_CONTROL, GPUS_CONTROL, EXPERTS_CONTROL, CARD_EXPERTS_CONTROL)
        )
    # ROCm or Vulkan per llama.cpp instance, only where an AMD card runs both
    amd = (BACKEND_CONTROL,) if engine is not Engine.VLLM and backend_offered() else ()
    every = (*catalog, *ROUTER_CONTROLS, *cards, *amd)
    return tuple(c for c in every if modality in c.modalities)


def current_value(recipe: Recipe, control: Control) -> str:  # noqa: PLR0911 - one per kind
    if control.key == BACKEND_CONTROL.key:  # The image's backend, not its tag
        return backend_current(recipe)
    owner: object = recipe if control.on_recipe else recipe.args
    if control.on_options:  # A music instance's options (none: nothing to show)
        owner = getattr(recipe.args, "options", None)
    value = getattr(owner, control.field, None)
    if value is None:
        return ""
    if isinstance(value, DraftModel):  # named by the instance it came from, or `mtp`
        return value.setting
    if isinstance(value, Placement):  # Expert blocks per card, in its order
        return ",".join(str(n) for n in value.counts)
    if isinstance(value, tuple):  # the cards an instance is pinned to
        return ",".join(str(v) for v in value)
    if control.kind is Kind.BOOL:
        return "on" if value else "off"
    return str(value.value if isinstance(value, Enum) else value)


def model_note(control: Control, native_ctx: int | None) -> str:
    """What this model allows, next to the generic recommendation (empty when unknown)."""
    if control.key != "ctx" or not native_ctx:
        return ""
    return f"This model was trained for up to {native_ctx:,} tokens; more than that degrades."


def shown_controls(recipe: Recipe) -> tuple[Control, ...]:
    """The modality's controls less those this instance cannot use: a voice that reads at
    its own pace has no speed, a video model without a high-noise half no high-noise steps."""
    args = recipe.args
    hidden: set[str] = set()
    if isinstance(args, AudioCppArgs) and not args.paced:
        hidden.add("speed")
    if isinstance(args, AudioCppArgs) and args.music:  # The options this family takes
        options = args.options
        hidden.update(
            c.field
            for c in MUSIC_CONTROLS
            if c.on_options and (options is None or c.field not in type(options).model_fields)
        )
    if isinstance(args, SdCppArgs) and recipe.high_noise is None:
        hidden.add("high_noise_steps")
    if recipe.draft is None:
        hidden.update({"draft_max", "draft_p_min"})
    if isinstance(args, LlamaCppArgs) and not args.experts_in_ram:
        hidden.add("moe_cache_mib")  # nothing in RAM to cache
    if isinstance(args, LlamaCppArgs) and args.placement is None:
        hidden.update({"experts", "placement"})  # nothing placed per card

    return tuple(c for c in controls_for(recipe.modality, recipe.engine) if c.field not in hidden)


def settings_rows(recipe: Recipe, native_ctx: int | None) -> list[tuple[Control, str, str]]:
    """Each control with its current value and what this model allows: `mnm show` and the
    instance page print the same rows."""
    return [
        (c, current_value(recipe, c), model_note(c, native_ctx)) for c in shown_controls(recipe)
    ]


def parse_assignments(items: list[str]) -> dict[str, str]:
    """`["ctx=16384", "kv=q8_0"]` -> {"ctx": "16384", "kv": "q8_0"}."""
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SettingsError(f"'{item}' is not key=value")
        out[key.strip().lower()] = value.strip()
    return out


Draft = Callable[[Recipe, str], Recipe]  # the draft setting, resolved against the store
NUMA_KEY = "numa"  # `auto` and `node:N` read the box (`numa_setting`)
# `experts=` and `card-experts=`, placed again from the model's files
Place = Callable[[Recipe, dict[str, str]], Recipe]


def apply_settings(
    recipe: Recipe,
    updates: dict[str, str],
    draft: Draft | None = None,
    cards: Cards | None = None,
    place: Place | None = None,
    numa: NumaFacts | None = None,
) -> Recipe:
    """Return a copy of the recipe with the given controls changed. Validates everything.
    A draft model names another instance, so only a caller that can read the store (`draft`,
    `InstanceService.update_settings`) may set one; `cards` lists the cards for `gpus=`
    (vLLM's tensor parallel asks whether they are equal); `place` places a MoE's experts
    across cards again (`experts=`, `card-experts=`: it reads the model's files)."""
    controls = {c.key: c for c in controls_for(recipe.modality, recipe.engine)}
    saved = recipe  # what a tune measured: a save that moves it drops the tune
    if DRAFT_CONTROL.key in updates and DRAFT_CONTROL.key in controls:
        if draft is None:
            raise SettingsError("a draft model is set through `mnm set` or the instance page")
        recipe = draft(recipe, updates[DRAFT_CONTROL.key])
        updates = {k: v for k, v in updates.items() if k != DRAFT_CONTROL.key}
    if GPUS_CONTROL.key in updates and GPUS_CONTROL.key in controls:
        recipe = _pinned(recipe, updates.pop(GPUS_CONTROL.key), cards)
    recipe, updates = _backend(recipe, updates, controls)
    recipe, updates = _numa(recipe, updates, controls, numa)
    recipe, updates = _placed(recipe, updates, controls, place)
    own, options, fields = _sorted(recipe, updates, controls)
    recipe = _with_own(recipe, own)
    if options:
        fields["options"] = _with_options(recipe, options)
    ctx = fields.get("ctx")
    if recipe.modality in EMBEDDING_MODALITIES and isinstance(ctx, int):
        args = recipe.llama_args
        # The web form sends every field: only a batch that differs from the saved one was set.
        chosen = {k for k in ("batch", "ubatch") if k in fields and fields[k] != getattr(args, k)}
        follow = args.whole_input_batches(ctx)
        fields.update({key: value for key, value in follow.items() if key not in chosen})
    try:
        updated = recipe.with_args(**fields)
    except ValidationError as exc:
        raise SettingsError(worded(exc.errors())) from exc
    except ValueError as exc:  # a field the engine's args have no name for
        raise SettingsError(str(exc)) from exc
    new = updated.args
    _check_against_the_recipe(updated, recipe)
    try:
        check_split(updated)
    except CardSettingError as exc:
        raise SettingsError(str(exc)) from exc
    if isinstance(new, SdCppArgs) and new.diffusion_type is not None and not recipe.converts_dit:
        raise SettingsError(
            "diffusion applies to a single DiT safetensors file only: this instance's weights "
            "are a GGUF, a checkpoint or a diffusers folder, which load as stored."
        )
    if isinstance(new, LlamaCppArgs):
        updated = _moe_cache(updated, recipe)
        new = updated.args
    if isinstance(new, LlamaCppArgs) and new.kv_type is not KvType.F16 and new.flash_attn == "off":
        # llama-server stops on it: "quantized V cache requires flash_attn".
        raise SettingsError(
            f"kv={new.kv_type.value} needs flash attention: keep flash-attn=auto (or on), or "
            "set kv=f16 to turn flash attention off."
        )
    # The web form sends every field: a save that changes nothing keeps the planned estimate.
    same = updated.args == recipe.args and updated.gpus == recipe.gpus
    kept = recipe if same and updated.image == recipe.image else updated
    return untuned_if_changed(saved, kept)


def _placed(
    recipe: Recipe, updates: dict[str, str], controls: dict[str, Control], place: Place | None
) -> tuple[Recipe, dict[str, str]]:
    """`experts=` and `card-experts=` first: placed again by `place`; the web form
    posts every field, so only a changed one places again. The other updates come back."""
    asked = {k: updates[k] for k in PLACEMENT_KEYS if k in updates and k in controls}
    rest = {k: v for k, v in updates.items() if k not in asked}
    placing = {k: v for k, v in asked.items() if v.strip() != current_value(recipe, controls[k])}
    if not placing:
        return recipe, rest
    if place is None:
        raise SettingsError("experts across cards are set through `mnm set` or its page")
    return place(recipe, placing), rest


def _refuse_card_key(key: str, recipe: Recipe) -> None:
    """A setting that lays an instance over several cards, asked on one card: said so, never
    "not a setting"."""
    if key not in {c.key for c in CARD_CONTROLS} or several_cards():
        return
    raise SettingsError(
        f"'{key}' lays {recipe.name} out over several cards, and mnm runs on one card now: it "
        "is a setting once `MNM_GPU` names several (`MNM_GPU=0,1` or `MNM_GPU=all` in the "
        "secrets file)"
    )


def _backend(
    recipe: Recipe, updates: dict[str, str], controls: dict[str, Control]
) -> tuple[Recipe, dict[str, str]]:
    """`backend=`: the recipe on the backend's image; refused with the reason where
    the box offers no choice. The other updates come back."""
    if BACKEND_CONTROL.key not in updates:
        return recipe, updates
    rest = {k: v for k, v in updates.items() if k != BACKEND_CONTROL.key}
    raw = updates[BACKEND_CONTROL.key]
    if BACKEND_CONTROL.key not in controls:
        if raw.strip().lower() in ("", backend_current(recipe)):
            return recipe, rest  # the form posted what is there
        raise SettingsError(backend_not_here())
    try:
        return with_backend(recipe, raw), rest
    except BackendSettingError as exc:
        raise SettingsError(str(exc)) from exc


def _numa(
    recipe: Recipe, updates: dict[str, str], controls: dict[str, Control], facts: NumaFacts | None
) -> tuple[Recipe, dict[str, str]]:
    """`numa=`: `auto` and `node:N` read the box's nodes (`numa_setting`). The other
    updates come back."""
    if NUMA_KEY not in updates or NUMA_KEY not in controls:
        return recipe, updates
    rest = {k: v for k, v in updates.items() if k != NUMA_KEY}
    try:
        return with_numa(recipe, updates[NUMA_KEY], facts), rest
    except NumaSettingError as exc:
        raise SettingsError(str(exc)) from exc


def _pinned(recipe: Recipe, raw: str, cards: Cards | None) -> Recipe:
    """`gpus=` first: it decides what the engine's split flags may be."""
    if raw.strip() == current_value(recipe, GPUS_CONTROL):  # the form posts every field
        return recipe
    try:
        return with_gpus(recipe, raw, cards)
    except CardSettingError as exc:
        raise SettingsError(str(exc)) from exc


Values = dict[str, object]


def _sorted(
    recipe: Recipe, updates: dict[str, str], controls: dict[str, Control]
) -> tuple[Values, Values, Values]:
    """Each setting coerced and sorted by where it lives: the recipe's own fields (the
    router's), a music instance's `args.options`, the engine's args."""
    own: Values = {}
    options: Values = {}
    fields: Values = {}
    for key, raw in updates.items():
        control = controls.get(key)
        if control is None:
            _refuse_card_key(key, recipe)
            raise SettingsError(
                f"'{key}' is not a setting for {recipe.modality.label} instances "
                f"(try: {', '.join(controls)})"
            )
        target = own if control.on_recipe else options if control.on_options else fields
        target[control.field] = _coerce(control, raw)
    return own, options, fields


def _with_options(recipe: Recipe, changed: dict[str, object]) -> object:
    """A music instance's options with the changed ones set, validated."""
    current = getattr(recipe.args, "options", None)
    if current is None or not set(changed) <= set(type(current).model_fields):
        raise SettingsError(f"{recipe.name} takes no {', '.join(changed)} setting")
    try:
        return type(current).model_validate({**current.model_dump(), **changed})
    except ValidationError as exc:
        raise SettingsError(worded(exc.errors())) from exc


def _with_own(recipe: Recipe, own: dict[str, object]) -> Recipe:
    """The recipe with its own (router) fields set, validated; the same recipe when nothing
    changes (the web form posts every field)."""
    if all(getattr(recipe, k) == v for k, v in own.items()):
        return recipe
    try:
        return Recipe.model_validate({**recipe.model_dump(), **own})
    except ValidationError as exc:
        raise SettingsError(worded(exc.errors())) from exc


def _number(control: Control, raw: str, text: str) -> int | float:
    if control.kind is Kind.INT:
        if not _WHOLE.fullmatch(text):  # isdigit() takes "²" and lstrip takes "--8192"
            raise SettingsError(f"{control.key} needs a whole number, not '{raw}'")
        return int(text)
    try:
        number = float(text)
    except ValueError:
        raise SettingsError(f"{control.key} needs a number, not '{raw}'") from None
    if not math.isfinite(number):  # `inf` would be saved as null: silently dropped
        raise SettingsError(f"{control.key} needs a finite number, not '{raw}'")
    return number


_WHOLE = re.compile(r"-?[0-9]+")


_TRUE = {"on", "true", "yes", "1"}
_FALSE = {"off", "false", "no", "0"}
_CLEAR = {"", "auto", "none", "default"}


def _coerce(control: Control, raw: str) -> int | float | str | bool | None:
    text = raw.strip().lower()
    if control.kind is Kind.CHOICE and text in control.choices:
        return text  # a real choice named like a clear word (pooling=none) wins
    if control.nullable and text in _CLEAR:
        return None
    match control.kind:
        case Kind.INT | Kind.FLOAT:
            return _number(control, raw, text)
        case Kind.BOOL:
            if text not in _TRUE | _FALSE:
                raise SettingsError(f"{control.key} is on or off, not '{raw}'")
            return text in _TRUE
        case Kind.CHOICE:
            if text not in control.choices:
                raise SettingsError(f"{control.key} must be one of {', '.join(control.choices)}")
            return text
        case Kind.TEXT:
            return raw.strip()  # case matters in a prompt


def _moe_cache(updated: Recipe, before: Recipe) -> Recipe:
    """The expert cache lives beside experts kept in RAM: a save that brings every
    expert back to the card drops it with them; asking for one without any is refused."""
    args = updated.llama_args
    if not args.moe_cache_mib or args.experts_in_ram:
        return updated
    if before.llama_args.experts_in_ram:
        return updated.with_args(moe_cache_mib=None)
    raise SettingsError(
        "moe-cache caches experts kept in RAM, and this instance keeps none there "
        "(cpu-moe-layers is empty): every expert is on the card already."
    )


def _check_against_the_recipe(recipe: Recipe, before: Recipe) -> None:
    """Settings the args type alone cannot judge: they depend on the model's files. A voice
    is judged only when it changed (the web form posts every field)."""
    args = recipe.args
    if isinstance(args, AudioCppArgs) and args.voice != before.voice_args.voice:
        known = known_voices(recipe)
        if voices_are_files(recipe) and args.voice not in known:  # every request a 500
            raise SettingsError(
                f"'{args.voice}' is not one of this model's voices: {', '.join(known)}"
            )
    if isinstance(args, SdCppArgs):
        if args.high_noise_steps is not None and recipe.high_noise is None:
            raise SettingsError(
                "high-noise-steps applies to Wan 2.2 A14B only (a high-noise and a low-noise "
                "model); this instance has one diffusion model."
            )
        kit = kit_for(recipe.image_family)
        video = kit.video if kit is not None else None
        if args.frames is not None and video is not None:
            step = video.temporal_compression
            if (args.frames - 1) % step:
                below = args.frames - (args.frames - 1) % step
                raise SettingsError(
                    f"frames must be one more than a multiple of {step} (the VAE packs {step} "
                    f"frames into one): {below} or {below + step}, not {args.frames}."
                )
