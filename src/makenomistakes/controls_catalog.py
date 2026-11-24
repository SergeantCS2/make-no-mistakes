"""What a setting is (`Control`), the modality sets settings are offered for, and the
import-time check every engine's table runs (each control names a real field and choice
of its args class). The tables live per engine: `controls_llamacpp`, `controls_sdcpp`,
`controls_audio`, `controls_vllm`; `controls.py` applies them."""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import get_args

from pydantic import BaseModel

from makenomistakes.hub.facts import CHAT_LIKE, EMBEDDING_MODALITIES, Modality


class Kind(StrEnum):
    INT = "int"
    FLOAT = "float"
    CHOICE = "choice"
    BOOL = "bool"
    TEXT = "text"


CHAT_MODALITIES = CHAT_LIKE
ALL_MODALITIES = CHAT_MODALITIES | EMBEDDING_MODALITIES  # llama.cpp engines
IMAGE_MODALITIES = frozenset({Modality.IMAGE_GEN, Modality.IMAGE_EDIT})  # sd-server, pictures
VIDEO_MODALITIES = frozenset({Modality.VIDEO})
SD_MODALITIES = IMAGE_MODALITIES | VIDEO_MODALITIES  # every sd-server setting but the clip's
AUDIO_MODALITIES = frozenset({Modality.SPEECH_TO_TEXT})
VOICE_MODALITIES = frozenset({Modality.TEXT_TO_SPEECH})


class Control(BaseModel):
    key: str  # what the user types: ctx, kv, gpu-layers …
    field: str  # LlamaCppArgs attribute
    label: str
    help: str
    kind: Kind
    recommended: str
    choices: tuple[str, ...] = ()
    nullable: bool = False  # "auto"/empty clears the value
    modalities: frozenset[Modality] = ALL_MODALITIES  # where the control is offered
    on_recipe: bool = False  # a field of the recipe itself, not its engine's args
    on_options: bool = False  # A field of a music instance's `args.options`
    live: bool = False  # the router reads it at every request: it applies at once


def check_controls(pairs: list[tuple[Control, type[BaseModel]]]) -> None:
    """Fail at import if a control names a field or choice its args class does not have."""
    for control, args_cls in pairs:
        field = args_cls.model_fields[control.field]  # KeyError: field renamed
        if control.kind is Kind.CHOICE:
            annotation = field.annotation
            if control.nullable:  # Literal[...] | None: look inside the union
                annotation = next(a for a in get_args(annotation) if a is not type(None))
            allowed = (
                {m.value for m in annotation}
                if isinstance(annotation, type) and issubclass(annotation, Enum)
                else set(get_args(annotation))
            )
            if set(control.choices) != allowed:
                raise AssertionError(f"{control.key}: choices {control.choices} != {allowed}")
