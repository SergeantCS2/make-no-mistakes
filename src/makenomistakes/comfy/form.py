"""A workflow's form: the settings worth showing, with the workflow's own values as
their defaults, found on the converted prompt.

Node-class rules come first: a prompt and a negative prompt (by the sampler input each text
feeds), the seed, steps, CFG, sampler and scheduler, size and frames on latent nodes, pictures
and clips to upload, LoRA strengths, the frame rate. Then names alone (SwarmUI's idea, MIT):
any `seed`, `steps`, `width` … on any node. A song's own settings by their names
(`MUSIC_NAMES`: the style or tags, the lyrics, BPM, key, time signature, language, the
length, a repaint range, a stem), shown after the prompt and named in words; the Song
pipeline binds its stage's roles to them (`music_keys`). Every other widget is under *More
settings*, by the node's title; the user's picks (`WorkflowMeta.shown`) move one up.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from makenomistakes.comfy.object_info import ObjectInfo, choices, spec_type

SEED_MAX = 2**53 - 1  # what a browser's number holds exactly
SEED_NAMES = ("seed", "noise_seed")  # the inputs a seed field is (a run keeps it by its key)


class Kind(StrEnum):
    TEXT = "text"
    LONG_TEXT = "long_text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    CHOICE = "choice"
    SEED = "seed"
    IMAGE = "image"  # a file to upload
    VIDEO = "video"
    AUDIO = "audio"


class Role(StrEnum):
    PROMPT = "prompt"
    NEGATIVE = "negative"
    MUSIC = "music"  # A song's style, lyrics, tempo, key, length, range, stem
    SEED = "seed"
    SAMPLING = "sampling"
    SIZE = "size"
    INPUT = "input"
    LORA = "lora"
    OUTPUT = "output"
    MORE = "more"  # under More settings


@dataclass
class Field:
    key: str  # `<node id>.<input>`, the prompt's own address of the value
    node: str
    input: str  # the node's own input the value is written to
    label: str
    kind: Kind
    role: Role
    default: Any
    group: str  # the node's title
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    options: list[Any] = field(default_factory=list)
    picked: bool = False  # the user moved it out of More settings (`WorkflowMeta.shown`)
    # what the value is: the input itself, or for a primitive node (its own `value`, wired
    # on) the input it feeds (YuE2's lyrics), which names its role and label
    feeds: str = ""
    # a song setting a second node holds too (ACE-Step's length on its latent and its encoder):
    # the key of the one shown, whose value this takes unless given its own
    follows: str = ""

    @property
    def means(self) -> str:
        return self.feeds or self.input

    @property
    def shown(self) -> bool:
        return self.role is not Role.MORE or self.picked


TEXT_NAMES = {"text", "prompt", "positive_prompt", "negative_prompt", "positive", "negative"}
SAMPLING_NAMES = {"steps", "cfg", "sampler_name", "scheduler", "denoise", "shift"}
SIZE_NAMES = {"width", "height", "length", "num_frames", "frames", "batch_size", "frame_count"}
UPLOAD_CLASSES = {
    "LoadImage": Kind.IMAGE,
    "LoadImageMask": Kind.IMAGE,
    "LoadVideo": Kind.VIDEO,
    "VHS_LoadVideo": Kind.VIDEO,
    "LoadAudio": Kind.AUDIO,
    "VHS_LoadAudioUpload": Kind.AUDIO,
}
UPLOAD_INPUTS = {"image", "video", "audio"}
LORA_INPUTS = {"strength_model", "strength_clip", "strength", "lora_strength"}
OUTPUT_INPUTS = {"fps", "frame_rate", "filename_prefix"}
# A music workflow's own settings by input name (ComfyUI's ACE-Step 1.5 and YuE2
# nodes, the AceMusic and HeartMuLa packs), each to the role the Song pipeline binds
MUSIC_NAMES = {
    "tags": "style",
    "style": "style",
    "lyrics": "lyrics",
    "bpm": "bpm",
    "keyscale": "key",
    "key": "key",
    "timesignature": "time signature",
    "time_signature": "time signature",
    "language": "language",
    "duration": "seconds",
    "seconds": "seconds",
    "max_duration": "seconds",
    "audio_duration": "seconds",
    "repainting_start": "repaint from",
    "repaint_start": "repaint from",
    "repainting_end": "repaint to",
    "repaint_end": "repaint to",
    "stem": "stem",
    "stem_name": "stem",
    "track_name": "stem",
}
MUSIC_LABELS = {
    "style": "Style",
    "lyrics": "Lyrics",
    "bpm": "BPM",
    "key": "Key",
    "time signature": "Time signature",
    "language": "Language",
    "seconds": "Length (s)",
    "repaint from": "Repaint from (s)",
    "repaint to": "Repaint to (s)",
    "stem": "Stem",
}
LABELS = {
    "cfg": "CFG",
    "sampler_name": "Sampler",
    "noise_seed": "Seed",
    "filename_prefix": "File name",
    "frame_rate": "Frame rate",
    "fps": "Frame rate",
    "num_frames": "Frames",
    "length": "Frames",
}
ORDER = list(Role)


def build(prompt: dict[str, dict[str, Any]], info: ObjectInfo) -> list[Field]:
    """Every widget value of the converted prompt as a field, the ones worth showing first."""
    # a text feeding only a `negative` (Z-Image zeroes its one prompt for the negative too)
    negatives = _feeding(prompt, "negative") - _feeding(prompt, "positive")
    feeds = _primitive_feeds(prompt)
    found: list[Field] = []
    for node, entry in prompt.items():
        class_type = entry["class_type"]
        title = (entry.get("_meta") or {}).get("title") or class_type
        specs = dict(info.ordered_inputs(class_type))
        for name, value in entry["inputs"].items():
            if _is_link(value):
                continue
            spec = specs.get(name.split(".", 1)[0])
            kind = _kind(class_type, name, spec, value)
            means = feeds.get(node, name)
            role = _role(class_type, means, kind, node in negatives)
            kind = Kind.SEED if role is Role.SEED else kind
            _kind_type, options = spec_type(spec)
            found.append(
                Field(
                    key=f"{node}.{name}",
                    node=node,
                    input=name,
                    feeds="" if means == name else means,
                    label=_label(name if role is Role.MORE else means, role, title),
                    kind=kind,
                    role=role,
                    default=value,
                    group=title,
                    minimum=options.get("min"),
                    maximum=options.get("max"),
                    step=options.get("step"),
                    options=choices(spec) or [],
                )
            )
    return _one_field_a_setting(sorted(found, key=_order))


def _order(item: Field) -> tuple[int, str, str, str]:
    return ORDER.index(item.role), item.group, item.node, item.means


def _one_field_a_setting(fields: list[Field]) -> list[Field]:
    """A song setting several nodes hold is one field: the first shown, the others under
    More settings by their node, taking its value unless given their own. The form and MCP
    had offered two lengths ("Length (s)", `length_s_2`), and setting one left the latent and
    the encoder disagreeing."""
    first: dict[str, Field] = {}
    for item in fields:
        if item.role is not Role.MUSIC:
            continue
        held = first.setdefault(MUSIC_NAMES[item.means], item)
        if held is not item:
            item.role, item.follows = Role.MORE, held.key
            item.label = f"{held.label} ({item.group})"
    return sorted(fields, key=_order)


def _primitive_feeds(prompt: dict[str, dict[str, Any]]) -> dict[str, str]:
    """A primitive node (`PrimitiveStringMultiline` …: a value of its own, wired on) named
    after the first input it feeds: YuE2's style and lyrics are primitives into its nodes."""
    named: dict[str, str] = {}
    for entry in prompt.values():
        for name, value in entry["inputs"].items():
            source = value[0] if _is_link(value) else None
            if source in prompt and prompt[source]["class_type"].startswith("Primitive"):
                named.setdefault(source, name)
    return named


def _is_link(value: object) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)


def _feeding(prompt: dict[str, dict[str, Any]], input_name: str) -> set[str]:
    """Nodes whose output reaches an input of that name (a sampler's `negative`), through
    any nodes in between (a zero-out, a combine)."""
    feeds: dict[str, set[str]] = {}
    targets: set[str] = set()
    for node, entry in prompt.items():
        for name, value in entry["inputs"].items():
            if _is_link(value):
                feeds.setdefault(node, set()).add(value[0])
                if name == input_name:
                    targets.add(value[0])
    found: set[str] = set()
    pending = list(targets)
    while pending:
        node = pending.pop()
        if node not in found:
            found.add(node)
            pending += feeds.get(node, ())
    return found


def _kind(class_type: str, name: str, spec: object, value: object) -> Kind:  # noqa: PLR0911
    if class_type in UPLOAD_CLASSES and name in UPLOAD_INPUTS:
        return UPLOAD_CLASSES[class_type]
    kind, options = spec_type(spec)
    if kind in ("COMBO", "COMFY_DYNAMICCOMBO_V3") or choices(spec):
        return Kind.CHOICE
    if kind == "STRING":
        return Kind.LONG_TEXT if options.get("multiline") else Kind.TEXT
    by_spec = {"BOOLEAN": Kind.BOOL, "INT": Kind.INT, "FLOAT": Kind.FLOAT}
    if kind in by_spec:
        return by_spec[kind]
    if isinstance(value, bool):
        return Kind.BOOL
    if isinstance(value, int):
        return Kind.INT
    return Kind.FLOAT if isinstance(value, float) else Kind.TEXT


def _role(class_type: str, name: str, kind: Kind, negative: bool) -> Role:  # noqa: PLR0911
    if kind in (Kind.IMAGE, Kind.VIDEO, Kind.AUDIO):
        return Role.INPUT
    if name in SEED_NAMES:
        return Role.SEED
    if name in MUSIC_NAMES and not (MUSIC_NAMES[name] == "style" and kind is Kind.CHOICE):
        return Role.MUSIC
    if name in TEXT_NAMES and kind in (Kind.TEXT, Kind.LONG_TEXT):
        return Role.NEGATIVE if negative or "negative" in name else Role.PROMPT
    if name in SAMPLING_NAMES:
        return Role.SAMPLING
    if name in SIZE_NAMES and kind is Kind.INT:
        return Role.SIZE
    if name in LORA_INPUTS and "lora" in class_type.lower():
        return Role.LORA
    if name in OUTPUT_INPUTS:
        return Role.OUTPUT
    return Role.MORE


def _label(name: str, role: Role, title: str) -> str:
    if role is Role.PROMPT:
        return "Prompt"
    if role is Role.NEGATIVE:
        return "Negative prompt"
    if role is Role.MUSIC:
        return MUSIC_LABELS[MUSIC_NAMES[name]]
    words = LABELS.get(name) or name.replace("_", " ").replace(".", " ").capitalize()
    return f"{words} ({title})" if role in (Role.LORA, Role.MORE) else words


class FormError(ValueError):
    """A value the form cannot take, worded for the user."""


def apply(
    prompt: dict[str, dict[str, Any]],
    fields: list[Field],
    values: dict[str, str],
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """The user's values into the prompt (in place); returns what was used, seeds included
    (a seed left as `random` is drawn now, and kept so a run can be repeated)."""
    by_key = {f.key: f for f in fields}
    used: dict[str, Any] = {}
    draw = rng or random.Random()
    for key, field_ in by_key.items():
        raw = values.get(key, values.get(field_.follows) if field_.follows else None)
        if field_.kind is Kind.SEED and (raw is None or raw.strip().lower() in ("", "random")):
            value: Any = draw.randint(0, SEED_MAX)
        elif raw is None:
            continue
        else:
            value = _coerce(field_, raw)
        prompt[field_.node]["inputs"][field_.input] = value
        used[key] = value
    for key in values:
        if key not in by_key:
            raise FormError(f"The workflow has no setting {key}.")
    return used


def _coerce(field_: Field, raw: str) -> Any:
    text = raw.strip()
    try:
        match field_.kind:
            case Kind.INT | Kind.SEED:  # exact for a whole number (a seed past 2**53 too)
                value: Any = int(text) if text.lstrip("-").isdigit() else int(float(text))
            case Kind.FLOAT:
                value = float(text)
                if not math.isfinite(value):  # nan passes both bounds; JSON has no NaN
                    raise FormError(f"{field_.label}: '{raw}' is not a finite number.")
            case Kind.BOOL:
                return text.lower() in ("1", "true", "on", "yes")
            case Kind.CHOICE:
                known = [str(o) for o in field_.options]
                if field_.options and raw not in known and raw != str(field_.default):
                    raise FormError(f"{field_.label}: '{raw}' is not one of its choices.")
                return next((o for o in field_.options if str(o) == raw), raw)
            case _:
                return raw
    except (ValueError, OverflowError) as exc:  # int(float("1e400")) overflows
        if isinstance(exc, FormError):
            raise
        raise FormError(f"{field_.label}: '{raw}' is not a number.") from None
    if field_.minimum is not None and value < field_.minimum:
        raise FormError(f"{field_.label}: {value} is below its minimum, {field_.minimum}.")
    if field_.maximum is not None and value > field_.maximum:
        raise FormError(f"{field_.label}: {value} is above its maximum, {field_.maximum}.")
    return value


def music_keys(fields: list[Field]) -> dict[str, list[str]]:
    """Each music role a workflow's form has, with every field that holds it (a template
    gives one length to its latent and its text encoder: both are set together)."""
    found: dict[str, list[str]] = {}
    for item in fields:
        if item.role is Role.MUSIC or item.follows:
            found.setdefault(MUSIC_NAMES[item.means], []).append(item.key)
    return found
