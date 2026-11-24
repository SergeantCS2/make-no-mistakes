"""`GET /object_info` typed: each node class's inputs in their order, and which of
them the frontend shows as widgets, which is what a UI-format node's `widgets_values` follow.

The frontend adds values of its own after some widgets, kept in `widgets_values` but never
sent: the seed's *control after generate* (any INT asking for it, and every `seed` and
`noise_seed`), a combo's control and its filter (when asked for), and an upload button's
value after an `image_upload` (or video, audio) combo.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"  # a choice whose options bring inputs of their own
WIDGET_TYPES = frozenset({"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", DYNAMIC_COMBO})
SEED_NAMES = frozenset({"seed", "noise_seed"})
UPLOAD_FLAGS = ("image_upload", "video_upload", "audio_upload", "animated_image_upload")


class NodeInfo(BaseModel):
    input: dict[str, dict[str, Any]] = Field(default_factory=dict)
    input_order: dict[str, list[str]] = Field(default_factory=dict)
    output: list[Any] = Field(default_factory=list)
    output_node: bool = False
    display_name: str | None = None
    category: str | None = None
    python_module: str | None = None
    api_node: bool | None = False


@dataclass(frozen=True)
class Widget:
    name: str
    extras: int  # the frontend's own values after it in `widgets_values`
    kind: str = ""
    options: dict[str, Any] | None = None
    choices: tuple[Any, ...] = ()  # a V1 combo's list (V3's are in `options`)

    @property
    def default(self) -> object:
        """What the frontend sets when a saved workflow has no value for it (an input added
        after it was saved): the spec's default, else a combo's first choice."""
        opts = self.options or {}
        if "default" in opts:
            return opts["default"]
        found = opts.get("options", self.choices)
        if isinstance(found, list | tuple) and found:
            first = found[0]
            return first.get("key") if isinstance(first, dict) else first
        return {"INT": 0, "FLOAT": 0.0, "STRING": "", "BOOLEAN": False}.get(self.kind)

    def nested(self, value: object) -> list[Widget]:
        """A dynamic combo's widgets for the option chosen (`format` → `format.codec`), or a
        VideoHelperSuite combo's (`formats`: the chosen format's own inputs, `crf`)."""
        formats = (self.options or {}).get("formats")
        if self.kind == "COMBO" and isinstance(formats, dict) and isinstance(value, str):
            specs = [
                (str(item[0]), list(item[1:]))
                for item in formats.get(value) or []
                if isinstance(item, list) and len(item) >= 2
            ]
            return widgets_of(specs)
        if self.kind != DYNAMIC_COMBO:
            return []
        for option in (self.options or {}).get("options") or []:
            if isinstance(option, dict) and option.get("key") == value:
                inputs = option.get("inputs") or {}
                specs = [
                    (name, spec)
                    for group in ("required", "optional")
                    for name, spec in (inputs.get(group) or {}).items()
                ]
                return [
                    Widget(f"{self.name}.{w.name}", w.extras, w.kind, w.options, w.choices)
                    for w in widgets_of(specs)
                ]
        return []


def spec_type(spec: object) -> tuple[str, dict[str, Any]]:
    """An input spec's type (a combo's list of choices reads as COMBO) and its options."""
    if not isinstance(spec, list | tuple) or not spec:
        return "", {}
    kind, options = spec[0], spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(kind, list):
        return "COMBO", options
    return (kind if isinstance(kind, str) else ""), options


def choices(spec: object) -> list[Any] | None:
    """A combo's choices, in either shape (V1's list, V3's `options`)."""
    if isinstance(spec, list | tuple) and spec and isinstance(spec[0], list):
        return list(spec[0])
    kind, options = spec_type(spec)
    found = options.get("options") if kind == "COMBO" else None
    return list(found) if isinstance(found, list) else None


class ObjectInfo:
    """Every node class ComfyUI knows, by class name."""

    def __init__(self, classes: dict[str, NodeInfo]) -> None:
        self.classes = classes

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> ObjectInfo:
        return cls(
            {
                name: NodeInfo.model_validate(info)
                for name, info in raw.items()
                if isinstance(info, dict)
            }
        )

    def get(self, class_type: str) -> NodeInfo | None:
        return self.classes.get(class_type)

    def ordered_inputs(self, class_type: str) -> list[tuple[str, object]]:
        """Required then optional inputs, in `input_order` (each with its spec)."""
        info = self.classes.get(class_type)
        if info is None:
            return []
        found: list[tuple[str, object]] = []
        for group in ("required", "optional"):
            specs = info.input.get(group) or {}
            order = info.input_order.get(group) or list(specs)
            found += [(name, specs[name]) for name in order if name in specs]
        return found

    def widgets(self, class_type: str) -> list[Widget]:
        """The inputs the frontend shows as widgets, in `widgets_values` order (a dynamic
        combo's own inputs follow it, once its value is known: `Widget.nested`)."""
        return widgets_of(self.ordered_inputs(class_type))


def widgets_of(specs: Sequence[tuple[str, object]]) -> list[Widget]:
    found = []
    for name, spec in specs:
        kind, options = spec_type(spec)
        if kind not in WIDGET_TYPES or options.get("forceInput"):
            continue
        extras = 0
        control = options.get("control_after_generate")
        if kind == "INT" and (control or name in SEED_NAMES):
            extras = 1
        elif kind == "COMBO" and control:
            extras = 2  # the control, then its filter list
        if kind == "COMBO" and any(options.get(flag) for flag in UPLOAD_FLAGS):
            extras += 1
        listed = (
            tuple(spec[0]) if isinstance(spec, list | tuple) and isinstance(spec[0], list) else ()
        )
        found.append(Widget(name, extras, kind, options, listed))
    return found


@functools.lru_cache(maxsize=1)
def parse_file(path: Path, mtime_ns: int, size: int) -> ObjectInfo | None:
    """object_info.json parsed once per version of the file (`mtime_ns` and `size` are the
    key): a page asks for it once per workflow, and it is megabytes."""
    del mtime_ns, size
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ObjectInfo.parse(raw) if isinstance(raw, dict) else None
