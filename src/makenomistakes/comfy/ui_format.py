"""A ComfyUI workflow as sites share it (the UI format, schema 0.4 and 1), typed at
the edge. Lenient on purpose: every node pack adds its own properties, so unknown fields are
kept, never refused; only the shape the converter and the report read is checked.

The API format (`{id: {class_type, inputs}}`, what `POST /prompt` takes) is recognised too,
so a workflow exported with "Save (API)" imports as it is.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MUTED, BYPASSED = 2, 4  # LiteGraph's NEVER and BYPASS modes
SUBGRAPH_INPUT = -10  # a subgraph's input node, inside its definition (its output node is -20)


class FormatError(ValueError):
    """Not a workflow this version reads (the message says what it is instead)."""


class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class ModelFile(_Lenient):
    """`properties.models` (templates) or a workflow's top-level `models`."""

    name: str
    url: str = ""
    directory: str = ""
    hash: str | None = None
    hash_type: str | None = None


class NodeProperties(_Lenient):
    cnr_id: str | None = None  # the Comfy Registry pack, `comfy-core` for ComfyUI's own
    ver: str | None = None  # the pack's version (semver) or a git commit
    aux_id: str | None = None  # `owner/repo` on GitHub, for a pack outside the Registry
    models: list[ModelFile] = Field(default_factory=list)
    proxy_widgets: list[list[str]] = Field(default_factory=list, alias="proxyWidgets")

    @field_validator("models", mode="before")
    @classmethod
    def _models(cls, value: object) -> object:
        return (
            [m for m in value if isinstance(m, dict) and m.get("name")]
            if isinstance(value, list)
            else []
        )

    @field_validator("proxy_widgets", mode="before")
    @classmethod
    def _proxies(cls, value: object) -> object:
        if not isinstance(value, list):
            return []
        return [
            [str(p) for p in pair] for pair in value if isinstance(pair, list) and len(pair) == 2
        ]


class Slot(_Lenient):
    name: str = ""
    type: str | list[str] = ""
    link: int | None = None
    widget: dict[str, Any] | None = None  # set when the input is (or was) a widget
    label: str | None = None

    @property
    def widget_name(self) -> str | None:
        found = (self.widget or {}).get("name")
        return found if isinstance(found, str) else None


class OutSlot(_Lenient):
    name: str = ""
    type: str | list[str] = ""
    links: list[int] | None = None


class Node(_Lenient):
    id: int | str
    type: str
    mode: int = 0
    title: str | None = None
    inputs: list[Slot] = Field(default_factory=list)
    outputs: list[OutSlot] = Field(default_factory=list)
    widgets_values: list[Any] | dict[str, Any] | None = None
    properties: NodeProperties = Field(default_factory=NodeProperties)

    @field_validator("properties", mode="before")
    @classmethod
    def _properties(cls, value: object) -> object:
        return value if isinstance(value, dict) else {}

    @property
    def key(self) -> str:
        return str(self.id)


class Link(BaseModel):
    id: int
    origin_id: int | str
    origin_slot: int
    target_id: int | str
    target_slot: int
    type: str | list[str] | int = ""

    @classmethod
    def read(cls, raw: object) -> Link | None:
        """A link as schema 0.4 writes it (`[id, from, slot, to, slot, type]`) or schema 1
        and subgraphs do (an object); None for anything else (a corrupt entry)."""
        if isinstance(raw, list) and len(raw) >= 5:
            fields = dict(
                zip(
                    ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type"),
                    raw,
                    strict=False,
                )
            )
        elif isinstance(raw, dict):
            fields = raw
        else:
            return None
        try:
            return cls.model_validate(fields)
        except ValidationError:
            return None


class SubgraphSlot(_Lenient):
    name: str = ""
    type: str | list[str] = ""
    link_ids: list[int] = Field(default_factory=list, alias="linkIds")


class Graph(_Lenient):
    """What a workflow and a subgraph definition share: nodes and the links between them."""

    nodes: list[Node] = Field(default_factory=list)
    raw_links: list[Any] = Field(default_factory=list, alias="links")

    @property
    def links(self) -> dict[int, Link]:
        found = (Link.read(raw) for raw in self.raw_links)
        return {link.id: link for link in found if link is not None}

    @field_validator("nodes", mode="before")
    @classmethod
    def _nodes(cls, value: object) -> object:
        return (
            [n for n in value if isinstance(n, dict) and "type" in n]
            if isinstance(value, list)
            else []
        )


class Subgraph(Graph):
    id: str
    name: str = ""
    inputs: list[SubgraphSlot] = Field(default_factory=list)
    outputs: list[SubgraphSlot] = Field(default_factory=list)
    definitions: Definitions | None = None


class Definitions(_Lenient):
    subgraphs: list[Subgraph] = Field(default_factory=list)


class Workflow(Graph):
    version: float | int | str | None = None
    definitions: Definitions = Field(default_factory=Definitions)
    models: list[ModelFile] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("definitions", mode="before")
    @classmethod
    def _definitions(cls, value: object) -> object:
        return value if isinstance(value, dict) else {}

    @field_validator("models", mode="before")
    @classmethod
    def _top_models(cls, value: object) -> object:
        return (
            [m for m in value if isinstance(m, dict) and m.get("name")]
            if isinstance(value, list)
            else []
        )

    def subgraphs(self) -> dict[str, Subgraph]:
        """Every definition, nested ones included, by id."""
        found: dict[str, Subgraph] = {}
        pending = list(self.definitions.subgraphs)
        while pending:
            graph = pending.pop()
            found.setdefault(graph.id, graph)
            if graph.definitions is not None:
                pending += graph.definitions.subgraphs
        return found

    @property
    def group_nodes(self) -> list[str]:
        """Old group nodes (`workflow>Name`): the frontend migrates them, mnm refuses them."""
        groups = (self.extra or {}).get("groupNodes")
        return sorted(groups) if isinstance(groups, dict) else []


MAX_DEPTH = 100  # a workflow nests a dozen levels (nodes, inputs, a subgraph's own nodes)


def decode(data: bytes) -> object:
    """JSON from a file as sites serve it: UTF-8, with or without a byte-order mark, nested no
    deeper than a workflow goes (`MAX_DEPTH`): a file nested past Python's recursion limit
    had been a traceback, and one just under it would be one in every reader after this."""
    try:
        found = json.loads(data.decode("utf-8-sig"))
    except RecursionError:
        raise FormatError(f"nested deeper than a workflow goes (over {MAX_DEPTH} levels)") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"not JSON ({exc})") from None
    if _deeper_than(found, MAX_DEPTH):
        raise FormatError(f"nested deeper than a workflow goes (over {MAX_DEPTH} levels)")
    return found


def _deeper_than(found: object, limit: int) -> bool:
    """Whether JSON nests past `limit`, walked without recursion."""
    stack: list[tuple[object, int]] = [(found, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > limit:
            return True
        children = (
            item.values() if isinstance(item, dict) else item if isinstance(item, list) else ()
        )
        stack.extend((child, depth + 1) for child in children if isinstance(child, dict | list))
    return False


def is_api(data: object) -> bool:
    return (
        isinstance(data, dict)
        and bool(data)
        and "nodes" not in data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    )


def read_workflow(data: object) -> Workflow:
    """The UI format; FormatError naming what it is otherwise."""
    if is_api(data):
        raise FormatError("an API-format prompt, not a workflow")
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise FormatError("a JSON file without a workflow's `nodes`")
    try:
        return Workflow.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise FormatError(f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}") from None


Subgraph.model_rebuild()
