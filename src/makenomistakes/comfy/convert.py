"""A UI-format workflow → the API prompt ComfyUI runs, as the frontend's
`graphToPrompt` builds it, checked against that frontend's own output (workflows exported
once from ComfyUI v0.42.1's frontend).

Every input is resolved to where its value really comes from:
- a node's output, through any reroute, a bypassed node (wired through to its matching
  input), the boundary of a subgraph (its inputs from the instance's links, its outputs from
  the node inside that feeds them), or a Set/Get pair (KJNodes);
- a primitive node's value;
- nothing (a muted node's output, an unconnected subgraph input): the node's own widget
  value then applies, or the input is left out.

Nodes inside subgraphs get the frontend's ids, `outer:inner` (`57:27`). Virtual nodes (notes,
reroutes, primitives, Set/Get) and muted or bypassed nodes are not sent. The idea follows
SethRobinson/comfyui-workflow-to-api-converter-endpoint (Unlicense); the code is mnm's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from makenomistakes.comfy.object_info import ObjectInfo
from makenomistakes.comfy.ui_format import (
    BYPASSED,
    MUTED,
    SUBGRAPH_INPUT,
    Graph,
    Link,
    Node,
    Subgraph,
    Workflow,
)

REROUTES = frozenset({"Reroute", "Reroute (rgthree)"})
PRIMITIVE = "PrimitiveNode"
SETTER, GETTER = "SetNode", "GetNode"  # KJNodes: a named wire
# Frontend-only nodes: drawn in the editor, never sent.
VIRTUAL = frozenset(
    {
        *REROUTES,
        PRIMITIVE,
        SETTER,
        GETTER,
        "Note",
        "MarkdownNote",
        "Label (rgthree)",
        "Fast Groups Bypasser (rgthree)",
        "Fast Groups Muter (rgthree)",
        "Fast Bypasser (rgthree)",
        "Fast Muter (rgthree)",
        "Bookmark (rgthree)",
    }
)
MAX_DEPTH = 64  # a cycle of reroutes or Set/Get pairs ends here


class ConvertError(ValueError):
    """The workflow cannot be turned into a prompt; the message names the node and why."""


@dataclass(frozen=True)
class Source:
    node: str
    slot: int


@dataclass(frozen=True)
class Fixed:
    value: object


Resolved = Source | Fixed | None


@dataclass
class Scope:
    """One graph being read: the workflow itself, or a subgraph inside an instance node."""

    graph: Graph
    prefix: str = ""
    parent: Scope | None = None
    instance: Node | None = None
    nodes: dict[str, Node] = field(init=False)
    links: dict[int, Link] = field(init=False)

    def __post_init__(self) -> None:
        self.nodes = {node.key: node for node in self.graph.nodes}
        self.links = self.graph.links

    @property
    def definition(self) -> Subgraph:
        assert isinstance(self.graph, Subgraph)
        return self.graph


class Converter:
    def __init__(self, workflow: Workflow, info: ObjectInfo) -> None:
        self.workflow = workflow
        self.info = info
        self.subgraphs = workflow.subgraphs()

    def convert(self) -> dict[str, dict[str, Any]]:
        if self.workflow.group_nodes:
            raise ConvertError(
                "it uses old group nodes ("
                + ", ".join(self.workflow.group_nodes)
                + "): open it once in ComfyUI's editor, which turns them into subgraphs, and "
                "import what it saves"
            )
        prompt: dict[str, dict[str, Any]] = {}
        self._emit(Scope(self.workflow), prompt, 0)
        return prompt

    # -- nodes ---------------------------------------------------------------------------

    def _emit(self, scope: Scope, prompt: dict[str, dict[str, Any]], depth: int) -> None:
        if depth > MAX_DEPTH:
            raise ConvertError("its subgraphs nest deeper than mnm follows")
        for node in scope.graph.nodes:
            if node.mode in (MUTED, BYPASSED):
                continue
            if node.type in self.subgraphs:
                inner = Scope(self.subgraphs[node.type], f"{scope.prefix}{node.key}:", scope, node)
                self._emit(inner, prompt, depth + 1)
                continue
            if node.type in VIRTUAL:
                continue
            info = self.info.get(node.type)
            if info is None:
                raise ConvertError(f"node {node.key} is a {node.type}, which this ComfyUI lacks")
            prompt[scope.prefix + node.key] = {
                "class_type": node.type,
                "inputs": self._inputs(scope, node),
                "_meta": {"title": node.title or info.display_name or node.type},
            }

    def _inputs(self, scope: Scope, node: Node) -> dict[str, Any]:
        inputs: dict[str, Any] = {}
        for slot in node.inputs:
            if slot.link is None:
                continue
            found = self._link(scope, slot.link, 0)
            name = slot.widget_name or slot.name
            if isinstance(found, Source):
                inputs[name] = [found.node, found.slot]
            elif isinstance(found, Fixed):
                inputs[name] = _wrapped(found.value)
        for name, value in self._widget_values(node).items():
            inputs.setdefault(name, _wrapped(value))
        return inputs

    def _widget_values(self, node: Node) -> dict[str, Any]:
        """Each widget's saved value by input name: `widgets_values` by position (skipping
        the frontend's own values, a dynamic combo's chosen option followed by its inputs),
        or by key when a pack saves a dict; an input the save predates gets its default."""
        values = node.widgets_values
        pending = list(self.info.widgets(node.type))
        found: dict[str, Any] = {}
        position = 0
        while pending:
            widget = pending.pop(0)
            if isinstance(values, dict):
                value = values.get(widget.name, widget.default)
            elif isinstance(values, list) and position < len(values):
                value = values[position]
                position += 1 + widget.extras
            else:
                value = widget.default
            found[widget.name] = value
            pending[0:0] = widget.nested(value)
        return found

    # -- where an input's value comes from -----------------------------------------------

    def _link(self, scope: Scope, link_id: int, depth: int) -> Resolved:
        link = scope.links.get(link_id)
        if link is None:
            return None
        return self._output(scope, link.origin_id, link.origin_slot, depth)

    def _output(  # noqa: PLR0911 - one answer per kind of node a wire can come from
        self, scope: Scope, origin: int | str, slot: int, depth: int
    ) -> Resolved:
        if depth > MAX_DEPTH:
            raise ConvertError("a loop of reroutes or Set/Get nodes never reaches a node")
        if origin == SUBGRAPH_INPUT or str(origin) == str(SUBGRAPH_INPUT):
            return self._from_outside(scope, slot, depth)
        node = scope.nodes.get(str(origin))
        if node is None or node.mode == MUTED:
            return None
        if node.type in self.subgraphs and node.mode != BYPASSED:
            return self._from_inside(scope, node, slot, depth)
        if node.mode == BYPASSED:
            return self._through(scope, node, slot, depth)
        if node.type in REROUTES or node.type == SETTER:  # a SetNode passes its wire on too
            first = node.inputs[0].link if node.inputs else None
            return self._link(scope, first, depth + 1) if first is not None else None
        if node.type == PRIMITIVE:
            values = node.widgets_values
            return Fixed(values[0]) if isinstance(values, list) and values else None
        if node.type == GETTER:
            return self._getter(scope, node, depth)
        return Source(scope.prefix + node.key, slot)

    def _from_outside(self, scope: Scope, slot: int, depth: int) -> Resolved:
        """A subgraph input: what feeds the instance's matching input outside. Unlinked, None
        leaves the inner node's widget value, which is the instance's: the frontend keeps a
        promoted widget's value on the inner node (equal to its export on every workflow
        checked)."""
        if scope.parent is None or scope.instance is None:
            return None
        definition = scope.definition
        name = definition.inputs[slot].name if slot < len(definition.inputs) else None
        # by name: an instance shows only the inputs it promoted, so positions differ
        outer = next((s for s in scope.instance.inputs if s.name == name), None)
        if outer is None or outer.link is None:
            return None
        return self._link(scope.parent, outer.link, depth + 1)

    def _from_inside(self, scope: Scope, instance: Node, slot: int, depth: int) -> Resolved:
        definition = self.subgraphs[instance.type]
        if slot >= len(definition.outputs) or not definition.outputs[slot].link_ids:
            return None
        inner = Scope(definition, f"{scope.prefix}{instance.key}:", scope, instance)
        return self._link(inner, definition.outputs[slot].link_ids[0], depth + 1)

    def _through(self, scope: Scope, node: Node, slot: int, depth: int) -> Resolved:
        """A bypassed node passes on the input that matches the output asked for: the same
        slot when its type matches, else the first input of that type, else the first
        input that takes anything."""
        out_type = node.outputs[slot].type if slot < len(node.outputs) else ""
        candidates = [
            *(
                [node.inputs[slot]]
                if slot < len(node.inputs) and node.inputs[slot].type == out_type
                else []
            ),
            *[s for s in node.inputs if s.type == out_type],
            *[s for s in node.inputs if s.type in ("*", "") or _overlaps(s.type, out_type)],
        ]
        for candidate in candidates:
            if candidate.link is not None:
                return self._link(scope, candidate.link, depth + 1)
        return None

    def _getter(self, scope: Scope, getter: Node, depth: int) -> Resolved:
        name = _first_value(getter)
        for candidate in _every_node(scope):
            owner, setter = candidate
            if setter.type == SETTER and _first_value(setter) == name and setter.inputs:
                link = setter.inputs[0].link
                return self._link(owner, link, depth + 1) if link is not None else None
        return None


def _every_node(scope: Scope) -> list[tuple[Scope, Node]]:
    """This graph's nodes, then its parents' (a Get finds the Set it names in either)."""
    found: list[tuple[Scope, Node]] = []
    current: Scope | None = scope
    while current is not None:
        found += [(current, node) for node in current.graph.nodes]
        current = current.parent
    return found


def _first_value(node: Node) -> object:
    values = node.widgets_values
    return values[0] if isinstance(values, list) and values else None


def _overlaps(a: str | list[str], b: str | list[str]) -> bool:
    def names(t: str | list[str]) -> set[str]:
        return set(t) if isinstance(t, list) else set(str(t).split(","))

    return bool(names(a) & names(b))


def _wrapped(value: object) -> object:
    """The frontend sends a list value as `{"__value__": […]}`: a bare list would read as a
    link."""
    return {"__value__": value} if isinstance(value, list) else value


def to_prompt(workflow: Workflow, info: ObjectInfo) -> dict[str, dict[str, Any]]:
    return Converter(workflow, info).convert()
