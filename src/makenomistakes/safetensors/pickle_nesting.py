"""How deep a pickle nests what it would make, read from its opcodes before anything is made.

CPython hashes a tuple by recursing through it in C with no guard: a 1 MB pickle whose dict key
was a tuple nested a million deep overflowed the C stack while it loaded (exit 139), taking
`mnm voices add`, `mnm serve` or the router down with it, though the restricted unpickler ran
nothing it named. A checkpoint of weights nests a few levels (a state dict, a tensor's
arguments, a training checkpoint's dicts around them), so `refuse_deep` refuses anything nested
deeper than `MAX_DEPTH` before the pickle is loaded, whatever would have recursed through it.

`pickletools.genops` decodes the opcodes and constructs nothing. Each stack slot holds what the
object there nests to: a number once it is settled (a scalar, a tuple or frozenset of settled
parts), or an `_Open` node while it may still grow (a list, a dict, a set, whatever a call
returns), which knows what holds it, so adding to it later deepens everything around it. A loop
(a list that holds itself) deepens without end and is refused the same way. This counts on what
mnm's restricted unpickler calls: each call makes a new object and never hands back a list or
dict it was given (`torch_ckpt._rebuild_parameter` takes a tensor only).
"""

from __future__ import annotations

import pickletools
from dataclasses import dataclass, field
from typing import Final

MAX_DEPTH: Final = 64  # a RoFormer's or an RVC voice's pickle nests 6 or fewer
_SETTLED = frozenset({"EMPTY_TUPLE", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "FROZENSET"})
_OPEN = frozenset(
    {"EMPTY_LIST", "LIST", "EMPTY_DICT", "DICT", "EMPTY_SET"}
    | {"REDUCE", "NEWOBJ", "NEWOBJ_EX", "INST", "OBJ"}
)
_ADDS = frozenset({"APPEND", "APPENDS", "SETITEM", "SETITEMS", "ADDITEMS", "BUILD"})
_PUTS = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_GETS = frozenset({"GET", "BINGET", "LONG_BINGET"})
_UNCHANGED = frozenset({"PROTO", "FRAME", "READONLY_BUFFER"})


class NestingError(ValueError):
    """A pickle that nests what it holds deeper than `MAX_DEPTH` (or in a loop)."""


@dataclass(eq=False, slots=True)
class _Open:
    """An object the pickle may still add to, and the objects that hold it."""

    depth: int = 1
    holders: list[_Open] = field(default_factory=list)


_Slot = int | _Open


def _depth(slot: _Slot) -> int:
    return slot.depth if isinstance(slot, _Open) else slot


def _too_deep() -> NestingError:
    return NestingError(f"nests what it holds more than {MAX_DEPTH} levels deep")


def _deepen(node: _Open, depth: int) -> None:
    pending = [(node, depth)]
    while pending:
        node, depth = pending.pop()
        if depth <= node.depth:
            continue
        if depth > MAX_DEPTH:
            raise _too_deep()
        node.depth = depth
        pending.extend((holder, depth + 1) for holder in node.holders)


def _hold(node: _Open, parts: list[_Slot]) -> None:
    for part in parts:
        if isinstance(part, _Open):
            part.holders.append(node)
        _deepen(node, _depth(part) + 1)


def _settled(parts: list[_Slot]) -> _Slot:
    if any(isinstance(part, _Open) for part in parts):
        node = _Open()
        _hold(node, parts)
        return node
    depth = 1 + max(map(_depth, parts), default=0)
    if depth > MAX_DEPTH:
        raise _too_deep()
    return depth


@dataclass
class _Stack:
    """The unpickler's stack as depths: marks as positions, as CPython keeps them."""

    slots: list[_Slot] = field(default_factory=list)
    marks: list[int] = field(default_factory=list)

    def _above_mark(self, count: int) -> int:
        """Where the last `count` slots start, refused when they reach past the last mark
        (CPython's "unpickling stack underflow")."""
        start = len(self.slots) - count
        if start < (self.marks[-1] if self.marks else 0):
            raise ValueError("it takes more than it made")
        return start

    def take(self, count: int) -> list[_Slot]:
        start = self._above_mark(count)
        taken = self.slots[start:]
        del self.slots[start:]
        return taken

    def peek(self) -> _Slot:
        return self.slots[self._above_mark(1)]

    def to_mark(self) -> list[_Slot]:
        if not self.marks:
            raise ValueError("it closes a mark it never opened")
        at = self.marks.pop()
        taken = self.slots[at:]
        del self.slots[at:]
        return taken


def _taken(op: pickletools.OpcodeInfo, stack: _Stack, keep: int = 0) -> list[_Slot]:
    """What `op` takes off the stack: down to its mark, or its count less the `keep` it
    leaves in place (the list, dict or object it adds to)."""
    if pickletools.markobject in op.stack_before:
        return stack.to_mark()
    return stack.take(len(op.stack_before) - keep)


def _step(
    op: pickletools.OpcodeInfo, arg: object, stack: _Stack, memo: dict[object, _Slot]
) -> None:
    name = op.name
    if name == "MARK":
        stack.marks.append(len(stack.slots))
    elif name in _GETS:
        if arg not in memo:
            raise ValueError("it reads back what it never kept")
        stack.slots.append(memo[arg])
    elif name in _PUTS or name == "MEMOIZE":
        memo[arg if name in _PUTS else len(memo)] = stack.peek()
    elif name == "DUP":
        stack.slots.append(stack.peek())
    elif name == "POP" and stack.marks and stack.marks[-1] == len(stack.slots):
        stack.marks.pop()  # POP on an empty frame drops its mark, as CPython's does
    elif name in _ADDS:
        parts = _taken(op, stack, keep=1)
        target = stack.peek()
        if isinstance(target, _Open):  # else a scalar, a tuple or a sealed global: the
            _hold(target, parts)  # load itself refuses an add to it, in its own words
    elif name in _SETTLED:
        stack.slots.append(_settled(_taken(op, stack)))
    elif name in _OPEN:
        node = _Open()
        _hold(node, _taken(op, stack))
        stack.slots.append(node)
    elif name not in _UNCHANGED:  # a scalar, a global, a storage, STOP, POP: nothing nests
        _taken(op, stack)
        stack.slots += [0] * len(op.stack_after)


def refuse_deep(body: bytes) -> None:
    """Raise `NestingError` when `body` would make anything nested deeper than `MAX_DEPTH`
    (or a loop), and a plain `ValueError` when its opcodes do not add up (genops' own for a
    cut pickle or an unknown opcode); construct nothing."""
    stack, memo = _Stack(), dict[object, _Slot]()
    for op, arg, _pos in pickletools.genops(body):
        _step(op, arg, stack, memo)
