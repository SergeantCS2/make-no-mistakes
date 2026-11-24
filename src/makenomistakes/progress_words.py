"""A progress line's kind, for the CLI that prints it: most lines name a step under way, and
the CLI prints them with a trailing "…"; a `Whole` line is a finished sentence (a warning,
what was stopped) and is printed as it is. 7.10 printed "… which is slower. …"."""

from __future__ import annotations


class Whole(str):
    """A progress line that is a whole sentence: printed as it is, never followed by "…"."""

    __slots__ = ()


def with_ellipsis(message: str) -> str:
    """How a CLI prints a progress line: a step under way gets "…", a `Whole` one nothing."""
    return message if isinstance(message, Whole) else f"{message} …"


def parked_for_room(names: list[str], for_name: str | None = None) -> str:
    """What a start says it did with the models it moved aside: they were parked (unloaded,
    loaded again at their next request), never stopped (`mnm stop` is off): "Stopped qwen3-8b
    to make room." would contradict the *parked* the list then shows."""
    who = ", ".join(names)
    them = "it" if len(names) == 1 else "them"
    room = f"to make room for {for_name}" if for_name else "to make room"
    return f"Parked {who} {room} (the next request loads {them} again)."


def parking_for_room(name: str, for_name: str) -> str:
    """The progress line while one is being parked for room."""
    return f"Parking {name} to make room for {for_name}"
