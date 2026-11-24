"""Where each card is now: this box's cards by stable id (their uuids) -> nvidia-smi
index, as `machine.json` last recorded them (the router at its start, the doctor, `mnm ls`,
`mnm machine show`, a plan's save). A saved placement names its cards by id, so a card that
moved slot is found at its new index (`card_placement.Placement.remapped`), and one that
left is told apart from one that did not.

Read again only when the file changed (the web board asks every 10 s); None when no box was
recorded or the file cannot be read: a placement then renders by its saved indices."""

from __future__ import annotations

import threading
from pathlib import Path

from makenomistakes.machine_facts import FILE, MachineFactsError, load_file

_cache: dict[Path, tuple[tuple[int, int], dict[str, int] | None]] = {}
_lock = threading.Lock()


def live_ids(state_dir: Path) -> dict[str, int] | None:
    """This box's cards, stable id -> index; None when nothing was recorded."""
    path = state_dir / FILE
    try:
        info = path.stat()
    except OSError:
        return None
    stamp = (info.st_mtime_ns, info.st_size)
    with _lock:
        cached = _cache.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    try:
        found = load_file(state_dir)
    except (MachineFactsError, OSError):
        found = None
    current = found.current.fingerprint if found is not None and found.current else None
    ids = {c.stable_id: c.index for c in current.cards} if current is not None else None
    with _lock:
        _cache[path] = (stamp, ids)
    return ids


def live_ids_now() -> dict[str, int] | None:
    """`live_ids` of the state folder the settings name, read at the call."""
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    try:
        return live_ids(settings_now().expanded_state_dir())
    except ValueError:  # settings broken: they say so wherever they are read
        return None
