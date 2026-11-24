"""The vocals stem's layers: what the separator gave, what the vocals stage made of
it, what a voice sang from that. `stem-vocals.wav` always holds the newest; each stage that
replaces it keeps its own input beside it once (the vocals stage the separator's as
`vocals-raw.wav`, the voice stage its input as `vocals-as-sung.wav`) and puts it back when
undone, in reverse stage order (the voice's first, then the vocals stage's: `_undo`).

`original` is the one answer to "the vocals as the song model sang them, before anything
changed them": what an edit sums into a new take (a swap, a key change by stems), which is
split and cleaned again; reading the stem or the voice's input there would clean cleaned
vocals twice.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable
from pathlib import Path

from makenomistakes.pipeline import song
from makenomistakes.runs import RunRecord, RunStore

VOCALS = song.stem_file("vocals")
RAW = "vocals-raw.wav"  # the separator's, kept by the vocals stage
AS_SUNG = "vocals-as-sung.wav"  # the voice stage's input (the cleaned vocals, or the separator's)
ORIGINALS = (RAW, AS_SUNG)  # the oldest layer first


def original(names: Iterable[str], folder: Path) -> str:
    """The oldest layer the song's folder holds: the separator's own vocals."""
    present = set(names)
    return next(
        (n for n in ORIGINALS if n in present and (folder / n).is_file()),
        VOCALS,
    )


def is_vocals(name: str) -> bool:
    return name == VOCALS or name in ORIGINALS


def keep(store: RunStore, run: RunRecord, layer: str) -> Path | None:
    """The stem kept as `layer` once (a stage's input); None when there is no vocals stem."""
    folder = store.folder(run)
    kept = folder / layer
    if kept.is_file():
        return kept
    stem = folder / VOCALS
    if not stem.is_file():
        return None
    shutil.copyfile(stem, kept)
    store.keep(run, kept)
    return kept


def put_back(store: RunStore, run: RunRecord, layer: str) -> bool:
    """`layer` in the stem's place again, and dropped; False when it was never kept."""
    folder = store.folder(run)
    kept = folder / layer
    if not kept.is_file():
        return False
    shutil.copyfile(kept, folder / VOCALS)
    store.keep(run, folder / VOCALS)
    drop(store, run, layer)
    return True


def drop(store: RunStore, run: RunRecord, name: str) -> None:
    run.files = [f for f in run.files if f.name != name]
    (store.folder(run) / name).unlink(missing_ok=True)


def no_stem(run: RunRecord, what: str) -> str:
    """Why there is nothing to `what`: the stems the song has instead."""
    stems = ", ".join(song.stem_files([f.name for f in run.outputs])) or "none"
    return (
        f"No vocals stem to {what} (the stems are {stems}): the stems stage's tier gives "
        "vocals with HTDemucs or a RoFormer."
    )
