"""A folder of `<name>.json` files written whole, behind one lock across threads and
processes (`locking.FolderLock`). The instance store and the workflow store are both one
of these plus their own parsing; the shape of the files is theirs."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from makenomistakes.locking import shared_lock

NAME_MAX = 255  # bytes in one file name, on every filesystem mnm keeps its state on


class BadNameError(LookupError):
    """A name that is no file name here: no such entry, as far as the folder can say."""


def cause(exc: OSError) -> str:
    """What the system said, without the path it repeats (`Permission denied`, `Is a
    directory`, `Input/output error`), for a store's words that name the file or folder once."""
    return exc.strerror or str(exc) or type(exc).__name__


def unlisted(noun: str, folder: Path, exc: OSError) -> str:
    """The one clause for a folder the system will not list, for every store's words: "the
    runs folder /x could not be listed (Permission denied)". A store that raises it on its own
    makes it a `sentence`."""
    return f"the {noun} {folder} could not be listed ({cause(exc)})"


def sentence(clause: str) -> str:
    """A clause said on its own: a capital first, a full stop last (a path keeps its case)."""
    return f"{clause[:1].upper()}{clause[1:]}."


class JsonFolder:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        # the process's one lock for the folder: two stores of it nest instead of deadlocking
        # on their own `flock` (`shared_lock`'s hazard)
        self.locked = shared_lock(folder)

    @contextmanager
    def held(self, refused: Callable[[OSError], Exception]) -> Iterator[None]:
        """The folder's lock, a store's one door to it: a lock the system will not take (a
        `.lock` this user cannot open, a folder it cannot enter) is the store's own error,
        built by `refused`, like every other failure to read the store (a raw `OSError` would
        raise at the router's tick every 5 s). Only taking the lock is turned into the store's
        error: what fails while it is held stays what it is."""
        with ExitStack() as hold:
            try:
                hold.enter_context(self.locked())
            except OSError as exc:
                raise refused(exc) from exc
            yield

    def path(self, name: str) -> Path:
        """`<folder>/<name>.json`; BadNameError for a name that would leave the folder or hide in
        it (`../instances/x` would delete a recipe through the workflow store), and for one too
        long to be a file's (no such entry, never read as a disk error).
        A name is measured as the filesystem encodes it: one listed from a file whose name is not
        UTF-8 holds lone surrogates, which strict UTF-8 refuses."""
        if not name or name.startswith(".") or "/" in name or "\\" in name:
            raise BadNameError(f"'{name}' is not a file name")
        if len(os.fsencode(f"{name}.json")) > NAME_MAX:  # the bytes the filesystem sees
            raise BadNameError(f"'{name[:40]}…' is too long for a file name")
        return self.folder / f"{name}.json"

    def exists(self, name: str) -> bool:
        try:
            return self.path(name).exists()
        except BadNameError:
            return False

    def names(self) -> list[str]:
        """Every file's name, dot-files (the lock, a write in flight) left out; a folder that is
        not there has none. One the system will not list raises its OSError: `Path.glob`
        swallows it, so an unlistable folder would read as empty, and the quality tune's sweep
        would delete what that folder's entries hold."""
        try:
            with os.scandir(self.folder) as found:
                files = [entry.name for entry in found]
        except FileNotFoundError:
            return []
        return sorted(
            name.removesuffix(".json")
            for name in files
            if name.endswith(".json") and not name.startswith(".")
        )

    def write(self, name: str, text: str) -> Path:
        """Written to a sibling temp file and moved into place: a reader never sees half."""
        self.folder.mkdir(parents=True, exist_ok=True)
        target = self.path(name)
        tmp = target.with_name(f".{name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(target)
        return target

    def read(self, name: str) -> str:
        return self.path(name).read_text(encoding="utf-8")

    def unlink(self, name: str) -> None:
        self.path(name).unlink()
