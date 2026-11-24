"""Instance recipes on disk, one JSON file each, plus port allocation.

All writes go through this class under one lock, and updates re-read before writing, so a
background download can never overwrite a settings change made meanwhile. The lock is a file
lock on the state dir (the web service and a terminal are two processes: a thread lock alone
would let `mnm set` lose to the service's download checkpoint).
"""

from __future__ import annotations

import contextlib
import difflib
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from pydantic import ValidationError

from makenomistakes.hub.facts import Modality
from makenomistakes.json_folder import BadNameError, JsonFolder, cause, sentence, unlisted
from makenomistakes.recipes.model import NewerRecipeError, Recipe, load_recipe_json

PORT_BLOCKS: dict[Modality, range] = {
    Modality.CHAT: range(7810, 7820),
    Modality.CODING: range(7810, 7820),
    Modality.VISION_CHAT: range(7810, 7820),
    Modality.IMAGE_GEN: range(7820, 7830),
    Modality.IMAGE_EDIT: range(7820, 7830),
    Modality.SPEECH_TO_TEXT: range(7830, 7840),
    Modality.TEXT_TO_SPEECH: range(7830, 7840),
    Modality.EMBEDDING: range(7840, 7850),
    Modality.RERANK: range(7840, 7850),
    Modality.UNKNOWN: range(7850, 7860),
    Modality.VIDEO: range(7860, 7870),  # its own block: the video route is not OpenAI's
    # The studio: audio.cpp's task route, one block per type above the overflow
    Modality.SONG: range(7900, 7910),
    Modality.STEMS: range(7910, 7920),
    Modality.VOICE_CONVERT: range(7920, 7930),
    Modality.MUSIC_SCORE: range(7930, 7940),
}
# When a type's block is full (ten image instances take 7820-7829), the next ones share these.
OVERFLOW_PORTS = range(7870, 7900)


class StoreError(RuntimeError):
    pass


class NoSuchInstanceError(StoreError):
    """The name has no recipe file (the CLI adds where to find the names)."""


def _unreadable(path: Path, why: str) -> str:
    return (
        f"The recipe file {path} cannot be read by this version ({why}). Delete that file and "
        "`mnm add` the model again; the downloaded weights are kept."
    )


def _unread_from_disk(path: Path, exc: OSError) -> str:
    """A recipe the system would not give back (permission denied, an I/O error, a folder
    where the file belongs): never the "this version" words, whose remedy deletes a file that
    may be fine, and no remedy of its own, since each cause has another (else "reads it
    again once it can (…the disk)" is said of a folder in its place)."""
    return f"The recipe file {path} could not be read ({cause(exc)})."


class InstanceStore:
    def __init__(self, state_dir: Path) -> None:
        self._files = JsonFolder(state_dir / "instances")
        self._dir = self._files.folder
        self._lock = self._files.locked  # threads and processes (locking.py)

    def _locked(self) -> AbstractContextManager[None]:
        """The folder's lock through its door (`JsonFolder.held`): one the system will not
        take is a StoreError naming the folder, like every other failure to read a recipe
        (else the router's tick raises through this door every 5 s)."""
        return self._files.held(
            lambda exc: StoreError(
                f"The recipes folder {self._dir} could not be locked ({cause(exc)})."
            )
        )

    def path(self, name: str) -> Path:
        try:
            return self._files.path(name)
        except BadNameError:
            raise self._missing(name) from None

    def exists(self, name: str) -> bool:
        return self._files.exists(name)

    def save(self, recipe: Recipe) -> Path:
        with self._locked():
            return self._write(recipe)

    def create(
        self,
        modality: Modality,
        in_use: Callable[[int], bool],
        build: Callable[[int], Recipe],
    ) -> Recipe:
        """A new recipe on a free port, allocated, built and written under the one lock: two
        adds at once (two web tabs) would each get 7810 if the write came after the lock."""
        with self._locked():
            port = self.allocate_port(modality, in_use)
            recipe = build(port)
            if self.exists(recipe.name):
                raise StoreError(f"An instance named '{recipe.name}' already exists.")
            self._write(recipe)
            return recipe

    def update(self, name: str, change: Callable[[Recipe], Recipe]) -> Recipe:
        """Re-read, apply `change`, write: the only safe way to modify from a background job."""
        with self._locked():
            updated = change(self.load(name))
            self._write(updated)
            return updated

    def merge_local_paths(self, name: str, local_paths: dict[str, str]) -> Recipe:
        return self.update(
            name,
            lambda r: r.model_copy(update={"local_paths": {**r.local_paths, **local_paths}}),
        )

    def load(self, name: str) -> Recipe:
        with self._locked():
            path = self.path(name)
            try:
                recipe = load_recipe_json(self._files.read(name))
            except FileNotFoundError as exc:
                raise self._missing(name) from exc
            except NewerRecipeError as exc:
                raise StoreError(
                    f"The recipe file {path} was written by a newer version of makenomistakes "
                    f"({exc}). Run `mnm update`, then try again."
                ) from exc
            except ValidationError as exc:
                first = exc.errors()[0]
                where = ".".join(str(part) for part in first["loc"]) or "file"
                raise StoreError(_unreadable(path, f"{where}: {first['msg']}")) from exc
            except ValueError as exc:  # not JSON, or no usable schema_version
                raise StoreError(_unreadable(path, str(exc))) from exc
            except OSError as exc:  # EACCES, EIO, EISDIR: the store's error naming the file,
                raise StoreError(_unread_from_disk(path, exc)) from exc  # never a raw one
            if (
                recipe.name != name
            ):  # renamed by hand: it would list under one name, load under another
                raise StoreError(_unreadable(path, f"it names the instance '{recipe.name}'"))
            return recipe

    def delete(self, name: str) -> None:
        with self._locked():
            try:
                self._files.unlink(name)
            except (FileNotFoundError, BadNameError) as exc:
                raise self._missing(name) from exc

    def _missing(self, name: str) -> NoSuchInstanceError:
        """Names the closest saved one, so `mnm stop klien` says `klein` (no hint when the
        folder cannot be listed: the hint is all it would lose)."""
        names: list[str] = []
        with contextlib.suppress(OSError):
            names = self.names()
        close = difflib.get_close_matches(name, names, n=1)
        hint = f"; did you mean '{close[0]}'?" if close else ""
        return NoSuchInstanceError(f"no instance named '{name}'{hint}")

    def names(self) -> list[str]:
        """The saved instances' names, no recipe read (a hint, shell completion); OSError when
        the folder will not list."""
        return self._files.names()

    def list(self) -> list[Recipe]:
        """Every recipe; a folder the system will not list is a StoreError naming it, never
        an empty list (read as empty, the quality tune's sweep would delete a file a saved
        recipe runs, as would `mnm rm --files` and the disk policy)."""
        with self._locked():
            try:
                names = self._files.names()
            except OSError as exc:
                raise StoreError(sentence(unlisted("recipes folder", self._dir, exc))) from exc
            return sorted((self.load(name) for name in names), key=lambda r: r.created_at)

    def allocate_port(
        self, modality: Modality, in_use: Callable[[int], bool] = lambda _port: False
    ) -> int:
        """The first port of the modality's block, then of the shared overflow, that no
        recipe has and `in_use` (another program listening) does not report (else 7810 is
        handed out again while something else holds it)."""
        with self._locked():
            taken = {r.port for r in self.list()}
            candidates: list[int] = [*PORT_BLOCKS[modality], *OVERFLOW_PORTS]
            for port in candidates:
                if port not in taken and not in_use(port):
                    return port
        block = PORT_BLOCKS[modality]
        raise StoreError(
            f"every port mnm gives {modality.label} instances is taken ({block.start}-"
            f"{block.stop - 1} and the shared {OVERFLOW_PORTS.start}-{OVERFLOW_PORTS.stop - 1}): "
            "`mnm rm` an instance you no longer use (its downloads stay)."
        )

    def _write(self, recipe: Recipe) -> Path:
        return self._files.write(recipe.name, recipe.model_dump_json(indent=2))
