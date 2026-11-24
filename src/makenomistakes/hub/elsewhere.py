"""The sources whose repos are not on the Hub and whose files mnm keeps as plain files:
Ollama's library and ModelScope. One row each, so the places that tell a source
apart (a recipe's page, `mnm replan`, the web's repo link, `mnm rm --files`, downloads, the
Hub client's refusal) read one table; Civitai keeps its own (with its archive and token)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.download_modelscope import modelscope_fetcher, modelscope_path
from makenomistakes.download_ollama import ollama_fetcher, ollama_path
from makenomistakes.hub import modelscope, ollama
from makenomistakes.hub.client import HubError

Lookup = Callable[[str, str, str, Path], Path | None]  # (repo, file, revision, cache) -> path
Place = Callable[[Path, str, str, str], Path]  # (models, repo, file, revision) -> its path
Fetcher = Callable[[str, str, str, Path, str | None], Path]


@dataclass(frozen=True)
class Elsewhere:
    name: str
    subdir: str  # under the models folder
    owns: Callable[[str], bool]
    page: Callable[[str, str], str]  # (repo id, revision) -> the model's page there
    place: Place  # where a file goes; ValueError for a name that would leave its folder
    fetcher: Callable[[Path], Fetcher]  # (models folder) -> fetch one file, sha256-checked
    error: type[HubError]  # what asking the Hub about one of its repos raises

    def lookup(self, models_dir: Path) -> Lookup:
        """Where a file already is: its place, once it is there (a `.part` is not)."""

        def found(repo_id: str, filename: str, revision: str, _cache: Path) -> Path | None:
            path = self.place(models_dir, repo_id, filename, revision)
            return path if path.is_file() else None

        return found


SOURCES = (
    Elsewhere(
        "Ollama",
        ollama.SUBDIR,
        ollama.is_ollama,
        lambda repo, tag: ollama.ref_of(repo, tag).page,
        ollama_path,
        ollama_fetcher,
        ollama.OllamaError,
    ),
    Elsewhere(
        "ModelScope",
        modelscope.SUBDIR,
        modelscope.is_modelscope,
        lambda repo, rev: modelscope.ref_of(repo, rev).page,
        modelscope_path,
        modelscope_fetcher,
        modelscope.ModelScopeError,
    ),
)


def elsewhere(repo_id: str) -> Elsewhere | None:
    """The source a repo id belongs to when it is not the Hub (nor Civitai)."""
    return next((s for s in SOURCES if s.owns(repo_id)), None)
