"""The weights a style training reads: ACE-Step 1.5's DiT for the variant trained
against (base, or turbo), its VAE and its text encoder (Qwen3-Embedding-0.6B), each file at a
pinned revision, downloaded into the models folder's Hugging Face cache as `mnm add` does and
mounted read-only into the training's container. The 5 Hz language model is not needed.

They are fetched when a training is queued (`mnm studio train style`, the web's job), never by
the router: a 6 GB download must not hold the card. Read from the Hub's tree API on 2026-11-02.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from makenomistakes.config import Settings
from makenomistakes.download import Fetcher, hub_cache_lookup, hub_fetcher
from makenomistakes.recipes.model import CONTAINER_MODELS_DIR, HF_CACHE_SUBDIR
from makenomistakes.studio.style_trainer import StyleTrainerError

StyleVariant = Literal["base", "turbo"]
SUITE, SUITE_REV = "ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22"
BASE, BASE_REV = "ACE-Step/acestep-v15-base", "e432212fec32b8965a14ffa57ae653438d6abd14"
SPARE_BYTES = 2 * 2**30  # the models disk keeps this much free after the download
TRAINER = "the style trainer"  # who lists its weights in `mnm disk`
GB = 1_000_000_000


@dataclass(frozen=True)
class HubFile:
    repo: str
    revision: str
    path: str
    size: int


def _dit(repo: str, revision: str, folder: str) -> tuple[HubFile, ...]:
    prefix = f"{folder}/" if folder else ""
    return (
        HubFile(repo, revision, f"{prefix}model.safetensors", 4_787_825_604),
        HubFile(repo, revision, f"{prefix}config.json", 1_968 if folder else 1_940),
        HubFile(repo, revision, f"{prefix}silence_latent.pt", 3_841_215),
    )


DIT: dict[StyleVariant, tuple[HubFile, ...]] = {
    "base": _dit(BASE, BASE_REV, ""),
    "turbo": _dit(SUITE, SUITE_REV, "acestep-v15-turbo"),
}
VAE = (
    HubFile(SUITE, SUITE_REV, "vae/config.json", 425),
    HubFile(SUITE, SUITE_REV, "vae/diffusion_pytorch_model.safetensors", 337_431_388),
)
TEXT_ENCODER = tuple(
    HubFile(SUITE, SUITE_REV, f"Qwen3-Embedding-0.6B/{name}", size)
    for name, size in (
        ("model.safetensors", 1_191_586_416),
        ("config.json", 1_359),
        ("tokenizer.json", 11_423_705),
        ("tokenizer_config.json", 5_404),
        ("vocab.json", 2_776_833),
        ("merges.txt", 1_671_853),
        ("added_tokens.json", 707),
        ("special_tokens_map.json", 613),
        ("chat_template.jinja", 4_116),
    )
)


def hub() -> Fetcher:
    """The Hub's downloader, the one seam the CLI's and the web's trainings share (tests
    replace it)."""
    return hub_fetcher


class StyleWeightsError(StyleTrainerError):
    """The weights could not be found or fetched, worded for the user."""


@dataclass(frozen=True)
class StyleWeights:
    """Each part's folder on this machine (a snapshot folder of the cache)."""

    dit_dir: Path
    vae_dir: Path
    text_encoder_dir: Path


def files_for(variant: StyleVariant) -> tuple[HubFile, ...]:
    return (*DIT[variant], *VAE, *TEXT_ENCODER)


def _cache(settings: Settings) -> Path:
    return settings.expanded_models_dir() / HF_CACHE_SUBDIR


def _found(settings: Settings, variant: StyleVariant) -> dict[HubFile, Path]:
    return _found_in(_cache(settings), variant)


def _found_in(cache: Path, variant: StyleVariant) -> dict[HubFile, Path]:
    found = {}
    for file in files_for(variant):
        path = hub_cache_lookup(file.repo, file.path, file.revision, cache)
        if path is not None:
            found[file] = path
    return found


def _weights(paths: dict[HubFile, Path], variant: StyleVariant) -> StyleWeights:
    return StyleWeights(
        dit_dir=paths[DIT[variant][0]].parent,
        vae_dir=paths[VAE[1]].parent,
        text_encoder_dir=paths[TEXT_ENCODER[0]].parent,
    )


def present(settings: Settings, variant: StyleVariant) -> StyleWeights | None:
    """Every file on disk already: their folders; else None."""
    found = _found(settings, variant)
    if len(found) < len(files_for(variant)):
        return None
    return _weights(found, variant)


def fetch(
    settings: Settings,
    variant: StyleVariant,
    fetcher: Fetcher | None = None,
    say: Callable[[str], None] = lambda _line: None,
) -> StyleWeights:
    """The weights on disk, downloading what is missing first (the models disk checked for
    room before anything is fetched)."""
    found = _found(settings, variant)
    wanted = [f for f in files_for(variant) if f not in found]
    need = sum(f.size for f in wanted)
    if wanted:
        folder = _cache(settings)
        folder.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(folder).free
        if free - need < SPARE_BYTES:
            raise StyleWeightsError(
                f"The {variant} checkpoint needs {need / GB:.1f} GB more in {folder}, which "
                f"has {free / GB:.1f} GB free (2 GB kept spare): free some space (`mnm disk` "
                "says what holds it), then train again."
            )
        say(f"Downloading ACE-Step 1.5 {variant}'s training weights: {need / GB:.1f} GB.")
    for file in wanted:
        try:
            found[file] = (fetcher or hub())(
                file.repo, file.path, file.revision, _cache(settings), None
            )
        except Exception as exc:
            raise StyleWeightsError(
                f"{file.repo}/{file.path} did not download ({exc}): train again to resume it."
            ) from exc
    return _weights(found, variant)


def listed(models_dir: Path) -> list[tuple[str, str]]:
    """Every variant's weights on disk, relative to the models folder, with who uses them, so
    `mnm disk` never calls them leftovers ("no instance or workflow") while a queued or
    paused style training reads them."""
    cache = models_dir / HF_CACHE_SUBDIR
    return [
        (path.relative_to(models_dir).as_posix(), TRAINER)
        for variant in DIT
        for path in _found_in(cache, variant).values()
    ]


def container_path(settings: Settings, path: Path) -> str:
    """A folder under the models folder as the training's container sees it."""
    relative = path.relative_to(settings.expanded_models_dir())
    return f"{CONTAINER_MODELS_DIR}/{relative.as_posix()}"
