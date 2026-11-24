"""The style library: the LoRA adapters that teach ACE-Step a style, one row each. A
row names the ACE-Step variant it was trained against (it plays on that variant's GGUF only:
upstream's advice, and the base and turbo models' noise schedules differ), its trigger word
(the tag the training prepended to every caption, which a song's style line carries to call
it up), its files by sha256, whose songs it learned from (`source`) and, for a trained one, how
(`row_library.Training`: settings, the songs by hash, the box).

The rows live under `<state>/studio/styles/<name>.json`; the files under
`<models>/styles/<name>/`, where audio.cpp's container reads them as `ace_step.lora` (it mounts
the models folder only). The uninstall keeps the files, as it keeps the voices.

An adapter is checked as audio.cpp reads it (`docs/lora.md`, `weight_adapter.cpp` at c0cd77f):
PEFT's `adapter_config.json` beside `adapter_model.safetensors`, no DoRA, no fan-in/fan-out
transpose, no QALoRA, no trained bias, and every tensor a `lora_A`/`lora_B` of the DiT's
decoder; anything else is refused before a row is written.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from makenomistakes.engine_args_music import STYLE_ADAPTER
from makenomistakes.engine_args_music import STYLES_FOLDER as FOLDER
from makenomistakes.gguf.cursor import FileSource
from makenomistakes.safetensors.reader import SafetensorsError, read_safetensors_header
from makenomistakes.studio.row_library import NAME_PATTERN, RowLibrary, Training, sha256_of

SCHEMA_VERSION = 1
ROWS = "studio/styles"
ADAPTER = STYLE_ADAPTER
CONFIG = "adapter_config.json"
REFUSED_CONFIG = ("use_dora", "fan_in_fan_out", "use_qalora")  # audio.cpp throws on these
_NAME = re.compile(NAME_PATTERN)
TRIGGER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")  # it leads a song's style line
LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight", ".lora_A.default.weight",
                 ".lora_B.default.weight")  # fmt: skip


class StyleError(ValueError):
    """A style mnm will not add or use, worded for the user."""


class StyleSource(StrEnum):
    OWN = "own"  # the user's own songs
    LICENSED = "licensed"  # songs licensed to the user for training
    GENERATED = "generated"  # songs a model made (mnm's own runs)


SOURCE_WORDS = {
    StyleSource.OWN: "your own songs",
    StyleSource.LICENSED: "songs licensed to you",
    StyleSource.GENERATED: "generated songs",
}
DISCLOSED_WORDS = {
    StyleSource.OWN: "the maker's own songs",
    StyleSource.LICENSED: "songs licensed to the maker",
    StyleSource.GENERATED: "generated songs",
}


class StyleFile(BaseModel):
    name: Literal["adapter_model.safetensors", "adapter_config.json"]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StyleRow(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str = Field(pattern=NAME_PATTERN)
    variant: Literal["base", "turbo"]
    trigger: str = Field(min_length=1, max_length=64)
    caption: str = ""
    files: list[StyleFile]
    rank: int
    alpha: int
    source: StyleSource
    training: Training | None = None
    licence: str = ""
    created_at: datetime

    def words(self) -> str:
        return (
            f"{self.name} (ACE-Step 1.5 {self.variant}, trigger '{self.trigger}', "
            f"{SOURCE_WORDS[self.source]})"
        )


class AdapterCheck(BaseModel):
    rank: int
    alpha: int
    tensors: int


def check_adapter(folder: Path) -> AdapterCheck:
    """`folder`'s PEFT adapter as audio.cpp reads it; refused in words otherwise."""
    weights, config_path = folder / ADAPTER, folder / CONFIG
    if not weights.is_file() or not config_path.is_file():
        raise StyleError(
            f"{folder} holds no PEFT adapter: it needs {ADAPTER} and {CONFIG} side by side "
            "(an ACE-Step LoRA trained with PEFT, as mnm's trainer and upstream's write them)."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rank, alpha = int(config["r"]), int(config["lora_alpha"])
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise StyleError(f"{config_path} is not PEFT's config ({exc}).") from exc
    refused = [key for key in REFUSED_CONFIG if config.get(key)]
    if config.get("bias", "none") != "none":
        refused.append(f"bias={config['bias']}")
    if refused:
        raise StyleError(
            f"{config_path} asks for {', '.join(refused)}, which audio.cpp does not load: "
            "train the adapter without it."
        )
    try:
        names = list(read_safetensors_header(FileSource(weights)).tensors)
    except (OSError, SafetensorsError) as exc:
        raise StyleError(f"{weights} is not a safetensors file ({exc}).") from exc
    odd = [n for n in names if not n.endswith(LORA_SUFFIXES)]
    if not names or odd:
        raise StyleError(
            f"{weights} is not a LoRA of ACE-Step's DiT: {len(odd)} of {len(names)} tensors are "
            f"no lora_A/lora_B ({odd[:1]} first)."
        )
    return AdapterCheck(rank=rank, alpha=alpha, tensors=len(names))


def check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise StyleError(
            f"'{name}' is not a style name: lower-case letters, digits, '.', '_' and '-', "
            "starting with a letter or digit, at most 64."
        )


def check_trigger(trigger: str) -> None:
    if not TRIGGER.fullmatch(trigger):
        raise StyleError(
            f"'{trigger}' is not a trigger word: letters, digits, spaces, '_' and '-', "
            "starting with a letter or digit, at most 64."
        )


class StyleLibrary(RowLibrary[StyleRow]):
    """The styles' rows (state) and their files (models folder): `row_library` reads them."""

    noun = "style"
    row_type = StyleRow
    error = StyleError
    schema_version = SCHEMA_VERSION
    rm_command = "mnm studio styles rm"
    add_command = "mnm studio styles add"

    def __init__(self, state_dir: Path, models_dir: Path) -> None:
        super().__init__(state_dir / ROWS, models_dir / FOLDER)

    def add(  # noqa: PLR0913 - the row's parts, each its own flag
        self,
        name: str,
        adapter: Path,
        variant: Literal["base", "turbo"],
        trigger: str,
        source: StyleSource,
        *,
        caption: str = "",
        licence: str = "",
        training: Training | None = None,
    ) -> StyleRow:
        """Check the adapter, copy it in (beside, then moved into place), save the row."""
        check_name(name)
        check_trigger(trigger)
        checked = check_adapter(adapter)
        with self.rows.locked():
            self.refuse_taken(name)
            files = []
            with self.staged(name) as staging:
                for file in (ADAPTER, CONFIG):
                    shutil.copyfile(adapter / file, staging / file)
                    files.append(StyleFile(name=file, sha256=sha256_of(staging / file)))
            row = StyleRow(
                name=name, variant=variant, trigger=trigger, caption=caption, files=files,
                rank=checked.rank, alpha=checked.alpha, source=source, training=training,
                licence=licence, created_at=datetime.now(UTC),
            )  # fmt: skip
            self.rows.write(name, row.model_dump_json(indent=2))
        return row
