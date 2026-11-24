"""The texts a quality tune reads: the use's, shipped with mnm and pinned by
sha256 (`texts/SOURCE.md` says where each came from and under which licence). A coding
instance is measured on code, a chat or vision instance on prose: a quant may lose more on
one than the other, and the user asks what the instance will be used for."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from makenomistakes.hub.facts import Modality

FOLDER = Path(__file__).parent / "texts"


class TextName(StrEnum):
    PROSE = "prose"
    CODE = "code"


@dataclass(frozen=True)
class TextRow:
    name: TextName
    file: str
    sha256: str
    words: str  # how a score names it: "measured on prose (WikiText-2)"

    @property
    def path(self) -> Path:
        return FOLDER / self.file

    @property
    def short(self) -> str:
        return self.sha256[:12]


TEXTS: dict[TextName, TextRow] = {
    TextName.PROSE: TextRow(
        TextName.PROSE,
        "prose.txt",
        "2e6175899e1205973eeba886caf19f173f6a010b1e7b0502d7f26071c9f248e0",
        "prose (WikiText-2)",
    ),
    TextName.CODE: TextRow(
        TextName.CODE,
        "code.txt",
        "0d37e50b589bd7a6325cb9d69dc4f9fa58bca3a2bd9f608063f546c647d8ecbb",
        "code (mnm's own Python)",
    ),
}


class TextError(RuntimeError):
    """A shipped text that is missing or changed (an install that lost or altered it)."""


def text_for(modality: Modality) -> TextRow:
    """The use's text: code for a coding instance, prose for chat and vision."""
    return TEXTS[TextName.CODE if modality is Modality.CODING else TextName.PROSE]


def checked(row: TextRow) -> bytes:
    """The text's bytes, refused unless they are the ones its scores were taken on."""
    try:
        data = row.path.read_bytes()
    except OSError as exc:
        raise TextError(
            f"the quality tune's {row.name} text is missing ({row.path}): {exc}"
        ) from exc
    if hashlib.sha256(data).hexdigest() != row.sha256:
        raise TextError(
            f"the quality tune's {row.name} text ({row.path}) is not the one mnm shipped: "
            "`mnm update` puts it back"
        )
    return data
