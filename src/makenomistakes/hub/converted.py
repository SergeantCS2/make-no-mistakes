"""The studio's checkpoints mnm converts for audio.cpp: models nobody packaged as a
GGUF, published as an MSST PyTorch checkpoint and its YAML config.

audio.cpp runs a BS- or Mel-Band RoFormer from a folder of `model.safetensors` and
`config.json`, and ships no converter in its images: mnm downloads the checkpoint and its YAML
like any package and converts them after the download (`converted_models.py`, with
`safetensors/torch_ckpt.py`, which never imports torch). Each row is one checkpoint mnm has run
through audio.cpp and checked against something its author published; a repository of the
same family that is not a row stays unservable, since a checkpoint's YAML may ask for what
audio.cpp's runtime refuses (a linear transformer, skip connections) and nobody would know
until it played.

The kit is the same `MusicKit` the packaged models have, so the planner, `mnm try`, the
song's sample rate and the export's licence line read a converted model as they read any.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from pydantic import BaseModel

from makenomistakes.hub.modality import Modality
from makenomistakes.hub.music import MusicFamily, MusicKit, kit_for, music_family
from makenomistakes.hub.variants import GgufVariant

FOLDER = "converted"  # beside the checkpoint: `converted/<checkpoint stem>/`


def folder_of(checkpoint: str) -> PurePosixPath:
    """The converted folder for a checkpoint at `checkpoint` (relative to the models folder)."""
    path = PurePosixPath(checkpoint)
    return path.parent / FOLDER / path.stem


class ConvertedKit(BaseModel):
    repo_id: str
    checkpoint: str  # the .ckpt, at the repository's root
    config: str  # its MSST YAML
    size_bytes: int  # the checkpoint's, as the Hub listed it (2026-11-01)
    # what mnm checked, by sha256: anything else is refused before it is converted
    checkpoint_sha256: str
    config_sha256: str
    revision: str  # the repository's commit when it was checked (the record, not the fetch)
    purpose: str  # what the vocals stage asks of it (`dereverb`)
    kit: MusicKit
    checked: str  # how it was checked against its author's own output

    @property
    def stem(self) -> str:
        return PurePosixPath(self.checkpoint).stem

    @property
    def split_qkv(self) -> bool:
        """audio.cpp's Mel-Band runtime reads q, k and v apart; its BS runtime fused."""
        return self.kit.family is MusicFamily.MEL_BAND_ROFORMER


CONVERTED: tuple[ConvertedKit, ...] = (
    ConvertedKit(
        repo_id="anvuew/dereverb_room",
        checkpoint="dereverb_room_anvuew_sdr_13.7432.ckpt",
        config="dereverb_room_anvuew.yaml",
        size_bytes=118_128_452,
        checkpoint_sha256="2edec521f09e26341c1923dc82c8c52dbc86478b42b9999f679535743c970cb3",
        config_sha256="c37e3039521d79cd1daff129857f69fa80c6a1f383a0fe8cda757f2dfc5032f8",
        revision="0b85f5b80b7f779b2dfe80f33a1b35b38af9376d",
        purpose="dereverb",
        kit=MusicKit(
            family=MusicFamily.BS_ROFORMER,
            label="anvuew's room dereverb (BS-RoFormer, mono)",
            modality=Modality.STEMS,
            task="sep",
            served_id="dereverb-room",
            licence="GPL-3.0 (the checkpoint, as its Hub card says)",
            commercial="yes",
            stems=("vocals", "instrumental"),  # audio.cpp's names: the dry voice, the room
            sample_rate=44_100,
            channels=1,  # its YAML: stereo false; audio.cpp refuses two channels
            note="Takes a room's reverb off a voice; the vocals stage's dereverb.",
        ),
        checked=(
            "converted by mnm and run on audio.cpp's full-cpu image: its answer "
            "to the repository's example/input.flac matched the author's example/noreverb.flac "
            "to 51 dB (SI-SDR)"
        ),
    ),
)


def converted_kit(repo_id: str, path: str | None = None) -> ConvertedKit | None:
    """The row for a pasted repository (or one file of it), if mnm converts it."""
    for row in CONVERTED:
        if row.repo_id.lower() == repo_id.lower() and path in (None, "", row.checkpoint):
            return row
    return None


def converted_by_file(file: str) -> ConvertedKit | None:
    """The row whose checkpoint a recipe's file or a credit names."""
    name = PurePosixPath(file).name
    return next((row for row in CONVERTED if row.checkpoint == name), None)


def converted_for_purpose(purpose: str) -> tuple[ConvertedKit, ...]:
    return tuple(row for row in CONVERTED if row.purpose == purpose)


def converted_variant(row: ConvertedKit, sizes: dict[str, int]) -> GgufVariant:
    """The checkpoint and its YAML as one variant (the checkpoint first: the config's model),
    labelled F32, the precision MSST checkpoints are saved in."""
    files = [row.checkpoint, row.config]
    file_bytes = [sizes.get(f, row.size_bytes if f == row.checkpoint else 0) for f in files]
    return GgufVariant(quant="F32", files=files, size_bytes=sum(file_bytes), file_bytes=file_bytes)


def kit_of(family: MusicFamily, file: str | None = None) -> MusicKit:
    """A converted checkpoint's own kit when `file` is one (its label and licence are not its
    family's packaged model's), else the family's."""
    row = converted_by_file(file) if file else None
    return row.kit if row is not None else kit_for(family)


def kit_named_by(file: str) -> MusicKit | None:
    """The kit a model file names, for words about it (the export's licences): a converted
    checkpoint's own row, else the packaged family its folder names, else None."""
    row = converted_by_file(file)
    if row is not None:
        return row.kit
    family = music_family(file)
    return kit_for(family) if family is not None else None
