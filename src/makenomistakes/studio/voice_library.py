"""The voice library: the voices the user may put on a song, one row each.

A row names its engine (RVC with the user's checkpoint; Seed-VC, Vevo2 and SoulX-Singer
from a reference recording), its files by sha256, where the voice comes from (`source`) and,
for someone else's voice, the hash and date of a consent document the user keeps (never
the document). The rows live under `<state>/studio/voices/<name>.json`; the files under
`<models>/voice-library/<name>/`, where the converter's container reads them (it mounts the
models folder only). The uninstall removes the rows with the rest of the state and keeps the
files, as it keeps the outputs: they are the user's recordings and checkpoints.

A row without the consent its source asks for is complete and usable: the export says so in
its disclosure, mnm never refuses. audio.cpp's packaged RVC voices are refused as rows:
whose voices they are and on what terms is not published.
"""

from __future__ import annotations

import re
import shutil
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.media_bytes import wav_file_seconds
from makenomistakes.music_api import VOICE_FOLDER as FOLDER
from makenomistakes.studio import rvc_checkpoint
from makenomistakes.studio.row_library import NAME_PATTERN, RowLibrary, Training, sha256_of

SCHEMA_VERSION = 1
ROWS = "studio/voices"  # under the state folder: one `<name>.json` per voice
_NAME = re.compile(NAME_PATTERN)
NO_VOICE = "none"  # `mnm song voice <run> none`: the song model's own voice, never a row
# audio.cpp's RVC package ships these four voices with no source (`docs/audio_tools.md`)
PACKAGED_RVC = ("default", "manthos", "chocola", "fraise")
REFERENCE_SUFFIXES = (".wav",)  # the converters' FLAC reading is unverified
CHECKPOINT_SUFFIXES = (".pth", ".pt")
INDEX_SUFFIXES = (".index",)
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 30.0  # Seed-VC takes 1 to 30 s of reference


class VoiceError(ValueError):
    """A voice mnm will not add or use, worded for the user."""


class VoiceEngine(StrEnum):
    RVC = "rvc"  # the user's own checkpoint (`.pth`), trained or theirs to use
    SEED_VC = "seed_vc"  # zero-shot from a reference recording
    VEVO2 = "vevo2"  # zero-shot from a reference recording (CC-BY-NC-ND weights)
    SOULX = "soulx"  # SoulX-Singer through a ComfyUI workflow, from a reference recording


class VoiceSource(StrEnum):
    OWN = "own"  # the user's own voice
    GENERATED = "generated"  # a voice no person has (a model's)
    HIRED = "hired"  # a singer the user hired
    LICENSED = "licensed"  # a voice licensed from someone
    LIKENESS = "likeness"  # someone's likeness, with their permission


# whose consent a release would want on record (the export notes its absence)
NEEDS_CONSENT = frozenset({VoiceSource.HIRED, VoiceSource.LICENSED, VoiceSource.LIKENESS})
SOURCE_WORDS = {  # to the user (the CLI, the web)
    VoiceSource.OWN: "your own voice",
    VoiceSource.GENERATED: "a generated voice no person has",
    VoiceSource.HIRED: "a singer you hired",
    VoiceSource.LICENSED: "a voice licensed to you",
    VoiceSource.LIKENESS: "someone's likeness",
}
DISCLOSED_WORDS = {  # in the export's disclosure, read by anyone
    VoiceSource.OWN: "the maker's own voice",
    VoiceSource.GENERATED: "a generated voice no person has",
    VoiceSource.HIRED: "a hired singer's voice",
    VoiceSource.LICENSED: "a voice licensed to the maker",
    VoiceSource.LIKENESS: "a person's likeness",
}


class VoiceFile(BaseModel):
    role: Literal["model", "index", "reference"]
    name: str = Field(pattern=r"^(model|index|reference)\.[a-z0-9]{1,8}$")  # in the row's folder
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Consent(BaseModel):
    """A consent document's fingerprint: the user keeps the document itself."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    date: date
    note: str = ""


class VoiceRow(BaseModel):
    schema_version: int = SCHEMA_VERSION
    name: str = Field(pattern=NAME_PATTERN)
    engine: VoiceEngine
    files: list[VoiceFile]
    source: VoiceSource
    consent: Consent | None = None
    training: Training | None = None
    licence: str = ""  # the terms the voice is used under, in the user's words
    note: str = ""
    created_at: datetime

    @property
    def needs_consent(self) -> bool:
        """Someone else's voice with no consent document recorded."""
        return self.source in NEEDS_CONSENT and self.consent is None

    def file(self, role: str) -> VoiceFile | None:
        return next((f for f in self.files if f.role == role), None)

    def words(self) -> str:
        """`alto (Seed-VC, your own voice)`, the consent's state after a voice not yours."""
        consent = ""
        if self.source in NEEDS_CONSENT:
            consent = (
                f"; consent recorded {self.consent.date}" if self.consent else "; consent: none"
            )
        return f"{self.name} ({ENGINE_WORDS[self.engine]}, {SOURCE_WORDS[self.source]}{consent})"


class VoiceTake(BaseModel):
    """What sang a song's vocals, kept beside them (`voice.json`, the voice stage writes it):
    the voice as it was then, so an export says what sang after the library's row changed."""

    voice: str
    engine: VoiceEngine
    source: VoiceSource
    files: list[VoiceFile]  # the voice's files as they were then, by sha256
    converter: str
    shift: int = 0
    # how alike the vocals and the vocals as sung sound to the reference (speaker cosines)
    similarity: dict[str, float] = Field(default_factory=dict)
    reader: str | None = None  # what read it

    def same_voice(self, row: VoiceRow | None) -> bool:
        """Whether the library's row is still the voice that sang (same name, same files)."""
        return row is not None and [f.sha256 for f in row.files] == [f.sha256 for f in self.files]


TAKE_FILE = "voice.json"  # beside a song's vocals: what sang them


def read_take(folder: Path) -> VoiceTake | None:
    """A song's take from its run folder; None when there is none or it cannot be read
    (the export then says the voice is not known here)."""
    try:
        return VoiceTake.model_validate_json((folder / TAKE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return None


ENGINE_WORDS = {
    VoiceEngine.RVC: "RVC",
    VoiceEngine.SEED_VC: "Seed-VC",
    VoiceEngine.VEVO2: "Vevo2",
    VoiceEngine.SOULX: "SoulX-Singer",
}


def refuse_packaged(voice_id: str) -> None:
    """audio.cpp's packaged RVC voices name no source: never a row."""
    raise VoiceError(
        f"'{voice_id}' is one of audio.cpp's packaged RVC voices ({', '.join(PACKAGED_RVC)}): "
        "whose voices they are and on what terms is not published, so mnm keeps no row for "
        "them. Add a checkpoint you have the rights to (`--model voice.pth`), or a reference "
        "recording for Seed-VC."
    )


class VoiceLibrary(RowLibrary[VoiceRow]):
    """The voices' rows (state) and their files (models folder): `row_library` reads them."""

    noun = "voice"
    row_type = VoiceRow
    error = VoiceError
    schema_version = SCHEMA_VERSION
    rm_command = "mnm voices rm"
    add_command = "mnm voices add"

    def __init__(self, state_dir: Path, models_dir: Path) -> None:
        super().__init__(state_dir / ROWS, models_dir / FOLDER)

    def add(  # noqa: PLR0913 - the row's parts, each its own flag
        self,
        name: str,
        engine: VoiceEngine,
        source: VoiceSource,
        *,
        reference: Path | None = None,
        model: Path | None = None,
        index: Path | None = None,
        licence: str = "",
        note: str = "",
        training: Training | None = None,
        consent: Consent | None = None,
    ) -> VoiceRow:
        """Copy the files in, hash them, save the row; refused before anything is written,
        and the files appear whole or not at all (copied beside, then moved into place). An
        RVC checkpoint is read as audio.cpp will read it first (`rvc_checkpoint`), and
        the library's copy carries `f0` as the integer audio.cpp asks for."""
        self.check_name(name)
        with self.rows.locked():
            self.refuse_taken(name)
            given = _files_for(engine, reference=reference, model=model, index=index)
            files = self._copy_in(name, given)
            row = VoiceRow(
                name=name,
                engine=engine,
                files=files,
                source=source,
                consent=consent,
                training=training,
                licence=licence,
                note=note,
                created_at=datetime.now(UTC),
            )
            self.rows.write(name, row.model_dump_json(indent=2))
            return row

    @staticmethod
    def check_name(name: str) -> None:
        if not _NAME.fullmatch(name) or name == NO_VOICE:
            raise VoiceError(
                f"'{name}' is not a voice name: lower-case letters, digits, '.', '_' and '-' "
                f"('{NO_VOICE}' means the song model's own voice)."
            )

    def _copy_in(
        self, name: str, given: list[tuple[Literal["model", "index", "reference"], Path]]
    ) -> list[VoiceFile]:
        files = []
        with self.staged(name) as staging:
            for role, source_path in given:
                target = staging / f"{role}{source_path.suffix.lower()}"
                shutil.copyfile(source_path, target)
                if role == "model":  # the copy says `f0` as audio.cpp reads it
                    try:
                        rvc_checkpoint.normalise(source_path, target)
                    except rvc_checkpoint.RvcCheckpointError as exc:
                        raise VoiceError(str(exc)) from exc
                files.append(VoiceFile(role=role, name=target.name, sha256=sha256_of(target)))
        return files

    def consent(self, name: str, document: Path, note: str = "") -> VoiceRow:
        """Record a consent document's sha256 and today's date; the document is never kept."""
        if not document.is_file():
            raise VoiceError(f"{document} is not a file.")
        with self.rows.locked():
            row = self.get(name)
            row.consent = Consent(sha256=sha256_of(document), date=date.today(), note=note)
            self.rows.write(name, row.model_dump_json(indent=2))
            return row


def _files_for(
    engine: VoiceEngine, *, reference: Path | None, model: Path | None, index: Path | None
) -> list[tuple[Literal["model", "index", "reference"], Path]]:
    """What each engine needs, checked before anything is copied."""
    for given in (reference, model, index):
        if given is not None and not given.is_file():
            raise VoiceError(f"{given} is not a file.")
    if engine is VoiceEngine.RVC:
        return _rvc_files(reference=reference, model=model, index=index)
    if reference is None:
        raise VoiceError(
            f"A {ENGINE_WORDS[engine]} voice is a reference recording: `--reference voice.wav` "
            f"(1 to {MAX_REFERENCE_SECONDS:g} s of the voice singing or speaking, alone)."
        )
    if reference.suffix.lower() not in REFERENCE_SUFFIXES:
        raise VoiceError(
            f"{reference.name}: a reference is a WAV file (from another format: "
            f"`ffmpeg -i {reference.name} {reference.stem}.wav`)."
        )
    seconds = wav_file_seconds(reference)
    if seconds is None:
        raise VoiceError(f"{reference.name} is not a WAV file mnm can read (its header).")
    if not MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
        raise VoiceError(
            f"{reference.name} is {seconds:.1f} s: a reference is {MIN_REFERENCE_SECONDS:g} to "
            f"{MAX_REFERENCE_SECONDS:g} s of the voice alone (the converters take no more)."
        )
    if model is not None or index is not None:
        raise VoiceError(f"A {ENGINE_WORDS[engine]} voice takes only a reference recording.")
    return [("reference", reference)]


def _rvc_files(
    *, reference: Path | None, model: Path | None, index: Path | None
) -> list[tuple[Literal["model", "index", "reference"], Path]]:
    """An RVC voice's checkpoint (read as audio.cpp reads it) and its optional index."""
    if model is None:
        raise VoiceError(
            "An RVC voice is a checkpoint: `--model voice.pth` (and `--index voice.index` "
            "for retrieval blending). audio.cpp's packaged voices are not offered."
        )
    if model.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise VoiceError(
            f"{model.name} is not an RVC checkpoint (`.pth` or `.pt`); the RVC GGUF package "
            "is the converter itself, already an instance (`mnm add`)."
        )
    if reference is not None:
        raise VoiceError("An RVC voice takes no reference recording: its checkpoint is the voice.")
    if index is not None and index.suffix.lower() not in INDEX_SUFFIXES:
        raise VoiceError(f"{index.name} is not an RVC retrieval index (`.index`).")
    try:
        rvc_checkpoint.read(model)  # What audio.cpp would refuse, refused now
    except rvc_checkpoint.RvcCheckpointError as exc:
        raise VoiceError(str(exc)) from exc
    if index is not None and not rvc_checkpoint.ivf_index(index):
        raise VoiceError(
            f"{index.name} is not a FAISS IVF index (IndexIVFFlat), the only kind audio.cpp's "
            "RVC reads: add the voice without it (`--index` left out)."
        )
    files: list[tuple[Literal["model", "index", "reference"], Path]] = [("model", model)]
    return [*files, ("index", index)] if index is not None else files
