"""An RVC voice's checkpoint read without torch, as audio.cpp's RVC converter will read it:
what `mnm voices add --engine rvc` and the trainer's result are checked against before
a row is written, so a voice that cannot sing is refused in words at once, never five minutes
into a song's voice stage.

audio.cpp (c0cd77f, `src/models/rvc/voice_model.cpp`) reads the RVC web UI's inference format: a
`torch.save` zip whose pickle is a dict of the weights (`weight`) and metadata: `f0` as an
integer, the sample rate as `config`'s last item (else `sr`, "32k" | "40k" | "48k"), `version`
(else the phone embedding's width: 256 v1, 768 v2), a HiFi-GAN decoder (`dec.ups.<n>`), a
speaker table (`emb_g.weight`). Applio writes the same format with `f0` as a Python bool, which
audio.cpp refuses ("missing integer metadata: f0"): `normalise` rewrites that one opcode as the
integer it means, in a copy (the user's own file is never touched). Applio also trains voices
on vocoders and embedders audio.cpp does not run (RefineGAN, MRF HiFi-GAN; SPIN, the Chinese
HuBERT): those are refused by name.

The pickle is read by `torch_ckpt.load`'s restricted unpickler (nothing it names is run); the
rewrite walks it with `pickletools.genops`, which decodes opcodes and constructs nothing.
"""

from __future__ import annotations

import pickletools
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.safetensors.torch_ckpt import (
    CheckpointError,
    is_tensor,
    load,
    tensor_shape,
)

RATES = {"32k": 32_000, "40k": 40_000, "48k": 48_000}
VOCODER = "HiFi-GAN"  # the only decoder audio.cpp's RVC synthesizer builds
EMBEDDER = "contentvec"  # the features audio.cpp's RVC HuBERT gives
PHONE_WIDTHS = {256: "v1", 768: "v2"}
IVF = b"IwFl"
ALIGN = 64  # torch.save aligns each record's data to 64 bytes; the copy keeps that
_STRINGS = frozenset({"BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE"})
_MEMO = frozenset({"BINPUT", "LONG_BINPUT", "PUT", "MEMOIZE"})


class RvcCheckpointError(ValueError):
    """A checkpoint audio.cpp's RVC converter would not sing with, worded for the user."""


@dataclass(frozen=True)
class RvcCheckpoint:
    """What the checkpoint says about itself."""

    version: str
    sample_rate: int
    pitch: bool  # trained with pitch guidance (f0): a singing voice needs it
    f0_is_bool: bool  # written by Applio: `normalise` makes it the integer audio.cpp reads
    vocoder: str | None
    embedder: str | None
    epochs: int | None
    tensors: int

    @property
    def words(self) -> str:
        made = f", {self.epochs} epochs" if self.epochs else ""
        return f"RVC {self.version}, {self.sample_rate // 1000} kHz{made}"


def read(path: Path) -> RvcCheckpoint:
    """Read `path` as audio.cpp reads an RVC voice, or a refusal saying what it lacks."""
    try:
        loaded = load(path)
    except CheckpointError as exc:
        raise RvcCheckpointError(f"{path.name} is not an RVC voice mnm reads: {exc}") from exc
    root = loaded.obj
    if not isinstance(root, dict) or not isinstance(root.get("weight"), dict):
        raise RvcCheckpointError(
            f"{path.name} is not an RVC voice's inference checkpoint (no `weight` table): a "
            "training checkpoint (G_*.pth, D_*.pth) is not one; Applio's voice is the "
            "`<name>_<epochs>e_<steps>s.pth` file beside it."
        )
    weights: dict[object, object] = root["weight"]
    tensors = {str(k): v for k, v in weights.items() if is_tensor(v)}
    _refuse_damaged(path)
    f0 = root.get("f0")
    if not isinstance(f0, int):  # bool is an int to Python; audio.cpp tells them apart
        raise RvcCheckpointError(f"{path.name} names no pitch guidance (`f0`): not an RVC voice.")
    checkpoint = RvcCheckpoint(
        version=_version(path, root, tensors),
        sample_rate=_rate(path, root),
        pitch=bool(f0),
        f0_is_bool=isinstance(f0, bool),
        vocoder=_text(root.get("vocoder")),
        embedder=_text(root.get("embedder_model")),
        epochs=root.get("epoch") if isinstance(root.get("epoch"), int) else None,
        tensors=len(tensors),
    )
    _refuse_unrunnable(path, checkpoint, tensors)
    return checkpoint


def _refuse_damaged(path: Path) -> None:
    """Every member read against its checksum (`load` read only the pickle, and refused
    compressed members, so this reads the file's own size): a damaged download is refused
    here, never a traceback from `normalise` or a voice that sings noise."""
    again = "get the voice again from where it came"
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
    except (OSError, EOFError, zipfile.BadZipFile) as exc:
        raise RvcCheckpointError(f"{path.name} is damaged ({exc}): {again}.") from exc
    if bad is not None:
        raise RvcCheckpointError(f"{path.name} is damaged ({bad} fails its checksum): {again}.")


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _rate(path: Path, root: dict[object, object]) -> int:
    config = root.get("config")
    if isinstance(config, list) and config and isinstance(config[-1], int) and config[-1] > 0:
        rate = config[-1]
    else:
        rate = RATES.get(str(root.get("sr")), 0)
    if rate not in RATES.values():
        raise RvcCheckpointError(
            f"{path.name} is trained at {rate or 'an unnamed'} Hz: audio.cpp's RVC sings voices "
            "trained at 32, 40 or 48 kHz."
        )
    return rate


def _version(path: Path, root: dict[object, object], tensors: dict[str, object]) -> str:
    named = root.get("version")
    if isinstance(named, str) and named:
        version = named
    else:
        shape = tensor_shape(tensors.get("enc_p.emb_phone.weight"))
        version = PHONE_WIDTHS.get(shape[1], "") if shape and len(shape) == 2 else ""
    if version not in PHONE_WIDTHS.values():
        raise RvcCheckpointError(
            f"{path.name} is RVC {version or 'of an unknown version'}: audio.cpp sings v1 and v2."
        )
    return version


def _refuse_unrunnable(path: Path, checkpoint: RvcCheckpoint, tensors: dict[str, object]) -> None:
    if checkpoint.vocoder is not None and checkpoint.vocoder != VOCODER:
        raise RvcCheckpointError(
            f"{path.name} was trained with Applio's {checkpoint.vocoder} vocoder: audio.cpp's "
            f"RVC builds the {VOCODER} decoder only. Train it again with {VOCODER} (mnm's "
            "trainer always does)."
        )
    if checkpoint.embedder is not None and checkpoint.embedder != EMBEDDER:
        raise RvcCheckpointError(
            f"{path.name} was trained on {checkpoint.embedder} features: audio.cpp's RVC "
            f"listens with ContentVec ({EMBEDDER}). Train it again on {EMBEDDER} (mnm's "
            "trainer always does)."
        )
    decoder = ("dec.ups.0.weight" in tensors) or ("dec.ups.0.weight_v" in tensors)
    if not decoder or "emb_g.weight" not in tensors:
        raise RvcCheckpointError(
            f"{path.name} has no HiFi-GAN decoder or speaker table (`dec.ups.0`, `emb_g`): "
            "not an RVC voice audio.cpp sings with."
        )


def ivf_index(path: Path) -> bool:
    """Whether `path` is a FAISS IndexIVFFlat (`IwFl`), the only index audio.cpp's RVC reads
    (Applio writes a flat one for a very short dataset)."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == IVF
    except OSError:
        return False


def normalise(path: Path, out: Path) -> bool:
    """Copy `path` to `out` with Applio's `f0` bool written as the integer it means; True when
    something changed (`out` is then the copy to keep), False when `path` is fine as it is."""
    checkpoint = read(path)
    if not checkpoint.f0_is_bool:
        return False
    with zipfile.ZipFile(path) as archive:
        members = [(info, archive.read(info)) for info in archive.infolist()]
    name = next(info.filename for info, _ in members if info.filename.endswith("/data.pkl"))
    tmp = out.with_name(out.name + ".part")
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as copy:
            for info, data in members:
                body = _f0_as_int(path, data) if info.filename == name else data
                _write_aligned(copy, info.filename, body)
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


def _f0_as_int(path: Path, pickled: bytes) -> bytes:
    """The one `NEWTRUE`/`NEWFALSE` after the key `f0` as `BININT1 1`/`0`. Protocol 2 (what
    torch.save writes) has no frame lengths and memo entries are numbered, never offsets, so
    one opcode a byte longer moves nothing else."""
    ops = list(pickletools.genops(pickled))
    if any(op.name == "FRAME" for op, _arg, _pos in ops):
        raise RvcCheckpointError(f"{path.name}'s pickle is protocol 4 or later: mnm cannot fix it.")
    found = []
    for at, (op, arg, _pos) in enumerate(ops):
        if op.name in _STRINGS and arg == "f0":
            nxt = at + 1
            while nxt < len(ops) and ops[nxt][0].name in _MEMO:
                nxt += 1
            if nxt < len(ops) and ops[nxt][0].name in ("NEWTRUE", "NEWFALSE"):
                found.append(ops[nxt])
    if len(found) != 1:
        raise RvcCheckpointError(f"{path.name}'s `f0` is not one plain bool: left as it is.")
    op, _arg, pos = found[0]
    assert pos is not None
    value = b"K\x01" if op.name == "NEWTRUE" else b"K\x00"
    return pickled[:pos] + value + pickled[pos + 1 :]


def _write_aligned(copy: zipfile.ZipFile, name: str, data: bytes) -> None:
    """A stored member whose data starts on a 64-byte boundary, as torch.save writes it (its
    extra field `FB` pads the local header)."""
    encoded = name.encode()
    start = copy.fp.tell() if copy.fp is not None else 0
    pad = (ALIGN - (start + 30 + len(encoded) + 4) % ALIGN) % ALIGN
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.extra = b"FB" + struct.pack("<H", pad) + b"\0" * pad
    copy.writestr(info, data)
