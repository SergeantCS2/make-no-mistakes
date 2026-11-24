"""A PyTorch checkpoint read without torch and written as safetensors.

`torch.save` (since 1.6) writes a zip: `<root>/data.pkl`, a pickle of the state dict whose
tensors point by a persistent id at raw storages `<root>/data/<key>`, plus `byteorder` and
`version`. A pickle runs whatever classes it names, so this one is read by an unpickler that
knows only what a weights-only state dict holds: `collections.OrderedDict`, the tensor rebuild
functions and the storage types (as `torch.load(weights_only=True)` allows); anything else is a
`CheckpointError` naming the class, and nothing is run. Each of those is sealed against a
pickle's BUILD (`_Sealed`), and the pickle's nesting is read before it loads
(`pickle_nesting.refuse_deep`: a dict key nested a million tuples deep crashed CPython). mnm
never imports torch (no inference code is installed on the host): the tensors' bytes are copied
from the zip into a safetensors file as they are, contiguous tensors only.

Music-source-separation checkpoints (MSST's BS- and Mel-Band RoFormers) are what this is for:
audio.cpp loads a RoFormer from a folder of `model.safetensors` and `config.json`. Its Mel-Band
runtime reads q, k and v apart (the MLX conversion's names) where the checkpoint keeps one fused
`to_qkv` (rows: q, then k, then v): `split_qkv` writes the three; the result equals
mlx-community/mel-roformer-mlx's own conversion of Kim's checkpoint, byte for byte.
"""

from __future__ import annotations

import collections
import io
import itertools
import json
import math
import pickle
import struct
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Final

from makenomistakes.safetensors.pickle_nesting import NestingError, refuse_deep

DTYPES: Final[dict[str, tuple[str, int]]] = {
    "FloatStorage": ("F32", 4),
    "HalfStorage": ("F16", 2),
    "BFloat16Storage": ("BF16", 2),
    "DoubleStorage": ("F64", 8),
    "LongStorage": ("I64", 8),
    "IntStorage": ("I32", 4),
    "ShortStorage": ("I16", 2),
    "CharStorage": ("I8", 1),
    "ByteStorage": ("U8", 1),
    "BoolStorage": ("BOOL", 1),
}
MAX_PICKLE = 8 * 1024 * 1024  # a state dict's pickle: names and shapes (a RoFormer's is 142 KB)
MAX_BYTEORDER = 16  # `little` or `big`
MAX_TENSORS = 100_000
SKIPPED = (".rotary_embed.freqs",)  # recomputed by the engine (audio.cpp's converter drops them)
FUSED = ".to_qkv.weight"
SPLIT = ("to_q", "to_k", "to_v")


class CheckpointError(ValueError):
    """A checkpoint mnm will not read, worded for the user."""


class _Sealed:
    """A pickle's BUILD writes an object's fields after it was made and checked: these
    refuse it, so what was checked is what is copied (and an OrderedDict's `get` or a rebuild
    function's attributes are never a pickle's)."""

    def __setstate__(self, *_state: object) -> None:  # on the class too: `self` is the state
        raise CheckpointError("the checkpoint rewrites an object after making it")


class _StateDict(_Sealed, collections.OrderedDict[object, object]):
    """`collections.OrderedDict` as the checkpoint may make it: a dict, sealed. torch's
    `Module.state_dict()` sets `_metadata` on the dict it returns, so `torch.save` of one (every
    MSST checkpoint, the room dereverb's included) ends with a BUILD of `{'_metadata': …}`:
    a dict state on a state dict is read and dropped, never set (mnm reads no attribute, and
    a `get` dropped shadows nothing). Any other BUILD, on the class itself or another shape,
    is still refused."""

    def __setstate__(self, *state: object) -> None:
        if isinstance(self, _StateDict) and len(state) == 1 and isinstance(state[0], dict):
            return
        _Sealed.__setstate__(self, *state)  # on the class, `self` is the state: no `super()`


@dataclass(frozen=True)
class _Storage(_Sealed):
    kind: str
    key: str
    numel: int


@dataclass(frozen=True)
class _Tensor(_Sealed):
    storage: _Storage
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass(frozen=True)
class _StorageType(_Sealed):
    name: str


def _rebuild_tensor_v2(
    storage: object,
    storage_offset: object,
    size: object,
    stride: object,
    _requires_grad: object = False,
    _backward_hooks: object = None,
    _metadata: object = None,
) -> _Tensor:
    if not isinstance(storage, _Storage) or not isinstance(storage_offset, int):
        raise CheckpointError("a tensor in the checkpoint points at no storage")
    if storage_offset < 0:
        raise CheckpointError("a tensor in the checkpoint starts before its storage")
    if not isinstance(size, tuple) or not isinstance(stride, tuple):
        raise CheckpointError("a tensor in the checkpoint has no shape")
    if not all(isinstance(n, int) and n >= 0 for n in (*size, *stride)):
        raise CheckpointError("a tensor in the checkpoint has a negative shape")
    return _Tensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data: object, _requires_grad: object, _hooks: object) -> _Tensor:
    """An nn.Parameter saved whole: its tensor (never anything else, so no call hands back a
    list or dict the pickle could still add to: `pickle_nesting` counts on it)."""
    if not isinstance(data, _Tensor):
        raise CheckpointError("a parameter in the checkpoint holds no tensor")
    return data


@dataclass(frozen=True)
class _Call(_Sealed):
    """A function the checkpoint may call, sealed: a bare function would take a BUILD's
    attributes onto itself, for every later checkpoint too."""

    run: Callable[..., _Tensor]

    def __call__(self, *args: object) -> _Tensor:
        return self.run(*args)


ALLOWED: Final[dict[tuple[str, str], object]] = {
    ("collections", "OrderedDict"): _StateDict,
    ("torch._utils", "_rebuild_tensor_v2"): _Call(_rebuild_tensor_v2),
    ("torch._utils", "_rebuild_parameter"): _Call(_rebuild_parameter),
    **{("torch", name): _StorageType(name) for name in DTYPES},
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        found = ALLOWED.get((module, name))
        if found is None:
            raise CheckpointError(
                f"the checkpoint names {module}.{name}, which a checkpoint of weights does "
                "not hold: mnm reads none of it (it would run code to load)"
            )
        return found

    def persistent_load(self, pid: object) -> _Storage:
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise CheckpointError("the checkpoint points at something that is not a storage")
        _tag, kind, key, _location, numel = pid
        if not isinstance(kind, _StorageType) or not isinstance(numel, int):
            raise CheckpointError("the checkpoint names a storage mnm does not know")
        return _Storage(kind.name, str(key), numel)


@dataclass(frozen=True)
class Piece:
    """One tensor of the output: where its bytes sit in the zip."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    member: str
    start: int
    nbytes: int


@dataclass(frozen=True)
class Converted:
    tensors: int
    nbytes: int


def _contiguous(size: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    """Row-major with no gaps; a dimension of one may carry any stride (torch leaves one
    after `unsqueeze` or a transpose of a vector)."""
    if len(size) != len(stride):
        return False
    step = 1
    for dim, given in zip(reversed(size), reversed(stride), strict=True):
        if dim != 1 and given != step:
            return False
        step *= dim
    return True


def _root(names: list[str]) -> str:
    pickles = [n for n in names if n.endswith("/data.pkl") and n.count("/") == 1]
    if len(pickles) != 1:
        raise CheckpointError(
            "not a checkpoint mnm reads: a zip with one `<name>/data.pkl` (torch.save since "
            "1.6); an older checkpoint is a bare pickle, which mnm does not read"
        )
    return pickles[0].split("/")[0]


def _state(loaded: object) -> dict[str, object]:
    """The state dict, unwrapped from a training checkpoint's own dict when it is one."""
    state = loaded
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict) and nested:
                state = nested
                break
    if not isinstance(state, dict) or not state:
        raise CheckpointError("the checkpoint holds no state dict")
    return {str(name): value for name, value in state.items()}


@dataclass(frozen=True)
class Loaded:
    """A checkpoint's pickle as the restricted unpickler made it, and its zip's members."""

    root: str
    obj: object
    sizes: dict[str, int]
    pickled: bytes


def load(path: Path) -> Loaded:
    """Read `path`'s pickle without running anything it names (`_Unpickler`), refusing what
    torch.save does not write (compressed members, big-endian, an oversized pickle)."""
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise CheckpointError(f"{path.name} is not a checkpoint mnm reads: {exc}") from exc
    with archive:
        root = _root(archive.namelist())
        infos = {info.filename: info for info in archive.infolist()}
        if packed := [n for n, i in infos.items() if i.compress_type != zipfile.ZIP_STORED]:
            raise CheckpointError(  # what a read may inflate to is then unbounded
                f"{path.name} holds compressed members ({packed[0]}): torch.save stores them "
                "uncompressed, and mnm reads only that"
            )
        order = infos.get(f"{root}/byteorder")  # torch may leave it out: its own byte order
        if order is not None and (
            order.file_size > MAX_BYTEORDER or archive.read(order).strip() != b"little"
        ):
            raise CheckpointError(f"{path.name} was saved big-endian: mnm reads little-endian only")
        pickled = infos[f"{root}/data.pkl"]
        if pickled.file_size > MAX_PICKLE:
            raise CheckpointError(f"{path.name}'s pickle is {pickled.file_size} bytes: too large")
        try:
            body = archive.read(pickled)
            refuse_deep(body)
            loaded = _Unpickler(io.BytesIO(body)).load()
        except NestingError as exc:
            raise CheckpointError(
                f"{path.name}'s pickle {exc}, where a checkpoint of weights nests a few: mnm "
                "leaves it unread (loading it could crash mnm). Use another copy of the "
                "model, from where it came."
            ) from exc
        except CheckpointError:
            raise
        except Exception as exc:  # a truncated or malformed pickle, a bad CRC: worded
            raise CheckpointError(f"{path.name}'s pickle is not a state dict: {exc}") from exc
        sizes = {name: info.file_size for name, info in infos.items()}
    return Loaded(root, loaded, sizes, body)


def is_tensor(value: object) -> bool:
    return isinstance(value, _Tensor)


def tensor_shape(value: object) -> tuple[int, ...] | None:
    return value.size if isinstance(value, _Tensor) else None


def pieces(path: Path, *, split_qkv: bool = False) -> list[Piece]:
    """Every tensor of `path` as safetensors will hold it, checked against the zip's sizes."""
    found = load(path)
    root, loaded, sizes = found.root, found.obj, found.sizes
    tensors = {n: t for n, t in _state(loaded).items() if isinstance(t, _Tensor)}
    if len(tensors) > MAX_TENSORS:
        raise CheckpointError(f"{path.name} holds {len(tensors)} tensors: too many")
    out: list[Piece] = []
    for name, tensor in tensors.items():
        if name.endswith(SKIPPED):
            continue
        out.extend(_pieces_of(name, tensor, root, sizes, split_qkv=split_qkv))
    if not out:
        raise CheckpointError(f"{path.name} holds no tensors")
    _refuse_overlaps(path, out)
    return out


def _refuse_overlaps(path: Path, found: list[Piece]) -> None:
    """Two tensors reading the same bytes of a storage would write them twice: a few KB of
    checkpoint could then write gigabytes."""
    spans = sorted((p.member, p.start, p.start + p.nbytes) for p in found)
    for (member, _start, end), (nxt, start, _end) in itertools.pairwise(spans):
        if member == nxt and start < end:
            raise CheckpointError(f"{path.name} has tensors that share bytes: mnm copies each once")


def _pieces_of(
    name: str, tensor: _Tensor, root: str, sizes: dict[str, int], *, split_qkv: bool
) -> list[Piece]:
    if tensor.storage.kind not in DTYPES:
        raise CheckpointError(f"{name} is stored as {tensor.storage.kind}, which mnm does not read")
    dtype, width = DTYPES[tensor.storage.kind]
    if tensor.size and not _contiguous(tensor.size, tensor.stride):
        raise CheckpointError(
            f"{name} is stored as a view of another tensor: mnm copies whole tensors"
        )
    member = f"{root}/data/{tensor.storage.key}"
    start, nbytes = tensor.offset * width, math.prod(tensor.size) * width
    if member not in sizes or start + nbytes > sizes[member]:
        raise CheckpointError(f"{name} runs past its storage in the checkpoint")
    if split_qkv and name.endswith(FUSED):
        if len(tensor.size) != 2 or not tensor.size[0] or tensor.size[0] % 3:
            raise CheckpointError(f"{name} is not three equal blocks of rows (q, k, v)")
        rows, cols = tensor.size
        third = rows // 3 * cols * width
        return [
            Piece(
                f"{name.removesuffix(FUSED)}.{part}.weight",
                dtype,
                (rows // 3, cols),
                member,
                start + i * third,
                third,
            )
            for i, part in enumerate(SPLIT)
        ]
    return [Piece(name, dtype, tensor.size, member, start, nbytes)]


def write_safetensors(path: Path, out: Path, *, split_qkv: bool = False) -> Converted:
    """`path`'s tensors into `out` (written beside it as `.part`, then moved into place)."""
    found = pieces(path, split_qkv=split_qkv)
    header: dict[str, object] = {}
    offset = 0
    for piece in found:
        header[piece.name] = {
            "dtype": piece.dtype,
            "shape": list(piece.shape),
            "data_offsets": [offset, offset + piece.nbytes],
        }
        offset += piece.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # the data starts 8-byte aligned
    part = out.with_name(f"{out.name}.part")
    try:
        with zipfile.ZipFile(path) as archive, part.open("wb") as sink:
            sink.write(struct.pack("<Q", len(blob)) + blob)
            for piece in found:
                with archive.open(piece.member) as source:
                    _copy(source, sink, piece.start, piece.nbytes)
    except zipfile.BadZipFile as exc:  # a storage whose CRC fails: a broken download
        part.unlink(missing_ok=True)
        raise CheckpointError(f"{path.name} is damaged: {exc}") from exc
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(out)
    return Converted(tensors=len(found), nbytes=offset)


def _copy(source: IO[bytes], sink: IO[bytes], start: int, nbytes: int) -> None:
    source.seek(start)
    left = nbytes
    while left:
        chunk = source.read(min(left, 16 * 1024 * 1024))
        if not chunk:
            raise CheckpointError("the checkpoint ended inside a tensor")
        sink.write(chunk)
        left -= len(chunk)
