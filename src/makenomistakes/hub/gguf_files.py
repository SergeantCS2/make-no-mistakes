"""Which files of a repo tree make one GGUF or whisper.cpp variant: shards grouped by stem,
vision projectors kept apart (and the f16 one preferred), quant labels read from names, and
what a pasted link to one file brings along (sibling shards, a projector, the other half of a
Wan 2.2 A14B pair). Files that only travel beside a model are never a variant: the importance
matrix a quantiser used (`imatrix_unsloth.gguf`, no architecture at all) and the draft heads
llama.cpp names by prefix (`mtp-`, `eagle3-`, `dflash-`, `dspark-`, or an `MTP/` folder); an
MTP head is paired with the variants it serves (`with_mtp_heads`). Plain functions over
`TreeEntry` lists; `facts.build_facts` calls them."""

from __future__ import annotations

import re
from dataclasses import dataclass

from makenomistakes.hub.client import TreeEntry
from makenomistakes.hub.variants import DraftHead, GgufVariant
from makenomistakes.hub.video_files import noise_half

_QUANT_RE = re.compile(
    r"(?i)(?:^|[-_.])((?:UD-)?(?:IQ\d_[A-Z0-9_]+|Q\d(?:_[A-Z0-9]+)+|TQ\d_\d|MXFP4|BF16|F16|FP16|F32))"
    r"(?=[-_.]|$)"
)
_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)


def gguf_model(variant: GgufVariant) -> str:
    """The model a GGUF variant belongs to: its file name without the quant and extension
    (`anima-preview2-Q4_K_M.gguf` -> `anima-preview2`; shards share their stem)."""
    name = basename(variant.files[0])
    shard = _SHARD_RE.match(name)
    stem = shard.group("stem") if shard else _gguf_stem(name)
    match = _QUANT_RE.search(stem)
    return (stem[: match.start(1)].rstrip("-_.") if match else stem) or stem


_PROJECTOR_PREFERENCE = tuple(
    re.compile(p) for p in (r"(?<![a-z])f16", r"bf16", r"q8_0", r"(?<![a-z])f32")
)


def projector_rank(entry: TreeEntry) -> tuple[int, str]:
    """f16 first: the projector is small, and f16 is what llama.cpp tests against."""
    name = basename(entry.path).lower()
    rank = next((i for i, tag in enumerate(_PROJECTOR_PREFERENCE) if tag.search(name)), 99)
    return rank, name


def is_projector(path: str) -> bool:
    """A vision projector: `mmproj-F16.gguf`, and second-state's `llava-…-mmproj-model-f16.gguf`
    (a projector read as a quant would break the plan)."""
    name = basename(path).lower()
    return name.endswith(".gguf") and "mmproj" in name


def is_imatrix(path: str) -> bool:
    """The importance matrix a quant was made with (unsloth's `imatrix_unsloth.gguf`,
    mradermacher's `<model>.imatrix.gguf` in every i1 repo): a GGUF without
    `general.architecture`, which the plan would otherwise read as the smallest variant. A
    quant named `…-imatrix.gguf` is a model."""
    name = basename(path).lower()
    return name.endswith(".gguf") and (
        name.startswith("imatrix") or name.removesuffix(".gguf").endswith(".imatrix")
    )


# llama.cpp's own sidecar keywords (common/download.cpp `sidecars`): files a model runs beside.
MTP_PREFIX = "mtp-"
OTHER_HEAD_PREFIXES = ("eagle3-", "dflash-", "dspark-")  # named, never run
DRAFT_HEAD_PREFIXES = (MTP_PREFIX, *OTHER_HEAD_PREFIXES)


def is_draft_head(path: str) -> bool:
    """A speculative-decoding head beside the model: named by llama.cpp's prefixes, or in an
    `MTP/` folder (unsloth's). Never a variant: one shadowed the real Q4_0 of Qwen3.8-27B."""
    name = basename(path).lower()
    folders = path.lower().split("/")[:-1]
    return name.endswith(".gguf") and (name.startswith(DRAFT_HEAD_PREFIXES) or "mtp" in folders)


def is_mtp_head(path: str) -> bool:
    """A multi-token-prediction head (`--spec-type draft-mtp`): the one draft kind mnm runs."""
    name = basename(path).lower()
    folders = path.lower().split("/")[:-1]
    in_folder = "mtp" in folders and not name.startswith(OTHER_HEAD_PREFIXES)
    return name.endswith(".gguf") and (name.startswith(MTP_PREFIX) or in_folder)


def _companion_only(path: str) -> bool:
    return is_projector(path) or is_imatrix(path) or is_draft_head(path)


def folder_of(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def within(path: str, folder: str | None) -> bool:
    return folder is None or path == folder or path.startswith(folder.rstrip("/") + "/")


def with_sibling_shards(
    files: list[TreeEntry], tree: list[TreeEntry], path: str | None
) -> list[TreeEntry]:
    """A `/blob` or `/resolve` link to one shard of a split GGUF (the Hub's own download
    link) means the whole model: every shard with that stem comes along, or the plan would
    size and download one third of it and llama-server would die looking for the rest."""
    if path is None or len(files) != 1 or not (shard := _SHARD_RE.match(basename(path))):
        return files
    folder, stem = folder_of(path), shard.group("stem")
    siblings = [
        e
        for e in tree
        if e.is_file
        and folder_of(e.path) == folder
        and (m := _SHARD_RE.match(basename(e.path))) is not None
        and m.group("stem") == stem
    ]
    return siblings or files


def with_companions(
    files: list[TreeEntry], tree: list[TreeEntry], path: str | None
) -> list[TreeEntry]:
    """A link to a model file or folder brings what runs with it: a vision GGUF's `mmproj`
    from the same folder (else the instance is text-only behind a vision card), and the
    other half of each Wan 2.2 A14B pair it holds, wherever that sits (`HighNoise/` beside
    `LowNoise/`): a high-noise half never runs alone. A pasted GGUF file or a folder of them
    (unsloth's sharded quants) also brings the repo's MTP heads (an `MTP/` folder beside
    it), which `with_mtp_heads` pairs by quant."""
    if path is None or not files:
        return files
    have = {e.path for e in files}
    pasted = _Pasted(
        file=path.lower().endswith(".gguf"),
        folder=folder_of(path),
        ggufs=any(e.path.lower().endswith(".gguf") for e in files),
        halves=frozenset(half[0] for e in files if (half := noise_half(e.path)) is not None),
    )
    extra = [e for e in tree if e.is_file and e.path not in have and pasted.brings(e.path)]
    return [*files, *extra]


@dataclass(frozen=True)
class _Pasted:
    """What a pasted file or folder holds, for `with_companions`."""

    file: bool  # a GGUF file was pasted, not a folder
    folder: str  # the pasted path's folder
    ggufs: bool  # the selection holds GGUF files
    halves: frozenset[str]  # the Wan 2.2 A14B pairs it holds

    def brings(self, path: str) -> bool:
        """Whether this file of the repo runs with the paste: its folder's projector, an MTP
        head of the repo, the other half of a noise pair."""
        if self.file and folder_of(path) == self.folder and is_projector(path):
            return True
        if self.ggufs and is_mtp_head(path):
            return True
        half = noise_half(path)
        return half is not None and half[0] in self.halves


def basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def gguf_variants(files: list[TreeEntry]) -> list[GgufVariant]:
    groups: dict[str, list[TreeEntry]] = {}
    for entry in files:
        name = basename(entry.path)
        if not name.lower().endswith(".gguf") or _companion_only(entry.path):
            continue
        shard = _SHARD_RE.match(name)
        key = f"{folder_of(entry.path)}/{shard.group('stem')}" if shard else entry.path
        groups.setdefault(key, []).append(entry)

    variants: list[GgufVariant] = []
    for key, group in groups.items():
        quant = _quant_of(basename(key)) or _quant_of(folder_of(key)) or _gguf_stem(basename(key))
        entries = sorted(group, key=lambda e: e.path)
        variants.append(
            GgufVariant(
                quant=quant,
                files=[e.path for e in entries],
                size_bytes=sum(e.size for e in entries),
                file_bytes=[e.size for e in entries],
            )
        )
    return sorted(variants, key=lambda v: v.size_bytes)


def with_mtp_heads(variants: list[GgufVariant], files: list[TreeEntry]) -> list[GgufVariant]:
    """Each variant with the MTP head it drafts with (`GgufVariant.mtp`): the head of its own
    quant label when the repo has one, else the labelled head nearest in bits, by
    llama.cpp's own sibling rule (`find_best_sibling`: the first number of the quant tag). A
    head is a model of its own (`-md`), so any quant of it serves any quant of the model:
    unsloth ships one Q4_0 head for every quant of Qwen3.8-27B. A head without a quant label
    pairs with nothing (`unused_mtp_heads` lists it)."""
    heads = sorted(
        (e for e in files if is_mtp_head(e.path) and _quant_of(basename(e.path))),
        key=lambda e: (e.size, e.path),
    )
    if not heads:
        return variants
    paired: list[GgufVariant] = []
    for variant in variants:
        own = [h for h in heads if _quant_of(basename(h.path)) == variant.quant.upper()]
        bits = _tag_bits(variant.quant)
        head = (own or sorted(heads, key=lambda h: abs(_tag_bits(h.path) - bits)))[0]
        paired.append(variant.model_copy(update={"mtp": head.path, "mtp_bytes": head.size}))
    return paired


def other_draft_heads(files: list[TreeEntry]) -> list[DraftHead]:
    """The repo's draft heads of the kinds mnm does not run (DSpark, DFlash, EAGLE-3)."""
    heads = []
    for entry in files:
        name = basename(entry.path).lower()
        kind = next((p for p in OTHER_HEAD_PREFIXES if name.startswith(p)), None)
        if kind is not None and name.endswith(".gguf"):
            heads.append(DraftHead(file=entry.path, size_bytes=entry.size, kind=kind[:-1]))
    return sorted(heads, key=lambda h: h.file)


def unused_mtp_heads(variants: list[GgufVariant], files: list[TreeEntry]) -> list[str]:
    """The repo's MTP heads no variant drafts with (`mnm resolve` lists them as unused)."""
    used = {v.mtp for v in variants}
    return sorted(e.path for e in files if is_mtp_head(e.path) and e.path not in used)


def head_quant(path: str) -> str:
    """A head file's quant label (`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` -> `Q4_0`), empty without."""
    return _quant_of(basename(path)) or ""


def _tag_bits(name: str) -> int:
    """llama.cpp's `extract_quant_bits`: the first number in a quant tag (`UD-IQ3_XXS` 3,
    `BF16` 16), 0 without one."""
    quant = _quant_of(basename(name)) or name
    digits = re.search(r"\d+", quant)
    return int(digits.group()) if digits else 0


_WHISPER_RE = re.compile(r"^ggml-(?P<size>[a-z0-9][a-z0-9._\-]*)\.bin$", re.IGNORECASE)


def is_whisper_bin(path: str) -> bool:
    return bool(_WHISPER_RE.match(basename(path)))


def whisper_variants(files: list[TreeEntry]) -> list[GgufVariant]:
    """whisper.cpp repos hold one `ggml-<size>[-<quant>].bin` per model; the size is the
    "quant" label (`base.en`, `large-v3-turbo-q8_0`). Core ML encoders (.zip) are skipped."""
    variants = [
        GgufVariant(quant=m.group("size"), files=[entry.path], size_bytes=entry.size)
        for entry in files
        if (m := _WHISPER_RE.match(basename(entry.path)))
    ]
    return sorted(variants, key=lambda v: v.size_bytes)


def _gguf_stem(name: str) -> str:
    """Fallback variant label for a GGUF whose name has no recognisable quant."""
    return name[: -len(".gguf")] if name.lower().endswith(".gguf") else name


def _quant_of(name: str) -> str | None:
    match = _QUANT_RE.search(name)
    return match.group(1).upper() if match else None


def extension(path: str) -> str:
    name = basename(path)
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""
