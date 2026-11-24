"""One quantisation of a GGUF repo, as the Hub lists it: a file or a set of shards, and for
a two-stage video model (Wan 2.2 A14B) the high-noise half that runs before it; for a
language model, the multi-token-prediction head it can draft with and the Engram shard
it keeps on disk."""

from __future__ import annotations

from pydantic import BaseModel

from makenomistakes.gguf.tensors import MXFP8_TYPE


class EngramShard(BaseModel):
    """The shard of a split GGUF that holds an Engram table (DeepSeek-V4.1's n-gram memory,
    `per_layer_token_embd.weight`), read from its tensor infos (`plan_large`): downloaded with
    the model and read row by row from disk as needed, never loaded into RAM or VRAM."""

    file: str
    size_bytes: int  # the table's bytes (the shard's size when its type is one ggml lacks)
    tensor_type: int  # ggml's type id; MXFP8 (43) is refused

    @property
    def unreadable(self) -> bool:
        return self.tensor_type == MXFP8_TYPE  # unsloth's MXFP8: llama.cpp never reads it


class DraftHead(BaseModel):
    """A speculative-decoding head the repo ships that mnm does not run: a
    DeepSeek-V4.1 DSpark head, a DFlash or EAGLE-3 one. Listed as unused, never downloaded."""

    file: str
    size_bytes: int = 0
    kind: str  # its llama.cpp prefix without the dash: dspark, dflash, eagle3


class GgufVariant(BaseModel):
    quant: str
    files: list[str]  # the model's own file(s): shards of one model, never another model
    size_bytes: int  # everything the variant downloads, the high-noise half included
    file_bytes: list[int] = []  # each of `files`' sizes, when the tree listed them
    # Wan 2.2 A14B: the high-noise diffusion model of the same quant (`hub/video_files`),
    # loaded beside `files[0]` as `--high-noise-diffusion-model`.
    high_noise: str | None = None
    # PocketTTS: the voice embeddings beside the GGUF, downloaded with it (`hub/voices`).
    voice_files: list[str] = []
    voice_bytes: int = 0  # their size, loaded beside the model
    # The MTP head beside the model (`hub/gguf_files.with_mtp_heads`): downloaded with it and
    # run only with `mnm set <name> draft=mtp` (`--spec-type draft-mtp -md <head>`). Not in
    # `size_bytes`, which the planner reads as the model's own weights.
    mtp: str | None = None
    mtp_bytes: int = 0
    # The shard whose Engram table stays on disk (read by `plan_large`); its bytes are in
    # `size_bytes` (downloaded) but never in `memory_bytes`, what RAM and the cards hold.
    engram: EngramShard | None = None

    @property
    def memory_bytes(self) -> int:
        """The weights RAM and the cards hold: everything but an Engram table."""
        return self.size_bytes - (self.engram.size_bytes if self.engram else 0)

    def largest_file(self) -> str:
        """The largest of its files (the first when the sizes are unknown)."""
        if len(self.file_bytes) != len(self.files):
            return self.files[0]
        return max(zip(self.file_bytes, self.files, strict=True))[1]
