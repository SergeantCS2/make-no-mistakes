"""The model families mnm knows the successors of: read from the makers'
and unsloth's Hub pages on 2026-10-26, never guessed. A row is one size class of one family,
its members oldest first; `mnm upgrade` offers the newest member after the instance's.

Each member matches a saved instance by its repo (the name after the org, lower case:
`unsloth/Qwen3-8B-GGUF` and `bartowski/Qwen_Qwen3-8B-GGUF` read alike) and, for a repo of
several models (whisper.cpp's sizes), its quant label. The repo planned for it is `gguf` for
llama.cpp, sd.cpp and whisper.cpp instances, `weights` (safetensors) for vLLM ones.

What the table does not cover the Hub may: `successors.py` reads a card's `new_version` and
`base_model` after it. A row whose last member is the instance's says "the newest the table
knows" (Qwen3-Embedding, Z-Image, FLUX.2 klein: nothing newer published on 2026-10-26)."""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel

READ = date(2026, 10, 26)  # every row below was read on the Hub that day


class Offer(StrEnum):
    NEWER = "newer"  # a newer family member
    FASTER = "faster"  # the same model made faster (whisper's turbo): never called better


class Member(BaseModel):
    label: str  # what the user reads: "Qwen3.5-9B"
    match: str  # a regex, fullmatch on `<repo name>` lower case (see `repo_key`)
    gguf: str | None = None  # the repo planned for a GGUF (or GGML) instance
    weights: str | None = None  # the repo planned for a vLLM instance
    quant: str | None = None  # the size label inside a repo of several (whisper.cpp)

    def repo_for(self, vllm: bool) -> str | None:
        return self.weights if vllm else self.gguf


class Family(BaseModel):
    name: str  # "Qwen 8B class"
    maker: str
    members: tuple[Member, ...]  # oldest first
    offer: Offer = Offer.NEWER
    note: str = ""


def _gguf(name: str) -> str:
    """`qwen3-8b` and its GGUF, AWQ or FP8 builds, with or without an org prefix
    (bartowski's `Qwen_Qwen3-8B-GGUF`)."""
    return rf"(?:[a-z0-9]+_)?{name}(?:-(?:gguf|awq|fp8|gptq-int[48]))?"


QWEN = "Alibaba (Qwen)"
GOOGLE = "Google (Gemma)"

FAMILIES: tuple[Family, ...] = (
    Family(
        name="Qwen 0.6B class",
        maker=QWEN,
        members=(
            Member(label="Qwen3-0.6B", match=_gguf(r"qwen3-0\.6b")),
            Member(
                label="Qwen3.5-0.8B",
                match=_gguf(r"qwen3\.5-0\.8b"),
                gguf="unsloth/Qwen3.5-0.8B-GGUF",
                weights="Qwen/Qwen3.5-0.8B",
            ),
        ),
    ),
    Family(
        name="Qwen 1.7B class",
        maker=QWEN,
        members=(
            Member(label="Qwen3-1.7B", match=_gguf(r"qwen3-1\.7b")),
            Member(
                label="Qwen3.5-2B",
                match=_gguf(r"qwen3\.5-2b"),
                gguf="unsloth/Qwen3.5-2B-GGUF",
                weights="Qwen/Qwen3.5-2B",
            ),
        ),
    ),
    Family(
        name="Qwen 4B class",
        maker=QWEN,
        members=(
            Member(label="Qwen3-4B", match=_gguf(r"qwen3-4b(?:-(?:instruct|thinking)-2507)?")),
            Member(
                label="Qwen3.5-4B",
                match=_gguf(r"qwen3\.5-4b"),
                gguf="unsloth/Qwen3.5-4B-GGUF",
                weights="Qwen/Qwen3.5-4B",
            ),
        ),
    ),
    Family(
        name="Qwen 8B class",
        maker=QWEN,
        members=(
            Member(label="Qwen3-8B", match=_gguf(r"qwen3-8b")),
            Member(
                label="Qwen3.5-9B",
                match=_gguf(r"qwen3\.5-9b"),
                gguf="unsloth/Qwen3.5-9B-GGUF",
                weights="Qwen/Qwen3.5-9B",
            ),
        ),
        note="Qwen3.6 and Qwen3.8 have no 8B or 9B; Qwen3.5-9B is the newest at this size.",
    ),
    Family(
        name="Qwen 27B class",
        maker=QWEN,
        members=(
            Member(label="Qwen3-32B", match=_gguf(r"qwen3-32b")),
            Member(
                label="Qwen3.5-27B",
                match=_gguf(r"qwen3\.5-27b"),
                gguf="unsloth/Qwen3.5-27B-GGUF",
                weights="Qwen/Qwen3.5-27B",
            ),
            Member(
                label="Qwen3.6-27B",
                match=_gguf(r"qwen3\.6-27b"),
                gguf="unsloth/Qwen3.6-27B-GGUF",
                weights="Qwen/Qwen3.6-27B",
            ),
            Member(
                label="Qwen3.8-27B",
                match=_gguf(r"qwen3\.8-27b"),
                gguf="unsloth/Qwen3.8-27B-GGUF",
                weights="Qwen/Qwen3.8-27B",
            ),
        ),
    ),
    Family(
        name="Qwen 30B-A3B class",
        maker=QWEN,
        members=(
            Member(
                label="Qwen3-30B-A3B",
                match=_gguf(r"qwen3-30b-a3b(?:-(?:instruct|thinking)-2507)?"),
            ),
            Member(
                label="Qwen3.5-35B-A3B",
                match=_gguf(r"qwen3\.5-35b-a3b"),
                gguf="unsloth/Qwen3.5-35B-A3B-GGUF",
                weights="Qwen/Qwen3.5-35B-A3B",
            ),
            Member(
                label="Qwen3.6-35B-A3B",
                match=_gguf(r"qwen3\.6-35b-a3b"),
                gguf="unsloth/Qwen3.6-35B-A3B-GGUF",
                weights="Qwen/Qwen3.6-35B-A3B",
            ),
        ),
        note="Qwen3.8 has no 35B-A3B.",
    ),
    Family(
        name="Qwen 14B",
        maker=QWEN,
        members=(Member(label="Qwen3-14B", match=_gguf(r"qwen3-14b")),),
        note="Qwen3.5 has no 14B: its 9B and 27B are either side (`mnm add` either).",
    ),
    Family(
        name="Qwen3 embeddings",
        maker=QWEN,
        members=(
            Member(
                label="Qwen3-Embedding",
                match=_gguf(r"qwen3-embedding-(?:0\.6|4|8)b"),
            ),
        ),
    ),
    Family(
        name="Qwen3 rerankers",
        maker=QWEN,
        members=(Member(label="Qwen3-Reranker", match=_gguf(r"qwen3-reranker-(?:0\.6|4|8)b")),),
    ),
    Family(
        name="Gemma E2B",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3n-E2B", match=_gguf(r"gemma-3n-e2b-it")),
            Member(
                label="gemma-4-E2B",
                match=_gguf(r"gemma-4-e2b-it"),
                gguf="unsloth/gemma-4-E2B-it-GGUF",
                weights="google/gemma-4-E2B-it",
            ),
        ),
    ),
    Family(
        name="Gemma 4B / E4B",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-4b", match=_gguf(r"gemma-3n?-(?:4b|e4b)-it")),
            Member(
                label="gemma-4-E4B",
                match=_gguf(r"gemma-4-e4b-it"),
                gguf="unsloth/gemma-4-E4B-it-GGUF",
                weights="google/gemma-4-E4B-it",
            ),
        ),
    ),
    Family(
        name="Gemma 4B QAT",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-4b QAT", match=_gguf(r"gemma-3-4b-it-qat(?:-int4)?")),
            Member(
                label="gemma-4-E4B QAT",
                match=_gguf(r"gemma-4-e4b-it-qat"),
                gguf="unsloth/gemma-4-E4B-it-qat-GGUF",
            ),
        ),
    ),
    Family(
        name="Gemma 12B",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-12b", match=_gguf(r"gemma-3-12b-it")),
            Member(
                label="gemma-4-12B",
                match=_gguf(r"gemma-4-12b-it"),
                gguf="unsloth/gemma-4-12b-it-GGUF",
                weights="google/gemma-4-12B-it",
            ),
        ),
    ),
    Family(
        name="Gemma 12B QAT",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-12b QAT", match=_gguf(r"gemma-3-12b-it-qat(?:-int4)?")),
            Member(
                label="gemma-4-12B QAT",
                match=_gguf(r"gemma-4-12b-it-qat"),
                gguf="unsloth/gemma-4-12B-it-qat-GGUF",
            ),
        ),
    ),
    Family(
        name="Gemma 27B / 31B",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-27b", match=_gguf(r"gemma-3-27b-it")),
            Member(
                label="gemma-4-31B",
                match=_gguf(r"gemma-4-31b-it"),
                gguf="unsloth/gemma-4-31B-it-GGUF",
                weights="google/gemma-4-31B-it",
            ),
        ),
    ),
    Family(
        name="Gemma 27B / 31B QAT",
        maker=GOOGLE,
        members=(
            Member(label="gemma-3-27b QAT", match=_gguf(r"gemma-3-27b-it-qat")),
            Member(
                label="gemma-4-31B QAT",
                match=_gguf(r"gemma-4-31b-it-qat"),
                gguf="unsloth/gemma-4-31B-it-qat-GGUF",
            ),
        ),
    ),
    Family(
        name="DeepSeek Flash",
        maker="DeepSeek",
        members=(
            Member(label="DeepSeek-V4-Flash", match=_gguf(r"deepseek-v4-flash(?:-0731)?")),
            Member(
                label="DeepSeek-V4.1-Flash-1017",
                match=_gguf(r"deepseek-v4\.1-flash(?:-1017)?"),
                gguf="unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
                weights="deepseek-ai/DeepSeek-V4.1-Flash-1017",
            ),
        ),
        note="V4.1 needs llama.cpp 74bca39 or later (`mnm engines check`).",
    ),
    Family(
        name="Z-Image",
        maker="Alibaba (Tongyi-MAI)",
        members=(Member(label="Z-Image", match=r"(?:z-image(?:-turbo)?(?:-gguf)?|z_image_turbo)"),),
    ),
    Family(
        name="FLUX.2 klein",
        maker="Black Forest Labs",
        members=(
            Member(label="FLUX.2 klein", match=r"flux\.2-klein-(?:base-)?(?:4b|9b)(?:-gguf)?"),
        ),
    ),
    Family(
        name="Wan text-to-video, small",
        maker="Alibaba (Wan)",
        members=(
            Member(label="Wan 2.1 T2V 1.3B", match=r"wan-?2?\.?1?-(?:t2v-)?1\.3b(?:-gguf)?"),
            Member(
                label="Wan 2.2 TI2V 5B",
                match=r"wan2\.2-ti2v-5b(?:-gguf)?",
                gguf="QuantStack/Wan2.2-TI2V-5B-GGUF",
            ),
        ),
    ),
    Family(
        name="Wan text-to-video, 14B",
        maker="Alibaba (Wan)",
        members=(
            Member(label="Wan 2.1 T2V 14B", match=r"wan2\.1-t2v-14b(?:-gguf)?"),
            Member(
                label="Wan 2.2 T2V A14B",
                match=r"wan2\.2-t2v-a14b(?:-gguf)?",
                gguf="QuantStack/Wan2.2-T2V-A14B-GGUF",
            ),
        ),
    ),
    Family(
        name="Wan image-to-video, 14B",
        maker="Alibaba (Wan)",
        members=(
            Member(label="Wan 2.1 I2V 14B", match=r"wan2\.1-i2v-14b-(?:480|720)p(?:-gguf)?"),
            Member(
                label="Wan 2.2 I2V A14B",
                match=r"wan2\.2-i2v-a14b(?:-gguf)?",
                gguf="QuantStack/Wan2.2-I2V-A14B-GGUF",
            ),
        ),
    ),
    Family(
        name="Whisper large",
        maker="OpenAI (Whisper), whisper.cpp's files",
        members=(
            Member(label="large-v2", match="whisper.cpp", quant="large-v2"),
            Member(
                label="large-v3",
                match="whisper.cpp",
                gguf="ggerganov/whisper.cpp",
                quant="large-v3",
            ),
        ),
    ),
    Family(
        name="Whisper large, turbo",
        maker="OpenAI (Whisper), whisper.cpp's files",
        offer=Offer.FASTER,
        members=(
            Member(label="large-v3", match="whisper.cpp", quant="large-v3"),
            Member(
                label="large-v3-turbo",
                match="whisper.cpp",
                gguf="ggerganov/whisper.cpp",
                quant="large-v3-turbo",
            ),
        ),
        note=(
            "Turbo prunes the decoder to 4 layers: several times faster, a little less "
            "accurate. A speed offer; large-v3 stays the more accurate model."
        ),
    ),
)


def repo_key(repo_id: str) -> str:
    """The repo's name after its owner, lower case (`unsloth/Qwen3-8B-GGUF` → `qwen3-8b-gguf`)."""
    return repo_id.rsplit("/", 1)[-1].lower()


_QUANT_TAIL = re.compile(r"-(?:q\d[\w]*|f16|f32|bf16)$")


def size_label(quant: str) -> str:
    """A whisper.cpp size without its quant: `large-v3-q5_0` → `large-v3`."""
    return _QUANT_TAIL.sub("", quant.lower())


class Placed(BaseModel):
    """Where an instance's repo sits in the table."""

    family: Family
    index: int  # its member's position

    @property
    def member(self) -> Member:
        return self.family.members[self.index]

    @property
    def newest(self) -> Member | None:
        """The newest member after the instance's, or None when it is the newest known."""
        later = self.family.members[self.index + 1 :]
        return later[-1] if later else None


def place(repo_id: str, quant: str | None = None) -> list[Placed]:
    """Every row whose member matches the repo (and size label); whisper's large-v3 is in two
    (a newer member for large-v2, a faster one for itself)."""
    key = repo_key(repo_id)
    size = size_label(quant) if quant else None
    found = []
    for family in FAMILIES:
        for index, member in enumerate(family.members):
            if not re.fullmatch(member.match, key):
                continue
            if member.quant is not None and member.quant != size:
                continue
            found.append(Placed(family=family, index=index))
    return found


def turbo_quant(old: str, wanted: str) -> str:
    """The successor's size label with the old one's quant kept where whisper.cpp ships it
    (`large-v3-q5_0` → `large-v3-turbo-q5_0`; `large-v2-q8_0` → `large-v3`: no v3 q8_0)."""
    tail = old.lower().removeprefix(size_label(old))
    return f"{wanted}{tail}" if f"{wanted}{tail}" in WHISPER_FILES else wanted


# ggerganov/whisper.cpp's large files, as the Hub listed them on 2026-10-26
WHISPER_FILES = frozenset(
    {
        "large-v1",
        "large-v2",
        "large-v2-q5_0",
        "large-v2-q8_0",
        "large-v3",
        "large-v3-q5_0",
        "large-v3-turbo",
        "large-v3-turbo-q5_0",
        "large-v3-turbo-q8_0",
    }
)
