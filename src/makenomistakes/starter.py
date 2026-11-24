"""Starter models: a short menu of instances that have run on a 12 GB card (an RTX 4070), for
`mnm setup`'s last step and `mnm starter`. Each is an ordinary `mnm add`; nothing here
bypasses the planner, which still fits every one to this machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.fit.budget import BUSY_CARD_BYTES
from makenomistakes.hardware.fingerprint import UNIFIED_MARGIN
from makenomistakes.ladder import BOLT, DS41, GLM47, POOL_EXTRA
from makenomistakes.model_uses import Use

if TYPE_CHECKING:
    from makenomistakes.hardware.probe import MemorySnapshot


@dataclass(frozen=True)
class Item:
    """One `mnm add`."""

    reference: str
    name: str
    quant: str | None = None
    for_agents: bool = False
    kind: str | None = None  # `--type` (a `Modality` value): Bolt's repo reads as vision
    use: Use | None = None  # Its use, a curated row of `mnm what` (the menu's: every one)


@dataclass(frozen=True)
class Starter:
    key: str
    title: str
    what: str
    items: tuple[Item, ...]
    download_gb: float  # what it downloads, its parts included (from the Hub's listing)
    min_vram_gb: float  # below this its balanced profile does not fit whole on the card
    note: str = ""  # said before anything is added: a large download, RAM beside the card
    # A large starter's RAM floor (`ladder.Rung.need_gb`): RAM + VRAM in GB, 0 for none
    floor_gb: float = 0.0


# The two coders, sizes from the Hub's trees:
# Qwen3.6-35B-A3B UD-Q4_K_M is 20.61 GiB (22.1 GB); Qwen3.8-27B UD-IQ3_XXS 10.18 GiB plus its
# MTP head (1.28 GiB, downloaded with it, used only with `draft=mtp`): 12.3 GB. Both drop the
# projector (`--for-agents` adds them text-only). The row's GB come first, the note after.
CODER36_NOTE = (
    "About 10.5 GiB of its experts stay in RAM beside the card: under WSL set memory=24GB in "
    "%UserProfile%\\.wslconfig, then `wsl --shutdown` (`mnm doctor` reads what WSL has)."
)
CODER38_NOTE = "The model and its MTP head, all of it on the card."


CATALOG: tuple[Starter, ...] = (
    Starter(
        "chat",
        "Chat",
        "Qwen3 8B: a capable all-round chat model, tools and thinking (~76 tok/s on a 4070)",
        (Item("unsloth/Qwen3-8B-GGUF", "qwen3-8b", use=Use.CHAT),),
        5.0,
        8,
    ),
    Starter(
        "small",
        "Small chat",
        "Qwen3 0.6B: tiny and fast, for quick tests and small cards",
        (Item("unsloth/Qwen3-0.6B-GGUF", "tiny", use=Use.CHAT),),
        0.4,
        2,
    ),
    Starter(
        "coding",
        "Coding agent",
        "Qwen3 8B with a 32k context, for Claude Code, Codex, OpenCode and Aider",
        (Item("unsloth/Qwen3-8B-GGUF", "coder8b", for_agents=True, use=Use.CODER),),
        5.0,
        8,
    ),
    Starter(
        "coder36",
        "Coder (MoE)",
        "Qwen3.6 35B-A3B: the bigger coding agent, 32k a request, experts in RAM (about 25-28 "
        "tok/s on an RTX 4070)",
        (
            Item(
                "unsloth/Qwen3.6-35B-A3B-GGUF",
                "coder36",
                quant="UD-Q4_K_M",
                for_agents=True,
                use=Use.CODER,
            ),
        ),
        22.1,
        12,
        CODER36_NOTE,
    ),
    Starter(
        "coder38",
        "Coder (dense)",
        "Qwen3.8 27B at 3 bits: the lighter coder, all on the card (about 21 tok/s on an RTX 4070)",
        (
            Item(
                "unsloth/Qwen3.8-27B-GGUF",
                "coder38",
                quant="UD-IQ3_XXS",
                for_agents=True,
                use=Use.CODER,
            ),
        ),
        12.3,
        12,
        CODER38_NOTE,
    ),
    Starter(
        "vision",
        "Vision",
        "Qwen3-VL 8B: chat about pictures and screenshots",
        (Item("Qwen/Qwen3-VL-8B-Instruct-GGUF", "qwen3-vl", use=Use.VISION),),
        6.2,
        8,
    ),
    Starter(
        "memory",
        "Memory",
        "Qwen3 Embedding 0.6B: memory and document search for chat apps",
        (Item("Qwen/Qwen3-Embedding-0.6B-GGUF", "embed", use=Use.MEMORY),),
        0.6,
        2,
    ),
    Starter(
        "pictures",
        "Pictures",
        "Z-Image Turbo: text-to-image at 1024x1024 (about 20 s on an RTX 4070)",
        (Item("unsloth/Z-Image-Turbo-GGUF", "zimage", use=Use.PICTURES),),
        10.0,
        8,
    ),
    Starter(
        "speech",
        "Speech",
        "Whisper large-v3-turbo (speech to text) and PocketTTS (text to speech, 26 voices)",
        (
            Item("ggerganov/whisper.cpp", "stt", quant="large-v3-turbo-q8_0", use=Use.EAR),
            Item("audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english", "pocket", use=Use.VOICE),
        ),
        1.2,
        3,
    ),
    Starter(
        "video",
        "Video",
        "Wan 2.2 5B: 2-second clips from text or from a picture (2-3 minutes each on an RTX 4070)",
        (Item("QuantStack/Wan2.2-TI2V-5B-GGUF", "wan5b", use=Use.VIDEO),),
        8.0,
        10,
    ),
)
# Offered only where the cards hold 64 GiB or more together (four 16 GB cards) and the
# box's RAM and VRAM hold the model (`ladder`'s floor). The quant is the one the planner is
# asked for; the plan still decides the rest. Sizes and floors from `ladder.LADDER`.
LARGE_CARDS_GB = 64
LARGE: tuple[Starter, ...] = (
    Starter(
        "glm47",
        "GLM-4.7-Flash",
        "GLM-4.7-Flash 30B-A3B: DeepSeek's architecture in small, tools and thinking",
        (Item(GLM47, "glm47", quant="Q4_K_M"),),
        18.3,
        LARGE_CARDS_GB,
        floor_gb=23.1,
    ),
    Starter(
        "bolt",
        "Nemotron Bolt",
        "Nemotron-3.5-Bolt 74B-A9B: a hybrid Mamba mixture of experts, tools and thinking",
        (Item(BOLT, "bolt", quant="UD-Q4_K_XL", kind="chat"),),
        50.2,
        LARGE_CARDS_GB,
        floor_gb=55.0,  # the text model, as it is added (`ladder.LADDER`)
    ),
    Starter(
        "ds41",
        "DeepSeek-V4.1",
        "DeepSeek-V4.1-Flash 763B at 1.6 bits: experts in RAM, about 5-12 tok/s estimated",
        (Item(DS41, "ds41", quant="UD-IQ1_S"),),
        233.0,
        LARGE_CARDS_GB,
        "233 GB to download, 111 GB of it an Engram table that stays on disk (put the models "
        "folder on a fast SSD); it needs a llama.cpp image with DeepSeek-V4.1 "
        "(`mnm engines update llamacpp`).",
        floor_gb=127.0,
    ),
)
DEFAULT = ("chat",)
BY_KEY = {starter.key: starter for starter in (*CATALOG, *LARGE)}


GIB = 1024**3


def fits(starter: Starter, vram_bytes: int) -> bool:
    """The one rule: the floor within 2% of the VRAM given (a 12 GB card reads 11.99 GiB)."""
    return starter.min_vram_gb * GIB <= vram_bytes * 1.02


def fitting(
    vram_bytes: int | None,
    ram_bytes: int | None = None,
    cards_bytes: int | None = None,
    pool: bool = False,
) -> list[Starter]:
    """What this card runs (all of it when the card is unknown: the planner decides);
    with the RAM and every chosen card's VRAM together (`cards_bytes`), the large starters
    those hold."""
    if vram_bytes is None:
        return list(CATALOG)
    found = [s for s in CATALOG if fits(s, vram_bytes)]
    if ram_bytes is None or cards_bytes is None:
        return found
    return found + [s for s in LARGE if large_fits(s, ram_bytes, cards_bytes, pool)]


def offered_on(snapshot: MemorySnapshot) -> list[Starter]:
    """`fitting` for the machine as the probe reads it: the first card, the RAM and every
    chosen card (`mnm starter`, the web's Get started)."""
    gpu = snapshot.gpu
    if gpu is None:  # no card read: the whole menu, the planner decides
        return fitting(None)
    ram = snapshot.memory.total_bytes if snapshot.memory is not None else None
    cards = sum(c.vram_total_bytes for c in snapshot.cards) or None
    if gpu.unified or gpu.carve_out:  # One pool (its card's total is the pool less
        ram = 0  # the margin, `unified_pool`): the RAM floor counts it once
    return fitting(gpu.vram_total_bytes, ram, cards, pool=gpu.unified)


def large_fits(starter: Starter, ram_bytes: int, cards_bytes: int, pool: bool = False) -> bool:
    """The cards hold 64 GiB or more together, and the RAM floor fits the RAM and VRAM. On a
    unified pool (`cards_bytes` the pool less its margin) the floor is judged as the ladder
    judges it (`ladder.holds(pool=True)`: the whole pool, the firmer margin once), never with
    the margin counted twice."""
    if not fits(starter, cards_bytes):
        return False
    if pool:
        return starter.floor_gb * 10**9 + POOL_EXTRA <= cards_bytes + UNIFIED_MARGIN
    return starter.floor_gb * 10**9 <= ram_bytes + cards_bytes


def floor_words(starter: Starter) -> str:
    """The VRAM floor as the menu says it (what the card needs, beside each download); a large
    starter's RAM + VRAM too."""
    if starter.floor_gb:
        return f"needs {starter.min_vram_gb:g} GiB of cards, {starter.floor_gb:.0f} GB with RAM"
    return f"needs about {starter.min_vram_gb:g} GiB of VRAM"


def room_words(starter: Starter, free_bytes: int | None) -> str | None:
    """Why a starter this card runs will not fit as the card is now, against the free VRAM a
    plan gets (mnm's own running models count as free, `planning_budget`); None when it fits
    or the card cannot be read. The floor is a whole card's, so what other programs hold up
    to `BUSY_CARD_BYTES` (a desktop's share, the busy-card threshold) is allowed for: an
    idle 12 GB card with 11.5 GiB free still runs a 12 GiB starter."""
    if free_bytes is None or fits(starter, free_bytes + BUSY_CARD_BYTES):
        return None
    return f"won't fit now: {free_bytes / GIB:.1f} GiB free"


def parse_picks(answer: str, offered: list[Starter]) -> list[Starter] | None:
    """`1 3`, `1,3`, `chat memory`, `all` (every starter but the large ones) or `none`
    (empty: the default); None when a word names nothing offered (asked again)."""
    words = answer.replace(",", " ").lower().split()
    if not words:
        return [s for s in offered if s.key in DEFAULT]
    if words == ["none"]:
        return []
    if words == ["all"]:  # the menu's own; a large starter only by name or number
        return [s for s in offered if not s.floor_gb]
    picked: list[Starter] = []
    for word in words:
        if word.isdigit() and 1 <= int(word) <= len(offered):
            found = offered[int(word) - 1]
        elif word in BY_KEY and BY_KEY[word] in offered:
            found = BY_KEY[word]
        else:
            return None
        if found not in picked:
            picked.append(found)
    return picked


def download_gb(picked: list[Starter]) -> float:
    """The total, counting a file two picks share once (chat and coding are one GGUF)."""
    seen: set[str] = set()
    total = 0.0
    for starter in picked:
        refs = {item.reference for item in starter.items}
        if refs <= seen:
            continue
        seen |= refs
        total += starter.download_gb
    return total
