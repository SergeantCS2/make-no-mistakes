"""A card's identifiers are secrets (a card's uuid identifies the user's hardware as a token
identifies its holder): this module is the one place that says what one looks like and how it
is kept out of sight. `masked` is a stable id as anything but the user's own terminal shows
it (`mnm machine show` without `--reveal`); `masked_cards` the same over a report's card lists
(`mnm doctor --json`); `without_card_ids` the ids taken out of text a tool printed, which a
doctor line (`doctor_types.Check`), the GPU smoke test's output (`probe.SmokeResult`) and
whatever leaves the box (`report.redact`) all pass through (else a log pasted from `mnm doctor
--deep`'s smoke-test row carries `nvidia-smi -L`'s uuid).
A leaf: nothing here imports mnm.
"""

from __future__ import annotations

import re

PRIVATE = "uuid kept private (`mnm machine show --reveal`)"
AMD_ID = "AMD-"  # an AMD card's stable id: its driver's `unique_id`
# NVIDIA's card and MIG instance uuids
GPU_UUID = re.compile(
    r"\b(?:GPU|MIG)-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
# An AMD card's `unique_id` (sysfs prints it as 16 hex digits; mnm's stable id carries
# it as `AMD-<id>`, rocminfo as `GPU-<id>`), and the same number where a tool names it
# (`unique_id`, amd-smi's `asic_serial`)
AMD_UNIQUE_ID = re.compile(
    r"\b(?:AMD|GPU)-[0-9A-Fa-f]{16}\b|"
    r"(?<=unique_id)([\"':=\s]+)(?:0x)?[0-9A-Fa-f]{16}\b|"
    r"(?<=asic_serial)([\"':=\s]+)(?:0x)?[0-9A-Fa-f]{16}\b",
    re.IGNORECASE,
)
# nvidia-smi -L names each card and MIG instance with its uuid in brackets: the brackets go
# with it (`GPU 0: NVIDIA GeForce RTX 4070`), on that line only
LISTED_UUID = re.compile(r"[ \t]*\(UUID:[^)\n]*\)")
UUID_REMOVED = "<card uuid removed>"
ID_REMOVED = "<card id removed>"
CARD_LISTS = ("gpus", "unchosen")  # a hardware report's card lists, as `model_dump` gives them


def masked(stable_id: str) -> str:
    """A card's stable id as anything but the user's own terminal shows it: a uuid (NVIDIA's,
    or AMD's `unique_id`) is a secret like a token; a slot or a name is not."""
    return PRIVATE if stable_id.startswith(("GPU-", AMD_ID)) else stable_id


def masked_cards(dumped: dict[str, object]) -> dict[str, object]:
    """A dumped hardware report with every card's uuid and AMD id masked, in each card list
    (`unchosen` holds every card a stale `MNM_GPU` left out)."""
    for name in CARD_LISTS:
        cards = dumped.get(name)
        for card in cards if isinstance(cards, list) else []:
            for key in ("uuid", "unique_id"):
                if isinstance(card, dict) and card.get(key):
                    card[key] = PRIVATE  # whatever its shape: a uuid is never printed
    return dumped


def without_card_ids(text: str) -> str:
    """`text` with every card's uuid and AMD id left out: nvidia-smi's `(UUID: …)` dropped
    whole, any other uuid or id replaced by a word saying it was removed."""
    text = LISTED_UUID.sub("", text)
    text = GPU_UUID.sub(UUID_REMOVED, text)
    return AMD_UNIQUE_ID.sub(_amd_id_removed, text)


def _amd_id_removed(found: re.Match[str]) -> str:
    """The id's place kept (`unique_id: `), the number gone."""
    lead = next((g for g in found.groups() if g is not None), "")
    return f"{lead}{ID_REMOVED}"
