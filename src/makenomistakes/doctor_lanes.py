"""The doctor's lines about the cards' PCIe links on a box with several: per card its
link now and at most and what that slot is good for, a WARN for an x1 riser, a WARN when the
largest card is not the one on the widest link (the main card, `fit.cards.ranked`: prompt
processing copies the RAM-held experts to it), and `MNM_GPU=all` suggested while mnm uses one
card of several. Under WSL nvidia-smi reads the links as `[N/A]`: one line says so.

Which slot is the CPU's is not something Linux says; the width is. On AM5 the CPU gives the
graphics cards x16 or x8/x8 of PCIe 5.0 and the chipset gives x4 PCIe 4.0 slots that share one
uplink (TechSpot's AM5 board guide, read 2026-10-25), so the words say "most
likely". The planner's own ranking decides the main card; this only words it.
"""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit.cards import ranked
from makenomistakes.fit.devices import device_for
from makenomistakes.hardware.cards import CHIPSET_LANES, RISER_LANES, mix_problems
from makenomistakes.hardware.probe import Gpu, HardwareReport, chosen_cards
from makenomistakes.plan import budget_on

GIB = 2**30
MNM_GPU_REMEDY = (
    "Add `MNM_GPU=all` to ~/.config/makenomistakes/secrets.env (or the cards you want, "
    "`MNM_GPU=0,1`), then `mnm service install` so the router and the web UI use them, and "
    "`mnm replan --all` so the saved models are planned for every card."
)


@dataclass(frozen=True)
class Role:
    """One card's link and what its slot is good for."""

    card: Gpu
    link: str  # `x8 Gen5 now, x16 Gen5 at most`
    words: str  # `the CPU's lanes, most likely: the main card …`
    riser: bool = False
    main: bool = False


def lane_cards(report: HardwareReport) -> list[Gpu]:
    """The cards the lines speak of: the chosen ones when `MNM_GPU` names several, else every
    card nvidia-smi lists (what `MNM_GPU=all` would use)."""
    listed = report.gpus if isinstance(report.gpus, list) else []
    chosen = chosen_cards(report.gpus, report.chosen)
    return chosen if len(chosen) > 1 else listed


def link_text(gpu: Gpu) -> str | None:
    """`x16 Gen5`; `x8 Gen5 now, x16 Gen5 at most` when it runs narrower or slower than it
    can (a card idles at a lower generation); None when nvidia-smi read `[N/A]`."""
    if gpu.lanes is None:
        return None
    now = f"x{gpu.lanes}" + (f" Gen{gpu.pcie_gen}" if gpu.pcie_gen else "")
    top_gen = gpu.pcie_gen_max or gpu.pcie_gen
    top = f"x{gpu.lanes_max or gpu.lanes}" + (f" Gen{top_gen}" if top_gen else "")
    return now if top == now else f"{now} now, {top} at most"


def main_card(cards: list[Gpu]) -> Gpu:
    """The card the planner makes the main one (`-mg`; `fit.cards.ranked`'s first)."""
    first = ranked(list(budget_on(cards, 0).cards))[0]
    return next(c for c in cards if c.index == first.index)


def roles(cards: list[Gpu]) -> list[Role]:
    """Each card's link and role, in nvidia-smi's order (empty when no link was read)."""
    if len(cards) < 2 or all(c.lanes is None for c in cards):
        return []
    main = main_card(cards)
    narrow = [c.index for c in cards if c.lanes is not None and 1 < c.lanes <= CHIPSET_LANES]
    found = []
    for card in sorted(cards, key=lambda c: c.index):
        link = link_text(card) or "link unknown ([N/A])"
        if card.lanes is not None and card.lanes <= RISER_LANES:
            words = (
                "a riser, most likely: layers only, slow for prompt processing (mnm puts no "
                "experts there and places it last)"
            )
            found.append(Role(card, link, words, riser=True))
            continue
        if card is main:
            words = (
                "the CPU's lanes, most likely: the main card (it takes `-mg`, the largest "
                "buffers and prompt processing's copies of the experts kept in RAM)"
            )
        elif card.lanes is not None and card.lanes <= CHIPSET_LANES:
            others = [f"GPU {i}" for i in narrow if i != card.index]
            share = f"; it shares the chipset's uplink with {', '.join(others)}" if others else ""
            words = (
                f"on the chipset, most likely: fine for layers, slow for prompt processing{share}"
            )
        else:
            words = "the CPU's lanes, most likely: layers and experts at full speed"
        found.append(Role(card, link, words, main=card is main))
    return found


def lane_checks(report: HardwareReport, gpu_set: bool = True) -> list[Check]:
    """The lines; nothing on a box with one card. `gpu_set`: `MNM_GPU` is in the secrets file
    or the environment (unset on a box of several: the suggestion)."""
    cards = lane_cards(report)
    if len(cards) < 2:
        return []
    checks = [] if gpu_set else [_in_use(report)]
    if all(c.lanes is None for c in cards):
        where = "not readable under WSL" if report.is_wsl else "not read (nvidia-smi said [N/A])"
        checks.append(
            Check(
                name="PCIe links",
                status=Status.OK,
                detail=f"{where}: the card with the most free VRAM is the main card",
            )
        )
        return checks
    for role in roles(cards):
        check = Check(name=f"GPU {role.card.index} link", status=Status.OK, detail="")
        detail = f"{role.link}: {role.words}"
        if role.riser:
            check = check.model_copy(
                update={
                    "status": Status.WARN,
                    "remedy": "A riser with more lanes (x4, or an M.2-to-PCIe x4 adapter) would "
                    "let it hold experts too; or move the card to an x4 or wider slot.",
                }
            )
        checks.append(check.model_copy(update={"detail": detail}))
    misplaced = _misplaced(cards)
    if misplaced is not None:
        checks.append(misplaced)
    return checks


def strongest(cards: list[Gpu]) -> Gpu:
    """The card that should be the main one: the most VRAM, then the fastest memory."""
    return max(
        cards,
        key=lambda c: (c.vram_total_bytes, device_for(c.name).bandwidth, -c.index),
    )


def _misplaced(cards: list[Gpu]) -> Check | None:
    """A WARN when the largest card is not on the widest link (the main card is)."""
    best, main = strongest(cards), main_card(cards)
    if best is main or (
        best.vram_total_bytes == main.vram_total_bytes
        and device_for(best.name).bandwidth == device_for(main.name).bandwidth
    ):
        return None
    return Check(
        name="Main card",
        status=Status.WARN,
        detail=(
            f"GPU {best.index} ({best.name}, {best.vram_total_bytes / GIB:.0f} GiB) is the "
            f"largest card but sits on {link_text(best) or 'an unread link'}; GPU {main.index} "
            f"({main.name}) has the widest link ({link_text(main)}), so it is the main card: "
            "the largest buffers and prompt processing's copies of the RAM-held experts go to "
            "the smaller card"
        ),
        remedy=(
            f"Swap them: put the {best.name} in the slot GPU {main.index} sits in "
            f"({link_text(main)}: the CPU's lanes, on most boards the full-length slot nearest "
            f"the CPU) and the {main.name} where the {best.name} is now. mnm follows the links "
            "by itself; `mnm replan --all` afterwards."
        ),
    )


def _in_use(report: HardwareReport) -> Check:
    listed = report.gpus if isinstance(report.gpus, list) else []
    total = sum(c.vram_total_bytes for c in listed) / GIB
    first = chosen_cards(report.gpus, report.chosen)
    which = f"card {first[0].index}" if first else "one card"
    problems = mix_problems(listed)
    odd = f". Leave the odd card out of the list: {problems[0]}" if problems else ""
    makers = {g.vendor for g in listed}
    # AMD's cards are read from sysfs
    who = "mnm reads" if any(g.is_amd for g in listed) else "nvidia-smi lists"
    remedy = MNM_GPU_REMEDY
    if len(makers) > 1:  # One maker's cards per instance, so never `all`
        remedy = (
            "Set MNM_GPU to one maker's cards in ~/.config/makenomistakes/secrets.env (an "
            "instance never spans NVIDIA's and AMD's images), then `mnm service install` and "
            "`mnm replan --all`; pin instances to the others with `mnm set <name> gpus=`."
        )
    return Check(
        name="Cards in use",
        status=Status.WARN,
        detail=(
            f"{who} {len(listed)} cards ({total:.0f} GiB together); mnm plans for "
            f"and runs on {which} only: MNM_GPU is not set{odd}"
        ),
        remedy=remedy,
    )
