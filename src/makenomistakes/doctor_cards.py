"""The doctor's lines about the cards `MNM_GPU` names: each card with its compute
capability (and its PCIe link, worded by `doctor_lanes`) when there are several, a mix
the user should not run, and a card too old for some engine images (`hardware.cards`).
One card of Turing or newer adds no line. Each card's row in the device table says how
it is priced in one word (measured, family, unknown); one card's is on the NVIDIA GPU line."""

from __future__ import annotations

from makenomistakes.doctor_types import Check, Status
from makenomistakes.fit.devices import device_for
from makenomistakes.hardware.cards import (
    TURING,
    capability,
    card_line,
    equal,
    mix_problems,
)
from makenomistakes.hardware.cards import refusals as card_refusals
from makenomistakes.hardware.probe import HardwareReport, chosen_cards


def card_checks(report: HardwareReport) -> list[Check]:
    if not isinstance(report.gpus, list) or not report.gpus:
        return []
    cards = chosen_cards(report.gpus, report.chosen)
    checks: list[Check] = []
    if len(cards) > 1:
        detail = "; ".join(f"{card_line(c)}, priced: {device_for(c).source_word}" for c in cards)
        if not equal(cards):
            detail += (
                ". Unequal cards: llama.cpp splits layers by each card's free VRAM; vLLM "
                "runs on one card (its tensor parallel needs equal ones)"
            )
        checks.append(Check(name="Cards", status=Status.OK, detail=detail))
        problems = mix_problems(cards)
        if problems:
            checks.append(
                Check(
                    name="Card mix",
                    status=Status.WARN,
                    detail="; ".join(problems),
                    remedy="Leave the odd card out of `MNM_GPU` (it lists the cards mnm "
                    "uses), or pin instances to the cards that suit them "
                    "(`mnm set <name> gpus=0`).",
                )
            )
    for card in cards:
        cc = capability(card)
        if cc is None or cc >= TURING:
            continue
        checks.append(
            Check(
                name=f"Card {card.index}",
                status=Status.WARN,
                detail=f"{card_line(card)}: " + "; ".join(card_refusals(card)),
                remedy="Run llama.cpp and sd.cpp instances on it; the others on a Turing or "
                "newer card. NVIDIA's 580 driver branch is the last that supports it.",
            )
        )
    return checks
