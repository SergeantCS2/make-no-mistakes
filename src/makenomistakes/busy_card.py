"""VRAM held outside mnm while a model is planned: a plan is sized to what is free when it
is made, and the recipe keeps it. Beside another program holding 8 GiB, every starter is
planned around it (qwen3-8b with 7 of 36 layers on the GPU).
One wording here for every place that says so: the plan table, the web Add and Get started
pages, a refusal, a re-plan and the guided setup."""

from __future__ import annotations

from makenomistakes.fit import Budget
from makenomistakes.hardware import platform as platforms

GIB = 1024**3


def held_elsewhere(budget: Budget) -> str | None:
    """`8.2 GiB of the card is in use by Windows and other apps`, or None when no more than a
    desktop's share is. With mnm's own instances running, what is left over is only what
    their estimates do not cover. Never what to do about it: what holds the card is the
    user's business; mnm says how much is in use and plans around it."""
    used = budget.busy_bytes
    if used is None:
        return None
    who = (
        "beyond what mnm's running instances are estimated to hold"
        if budget.held_by
        else f"by {platforms.current().others}"
    )
    if budget.pool:  # The pool's pages, the system's margin counted out of it
        return (
            f"{used / GIB:.1f} GiB of the unified memory is in use {who} "
            f"({budget.margin / GIB:.0f} GiB of it kept for the system anyway)"
        )
    return f"{used / GIB:.1f} GiB of the card is in use {who}"


def plan_note(budget: Budget) -> str | None:
    """Under a plan that a busy card shaped: what it means for the saved instance."""
    held = held_elsewhere(budget)
    if held is None:
        return None
    return (
        f"{held}, so this plan is sized around it. An instance added now keeps these settings "
        "until `mnm replan <name>` (or `mnm replan --all`; `--any-quant` to choose the quant "
        f"again too) when more of {_where(budget)} is free."
    )


def replan_note(budget: Budget) -> str | None:
    """Before a re-plan on a busy card: it would be sized around it too."""
    held = held_elsewhere(budget)
    if held is None:
        return None
    return f"{held}: a re-plan now is sized around it too."


def refusal_note(budget: Budget) -> str | None:
    """Inside a refusal: the card is busy, so the model may fit once it is free."""
    held = held_elsewhere(budget)
    if held is None:
        return None
    return f"{held}; it may fit when more of {_where(budget)} is free"


def _where(budget: Budget) -> str:
    return "the unified memory" if budget.pool else "the card"
