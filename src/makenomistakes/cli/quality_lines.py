"""What `mnm tune --quality` measured on this box, as `mnm fit`, `mnm add`,
`mnm starter` and `mnm what` print it beside the planner's estimates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rich.markup import escape

from makenomistakes.cli._app import console
from makenomistakes.quality.readings import QuantScore


def print_measured(measured: Mapping[str, QuantScore], quants: Iterable[str] = ()) -> None:
    """One line a measured quant of the repo, the plan's own quants first."""
    if not measured:
        return
    order = dict.fromkeys([q for q in quants if q in measured] + sorted(measured))
    console.print("Measured on this box (`mnm tune <name> --quality`):")
    for quant in order:
        console.print(f"  {escape(quant)}: {escape(measured[quant].words)}")
