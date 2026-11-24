"""One read of the run records (`RunsRead`) and the prune's size; `runs` re-exports both."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from makenomistakes.run_errors import RunError

if TYPE_CHECKING:
    from makenomistakes.runs import RunRecord


def size_limit(gib: str | float) -> int:
    """`--max-gb` / the Prune form's size as bytes; anything but a finite size of 0 or more
    is a worded refusal (`inf` and `1e400` would overflow into a traceback)."""
    refused = RunError(f"'{gib}' is not a size in GiB (a number, 0 or more).")
    try:
        value = float(gib)
    except ValueError:
        raise refused from None
    if not math.isfinite(value) or value < 0:  # inf and 1e400 (float makes it inf) included
        raise refused
    return int(value * 2**30)


@dataclass(frozen=True)
class RunsRead:
    """One read of the runs: every readable run, newest first; the files this version cannot
    read, each said whole with what clears it (`problems`) and by name (`unread`); and, when
    the folder itself could not be listed, why (then no records at all). What keeps the
    files a run uses (`mnm clean comfy`, `mnm clean imports`, `mnm comfyui workflow rm`)
    reads `unsure` as "any run may use anything", never as no run in use (else ComfyUI's
    folders could be cleared under a going run, or a waiting film's workflow removed).
    `run_lineage.kept` keeps the runs under an `unread` record."""

    records: list[RunRecord]
    problems: list[str]
    unlisted: str = ""
    unread: tuple[str, ...] = ()  # the names of the records that could not be read

    @property
    def unsure(self) -> str:
        """Why what the runs use cannot be known now, "" when it can: the folder could not be
        listed, or a record could not be read (a run going may be that one: a record a
        `sudo mnm` wrote, an I/O error, a newer mnm's). Every keep-check reads this, never
        `records` alone. Each record's own words, with what clears
        it, are `mnm outputs ls`'s and the gallery's: said here once, never nested."""
        if self.unlisted:
            return self.unlisted
        if not self.unread:
            return ""
        if len(self.unread) == 1:
            return (
                f"the run record {self.unread[0]} could not be read "
                "(`mnm outputs ls` says why and what clears it)"
            )
        return (
            f"{len(self.unread)} run records could not be read "
            "(`mnm outputs ls` says why and what clears each)"
        )
