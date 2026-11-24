"""What a re-plan says before anything is saved: `mnm replan`'s lines, and `mnm why`'s under
*Now*, which shares them (a copy drifts: without the files a start downloads or what gets
smaller, or naming `--shrink` twice beside a withheld plan). The web's re-plan page and
`mnm machine adopt` still word a re-plan their own way."""

from __future__ import annotations

from makenomistakes.backup_auto import Said, Tone
from makenomistakes.replan import Replanned


def replan_lines(done: Replanned, name: str, header: str) -> Said:
    """The withheld smaller plan (alone when nothing else moves), the tune kept or dropped,
    then `header` and each change, the files the next start downloads and what gets smaller;
    "planned as it is" when nothing moves."""
    said: Said = []
    if done.withheld is not None:
        said.append((Tone.WARN, done.withheld.words(name)))
        if not done.changed:
            return said
    if done.tune_note is not None:  # A tuned recipe's tune, kept or dropped
        said.append((Tone.PLAIN, done.tune_note))
    if not done.changed:
        said.append((Tone.PLAIN, f"{name} is planned as it is: nothing to change."))
        return said
    said.append((Tone.PLAIN, header))
    said += [(Tone.PLAIN, f"  {change}") for change in done.changes]
    if done.new_files:
        said.append((Tone.PLAIN, f"The next start downloads: {', '.join(done.new_files)}."))
    if done.shrinks:
        said.append((Tone.WARN, f"{name} gets smaller: {', '.join(done.shrinks)}."))
    return said
