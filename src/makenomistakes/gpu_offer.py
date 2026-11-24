"""`MNM_GPU` after a card change: the setting names cards by nvidia-smi index, and an
index may name another card once a card is swapped, added, removed or moved. This says what
the setting selects now, what it selected on the box the plans were made on, and the value
that keeps the same cards (by their uuids) or `all`. It is printed, never written: the
secrets file is the user's (`mnm machine adopt`, the doctor's *Machine* line).

Pure: the caller passes the setting as saved (None when unset: card 0) and both boxes."""

from __future__ import annotations

from dataclasses import dataclass, field

from makenomistakes.hardware.fingerprint import CardId, MachineFingerprint
from makenomistakes.hardware.gpu_select import ALL, CardChoice, parse_choice

# The one place every service reads it (the units do not carry it); a restart picks it
# up. Adopt and the doctor only offer the value.
WHERE = (
    "set it in the secrets file (~/.config/makenomistakes/secrets.env), then `mnm service "
    "install` so the services start on it; adopt never writes it"
)


@dataclass(frozen=True)
class GpuNote:
    selects: str  # `MNM_GPU=0,1 selects RTX 4070 (GPU0) and RTX 3060 (GPU1)`
    changed: list[str] = field(default_factory=list)  # `MNM_GPU=1 now selects …, was …`
    offer: str | None = None  # `MNM_GPU=1,0 keeps the same cards: set it …`

    @property
    def lines(self) -> list[str]:
        return [self.selects, *self.changed, *([self.offer] if self.offer else [])]


def worth_saying(raw: str | None, old: MachineFingerprint | None, new: MachineFingerprint) -> bool:
    """Whether the per-index change is news: the user set `MNM_GPU`, or either box has
    several cards (one card swapped with the setting unset is the cards' change, said)."""
    return raw is not None or len((old or new).cards) > 1 or len(new.cards) > 1


def _named(raw: str | None) -> str:
    return f"MNM_GPU={raw}" if raw is not None else "MNM_GPU (unset: card 0)"


def _at(cards: list[CardId], index: int) -> CardId | None:
    return next((c for c in cards if c.index == index), None)


def _card(card: CardId | None) -> str:
    return card.short if card is not None else "no card"


def gpu_note(raw: str | None, old: MachineFingerprint | None, new: MachineFingerprint) -> GpuNote:
    """What `MNM_GPU` selects on `new`, and against `old` (the box the plans were made on)
    what changed at each index it names and the value that keeps the same cards."""
    named = _named(raw)
    try:
        choice = parse_choice(raw if raw is not None else "0")
    except ValueError as exc:
        return GpuNote(selects=f"{named}: {exc}")
    selects = _selects(named, choice, new)
    if old is None or choice.every:
        return GpuNote(selects=selects)
    changed = []
    for index in choice.indices if worth_saying(raw, old, new) else ():
        was, now = _at(old.cards, index), _at(new.cards, index)
        if _id(was) == _id(now):
            continue
        if len(choice.indices) == 1:
            changed.append(f"{named} now selects {_card(now)}, was {_card(was)}")
        else:
            changed.append(f"{named}: card {index} is now {_card(now)}, was {_card(was)}")
    return GpuNote(selects=selects, changed=changed, offer=_offer(raw, choice, old, new))


def _id(card: CardId | None) -> str | None:
    return card.stable_id if card is not None else None


def _selects(named: str, choice: CardChoice, box: MachineFingerprint) -> str:
    if not box.cards:
        return f"{named}: this box has no card"  # of any maker
    indices = [c.index for c in box.cards] if choice.every else list(choice.indices)
    words = [f"{_card(_at(box.cards, i))} (GPU{i})" for i in indices]
    return f"{named} selects {' and '.join(words) or 'no card'}"


def _offer(
    raw: str | None, choice: CardChoice, old: MachineFingerprint, new: MachineFingerprint
) -> str | None:
    """The value to set when cards came, went or moved under the setting; None when the
    setting still selects what it did and no card is new."""
    after = {c.stable_id: c.index for c in new.cards}
    came = {c.stable_id for c in new.cards} - {c.stable_id for c in old.cards}
    chosen = [_at(old.cards, i) for i in choice.indices]
    current = raw if raw is not None else "0"
    if chosen and all(c is not None and c.stable_id in after for c in chosen):
        value = ",".join(str(after[c.stable_id]) for c in chosen if c is not None)
        if value != current:
            names = ", ".join(_card(c) for c in chosen)
            return f"MNM_GPU={value} keeps the same cards ({names}): {WHERE}"
        if not came:
            return None
    if len(new.cards) > 1:
        return f"MNM_GPU={ALL} uses every card this box has: {WHERE}"
    if new.cards and current != str(new.cards[0].index):
        return f"MNM_GPU={new.cards[0].index}, the one card now: {WHERE}"
    return None
