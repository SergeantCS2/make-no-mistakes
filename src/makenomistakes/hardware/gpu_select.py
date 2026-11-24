"""`MNM_GPU` as a list of cards: one index (the default 0), several (`0,1`) or `all`; and
what Docker is told for each engine.

One card renders `--gpus device=N` and nothing else. Several are
`--gpus '"device=0,1"'` (the argv element carries the double quotes: Docker splits the
flag's value at commas otherwise; NVIDIA's container-toolkit docs, docker-specialized.html)
with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so that index N inside the container is the Nth card in
nvidia-smi's order and `-ts`/`-mg` count the cards the way the plan did (CUDA's default
order is fastest first; the CUDA programming guide, environment variables appendix). Inside
the container the exposed cards are numbered from 0 in that order.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

MAX_CARD = 15
ALL = "all"
SPEC_ERROR = (
    "MNM_GPU must be a card number from 0 to 15 (nvidia-smi's index), several separated by "
    "commas (0,1), or all; fix it in the secrets file or the environment"
)


class CardChoice(BaseModel):
    """The cards mnm plans for and runs on, in the order `MNM_GPU` names them; `every`
    for `all` (resolved against what nvidia-smi lists)."""

    indices: tuple[int, ...] = (0,)
    every: bool = False

    @property
    def first(self) -> int:
        """The card a one-card engine (sd.cpp, whisper, audio.cpp, ComfyUI) gets."""
        return self.indices[0] if self.indices else 0

    @property
    def several(self) -> bool:
        return self.every or len(self.indices) > 1

    def resolve(self, listed: list[int]) -> tuple[int, ...]:
        """The chosen indices among the cards nvidia-smi lists (`all`: every one of them)."""
        return tuple(listed) if self.every else self.indices


def parse_choice(text: str) -> CardChoice:
    """`0`, `1`, `0,1`, `all`; anything else a ValueError (never card 0 by mistake)."""
    value = text.strip().lower()
    if value == ALL:
        return CardChoice(indices=(), every=True)
    parts = [p.strip() for p in value.split(",")]
    if not parts or not all(p.isdigit() for p in parts):
        raise ValueError(SPEC_ERROR)
    indices = tuple(int(p) for p in parts)
    if any(i > MAX_CARD for i in indices) or len(set(indices)) != len(indices):
        raise ValueError(SPEC_ERROR)
    return CardChoice(indices=indices)


def chosen() -> CardChoice:
    """`MNM_GPU`, read at the call (like the tokens). A value that is not a card list is a
    ValueError naming it; settings broken for another reason say so wherever they are read,
    so the card is 0 here."""
    from makenomistakes.config import settings_now  # noqa: PLC0415 - read at the call

    try:
        return parse_choice(settings_now().gpu)
    except ValidationError as exc:
        if any(error["loc"][:1] == ("gpu",) for error in exc.errors()):
            raise ValueError(SPEC_ERROR) from exc
        return CardChoice()


def device_spec(cards: tuple[int, ...] | None) -> str:
    """`docker run --gpus`'s value: `device=N` for one card,
    `"device=0,1"` for several (in nvidia-smi's order), `all` for `MNM_GPU=all` (None)."""
    if cards is None:
        return ALL
    ordered = sorted(cards)
    if len(ordered) == 1:
        return f"device={ordered[0]}"
    return '"device=' + ",".join(str(c) for c in ordered) + '"'


def device_env(cards: tuple[int, ...] | None) -> list[str]:
    """The `--env` pair several cards need (none for one: its order cannot differ)."""
    if cards is not None and len(cards) == 1:
        return []
    return ["--env", "CUDA_DEVICE_ORDER=PCI_BUS_ID"]
