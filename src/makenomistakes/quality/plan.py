"""What a quality tune measures: the reference and the candidates, chosen from
the quants the planner runs on this box, as the user approved them before anything was
downloaded. The plan travels with the queued tune (`tune.store.Queued.quality`); the router
measures exactly these, never downloads, and refuses a plan whose instance moved since.

The choice of candidates is pure (`pick_candidates`, `pick_reference`) over what the planner
said of each quant; `quality.prepare` asks the planner and the disk."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from makenomistakes.fit.quant_ladder import quant_bits
from makenomistakes.measurements import settings_key
from makenomistakes.quality import FLOOR_MAX, FLOOR_MIN
from makenomistakes.quality.texts import TextName
from makenomistakes.recipes import Recipe
from makenomistakes.recipes.model import load_recipe_data

MAX_CANDIDATES = 4  # the reference counts when it also serves; `--quants` names its own
BELOW_FLOOR_TIERS = 2  # tiers looked at below the current quant when it reads under the floor
REFERENCE = "Q8_0"  # scored against Q8_0 where it runs
CHUNKS = 24  # 512-token chunks the base run reads (6,120 scored tokens)
MEASURE_CTX = 8192  # a reference that does not serve is planned for this many tokens

Use = Literal["chat", "agents"]


# What a candidate keeps of the instance as it is: the KV cache (its precision is in the
# score), flash attention and the batches; only the quant and where it is placed move.
HELD = frozenset({"kv_type", "flash_attn", "batch", "ubatch"})


def _migrated(value: object) -> object:
    """A recipe read back from the tuner's queue or the downloads ledger goes through the
    recipe files' migrations, as a saved recipe does (a plan queued before an `mnm update`)."""
    return load_recipe_data(value) if isinstance(value, dict) else value


SavedRecipe = Annotated[Recipe, BeforeValidator(_migrated)]


class QualityCandidate(BaseModel):
    quant: str
    # The current quant: the saved recipe itself. Another: its serving plan with the user's
    # settings kept (`replan` pinned to the quant, `HELD` kept; a reference that does not serve:
    # its plan for `MEASURE_CTX`), its files' local paths once downloaded: the router runs it
    # as it is, never asking the Hub.
    recipe: SavedRecipe
    size_bytes: int | None = None
    serves: bool = True  # False: a reference only measured (it does not serve on this box)


class QualityPlan(BaseModel):
    """The approved plan (schema 1)."""

    schema_version: Literal[1] = 1
    repo_id: str
    current: str  # the instance's quant when planned
    settings: str  # `measurements.settings_key` of the recipe planned from
    reference: QualityCandidate
    candidates: list[QualityCandidate]  # the most faithful first, the current among them
    floor: float = Field(ge=FLOOR_MIN, le=FLOOR_MAX)  # decode tok/s
    use: Use
    text: TextName
    chunks: int = Field(default=CHUNKS, ge=4, le=200)
    keep: bool = False  # `--keep`: the quants not chosen stay on disk

    @property
    def instance(self) -> str:
        """The instance it was planned for (every candidate's recipe carries its name)."""
        return self.reference.recipe.name

    def candidate(self, quant: str) -> QualityCandidate | None:
        return next((c for c in self.candidates if c.quant == quant), None)

    def candidate_or_reference(self, quant: str) -> QualityCandidate | None:
        """The row the tune measures `quant` with (a reference only measured among them)."""
        return next((c for c in self.measured if c.quant == quant), None)

    @property
    def measured(self) -> list[QualityCandidate]:
        """Every quant the tune runs: the reference first when it is not a candidate."""
        among = self.candidate(self.reference.quant) is not None
        return self.candidates if among else [self.reference, *self.candidates]

    @property
    def fingerprint(self) -> str:
        """What the user approved, short: the web's page posts it back, and a plan that moved
        between the page and the click (the card, the floor, the instance) is refused."""
        quants = ",".join(f"{c.quant}:{settings_key(c.recipe)}" for c in self.measured)
        said = "|".join(
            (self.instance, self.repo_id, self.settings, self.reference.quant, quants,
             f"{self.floor:g}", self.use, self.text.value, str(self.chunks), str(self.keep))
        )  # fmt: skip
        return hashlib.sha256(said.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class QuantFact:
    """What the planner said of one quant on this box."""

    quant: str
    size_bytes: int  # the variant's files
    serves: bool  # it plans at the instance's own context and type
    measures: bool  # it plans at the reference's measuring context (any fit, offload included)
    decode: float | None = None  # the planner's tok/s at the serving plan (None: not planned)

    @property
    def bits(self) -> float:
        return quant_bits(self.quant)

    @property
    def faithful(self) -> tuple[float, int]:
        """The ladder's own order (`quant_ladder.below`): bits, then size."""
        return (self.bits, self.size_bytes)


def pick_reference(facts: Sequence[QuantFact], current: str | None = None) -> QuantFact | None:
    """Q8_0 when the box can measure it, else the most faithful quant it can (never a 16-bit
    one to download: the reference is Q8_0 by design, and a larger download for a yardstick buys
    little). An instance already on a more faithful quant (BF16, unsloth's UD-Q8_K_XL) is its
    own reference: it runs, it is on disk, and nothing it could move to is above it."""
    runnable = [f for f in facts if f.measures and f.bits <= quant_bits(REFERENCE)]
    named = next((f for f in runnable if f.quant.upper() == REFERENCE), None)
    picked = named or max(runnable, key=lambda f: f.faithful, default=None)
    now = next((f for f in facts if f.quant == current), None)
    if now is not None and (picked is None or now.faithful > picked.faithful):
        return now
    return picked


def _family(quant: str) -> str:
    """`UD-` for unsloth's dynamic quants, `I` for the codebook ones, else plain."""
    upper = quant.upper()
    return "UD-" if upper.startswith("UD-") else "I" if upper.startswith("IQ") else ""


def _tier_pick(tier: list[QuantFact], current: str) -> QuantFact:
    """One quant of a bit tier: the current's family first, then `_K_M`, then the most
    faithful (the tier's quants differ little in bits, a family keeps one maker's recipe)."""
    family = _family(current)
    same = [f for f in tier if _family(f.quant) == family]
    medium = [f for f in same or tier if f.quant.upper().endswith(("_K_M", "_K_XL", "_0"))]
    return max(medium or same or tier, key=lambda f: f.faithful)


def pick_candidates(
    facts: Sequence[QuantFact],
    current: str,
    reference: QuantFact,
    floor: float,
    limit: int = MAX_CANDIDATES,
) -> list[QuantFact]:
    """The quants a tune measures, the most faithful first: one a bit tier from the
    reference's down to the current's (the current itself standing for its tier), then, when
    the current reads under the floor on the planner's estimate, tiers below it until one
    reads over; at most `limit`, the top and the current always kept."""
    serving = [f for f in facts if f.serves and f.faithful <= reference.faithful]
    now = next((f for f in serving if f.quant == current), None)
    tiers: dict[int, list[QuantFact]] = {}
    for fact in serving:
        tiers.setdefault(int(fact.bits), []).append(fact)
    above = [
        now if now is not None and int(now.bits) == tier else _tier_pick(found, current)
        for tier, found in sorted(tiers.items(), reverse=True)
        if now is None or tier >= int(now.bits)
    ]
    picked = [f for f in above if f.quant != current] + ([now] if now is not None else [])
    picked.sort(key=lambda f: f.faithful, reverse=True)
    if now is not None and (now.decode or 0.0) < floor:
        below = sorted(
            (found for tier, found in tiers.items() if tier < int(now.bits)),
            key=lambda found: -max(f.bits for f in found),
        )
        for found in below[:BELOW_FLOOR_TIERS]:
            pick = _tier_pick(found, current)
            picked.append(pick)
            if (pick.decode or 0.0) >= floor:
                break
    return _spread(picked, now, limit)


def _spread(picked: list[QuantFact], now: QuantFact | None, limit: int) -> list[QuantFact]:
    """At most `limit`: the most faithful, the current and whatever reads over the floor below
    it kept, the tiers between thinned evenly."""
    if len(picked) <= limit:
        return picked
    keep = {0, len(picked) - 1}
    if now is not None:
        keep.add(picked.index(now))
    rest = [i for i in range(len(picked)) if i not in keep]
    room = max(limit - len(keep), 0)
    if room:
        step = len(rest) / room
        keep |= {rest[int(i * step)] for i in range(room)}
    return [picked[i] for i in sorted(keep)][:limit]


def named_candidates(
    facts: Sequence[QuantFact], wanted: Sequence[str], current: str, reference: QuantFact
) -> list[QuantFact]:
    """`--quants Q6_K,Q5_K_M`: those, and the current, each one the box serves; ValueError
    naming a quant it does not (or one more faithful than the reference)."""
    by_name = {f.quant.upper(): f for f in facts}
    chosen: list[QuantFact] = []
    for name in [*wanted, current]:
        fact = by_name.get(name.upper())
        if fact is None:
            raise ValueError(f"{name}: not a quant of this repo")
        if not fact.serves:
            raise ValueError(f"{name}: does not run on this box at the instance's context")
        if fact.faithful > reference.faithful:
            raise ValueError(
                f"{name}: more faithful than the reference ({reference.quant}), which scores it"
            )
        if fact not in chosen:
            chosen.append(fact)
    return sorted(chosen, key=lambda f: f.faithful, reverse=True)
