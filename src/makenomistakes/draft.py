"""A draft model for speculative decoding: `mnm set <name> draft=<instance>`.

A small model of the same vocabulary drafts a few tokens; the instance's own model checks
them in one pass and keeps the ones it agrees with, so a reply comes faster with the same
words. The draft is another saved llama.cpp instance (chat, coding or vision), downloaded;
its main file is kept on the recipe (`DraftModel`), so the draft runs whatever becomes of
that instance later. llama.cpp refuses to start with a draft whose vocabulary differs, so the
same rule (`gguf.vocab.incompatible`) refuses it here first, with the reason.

`draft=mtp` drafts with the instance's own multi-token-prediction head instead:
the file its repo ships beside the quant (`Recipe.mtp`, downloaded with the model), run as
`--spec-type draft-mtp`. It needs no second instance and shares the model's vocabulary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from makenomistakes.engine_args import DRAFT_MAX, DRAFT_P_MIN
from makenomistakes.gguf.reader import FileSource, GgufError
from makenomistakes.gguf.vocab import Vocab, incompatible, read_vocab
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.hub.gguf_files import head_quant
from makenomistakes.measurements import Reading, SpeedUnit, settings_key
from makenomistakes.recipes.model import DraftModel, Recipe
from makenomistakes.try_figures import Figure

DRAFT_KEY = "draft"  # the setting's key (`mnm set <name> draft=…`)
MTP = "mtp"  # `draft=mtp`: the instance's own MTP head, never an instance of that name
_CLEAR = frozenset({"", "none", "off"})
EXPERIMENT = (
    "A draft is an experiment: the planner's speed is the model's alone, and the next "
    "`mnm try {name} --speed` measures what the draft gains (its accepted share included)."
)


class DraftError(ValueError):
    """A draft that cannot run beside this instance, and why."""


def with_draft(
    recipe: Recipe, raw: str, find: Callable[[str], Recipe | None], models_dir: Path
) -> Recipe:
    """The recipe with `raw` as its draft: an instance name, or empty (none, off) to run it
    alone. The same recipe when nothing changes (the web form posts every field); a new
    draft forgets the planned estimate, which priced the model alone."""
    wanted = raw.strip().lower()
    current = draft_name(recipe.draft)
    if wanted in _CLEAR:
        return recipe if recipe.draft is None else _saved(recipe, None)
    if wanted == current:
        return recipe
    if wanted == MTP:
        return _saved(recipe, mtp_draft(recipe, models_dir))
    draft = find(wanted)
    if draft is None:
        raise DraftError(f"No instance is named {wanted!r}: `mnm ls` lists them.")
    return _saved(recipe, choose_draft(recipe, draft, models_dir))


def draft_name(draft: DraftModel | None) -> str:
    """The setting's value for a draft: the instance it came from, `mtp`, or empty."""
    if draft is None:
        return ""
    return draft.setting


def mtp_draft(recipe: Recipe, models_dir: Path) -> DraftModel:
    """The instance's own MTP head as its draft, or DraftError naming what is missing."""
    if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
        raise DraftError(
            f"{recipe.name} is {recipe.modality.a_label} model on {recipe.engine.value}: only "
            "llama.cpp chat, coding and vision instances draft with an MTP head."
        )
    if recipe.mtp is None:
        raise DraftError(
            f"{recipe.name} has no MTP head: {recipe.repo_id} ships none beside its GGUF files "
            f"(`mnm resolve {recipe.repo_id}` lists the heads it has), or {recipe.name} was "
            f"added before mnm kept them (`mnm replan {recipe.name}` fetches it)."
        )
    rel = recipe.local_paths.get(recipe.mtp)
    if rel is None:
        raise DraftError(
            f"{recipe.name}'s MTP head is not downloaded yet: `mnm start {recipe.name}` fetches it."
        )
    try:
        size = (models_dir / rel).stat().st_size
    except OSError as exc:
        raise DraftError(f"{recipe.name}'s MTP head cannot be read: {exc}") from exc
    return DraftModel(
        kind="mtp",
        instance=recipe.name,
        repo_id=recipe.repo_id,
        file=recipe.mtp,
        path=rel,
        quant=head_quant(recipe.mtp),
        size_bytes=size,
    )


def drafting_with(recipes: list[Recipe], name: str) -> str | None:
    """What `mnm rm <name>` says when other instances draft with its file, or
    None: the file stays (`mnm rm --files` keeps it) while one lists it. An MTP
    head is the instance's own file and goes with it."""
    users = sorted(
        r.name
        for r in recipes
        if r.draft is not None and not r.draft.is_mtp and r.draft.instance == name
    )
    if not users:
        return None
    who = ", ".join(users)
    return (
        f"{who} {'drafts' if len(users) == 1 else 'draft'} with {name}'s file; the file stays "
        f"while {who} {'lists' if len(users) == 1 else 'list'} it (`mnm set {users[0]} draft=` "
        "runs it alone)."
    )


def _saved(recipe: Recipe, draft: DraftModel | None) -> Recipe:
    """A draft attached or dropped: its per-step settings start again at the defaults
    (llama.cpp's 3 tokens, p-min 0.75: a recipe saved with schema 12's 16 drafts at 3 once
    it pairs again); a `draft-max=` or `draft-p-min=` in the same command is applied after."""
    args = recipe.llama_args.model_copy(update={"draft_max": DRAFT_MAX, "draft_p_min": DRAFT_P_MIN})
    return recipe.model_copy(update={"draft": draft, "args": args, "vram_estimate_bytes": None})


def choose_draft(target: Recipe, draft: Recipe, models_dir: Path) -> DraftModel:
    """`draft`'s main file as a draft for `target`, or DraftError naming what is wrong."""
    if draft.name == target.name:
        raise DraftError(
            f"{target.name} cannot draft for itself: name a smaller model of the same family "
            "(Qwen3-0.6B for a Qwen3 model), added with `mnm add`."
        )
    for recipe, role in ((target, "have a draft"), (draft, "draft")):
        if recipe.engine is not Engine.LLAMACPP or recipe.modality not in CHAT_LIKE:
            raise DraftError(
                f"{recipe.name} is {recipe.modality.a_label} model on {recipe.engine.value}: "
                f"only llama.cpp chat, coding and vision instances {role} (llama-server's "
                "speculative decoding)."
            )
    if not draft.downloaded:
        raise DraftError(
            f"{draft.name} is not downloaded yet: `mnm start {draft.name}` fetches it."
        )
    target_vocab = _vocab(target, models_dir)
    draft_vocab = _vocab(draft, models_dir)
    reason = incompatible(target_vocab, draft_vocab)
    if reason is not None:
        raise DraftError(
            f"{draft.name} cannot draft for {target.name}: {reason}. llama.cpp drafts only with "
            "a model of the same vocabulary: one of the same family, usually its smallest size."
        )
    size, own = _size(draft, models_dir), _size(target, models_dir)
    if size >= own:  # `mnm set tiny draft=main` is refused here
        raise DraftError(
            f"{draft.name} ({size / 2**30:.2f} GiB) is not smaller than {target.name} "
            f"({own / 2**30:.2f} GiB): a draft saves time only when it runs at a fraction of the "
            "model's cost. Name a smaller model of the same family."
        )
    return DraftModel(
        instance=draft.name,
        repo_id=draft.repo_id,
        file=draft.files[0],
        path=draft.local_paths[draft.files[0]],
        quant=draft.quant,
        size_bytes=size,
    )


def _size(recipe: Recipe, models_dir: Path) -> int:
    """Its model files on disk (every shard)."""
    try:
        return sum((models_dir / recipe.local_paths[f]).stat().st_size for f in recipe.files)
    except (OSError, KeyError) as exc:
        raise DraftError(f"{recipe.name}'s model files cannot be read: {exc}") from exc


def _vocab(recipe: Recipe, models_dir: Path) -> Vocab:
    rel = recipe.local_paths.get(recipe.files[0])
    if rel is None:
        raise DraftError(f"{recipe.name} is not downloaded yet: `mnm start {recipe.name}`.")
    try:
        return read_vocab(FileSource(models_dir / rel))
    except (GgufError, OSError) as exc:
        raise DraftError(f"{recipe.name}'s model file cannot be read as a GGUF: {exc}") from exc


# Below this share of drafted tokens kept, checking the draft costs more than it saves: at
# 8% kept, 16 a step, qwen3-8b fell from 75.6 to 27.1 tok/s on an RTX 4070.
LOSING_SHARE = 30.0


def alone_speed(readings: list[Reading], recipe: Recipe) -> float | None:
    """The instance's last tok/s reading without a draft (the same settings otherwise)."""
    key = settings_key(recipe.model_copy(update={"draft": None}))
    found = [
        r for r in readings if r.settings == key and r.speed and r.speed_unit is SpeedUnit.TOKENS
    ]
    return max(found, key=lambda r: r.at).speed if found else None


def verdict(recipe: Recipe, numbers: Mapping[Figure, float], alone: float | None) -> str | None:
    """One line after `mnm try --speed` with a draft: whether it gains or loses here, and the
    setting that removes it when it loses. None without a draft or its counts."""
    if recipe.draft is None:
        return None
    speed = numbers.get(Figure.PREDICTED_PER_SECOND)
    share = numbers.get(Figure.DRAFT_ACCEPTED_PERCENT)
    if speed is None or share is None:
        return None
    against = f" against {alone:.0f} alone" if alone else ""
    removes = f"`mnm set {recipe.name} draft=` removes it"
    if (alone and speed < alone) or share < LOSING_SHARE:
        why = f"{speed:.0f} tok/s{against}, {share:.0f}% of its tokens kept"
        return f"The draft loses here ({why}): {removes}."
    if alone:
        return f"The draft gains here ({speed:.0f} tok/s{against}, {share:.0f}% kept)."
    return (
        f"{share:.0f}% of the draft's tokens kept; `mnm set {recipe.name} draft=` then "
        f"`mnm try {recipe.name} --speed` gives the speed alone to compare."
    )
