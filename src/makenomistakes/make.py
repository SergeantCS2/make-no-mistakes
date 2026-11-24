"""A Run: the user's prompt to one instance, kept as a run with its files.

The request is `mnm try`'s (`try_run`) with the user's words in place of the card's, sent
through the router (which loads the model and makes room), and what comes back is saved in
the run's folder instead of the current one. The CLI's `mnm run` and the web Run form share
this; pipelines call it for their steps.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import httpx

from makenomistakes.engine_args import AudioCppArgs, SdCppArgs
from makenomistakes.harness_media import image_size
from makenomistakes.hub.facts import Modality
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.recipes import Recipe
from makenomistakes.run_parts import InputRecord, ModelCredit
from makenomistakes.runs import RunKind, RunRecord, RunStore
from makenomistakes.smoke import Picture
from makenomistakes.try_run import TryInputs, run_try

# The types a Run makes something with, and what each asks for: a prompt (text), a picture,
# or a clip to transcribe. Embeddings and rerankers answer numbers, which `mnm try` covers.
PROMPTED = frozenset(
    {
        Modality.CHAT,
        Modality.CODING,
        Modality.VISION_CHAT,
        Modality.IMAGE_GEN,
        Modality.IMAGE_EDIT,
        Modality.VIDEO,
        Modality.TEXT_TO_SPEECH,
        Modality.SONG,  # Tags, and lyrics beside them
    }
)
# The studio's other types work on a clip: a mix to split, a vocal, a take to score.
ON_A_CLIP = frozenset({Modality.STEMS, Modality.VOICE_CONVERT, Modality.MUSIC_SCORE})
RUNNABLE = PROMPTED | {Modality.SPEECH_TO_TEXT} | ON_A_CLIP
MAX_PROMPT = 20_000  # characters: a long brief, never a file pasted by accident


PICTURE, MASK, CLIP = "picture", "mask", "clip"  # the roles a run's inputs are kept by


@dataclass(frozen=True)
class Ask:
    """What the user gave a Run: the words, a picture (and mask), a clip, each with the
    name it came under (kept in the run's folder as `input-<role>-<name>`)."""

    prompt: str | None = None
    picture: tuple[str, bytes] | None = None
    mask: tuple[str, bytes] | None = None
    clip: tuple[str, bytes] | None = None
    lyrics: str | None = None  # What a song instance sings to the prompt's tags


MAX_REPEAT = 20  # `--repeat N`: one run each, one after another


def fixed_seed(recipe: Recipe) -> int | None:
    """The seed a picture instance draws with every time, if the user set one (else every
    request draws a new one): a repeat then makes the same picture again."""
    args = recipe.args
    return args.seed if isinstance(args, SdCppArgs) else None


def check(recipe: Recipe, ask: Ask) -> None:
    """Refused before anything is sent or saved, worded for the user."""
    if recipe.modality not in RUNNABLE:
        raise InstanceError(
            f"{recipe.name} is {recipe.modality.a_label} model: it answers numbers, not "
            f"something to keep (`mnm try {recipe.name}` checks it)."
        )
    words = (ask.prompt or "").strip()
    if recipe.modality in PROMPTED - {Modality.IMAGE_EDIT, Modality.VIDEO} and not words:
        raise InstanceError("Write what you want first: the prompt is empty.")
    if len(words) > MAX_PROMPT:
        raise InstanceError(f"The prompt is over {MAX_PROMPT:,} characters.")
    if recipe.modality is Modality.SPEECH_TO_TEXT and ask.clip is None:
        raise InstanceError("Give it a clip to transcribe.")
    if recipe.modality in ON_A_CLIP and ask.clip is None:
        raise InstanceError(f"Give it a clip to work on: `mnm run {recipe.name} --audio song.wav`.")
    if ask.lyrics is not None and recipe.modality is not Modality.SONG:
        raise InstanceError(f"{recipe.name} sings nothing: --lyrics is for a song instance.")
    if len(ask.lyrics or "") > MAX_PROMPT:
        raise InstanceError(f"The lyrics are over {MAX_PROMPT:,} characters.")
    if ask.mask is not None and ask.picture is None:
        raise InstanceError("A mask needs the picture it masks: add the picture too.")


def settings_used(recipe: Recipe) -> dict[str, str]:
    """What the run is kept with beside the prompt: the model and the instance's own size or
    voice, so "Run again" means the same thing a week later."""
    used = {"model": recipe.repo_id, "quant": recipe.quant}
    if recipe.modality in {Modality.IMAGE_GEN, Modality.IMAGE_EDIT, Modality.VIDEO}:
        width, height = image_size(recipe)
        used["size"] = f"{width}x{height}"
    if isinstance(recipe.args, AudioCppArgs) and recipe.args.music:
        if recipe.args.options is not None:
            used["options"] = recipe.args.options.summary
    elif isinstance(recipe.args, AudioCppArgs):
        used["voice"] = recipe.args.voice
    return used


def credit(recipe: Recipe) -> list[ModelCredit]:
    """The model that made a run: the instance, its repo, file, quant and image."""
    return [
        ModelCredit(
            instance=recipe.name,
            repo=recipe.repo_id,
            file=recipe.files[0] if recipe.files else None,
            quant=recipe.quant,
            build=recipe.image,
        )
    ]


def make(
    service: InstanceService,
    store: RunStore,
    recipe: Recipe,
    http: httpx.Client,
    ask: Ask,
    record: RunRecord | None = None,
    parent: RunRecord | None = None,
) -> RunRecord:
    """One Run, start to finish: a record (or the one given, already shown as running), the
    inputs kept, the request, the files. A failure is kept as a failed run and raised. With
    `parent` it is one candidate under that run."""
    check(recipe, ask)
    words = (ask.prompt or "").strip() or None
    run = record or store.new(
        RunKind.INSTANCE, recipe.name, words, settings_used(recipe),
        parent=parent.id if parent is not None else None,
        role="candidate" if parent is not None else None,
    )  # fmt: skip
    run.models = credit(recipe)
    if parent is not None:
        run.parent, run.role = parent.id, "candidate"
    if ask.lyrics is not None:
        run.settings["lyrics"] = ask.lyrics
    store.save(run)
    try:
        inputs = _inputs(store, run, ask, words)
        result = run_try(service, recipe, http, store.folder(run), inputs).result
    except BaseException as exc:  # whatever stops it, the run is never left running
        store.fail(run, str(exc) or f"Interrupted ({type(exc).__name__}).")
        raise
    made = [*([result.saved] if result.saved is not None else []), *result.extra]
    numbers = {figure.value: value for figure, value in result.numbers.items()}
    return store.finish(run, made, result.answer, result.seconds, numbers)


def _inputs(store: RunStore, run: RunRecord, ask: Ask, words: str | None) -> TryInputs:
    picture = mask = clip = None
    if ask.picture is not None:
        picture = store.add_input(run, *ask.picture, role=PICTURE).read_bytes()
    if ask.mask is not None:
        mask = store.add_input(run, *ask.mask, role=MASK).read_bytes()
    if ask.clip is not None:
        clip = store.add_input(run, *ask.clip, role=CLIP)
    for role, given in ((PICTURE, ask.picture), (MASK, ask.mask), (CLIP, ask.clip)):
        if given is not None:  # every input by hash (run record schema 2)
            digest = hashlib.sha256(given[1]).hexdigest()
            run.inputs_recorded.append(InputRecord(role=role, name=given[0], sha256=digest))
    return TryInputs(
        audio=clip,
        picture=Picture(picture, mask) if picture else None,
        prompt=words,
        lyrics=ask.lyrics,
    )


def again(store: RunStore, run: RunRecord) -> Ask:
    """A finished run's prompt and inputs, for "Run again": each input by the role it was
    kept under (`input-picture-…`, `input-mask-…`, `input-clip-…`)."""
    found = store.inputs(run, [PICTURE, MASK, CLIP])
    return Ask(
        prompt=run.prompt,
        picture=found.get(PICTURE),
        mask=found.get(MASK),
        clip=found.get(CLIP),
        lyrics=run.settings.get("lyrics"),
    )


def takes_candidates(recipe: Recipe, repeat: int) -> bool:
    """A song asked for more than once makes candidates under one parent: the
    user picks one of the four; any other repeat is N runs side by side."""
    return recipe.modality is Modality.SONG and repeat > 1


def open_parent(store: RunStore, recipe: Recipe, ask: Ask) -> RunRecord:
    """The run the candidates sit under: what was asked, no files of its own."""
    check(recipe, ask)
    words = (ask.prompt or "").strip() or None
    parent = store.new(RunKind.INSTANCE, recipe.name, words, settings_used(recipe))
    parent.models = credit(recipe)
    if ask.lyrics is not None:
        parent.settings["lyrics"] = ask.lyrics
    store.save(parent)
    return parent


def close_parent(
    store: RunStore, parent: RunRecord, made: list[RunRecord], wanted: int, error: str = ""
) -> RunRecord:
    """Done when every candidate was made; failed, saying how many were, when one failed."""
    seconds = sum(c.seconds or 0.0 for c in made)
    if error:
        return store.fail(parent, f"{len(made)} of {wanted} candidates were made: {error}")
    answer = f"{len(made)} candidates (`mnm outputs show {parent.id}` lists them)"
    return store.finish(parent, [], answer, seconds)
