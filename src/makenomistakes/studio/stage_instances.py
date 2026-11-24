"""Which saved instances play a stage's tier: the user's pick (`mnm studio stage`),
else the instance whose files are the registry's package. A package nobody added is a
refusal that names the `mnm add` that adds it; the registry never names a model in code."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.hub.music import music_family
from makenomistakes.recipes import Recipe
from makenomistakes.studio import stages
from makenomistakes.studio.stages import Package, StudioError, TierName


@dataclass(frozen=True)
class Player:
    """One package of a tier and the instance that plays it. `stands_in`: the user's pick of
    another family plays a tier of one package (BS-RoFormer for HTDemucs' fast stems), so
    what it gives is its own, never filtered by the package's `gives` (filtered, its
    `instrumental` would be dropped and the mix be the vocals alone)."""

    package: Package
    recipe: Recipe
    stands_in: bool = False

    def stems(self, given: list[str]) -> list[str]:
        """Which of the stems the instance gave are kept: the package's (refused, naming what
        is missing, when it gave fewer), or every one a stand-in gave."""
        wanted = [role.removeprefix("stem:") for role in self.package.gives]
        if self.stands_in or not wanted:
            return given
        missing = [name for name in wanted if name not in given]
        if missing:
            raise StudioError(
                f"{self.recipe.name} gave the stems {', '.join(given) or 'none'}, and its "
                f"package gives {', '.join(wanted)}: {', '.join(missing)} missing."
            )
        return [name for name in given if name in wanted]


def is_package(recipe: Recipe, package: Package) -> bool:
    """Whether the instance's files are the package (its repo, folder and quant)."""
    return (
        recipe.repo_id == package.repo
        and recipe.quant.upper() == package.quant.upper()
        and any(f.startswith(package.folder.rstrip("/") + "/") for f in recipe.files)
    )


def _family(recipe: Recipe) -> str | None:
    return recipe.args.family if isinstance(recipe.args, AudioCppArgs) else None


def players(stage: str, tier: TierName, state_dir: Path, recipes: list[Recipe]) -> list[Player]:
    """Each package of the tier with its instance, in the tier's order; StudioError naming
    what to add (or pick) when one has none."""
    resolved = stages.resolve(stage, tier, state_dir)
    by_name = {r.name: r for r in recipes}
    picked = by_name.get(resolved.instance) if resolved.instance else None
    if resolved.instance and picked is None:
        raise StudioError(
            f"The {stage} stage's {tier} tier is {resolved.instance}, which is no saved "
            f"instance any more: `mnm studio stage {stage} {tier}=<instance>` picks another, "
            f"`{tier}=` goes back to the registry's."
        )
    found: list[Player] = []
    missing: list[Package] = []
    alone = len(resolved.packages) == 1
    for package in resolved.packages:
        family = music_family(package.folder)
        own = _family(picked) == (family and family.value) if picked is not None else False
        if picked is not None and (alone or own):
            found.append(Player(package, picked, stands_in=not own))
            continue
        match = next((r for r in recipes if is_package(r, package)), None)
        if match is None:
            missing.append(package)
        else:
            found.append(Player(package, match))
    if missing:
        adds = "; ".join(f"`mnm add {p.reference}`" for p in missing)
        raise StudioError(
            f"The {stage} stage's {tier} tier runs {stages.tier_words(stages.row(stage), tier)}, "
            f"and no instance plays {'it' if len(missing) == 1 else 'them'}: {adds} "
            f"(or `mnm studio stage {stage} {tier}=<instance>` picks one you have)."
        )
    return found


def playing(stage: str, tier: TierName, state_dir: Path, recipes: list[Recipe]) -> list[str]:
    """The instances that play a package tier now (the user's pick, else the ones whose
    files are its packages), in order; empty when one is missing (`players` says which) or
    the picks file cannot be read (the page says that once)."""
    try:
        resolved = stages.resolve(stage, tier, state_dir)
        if resolved.workflow is not None:  # a ComfyUI workflow plays it: the pick
            return [resolved.instance or ""]
        if resolved.instance and stages.row(stage).host == "tool":  # a tool row's (MIDI's)
            return [resolved.instance]
        return list(dict.fromkeys(p.recipe.name for p in players(stage, tier, state_dir, recipes)))
    except StudioError:
        return []
