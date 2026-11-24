"""The box's freshness step: `mnm engines check` and the family table over every
saved instance, as findings the nightly's page offers (`freshness.Finding`). At most one line
per engine and per instance a night; the Hub asked once per repo a night (`successors.cached`);
the engine check saved for the doctor and System → Engines as `mnm engines check` saves it.

Whisper's turbo (a speed trade, `Offer.FASTER`) is never a nightly line: the page lists newer
models, and `mnm upgrade` offers the turbo when asked."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from makenomistakes.config import Settings
from makenomistakes.engine_check import (
    ENGINE_NAMES,
    EngineReport,
    State,
    check_engines,
    save_report,
    update_command,
)
from makenomistakes.families import Offer
from makenomistakes.hub.client import HubModel
from makenomistakes.hub.facts import Engine
from makenomistakes.instances import InstanceService
from makenomistakes.nightly.freshness import Finding, FindingKind
from makenomistakes.plan import open_hub
from makenomistakes.recipes import Recipe
from makenomistakes.successors import Lookup, Successor, find, hub_lookup
from makenomistakes.supervisor import SupervisorError
from makenomistakes.upgrade_apply import OLD

ENGINES = "the engines"


@dataclass
class BoxFreshness:
    service: InstanceService
    lookup: Lookup | None  # the Hub, cached for the night; None: the table alone
    engines: Callable[[list[Recipe]], EngineReport]  # `check_engines` on the box (`for_box`)

    def check(self) -> list[Finding]:
        recipes = self.service.all()
        once: dict[tuple[FindingKind, str], Finding] = {}
        for finding in self._engines(recipes) + self._models(recipes):
            once.setdefault((finding.kind, finding.subject), finding)  # one line each a night
        return list(once.values())

    def _engines(self, recipes: list[Recipe]) -> list[Finding]:
        try:
            report = self.engines(recipes)
        except SupervisorError as exc:
            words = f"could not be checked: Docker could not be asked ({exc})"
            return [Finding(kind=FindingKind.UNCHECKED, subject=ENGINES, figures=words)]
        save_report(self.service.settings.expanded_state_dir(), report)
        found = []
        for row in report.rows:
            name = ENGINE_NAMES[row.engine]
            if row.state is State.NEWER:
                figures = (
                    f"has a newer build (built {row.upstream_day}; this box's {row.installed_day})"
                )
                found.append(
                    Finding(
                        kind=FindingKind.ENGINE,
                        subject=name,
                        figures=figures,
                        offered=update_command(row.engine),
                    )
                )
            elif row.state is State.UNKNOWN:
                found.append(
                    Finding(
                        kind=FindingKind.UNCHECKED,
                        subject=name,
                        figures=f"could not be checked: {row.why}",
                    )
                )
        for line in report.instances:
            if line.runs:
                found.append(
                    Finding(
                        kind=FindingKind.ENGINE,
                        subject=line.instance,
                        figures=f"needs {line.architecture}, which the registry's llama.cpp runs",
                        offered=update_command(Engine.LLAMACPP),
                    )
                )
        return found

    def _models(self, recipes: list[Recipe]) -> list[Finding]:
        names = {r.name for r in recipes}
        found = []
        for recipe in recipes:
            base = recipe.name.removesuffix(OLD)
            if recipe.name.endswith(OLD) and base in names:
                found.append(
                    Finding(
                        kind=FindingKind.KEPT,
                        subject=recipe.name,
                        figures=f"is kept beside {base} since an upgrade",
                        offered=f"mnm upgrade {base} --replace",
                    )
                )
                continue
            try:
                successor = find(recipe, self.lookup)
            except Exception as exc:  # one model's lookup never ends the step
                words = f"could not be looked up: {exc}"
                found.append(
                    Finding(kind=FindingKind.UNCHECKED, subject=recipe.name, figures=words)
                )
                continue
            if isinstance(successor, Successor) and successor.offer is Offer.NEWER:
                found.append(model_finding(recipe, successor))
        return found


def for_box(service: InstanceService, lookup: Lookup | None) -> BoxFreshness:
    """The nightly's freshness on this box: the registry through Docker, the Hub's cards."""
    return BoxFreshness(service, lookup, lambda recipes: check_engines(service.supervisor, recipes))


def model_finding(recipe: Recipe, successor: Successor) -> Finding:
    return Finding(
        kind=FindingKind.MODEL,
        subject=recipe.name,
        figures=f"has a newer family member: {successor.target}",
        offered=f"mnm upgrade {recipe.name}",
    )


def hub_answers(settings: Settings) -> Lookup:
    """The Hub's card for a repo (None when it cannot say), a client a question: the nightly
    asks a few repos a night, each once (`successors.cached`)."""

    def ask(repo: str) -> HubModel | None:
        with open_hub(settings) as hub:
            return hub_lookup(hub.model_info)(repo)

    return ask
