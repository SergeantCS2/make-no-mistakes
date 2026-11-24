"""`mnm doctor --app`'s order: build the card the app's file comes from, pick where
to ask from, run the checks `doctor.py` holds. The model list goes first: when nothing answers
there, every other request would fail the same way, so only the key and file lines follow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from makenomistakes.apps import doctor as checks
from makenomistakes.apps.card import AppCard
from makenomistakes.apps.cards import card_for, local_manifest
from makenomistakes.apps.files import Ledger
from makenomistakes.apps.names import CONTAINERS, LABELS, VANTAGE, App, Vantage
from makenomistakes.apps.probe import (
    EnvRunner,
    HostProbe,
    Probe,
    SubprocessEnvRunner,
    inside,
    one_off,
    running,
)
from makenomistakes.doctor_types import Check
from makenomistakes.hardware import CommandRunner, SubprocessRunner
from makenomistakes.instances import InstanceService
from makenomistakes.manifest import Manifest


@dataclass(frozen=True)
class AppReport:
    where: str
    checks: list[Check]


def probes(
    app: App,
    key: str | None,
    container: str | None,
    runner: CommandRunner,
    env_runner: EnvRunner,
) -> tuple[Probe, Probe]:
    """Where the app's requests come from: with the key, and without it (the key check)."""
    if VANTAGE[app] is Vantage.HOST:
        return HostProbe(key), HostProbe(None)
    name = container or CONTAINERS.get(app)
    up = running(name, runner) if name else False
    if name and up:
        return inside(name, key, env_runner), inside(name, None, env_runner)
    label = LABELS[app]
    why = (
        f"{label}'s container {name} is not running"
        if name and up is False
        else f"Docker could not say whether {name} runs"
        if name
        else f"{label}'s container is one you name: `--container NAME` asks from inside it"
    )
    return one_off(key, env_runner, why), one_off(None, env_runner, why)


def _openai_base(manifest: Manifest, app: App) -> str:
    """The router's `/v1` from the app's side (a rerank client is given the full URL)."""
    return manifest.openai.loopback if VANTAGE[app] is Vantage.HOST else manifest.openai.containers


def run_checks(
    app: App,
    card: AppCard,
    manifest: Manifest,
    probe: Probe,
    bare: Probe,
    written: list[Path],
) -> list[Check]:
    vantage = VANTAGE[app]
    found, answer, listing = checks.models_check(card, probe, vantage)
    done = [found]
    if answer.status != 0:  # something answers: the rest are worth asking
        if card.uses.chat_model and answer.status == 200:
            done.append(checks.tool_check(card, probe, vantage))
        if card.uses.embed_model:
            done.append(checks.embed_check(card, probe, vantage))
        if "rerank" in manifest.aliases:
            done.append(checks.rerank_check(manifest, probe, vantage, _openai_base(manifest, app)))
        if card.uses.mcp_url:
            sends = probe if card.uses.mcp_takes_key else bare  # as the app's MCP client asks
            done.append(checks.mcp_check(card, sends, vantage))
        if (floor := checks.floor_check(card, listing, manifest)) is not None:
            done.append(floor)
    done.append(checks.key_check(card, manifest, bare))
    return [*done, *checks.file_check(card, written)]


def app_doctor(
    app: App,
    service: InstanceService,
    container: str | None = None,
    runner: CommandRunner | None = None,
    env_runner: EnvRunner | None = None,
    before: Callable[[str], None] = lambda _where: None,
) -> AppReport:
    """The checks; `before` is told where they ask from before the first request goes (a
    one-off container may pull its image first). InstanceError (an AppFileError, mnm's own
    files unreadable) before anything is asked."""
    settings = service.settings
    ledger = Ledger(settings.expanded_state_dir()).read()
    written = [Path(p).resolve() for p, w in ledger.files.items() if w.app == app.value]
    manifest = local_manifest(service)
    card = card_for(app, manifest, settings)
    probe, bare = probes(
        app,
        settings.api_key_value(),
        container,
        runner or SubprocessRunner(),
        env_runner or SubprocessEnvRunner(),
    )
    before(probe.where)
    return AppReport(probe.where, run_checks(app, card, manifest, probe, bare, written))
