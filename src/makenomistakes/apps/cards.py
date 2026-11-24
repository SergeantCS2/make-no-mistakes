"""Each app's card from the router's manifest, and the manifest as this process
builds it for the CLI: the same `manifest.build` the router serves, from the same recipes,
containers and desk."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import ValidationError

from makenomistakes import router_client
from makenomistakes.apps import goose, hermes, librechat, odysseus, open_webui, opencode, openhands
from makenomistakes.apps.card import AppCard
from makenomistakes.apps.names import App
from makenomistakes.client_view import ReachError
from makenomistakes.config import Settings
from makenomistakes.desk import read_aliases
from makenomistakes.hardware import SubprocessRunner
from makenomistakes.instances import InstanceService
from makenomistakes.manifest import PATH, SCHEMA_VERSION, Manifest, build, lan_reach
from makenomistakes.model_meta import MetaCache
from makenomistakes.workflows import WorkflowStore

CARDS: dict[App, Callable[[Manifest, Settings], AppCard]] = {
    App.OPENCODE: lambda manifest, _settings: opencode.card(manifest),
    App.LIBRECHAT: lambda manifest, settings: librechat.card(
        manifest, settings.expanded_apps_dir() / App.LIBRECHAT.value / "librechat.yaml"
    ),
    App.OPENHANDS: lambda manifest, _settings: openhands.card(manifest),
    App.OPEN_WEBUI: lambda manifest, settings: open_webui.card(
        manifest, open_webui.sizes(settings)
    ),
    App.GOOSE: lambda manifest, _settings: goose.card(manifest),
    App.ODYSSEUS: odysseus.card,  # its MCP card reads the settings (`bridge.card`)
    App.HERMES: lambda manifest, _settings: hermes.card(manifest),
}


def card_for(app: App, manifest: Manifest, settings: Settings) -> AppCard:
    return CARDS[app](manifest, settings)


def local_manifest(service: InstanceService) -> Manifest:
    """The manifest the router would serve now, built here (no router needed); on a client
    install (`MNM_ROUTER_URL`) the box's router's own, asked over the network."""
    settings = service.settings
    if settings.router_url:
        return remote_manifest(settings)
    store = WorkflowStore(settings.expanded_state_dir())
    statuses = service.statuses()
    aliases = read_aliases(store, (s.recipe.name for s in statuses))
    lan = lan_reach(settings.router_host, SubprocessRunner())
    return build(statuses, aliases, MetaCache(settings.expanded_models_dir), settings, lan)


def remote_manifest(settings: Settings) -> Manifest:
    """The box's `GET /__mnm/manifest` (behind its key); ReachError saying why not."""
    try:
        body = router_client.get(settings, PATH)
    except OSError as exc:
        said = str(exc).rstrip(".")
        raise ReachError(
            f"The box's router (MNM_ROUTER_URL) did not give its manifest: {said}. `mnm doctor "
            "--from lan` checks the way there; a 401 means this machine's MNM_API_KEY is not "
            "the box's."
        ) from None
    try:
        found = Manifest.model_validate_json(body)
    except ValidationError as exc:
        raise ReachError(
            f"The box's manifest is not one this mnm reads ({exc.errors()[0]['msg']}): "
            "`mnm update` here and on the box."
        ) from None
    if found.schema_version > SCHEMA_VERSION:
        raise ReachError("The box's mnm is newer than this one: `mnm update` here.")
    return found
