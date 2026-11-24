"""The router's manifest: `GET /__mnm/manifest`, behind the key like `/__mnm/busy`,
and `mnm router manifest`. What an app (or mnm's own files for it, `apps/`) needs to know to
use every model without being told by hand: each instance with its desk roles, type, the
paths it answers, the context one request gets, a reply budget, tools and vision, and where
the router is reached from this machine, from its containers and from the LAN.

Typed (`Manifest`), versioned (`schema_version`, raised when a field changes meaning or goes),
built from what the router itself reads: the recipes, the containers' states, the desk
(`desk.read_aliases`) and the model files (`MetaCache.tools`)."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

from pydantic import BaseModel

from makenomistakes import __version__
from makenomistakes.aliases import Aliases
from makenomistakes.apps import ledger
from makenomistakes.apps.defs import DEFS
from makenomistakes.bridge.protocol import PATH as MCP_PATH
from makenomistakes.client_view import From, Seen, resolve
from makenomistakes.config import Settings
from makenomistakes.consumers import ODYSSEUS_HOST, origin, output_tokens
from makenomistakes.doctor_network import reaches_lan
from makenomistakes.engine_args import AudioCppArgs
from makenomistakes.engine_args_music import AceRoute, AceStepOptions
from makenomistakes.hardware import CommandRunner, SubprocessRunner
from makenomistakes.hardware import platform as platforms
from makenomistakes.hardware.probe import networking_mode
from makenomistakes.harness_media import VIDEO_PATH, client_path
from makenomistakes.hub.facts import CHAT_LIKE, MUSIC_MODALITIES, Engine, Modality
from makenomistakes.instance_types import InstanceStatus
from makenomistakes.model_meta import MetaCache, serving_context, serving_slots
from makenomistakes.music_api import TASK_PATH
from makenomistakes.ollama_api import serves
from makenomistakes.recipes.audiocpp import voice_names
from makenomistakes.recipes.model import Recipe
from makenomistakes.router_bind import LOOPBACK, Exposure, check_setting, exposure
from makenomistakes.routing import UPSTREAM

if TYPE_CHECKING:
    from makenomistakes.router import Router

PATH = "/__mnm/manifest"
SCHEMA_VERSION = 1
State = Literal["loaded", "parked", "stopped", "not downloaded", "unknown"]


class Urls(BaseModel):
    """One of the router's doors, from each place a client may be."""

    loopback: str  # an app on this machine (OpenCode, a WSL shell, a Windows app)
    containers: str  # an app in a Docker container here (`host.docker.internal`)
    lan: list[str]  # another machine, when the router listens there (`lan_note` says)


class ModelEntry(BaseModel):
    name: str  # the instance: the id that always works
    aliases: list[str]  # the desk roles that name it (`mnm sets default`)
    type: str  # the model type (`Modality`)
    engine: str
    engine_id: str  # what the engine calls it; the router rewrites to it
    state: State
    endpoints: list[str]  # request paths from the router's root
    context: int | None  # the context one request gets (`model_meta.serving_context`)
    slots: int  # the slots its context is split into (llama-server's `-np`; one otherwise)
    max_output: int | None  # the reply budget mnm's cards give (a quarter, at most 8,192)
    tools: bool | None  # the template declares tool calls (None: not a chat model)
    vision: bool
    voices: list[str]  # a voice model's voices, its default first (a request names one)
    # A music instance's task and the operations it runs (ACE-Step's routes); the
    # router leaves it out of `/v1/models`, so this is where a client learns of it
    tasks: list[str] = []


class AppEntry(BaseModel):
    """A workspace app `mnm apps` installed: its version and where it answers."""

    name: str
    version: str
    url: str | None


class Manifest(BaseModel):
    schema_version: int = SCHEMA_VERSION
    mnm_version: str
    desk: str | None  # the default set, whose roles are aliases
    aliases: dict[str, str]  # role -> instance
    alias_problem: str | None  # why none is served although a desk is chosen
    shadowed: list[str]  # desk roles an instance is named after: no ids (`desk.aliases_of`)
    key_required: bool  # every request (and /mcp) must carry MNM_API_KEY
    openai: Urls  # base URLs, `/v1` included
    ollama: Urls  # Ollama's API (its own port when MNM_OLLAMA_PORT is set)
    mcp: Urls  # mnm's MCP tools, Streamable HTTP
    lan_note: str
    models: list[ModelEntry]
    apps: list[AppEntry] = []  # The apps `mnm apps` installed (added, not changed)


@dataclass(frozen=True)
class Lan:
    """Whether other machines reach the router, and at which addresses."""

    hosts: tuple[str, ...]
    note: str


def lan_reach(router_host: str, runner: CommandRunner) -> Lan:
    """Where another machine reaches the router: none with `local`, none under WSL's NAT
    networking (the VM's address is not on the LAN), else the listed addresses or, for
    0.0.0.0, the address of this machine's default route."""
    if exposure(router_host) is Exposure.LOCAL:
        return Lan((), "MNM_ROUTER_HOST=local: this machine and its Docker containers only.")
    wsl = platforms.current().is_wsl
    if not reaches_lan(networking_mode(runner) if wsl else None, router_host):
        return Lan(
            (),
            "WSL's NAT networking keeps the router on this PC (Windows and its WSL apps "
            "reach localhost); another machine comes over Tailscale (GUIDE 13.4).",
        )
    if exposure(router_host) is Exposure.LISTED:
        hosts = tuple(a for a in check_setting(router_host).split(",") if a != LOOPBACK)
    else:
        hosts = tuple(h for h in [default_route_address()] if h)
    if not hosts:
        return Lan((), "The router listens on this machine's loopback only.")
    return Lan(hosts, "Another machine reaches the router at these addresses (with the key).")


def default_route_address() -> str | None:
    """This machine's address on its default route: a UDP socket "connected" to a
    documentation address sends nothing and tells which source address would go out."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("192.0.2.1", 9))
            address = probe.getsockname()[0]
        except OSError:
            return None
    return str(address) if address and not str(address).startswith("127.") else None


def _urls(port: int, suffix: str, lan: Lan) -> Urls:
    return Urls(
        loopback=origin(LOOPBACK, port) + suffix,
        containers=origin(ODYSSEUS_HOST, port) + suffix,
        lan=[origin(host, port) + suffix for host in lan.hosts],
    )


def endpoints(status: InstanceStatus) -> list[str]:
    """The request paths that reach this instance through the router, from its root: the
    OpenAI routes its kind answers (the name or an alias as `model`), Ollama's for what that
    API serves, and its own `/upstream/<name>/` door."""
    recipe = status.recipe
    own = f"{UPSTREAM}{recipe.name}"
    if recipe.modality in CHAT_LIKE:
        found = ["/v1/chat/completions", "/v1/completions", "/v1/responses"]
        if recipe.engine is Engine.LLAMACPP:  # llama-server's own
            found += ["/v1/messages", f"{own}/infill"]
    elif recipe.modality is Modality.VIDEO:
        found = [f"{own}{VIDEO_PATH}"]
    elif recipe.modality is Modality.IMAGE_GEN:
        found = [client_path(recipe), "/v1/images/edits"]  # any image model repaints
    elif recipe.modality in MUSIC_MODALITIES:  # By its own door only
        found = [f"{own}{TASK_PATH}"]
    else:
        found = [client_path(recipe)]
    if serves(recipe):
        found += ["/api/chat", "/api/generate"] if recipe.modality in CHAT_LIKE else ["/api/embed"]
    return [*found, f"{own}/v1"]


def _state(status: InstanceStatus) -> State:
    if status.error:
        return "unknown"
    if status.running:
        return "loaded"
    if status.parked:
        return "parked"
    return "not downloaded" if not status.files_present else "stopped"


def model_entry(status: InstanceStatus, aliases: Aliases, meta: MetaCache) -> ModelEntry:
    recipe = status.recipe
    context = serving_context(recipe)
    return ModelEntry(
        name=recipe.name,
        aliases=aliases.of(recipe.name),
        type=recipe.modality.value,
        engine=recipe.engine.value,
        engine_id=recipe.served_name,
        state=_state(status),
        endpoints=endpoints(status),
        context=context,
        slots=serving_slots(recipe),
        max_output=output_tokens(recipe) if context and recipe.modality in CHAT_LIKE else None,
        tools=meta.tools(recipe),
        vision=recipe.modality is Modality.VISION_CHAT,
        voices=voice_names(recipe) if isinstance(recipe.args, AudioCppArgs) else [],
        tasks=music_tasks(recipe),
    )


def music_tasks(recipe: Recipe) -> list[str]:
    """`gen` and ACE-Step's routes, `sep`, `vc`, `midi`: empty for any other instance."""
    args = recipe.args
    if not isinstance(args, AudioCppArgs) or not args.music:
        return []
    routes = list(get_args(AceRoute)) if isinstance(args.options, AceStepOptions) else []
    return [args.task, *routes]


def build(
    statuses: list[InstanceStatus],
    aliases: Aliases,
    meta: MetaCache,
    settings: Settings,
    lan: Lan,
) -> Manifest:
    """The manifest from what the router reads now. Every saved instance is in it with its
    state, so an app's doctor can say why a model it names does not answer."""
    port = settings.router_port
    ollama_port = settings.ollama_port or port
    return Manifest(
        mnm_version=__version__,
        desk=aliases.desk,
        aliases=dict(aliases.members),
        alias_problem=aliases.problem,
        shadowed=list(aliases.shadowed),
        key_required=settings.api_key_value() is not None,
        openai=_urls(port, "/v1", lan),
        ollama=_urls(ollama_port, "", lan),
        mcp=_urls(port, MCP_PATH, lan),
        lan_note=lan.note,
        models=[model_entry(s, aliases, meta) for s in statuses],
        apps=installed_apps(settings),
    )


def view(
    where: From | None,
    settings: Settings,
    runner: CommandRunner | None = None,
    need_key: bool = True,
) -> Seen | None:
    """`--from`'s view after the rules (`client_view.resolve`): the LAN's addresses asked only
    when `lan` is named; ReachError for a place no card may name."""

    def hosts() -> tuple[tuple[str, ...], str]:
        found = lan_reach(settings.router_host, runner or SubprocessRunner())
        return found.hosts, found.note

    return resolve(where, settings, hosts, need_key)


def seen_by(manifest: Manifest, seen: Seen | None) -> Manifest:
    """The manifest with its doors as `seen` names them (`mnm url --from`, a client
    install): an app's file then names the router as that place reaches it."""
    if seen is None:
        return manifest

    def doors(urls: Urls) -> Urls:
        return Urls(
            loopback=seen.url(urls.loopback),
            containers=seen.url(urls.containers),
            lan=seen.urls(urls.lan),
        )

    return manifest.model_copy(
        update={
            "openai": doors(manifest.openai),
            "ollama": doors(manifest.ollama),
            "mcp": doors(manifest.mcp),
        }
    )


def installed_apps(settings: Settings) -> list[AppEntry]:
    """Each app's ledger that reads; one that does not is left out (`mnm apps ls` says why)."""
    found = []
    for app, definition in DEFS.items():
        try:
            installed = ledger.read(settings.expanded_state_dir(), app)
        except ledger.AppsError:
            continue
        if installed is not None:
            found.append(
                AppEntry(name=app.value, version=installed.pin.version, url=definition.url)
            )
    return found


def of_router(router: Router, runner: CommandRunner | None = None) -> Manifest:
    """The manifest a running router serves: its instances, desk and file facts. The LAN reach
    is asked each time (one `wslinfo` under WSL): a manifest is asked rarely (an app's setup,
    its doctor), never per message."""
    settings = router.instances.settings
    lan = lan_reach(settings.router_host, runner or SubprocessRunner())
    return build(router.instances.statuses(), router.aliases(), router.meta, settings, lan)
