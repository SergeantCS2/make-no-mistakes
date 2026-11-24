"""`mnm router`: one URL for every model, run by its systemd user unit
(`mnm service install`) or in a terminal. A thread beside it unloads idle instances."""

from __future__ import annotations

import errno
import os
import socket
import threading
from collections.abc import Callable
from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.apps.cards import local_manifest
from makenomistakes.bridge import bridge_tools
from makenomistakes.bridge.protocol import Toolbox
from makenomistakes.card_jobs import CardGate
from makenomistakes.cli._app import app, console, fail_with, print_block
from makenomistakes.comfy.engine import ComfyEngine
from makenomistakes.comfy.guard import ComfyGuard
from makenomistakes.comfy.idle import release_if_idle
from makenomistakes.config import Settings
from makenomistakes.hardware import CommandRunner, SubprocessRunner
from makenomistakes.healing import Healer, seams_for
from makenomistakes.health_now import HealthReader
from makenomistakes.idle import sweep_forever
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Lasting
from makenomistakes.machine import saved_recipes, start_line
from makenomistakes.notify import Notice, notifier
from makenomistakes.request_meter import RequestMeter
from makenomistakes.router import Router
from makenomistakes.router_bind import EVERYWHERE, BindError, bind_addresses
from makenomistakes.router_hosts import HostList, default_route_addresses, interface_addresses
from makenomistakes.router_http import RouterServer
from makenomistakes.router_queue import write as write_windows
from makenomistakes.service_watch import Watchdog, started_by_systemd
from makenomistakes.studio.train_runner import for_router as trainer_for_router
from makenomistakes.tune.wiring import for_router


def bind_problem(address: str, port: int) -> str | None:
    """Why the router cannot listen there, in words (None: it can)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((address, port))
        except OSError as exc:
            if exc.errno == errno.EADDRNOTAVAIL:
                return (
                    f"{address} is not an address of this machine (MNM_ROUTER_HOST): use "
                    "0.0.0.0, local, or addresses `ip -4 addr` lists."
                )
            return (
                f"Port {port} on {address} is already in use, most likely by a router that is "
                "still running. Stop it with Ctrl-C in its terminal, or pass --port."
            )
    return None


def listen(
    hosts: list[str],
    port: int,
    router: Router,
    tools: Toolbox | None = None,
    ollama_port: int | None = None,
    runner: CommandRunner | None = None,
) -> list[RouterServer]:
    """One server per address (and per address on Ollama's port, when set), every one in
    front of the same router; all but the first already serving on threads of their own
    (the first is the caller's to serve)."""
    settings = router.instances.settings
    guard = ComfyGuard(settings.expanded_state_dir())
    interfaces = interface_addresses(runner or SubprocessRunner()) if EVERYWHERE in hosts else []
    names = HostList.of(hosts, settings.router_hosts, [*interfaces, *default_route_addresses()])
    servers = [
        RouterServer((address, port), router, tools, guard, hosts=names) for address in hosts
    ]
    if ollama_port is not None:
        servers += [
            RouterServer((address, ollama_port), router, tools, guard, ollama=True, hosts=names)
            for address in hosts
        ]
    reader = HealthReader(settings, runner or SubprocessRunner())  # `/__mnm/health`
    for server in servers:
        server.health = reader
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    return servers


def heal_and_watch(healer: Healer | None, watchdog: Watchdog, said: Lasting) -> None:
    """The sweep's first part: neither one's failure keeps the idle sweep from running, and a
    failure that lasts (Docker away) is said once per look, not every sweep."""
    looks = {"healer": healer.sweep if healer is not None else None, "watchdog": watchdog.tick}
    for what, look in looks.items():
        if look is None:
            continue
        try:
            look()
        except Exception as exc:  # the sweep thread never dies of it
            said.failed(f"The router's health sweep ({what}) failed and tries again: {exc}", what)
        else:
            said.worked(what)


def ollama_banner(port: int | None) -> str:
    """Where else Ollama's API listens, for the banner."""
    if port is None:
        return " (MNM_OLLAMA_PORT=11434 adds Ollama's own port)"
    return f" and on port {port} (MNM_OLLAMA_PORT), the same addresses and key"


router_app = typer.Typer(
    invoke_without_command=True,
    help="Serve every instance on one URL (the systemd unit runs it); `mnm router manifest` "
    "prints what it tells apps.",
)
app.add_typer(router_app, name="router")


@router_app.command("manifest")
def manifest_cmd() -> None:
    """What the router's `GET /__mnm/manifest` answers now: every instance with its desk
    roles, paths, context, tools and vision, and the router's URLs from here, from containers
    and from the LAN. Built here, so it answers with the router down too."""
    service = InstanceService(Settings())
    try:
        found = local_manifest(service)
    except InstanceError as exc:  # a client install whose box did not answer
        fail_with(exc)
    print_block(found.model_dump_json(indent=2))


@router_app.callback()
def router(
    ctx: typer.Context,
    host: Annotated[
        str | None,
        typer.Option(
            help="Where to listen: 0.0.0.0, local (this machine and its Docker containers) or "
            "addresses (default MNM_ROUTER_HOST, 0.0.0.0)."
        ),
    ] = None,
    port: Annotated[int | None, typer.Option(help="Port (default MNM_ROUTER_PORT, 7801).")] = None,
) -> None:
    """Serve every instance on one URL: requests name their model, parked ones load."""
    if ctx.invoked_subcommand is not None:
        return
    settings = Settings()
    bind_port = port or settings.router_port
    try:
        hosts = bind_addresses(host or settings.router_host, SubprocessRunner())
    except (BindError, ValueError) as exc:
        fail_with(exc)
    ollama_port = settings.ollama_port
    if ollama_port == bind_port:
        fail_with(
            RuntimeError(f"MNM_OLLAMA_PORT is the router's own port ({bind_port}): use 11434")
        )
    for address in hosts:
        for each in (bind_port, *([ollama_port] if ollama_port is not None else [])):
            if (problem := bind_problem(address, each)) is not None:
                fail_with(RuntimeError(problem))
    service = InstanceService(settings, router_up=lambda _s: True)  # it is the router
    router = Router(service)
    runner = SubprocessRunner()
    if (restarted := started_by_systemd(runner, os.environ)) is not None:
        router.note(restarted)
    router.note(start_line(settings, service.memory(), saved_recipes(settings)))
    notices = notifier(settings)
    seams = seams_for(service, notices.post)
    router.healer = Healer(router, seams)  # dead engines noticed and restarted once
    watchdog = Watchdog(runner, settings.expanded_state_dir(), notices.post)
    # Readings for the planner; this process answers its own traffic, not over HTTP.
    router.meter = RequestMeter(service, router.container)
    service.card.traffic = router.traffic
    servers = listen(hosts, bind_port, router, bridge_tools(service), ollama_port)
    stopped = threading.Event()
    comfy = ComfyEngine(settings)
    health_said = Lasting(router.note)  # the health looks' lasting failures, said once

    def sweep() -> list[str]:
        """Dead engines first, idle models, then ComfyUI ten minutes after its last
        run, from any process; the services' restarts every five minutes; the queue's
        windows done as readings."""
        heal_and_watch(router.healer, watchdog, health_said)
        unloaded = router.unload_idle()
        write_windows(service, router.queue.closed())
        return [*unloaded, "ComfyUI's models"] if release_if_idle(comfy) else unloaded

    sweeps = threading.Thread(
        target=sweep_forever,
        args=(sweep, router.note, stopped),  # the journal, and the System page's activity
        daemon=True,
    )
    sweeps.start()
    _card_jobs(service, router, notices.post, stopped)
    console.print(
        f"Router on http://localhost:{bind_port}/v1 (Odysseus in Docker: "
        f"http://host.docker.internal:{bind_port}/v1), listening on "
        f"{escape(', '.join(hosts))}; Ctrl-C to stop."
    )
    console.print(
        f"[dim]Name an instance as the model, or ask {escape('/upstream/<name>/')} for one "
        "instance. `GET /v1/models` lists them. MCP tools: /mcp (`mnm mcp`). Ollama's API: "
        f"/api/… on this port{ollama_banner(ollama_port)}.[/dim]"
    )
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stopped.set()
        for server in servers:
            server.server_close()


def _card_jobs(
    service: InstanceService,
    router: Router,
    notify: Callable[[Notice], None],
    stopped: threading.Event,
) -> None:
    """The tuner's loop and the trainer's on threads of their own,
    one card gate between them: one job on the card at a time, both held by the nightly."""
    gate = CardGate(note=router.note)
    router.card_jobs = gate
    tuner = for_router(service, router.traffic, router.busy, router.note, gate)
    router.tuner = tuner
    threading.Thread(target=tuner.loop, args=(stopped,), daemon=True).start()
    trainer = trainer_for_router(service, gate, router.busy, router.traffic, router.note, notify)
    router.trainer = trainer
    threading.Thread(target=trainer.loop, args=(stopped,), daemon=True).start()
