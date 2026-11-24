"""`mnm doctor --from here|docker|lan` and `mnm doctor --remote-script`: the router as
a client sees it, run here (`remote_doctor`) or printed for another machine (`remote_script`)."""

from __future__ import annotations

import json
from typing import NoReturn

import typer
from rich.markup import escape

from makenomistakes.apps.cards import local_manifest
from makenomistakes.apps.probe import SubprocessEnvRunner
from makenomistakes.cli._app import console, fail_with
from makenomistakes.client_view import From, ReachError, Seen
from makenomistakes.config import Settings
from makenomistakes.doctor_types import Check, Status
from makenomistakes.hardware import platform as platforms
from makenomistakes.instance_types import InstanceError
from makenomistakes.instances import InstanceService
from makenomistakes.manifest import view
from makenomistakes.remote_doctor import probes, run, target
from makenomistakes.remote_script import Shell, script
from makenomistakes.store import StoreError

# Under WSL's NAT networking no other machine reaches the router, but Windows does, at
# localhost (WSL forwards it): the script's address then, for the user's PowerShell.
WINDOWS_HOST = "localhost"
PLACES = {From.HERE: "on this machine", From.DOCKER: "in a container here", From.LAN: "elsewhere"}


def checks_from(settings: Settings, where: From) -> tuple[str, list[Check]]:
    """Where the checks ask from, and their lines; one FAIL line when that place has no way
    to the router at all (no LAN address)."""
    try:
        seen = view(where, settings, need_key=False)
    except ReachError as exc:
        return f"from {where.value}", [Check(name="Reach", status=Status.FAIL, detail=str(exc))]
    assert seen is not None  # a place was named
    manifest = local_manifest(InstanceService(settings))
    found = target(manifest, seen)
    probe, bare = probes(seen.where, settings.api_key_value(), SubprocessEnvRunner())
    return f"{probe.where}, to {found.base}", run(found, probe, bare)


def run_from(settings: Settings, where: From, as_json: bool, show: object) -> NoReturn:
    try:
        said, found = checks_from(settings, where)
    except (InstanceError, StoreError) as exc:
        fail_with(exc)
    if as_json:
        console.print_json(json.dumps({"where": said, "checks": [c.model_dump() for c in found]}))
    else:
        console.print(f"The router as a client {PLACES[where]} sees it, {escape(said)}:")
        show(found, f"mnm doctor --from {where.value}")  # type: ignore[operator]
    raise typer.Exit(code=1 if any(c.status is Status.FAIL for c in found) else 0)


def script_seen(settings: Settings) -> tuple[Seen, bool]:
    """The address the script names, and whether the key is required there: the LAN's (or
    MNM_ROUTER_URL's) when the router is on it; Windows' localhost under WSL's NAT."""
    try:
        seen = view(From.LAN, settings, need_key=False)
    except ReachError:
        if not platforms.current().is_wsl:
            raise
        return Seen(From.HERE, every=WINDOWS_HOST), False
    assert seen is not None
    return seen, True


def print_script(settings: Settings, shell: Shell) -> NoReturn:
    try:
        seen, need_key = script_seen(settings)
        manifest = local_manifest(InstanceService(settings))
    except (InstanceError, StoreError) as exc:
        fail_with(exc)
    # Plain stdout, never rich: the user redirects it into a file (`> mnm-check.sh`).
    typer.echo(script(target(manifest, seen), shell, need_key), nl=False)
    raise typer.Exit(code=0)
