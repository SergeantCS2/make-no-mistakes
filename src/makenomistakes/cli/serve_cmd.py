"""`mnm serve`: the web UI."""

from __future__ import annotations

import os
import socket
from typing import Annotated

import typer

from makenomistakes.cli._app import app, console, fail_with
from makenomistakes.config import Settings


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address.")] = "127.0.0.1",
    port: Annotated[int | None, typer.Option(help="Port (default MNM_PORT, 7800).")] = None,
) -> None:
    """Start the web UI (instances, add model, doctor)."""
    import uvicorn  # noqa: PLC0415  (server only needed here)

    from makenomistakes.web import create_app  # noqa: PLC0415
    from makenomistakes.web.app import LOCAL_HOSTS  # noqa: PLC0415

    settings = Settings()
    bind_port = port or settings.port
    if not port_is_free(host, bind_port):
        fail_with(
            RuntimeError(
                f"Port {bind_port} is already in use, most likely by an earlier `mnm serve` "
                "that is still running. Stop it with Ctrl-C in its terminal (a web UI started "
                "before `git pull` keeps the old code and breaks on new pages), or pass "
                "--port to run a second one."
            )
        )
    if os.environ.get("INVOCATION_ID"):  # systemd sets it: this is the service, not a terminal
        console.print(
            f"Web UI service on http://{host}:{bind_port} (`mnm update` restarts it; "
            "`mnm service status` shows it)"
        )
    else:
        console.print(f"Web UI on http://{host}:{bind_port}  (Ctrl-C to stop)")
        console.print(
            "[dim]Restart this command after every update, or run `mnm service install` once "
            "and let `mnm update` do it.[/dim]"
        )
    if host in _WILDCARDS:
        console.print(
            "[dim]Other machines get 400 Invalid host header: the UI answers only to "
            "localhost and 127.0.0.1 (a check against DNS rebinding). To open it from another "
            "machine, pass the address it is reached at: `mnm serve --host <that IP>`.[/dim]"
        )
    app = create_app(settings, allowed_hosts=allowed_hosts(host, LOCAL_HOSTS))
    uvicorn.run(app, host=host, port=bind_port, log_level="warning")


_WILDCARDS = ("0.0.0.0", "::")  # every interface: not a name the UI is opened under


def allowed_hosts(host: str, local: tuple[str, ...]) -> tuple[str, ...]:
    """The names the UI answers to: the local ones, and the address it was bound to."""
    return local if host in ("127.0.0.1", "localhost", *_WILDCARDS) else (*local, host)


def port_is_free(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET  # `::` is not an IPv4 bind
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True
