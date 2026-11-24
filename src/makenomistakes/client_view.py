"""Where a client reaches the router from: one rule for every card. (Not
`comfy/reach.py`, which is what ComfyUI's network can reach.)

- *this machine* (`here`): `localhost` / `127.0.0.1`, which a Windows app reaches too
  through WSL's localhost forwarding;
- *this machine's containers* (`docker`): `host.docker.internal`, Docker's bridge gateway;
- *another machine* (`lan`): the address other machines reach the router at
  (`manifest.lan_reach`: `MNM_ROUTER_HOST`'s exposure), with the key required; on a client
  install (`MNM_ROUTER_URL` set) the box's router at that URL, which may be a Tailscale name.

Without `--from` every card names the place its client sits as it always did (Odysseus and
LibreChat in a container, OpenCode and the editors on this machine), unless this is a client
install: then every URL is the box's. `Seen.origin` is the one place a door's origin is
decided; `consumers.origin` asks it, so the chat clients', agents' and apps' cards all follow
(`seen_from` sets it for one command), and `Seen.manifest` re-points a manifest the same way
(the apps' files read the manifest's doors)."""

from __future__ import annotations

import urllib.parse
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum

from makenomistakes.config import Settings
from makenomistakes.instance_types import InstanceError

DOCKER_HOST = "host.docker.internal"
LOCAL_NAMES = ("localhost", "127.0.0.1")
KEY_STEP = (
    "Make one first: `mnm setup` (section 3, Network, makes MNM_API_KEY), or add "
    "MNM_API_KEY=<a long random string> to ~/.config/makenomistakes/secrets.env and run "
    "`mnm service install`; the client then sends it as its API key."
)


class From(StrEnum):
    HERE = "here"
    DOCKER = "docker"
    LAN = "lan"


class ReachError(InstanceError):
    """A place mnm will not print a URL for, worded with what to do instead."""


@dataclass(frozen=True)
class Seen:
    """How the router's doors look from one place: a host swapped for another, every door at
    one host (the LAN address), or every door at one origin (`MNM_ROUTER_URL`)."""

    where: From
    hosts: Mapping[str, str] = field(default_factory=dict)
    every: str | None = None
    fixed: str | None = None

    def origin(self, host: str, port: int) -> str:
        if self.fixed is not None:
            return self.fixed
        return f"http://{self.every or self.hosts.get(host, host)}:{port}"

    def url(self, url: str) -> str:
        """A URL of the router's (its scheme, host and port) as seen from here."""
        parts = urllib.parse.urlsplit(url)
        if not parts.hostname or parts.port is None:
            return url
        return self.origin(parts.hostname, parts.port) + parts.path

    def urls(self, found: list[str]) -> list[str]:
        return list(dict.fromkeys(self.url(u) for u in found))


HERE = Seen(From.HERE, hosts={DOCKER_HOST: "localhost"})
DOCKER = Seen(From.DOCKER, hosts=dict.fromkeys(LOCAL_NAMES, DOCKER_HOST))
_current: ContextVar[Seen | None] = ContextVar("mnm_seen_from", default=None)


def current() -> Seen | None:
    return _current.get()


@contextmanager
def seen_from(seen: Seen | None) -> Iterator[None]:
    """Every origin `consumers.origin` builds inside it, as `seen` names it."""
    token = _current.set(seen)
    try:
        yield
    finally:
        _current.reset(token)


def client_origin(settings: Settings) -> str | None:
    """`MNM_ROUTER_URL`'s origin on a client install; None on the box."""
    return settings.router_url or None


def lan(hosts: tuple[str, ...], note: str, keyed: bool, need_key: bool = True) -> Seen:
    """Another machine's view of the box's router; ReachError when no address is on the LAN,
    or no key is set: a card never prints a keyless LAN URL (`need_key=False`: the doctor,
    which asks there to show the router answering keyless)."""
    if not hosts:
        raise ReachError(
            f"No address of this machine's router is on your network: {note} `mnm setup` "
            "(section 3) lets every machine on your network in, with a key; Tailscale is the "
            "other way (GUIDE 13.4)."
        )
    if not keyed and need_key:
        raise ReachError(
            "The router's key is required off this machine, and none is set, so no card "
            f"prints a LAN URL. {KEY_STEP}"
        )
    return Seen(From.LAN, every=hosts[0])


def remote(origin: str, keyed: bool, need_key: bool = True) -> Seen:
    """A client install's view: every door at `MNM_ROUTER_URL`."""
    if not keyed and need_key:
        raise ReachError(
            f"MNM_ROUTER_URL is {origin}, another machine's router, and this machine's secrets "
            "file has no MNM_API_KEY: put the box's key there (the router asks for it off "
            "its own machine)."
        )
    return Seen(From.LAN, fixed=origin)


def refuse_on_client(where: From, origin: str) -> None:
    raise ReachError(
        f"MNM_ROUTER_URL is set ({origin}): this is a client of another machine's router, "
        f"where `--from {where.value}` means nothing. Leave `--from` out (or `--from lan`)."
    )


LanHosts = Callable[[], tuple[tuple[str, ...], str]]  # the LAN addresses and why (none)


def resolve(
    where: From | None, settings: Settings, lan_hosts: LanHosts, need_key: bool = True
) -> Seen | None:
    """The view a command's `--from` names (None: each card's own place),
    after the client install's rule: with `MNM_ROUTER_URL` set every door is the box's."""
    keyed = settings.api_key_value() is not None
    origin = client_origin(settings)
    if origin is not None:
        if where not in (None, From.LAN):
            refuse_on_client(where, origin)
        return remote(origin, keyed, need_key)
    if where is None:
        return None
    if where is From.HERE:
        return HERE
    if where is From.DOCKER:
        return DOCKER
    hosts, note = lan_hosts()
    return lan(hosts, note, keyed, need_key)
