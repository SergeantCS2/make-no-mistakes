"""Who may call `/mcp` without a key: this machine and its Docker containers.
Odysseus in Docker on this box keeps working with no key; Odysseus on another machine, or
anyone on a LAN the router listens on (a native box on 0.0.0.0, WSL's mirrored networking),
is asked for `MNM_API_KEY` first, since a tool starts GPU work and hands back run outputs.
Docker's networks are read from Docker itself (a minute's cache, read by one request while
the others keep the last answer). When Docker does not answer, no container is let in without
a key: a guessed pool would also hold WSL's port-proxied traffic from Windows. A Docker network
that overlaps the user's LAN lets that LAN in: Docker hands out 192.168.x.0/20 networks once
its 172.17-31 ones are used, and a home LAN is often there. Set MNM_API_KEY on such a box."""

from __future__ import annotations

import ipaddress
import threading
import time
from collections.abc import Callable

from makenomistakes.comfy.guard import Network, unmapped
from makenomistakes.hardware.probe import CommandRunner, SubprocessRunner

REFRESH_SECONDS = 60.0
LIST = ["docker", "network", "ls", "--quiet"]
NEEDS_KEY = (
    "mnm's MCP tools answer another machine only with a key: set MNM_API_KEY in "
    "~/.config/makenomistakes/secrets.env and send it as `Authorization: Bearer <key>`."
)
# Docker did not answer the router (right after setup its user manager may not be in the
# docker group yet): a container on this machine cannot be told from another machine
NO_DOCKER = (
    "mnm's router could not ask Docker for its networks, so it cannot tell a container on "
    "this machine from another machine, and answers neither without a key. `mnm doctor` says "
    "why Docker does not answer (under WSL after setup: `wsl --shutdown`, then `mnm service "
    "install`); or set MNM_API_KEY and send it as `Authorization: Bearer <key>`."
)


class LocalPeers:
    def __init__(
        self,
        runner: CommandRunner | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runner = runner or SubprocessRunner()
        self.clock = clock
        self._networks: tuple[Network, ...] = ()
        self._read_at: float | None = None
        self._lock = threading.Lock()  # the cache
        self._reading = threading.Lock()  # one Docker read at a time
        self._docker = False  # whether the last read had an answer from Docker

    def refusal(self) -> str:
        """Why a keyless call was refused: another machine, or Docker did not answer."""
        with self._lock:
            return NO_DOCKER if self._read_at is not None and not self._docker else NEEDS_KEY

    def allows(self, address: str) -> bool:
        """The peer is this machine (loopback) or one of its containers."""
        try:
            ip = unmapped(ipaddress.ip_address(address.split("%", 1)[0]))
        except ValueError:
            return False
        if ip.is_loopback:
            return True
        return any(ip in network for network in self.networks())

    def networks(self) -> tuple[Network, ...]:
        """The cached networks; a stale cache is read again by one request, never under the
        cache's lock, while any other keeps the last answer (a hung Docker stalls one)."""
        with self._lock:
            if self._read_at is not None and self.clock() - self._read_at <= REFRESH_SECONDS:
                return self._networks
            known = self._read_at is not None
        if not self._reading.acquire(blocking=not known):
            with self._lock:
                return self._networks
        try:
            with self._lock:  # another request may have read it while this one waited
                if self._read_at is not None and self.clock() - self._read_at <= REFRESH_SECONDS:
                    return self._networks
            found, answered = self._read()
            with self._lock:
                self._networks, self._read_at, self._docker = found, self.clock(), answered
                return found
        finally:
            self._reading.release()

    def _read(self) -> tuple[tuple[Network, ...], bool]:
        """Docker's subnets, and whether Docker answered at all."""
        code, out, _err = self.runner.run(LIST, timeout=10)
        if code != 0:
            return (), False
        ids = out.split()
        if not ids:
            return (), True
        template = "{{range .IPAM.Config}}{{.Subnet}} {{end}}"
        code, out, _err = self.runner.run(
            ["docker", "network", "inspect", "--format", template, *ids], timeout=10
        )
        found = []
        for word in out.split() if code == 0 else []:
            try:
                found.append(ipaddress.ip_network(word, strict=False))
            except ValueError:
                continue
        return tuple(found), code == 0
