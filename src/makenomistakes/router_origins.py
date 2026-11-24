"""Requests from web pages on other sites, a mixin of `router_http._Handler`. The `Host`
allow-list (`router_hosts`) stops DNS rebinding, not the plain case: any page the user opens
may POST to `http://localhost:7801` (a `text/plain` body needs no preflight, and the router
reads the JSON whatever the type), which would load and drive models.

A keyless router refuses every request but a look (GET, HEAD) whose `Origin` is not on the
names it answers to (`HostList.allows_origin`: localhost, 127.0.0.1, `[::1]`,
`host.docker.internal`, its own addresses, `MNM_ROUTER_HOSTS`): a worded 403, one journal
line per peer and origin. A request without an `Origin` (curl, the SDKs, Odysseus's server,
the apps' containers) is not a web page's and passes; one that carries the key passes
whatever its origin, since the key is the protection. `/mcp` keeps its own, older check.

CORS: every `Access-Control-Allow-Origin` the router sends, its own or an engine's passed
on, is decided here (`send_header`): the page's own origin when it may use the router, `*`
when no page asked, and none for a page that may not (so it cannot read a look's answer
either). No `Access-Control-Allow-Credentials` ever goes out: the key is a header, not a
cookie."""

from __future__ import annotations

from typing import TYPE_CHECKING

from makenomistakes.forward import error_body
from makenomistakes.journal import shown
from makenomistakes.ollama_http import OllamaRoutes

if TYPE_CHECKING:
    from makenomistakes.router_http import RouterServer

LOOKS = ("GET", "HEAD")  # never load or change anything (`router_refusals.not_loadable`)
ALLOW_ORIGIN = "access-control-allow-origin"
CREDENTIALS = "access-control-allow-credentials"
DROP_LIMIT = 16 << 20  # a refused body read and dropped before the 401 or 403, at most


class CrossSite(OllamaRoutes):
    server: RouterServer
    journaled: bool

    def _keyless(self) -> bool:
        return self.server.router.instances.settings.api_key_value() is None

    def _page_may_not(self) -> bool:
        """A web page on another site, and no key stands in for it."""
        origin = self.headers.get("Origin")
        return (
            origin is not None and not self.server.hosts.allows_origin(origin) and self._keyless()
        )

    def _cross_site(self) -> bool:
        """The 403 for a page on another site asking a keyless router for work; True when
        refused. `/mcp` refuses such a page itself (`McpRoutes._mcp`)."""
        if self.command in LOOKS or not self._page_may_not():
            return False
        origin = self.headers.get("Origin") or ""
        self._drop_body()
        seen = (self.client_address[0], origin)
        if seen in self.server.cross_site:
            self.server.counts.refused(403)
        else:
            self.server.cross_site.add(seen)
            self._refusal_note(403, f"Origin {shown(origin)}: a web page on another site")
        self.journaled = True
        self.json_reply(403, error_body(self.server.hosts.origin_refusal(origin)), close=True)
        return True

    def send_header(self, keyword: str, value: str) -> None:
        """Every CORS answer through one rule (the module's docstring)."""
        if keyword.lower() == CREDENTIALS:
            return
        if keyword.lower() == ALLOW_ORIGIN:
            origin = self.headers.get("Origin")
            if origin is None:
                value = "*"
            elif self._page_may_not():
                return
            else:
                value = origin
                super().send_header("Vary", "Origin")
        super().send_header(keyword, value)

    def _drop_body(self) -> None:
        """Read a refused request's body before the 401 or 403 (`drain_body`), up to
        DROP_LIMIT bytes; the connection closes anyway."""
        self.drain_body(DROP_LIMIT)
