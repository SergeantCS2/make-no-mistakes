"""The router's MCP endpoint and the key checks it shares with every route: a mixin
of `router_http._Handler`."""

from __future__ import annotations

import hmac
import urllib.parse
from typing import TYPE_CHECKING

from makenomistakes.bridge.protocol import PATH as MCP_PATH
from makenomistakes.bridge.protocol import handle
from makenomistakes.forward import TooLarge, error_body
from makenomistakes.journal import error_message, mcp_method
from makenomistakes.router_origins import CrossSite

if TYPE_CHECKING:
    from makenomistakes.router_http import RouterServer

LOCAL_ORIGINS = frozenset({"localhost", "127.0.0.1", "::1"})  # a page on this machine
MCP_BODY_LIMIT = 1 << 20  # JSON-RPC with a tool's text arguments; no file goes this way


class McpRoutes(CrossSite):
    """`/mcp`, and the key."""

    server: RouterServer

    def _key_given(self) -> bool:
        """With `MNM_API_KEY` set, the request carries it: `Authorization: Bearer <key>`, or
        `x-api-key` (Anthropic clients). Read per request: a new key applies at once."""
        key = self.server.router.instances.settings.api_key_value()
        if key is None:
            return True
        bearer = self.headers.get("Authorization", "")
        sent = bearer[7:] if bearer[:7].lower() == "bearer " else self.headers.get("x-api-key", "")
        return hmac.compare_digest(sent.strip().encode(), key.encode())

    def _mcp(self) -> None:
        """The MCP endpoint: JSON-RPC by POST, answered as JSON. A request from a web
        page on another site is refused (a rebound name would otherwise reach the tools)."""
        tools = self.server.tools
        origin = self.headers.get("Origin")
        if tools is None:
            self._drop_body()
            self._mcp_note(404, "no MCP tools in this router")
            self.json_reply(404, error_body("mnm's MCP server is not on in this router."))
        elif origin and urllib.parse.urlparse(origin).hostname not in LOCAL_ORIGINS:
            self._drop_body()
            self._mcp_note(403, f"a web page's origin, {origin!r}")
            self.json_reply(403, error_body(f"MCP requests from {origin} are refused."), close=True)
        elif self.command != "POST":
            self._drop_body()
            self._mcp_note(405, "no server stream (the client goes on without one)")
            self.json_reply(405, error_body("mnm's MCP server takes POST only (no streams)."))
        elif self._keyless() and not self.server.peers.allows(self.client_address[0]):
            self._drop_body()
            refusal = self.server.peers.refusal()
            self._mcp_note(403, refusal)
            self.json_reply(403, error_body(refusal), close=True)
        else:
            try:
                body = self.read_body(MCP_BODY_LIMIT)
            except TooLarge as exc:  # a tool call is a few hundred bytes of JSON
                self.drain_body()  # a few MiB, so the 413 reaches the client
                self._mcp_note(413, str(exc))
                self.json_reply(413, error_body(f"MCP request refused: {exc}."), close=True)
                return
            status, answer = handle(tools, body)
            why = error_message(answer) if status >= 400 else ""  # a 400 says why
            self._mcp_note(status, why, mcp_method(body))  # before the reply, as above
            if answer is None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.json_reply(status, answer)

    def _is_mcp(self) -> bool:
        return self.path.split("?", 1)[0].rstrip("/") == MCP_PATH

    def _mcp_note(self, status: int, why: str = "", method: str = "") -> None:
        """`/mcp POST tools/list from 172.18.0.2: 200`, or the refusal's reason."""
        asked = f" {method}" if method else ""
        said = f" ({why})" if why else ""
        peer = self.client_address[0]
        line = f"/mcp {self.command}{asked} from {peer}: {status}{said}"
        if status < 400:
            self.server.note(line)
        else:
            self.server.counts.refused(status)  # with the journal's line, as `_refusal_note`
            self.server.quiet.refusal(line, peer, status, "/mcp")
