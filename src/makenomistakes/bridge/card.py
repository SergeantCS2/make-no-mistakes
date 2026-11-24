"""The card for adding mnm's runs to Odysseus (Settings → Integrations → MCP Tool Server):
`mnm mcp` prints it, the web's System page shows it. The key is named, never printed.

What Odysseus 0.20.19 offers, from its source: the integration form "Add MCP Server" takes a
name, a transport and a URL (static/js/settings.js:4750-4795, admin only: routes/mcp/
mcp_routes.py:170-182); a Streamable HTTP server is reached with OAuth only, no custom header
(src/mcp_manager.py:368-386). So Odysseus can never send `MNM_API_KEY`: with a key set
the router refuses it, and keyless `/mcp` answers this machine and its Docker containers only
(`bridge/peers.py`), Odysseus's container among them."""

from __future__ import annotations

from makenomistakes.bridge.protocol import PATH
from makenomistakes.client_view import From, current
from makenomistakes.config import Settings
from makenomistakes.consumers import ODYSSEUS_HOST, router_origin


def mcp_card(settings: Settings) -> list[str]:
    """What to enter in Odysseus's MCP Tool Server form; the key is named, never printed.
    For an Odysseus on another machine (`mnm url --for odysseus --from lan`) only why
    it cannot: the key is required there and its form cannot send one."""
    seen = current()
    if seen is not None and seen.where is From.LAN:
        return [
            "Odysseus on another machine cannot use mnm's MCP tools: the router asks every "
            "request from there for MNM_API_KEY, and Odysseus's MCP form cannot send one. Its "
            "chat endpoints above work (each takes the key as its API key).",
        ]
    lines = [
        "Odysseus → Settings → Integrations → Add Integration → MCP Tool Server (an admin):",
        "  Name:      mnm",
        "  Transport: Streamable HTTP",
        f"  URL:       {router_origin(ODYSSEUS_HOST)}{PATH}",
        f"  (from this machine: {router_origin()}{PATH})",
        "Save. Odysseus's form has no header or key field, so it sends no key.",
    ]
    keyed = settings.api_key_value() is not None
    if keyed:
        lines.append(
            "MNM_API_KEY is set: the router asks every request for it, so Odysseus cannot reach "
            "/mcp while it is. Other MCP clients send `Authorization: Bearer <your MNM_API_KEY>` "
            "(the key is in ~/.config/makenomistakes/secrets.env)."
        )
    else:
        lines.append(
            "No key is set: /mcp answers this machine and its Docker containers (Odysseus's "
            "among them) and refuses every other machine."
        )
    lines.append(
        "Then a chat can run a workflow or make a short film and ask run_status for the files."
    )
    lines.append(
        'If Odysseus shows the integration as "error" (it never says why), the router\'s '
        "journal has one line per /mcp call with its answer: `journalctl --user -u "
        "makenomistakes-router -n 100 | grep /mcp`."
    )
    lines.append(
        "Odysseus on another machine cannot use it (it would need the key it cannot send). "
        "Another MCP client there: this machine's LAN address in the URL above, "
        f"MNM_API_KEY{'' if keyed else ' (set one first)'} sent as `Authorization: Bearer <key>`, "
        "and MNM_WEB_URL "
        "(where that machine opens mnm's web UI) for run_status's links"
        + ("" if not settings.web_url else f" (now {settings.web_url})")
        + "."
    )
    return lines
