"""`mnm mcp [tools]`: the card for adding mnm's runs to Odysseus (Settings →
Integrations), and the tools the router's MCP endpoint offers right now."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.markup import escape

from makenomistakes.bridge import bridge_tools
from makenomistakes.bridge.card import mcp_card
from makenomistakes.cli._app import app, console
from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService

mcp_app = typer.Typer(
    help="mnm's workflows and pipelines as MCP tools, for Odysseus (or any MCP client).",
    invoke_without_command=True,
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.callback()
def mcp_cmd(ctx: typer.Context) -> None:
    """The card for Odysseus's MCP Tool Server integration (the router serves it at /mcp)."""
    if ctx.invoked_subcommand is None:
        for line in mcp_card(Settings()):
            console.print(escape(line))


@mcp_app.command("tools")
def tools_cmd(
    schema: Annotated[bool, typer.Option("--schema", help="Each tool's arguments too.")] = False,
) -> None:
    """The tools the router offers now: one per workflow that runs here, the film, run_status."""
    for tool in bridge_tools(InstanceService(Settings())).tools():
        console.print(f"[bold]{escape(tool.name)}[/bold]: {escape(tool.description)}")
        if schema:
            for name, prop in tool.input_schema.get("properties", {}).items():
                console.print(f"    {escape(name)} ({escape(str(prop.get('type')))})")
