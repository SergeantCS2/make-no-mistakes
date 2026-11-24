"""mnm's runs as MCP tools, for Odysseus's MCP client (or any other)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from makenomistakes.bridge.tools import MnmTools
    from makenomistakes.instances import InstanceService


def bridge_tools(service: InstanceService) -> MnmTools:
    """The tools for this box: workflows through ComfyUI, the film through the router."""
    from makenomistakes.bridge.tools import MnmTools  # noqa: PLC0415 - the router imports this
    from makenomistakes.consumers import web_origin  # noqa: PLC0415
    from makenomistakes.pipeline.runner import comfy_runner_for, film_runner  # noqa: PLC0415

    comfy = comfy_runner_for(service)
    origin = web_origin(service.settings.port, service.settings.web_url)
    return MnmTools(comfy, film_runner(service, comfy), comfy.runs, origin)
