"""What every route module shares: the template renderer and the back-link check."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

Render = Callable[..., HTMLResponse]


class Pages:
    """Renders templates; the only place that knows how a template becomes a response."""

    def __init__(self, templates: Jinja2Templates) -> None:
        self._templates = templates

    def render(
        self, request: Request, name: str, status_code: int = 200, **ctx: object
    ) -> HTMLResponse:
        return self._templates.TemplateResponse(request, name, ctx, status_code=status_code)


def local(path: str, default: str) -> str:
    """Where a form asked to go back to, if it is a page of this app (never another site)."""
    clean = all(ord(c) >= 32 for c in path)  # a tab or newline a browser would drop
    if clean and path.startswith("/") and not path.startswith("//") and "\\" not in path:
        return path
    return default
