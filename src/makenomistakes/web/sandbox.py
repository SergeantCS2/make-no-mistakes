"""The web UI keeps ComfyUI's container out, as the router does. On 127.0.0.1, the
service's bind, the container never reaches it; `mnm serve --host 0.0.0.0` puts it at the
gateway of ComfyUI's network, where a script sets any Host header and no Sec-Fetch-Site, so
neither TrustedHost nor the CSRF check would stop it approving packs or starting runs. A pure
ASGI middleware added last, so it runs first: before every other middleware (no rendered error
page, no body read), for HTTP and websockets alike."""

from __future__ import annotations

from starlette.types import ASGIApp, Receive, Scope, Send

from makenomistakes.comfy.guard import SANDBOXED, ComfyGuard


class SandboxGuard:
    def __init__(self, app: ASGIApp, guard: ComfyGuard) -> None:
        self.app = app
        self.guard = guard

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            client = scope.get("client")
            if self.guard.blocks(client[0] if client else None):
                await self._refuse(scope, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(scope: Scope, send: Send) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})  # before accept: a 403
            return
        body = SANDBOXED.encode()
        headers = [(b"content-type", b"text/plain; charset=utf-8")]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({"type": "http.response.start", "status": 403, "headers": headers})
        await send({"type": "http.response.body", "body": body})
