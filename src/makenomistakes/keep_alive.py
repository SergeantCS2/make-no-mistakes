"""A request's `keep_alive` on the router's OpenAI routes, as its Ollama API reads it
(`ollama_api.keep_alive_seconds`, one meaning for both): a chat, completion or embedding
request's JSON body field, or the `X-Mnm-Keep-Alive` header, which wins.

- `0`: the model is parked once the reply is done, unless the router keeps it (pinned, a
  member of a set in use, other replies in flight: `Router.park` says which);
- a negative number (`-1`): no idle unload for it from now on;
- a duration (`5m`, `2h`, `1h30m`, or seconds): its idle limit from now on.

Held in the router's memory (`IdleClock.keep`), never saved: a router restart goes by the
recipe again. The field is taken out before the engine sees the request.
"""

from __future__ import annotations

import json
from email.message import Message

from makenomistakes.forward import without_fields
from makenomistakes.ollama_api import KeepAliveError, keep_alive_seconds

HEADER = "X-Mnm-Keep-Alive"
FIELD = "keep_alive"
PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
# POST here: the user saved this instance's idle or keep; its client keep_alive goes
SAVED_PATH = "/__mnm/saved/"

__all__ = ["FIELD", "HEADER", "PATHS", "SAVED_PATH", "KeepAliveError", "asked"]


def asked(path: str, body: bytes, headers: Message) -> tuple[int | None, bytes]:
    """What this request asks, in seconds (`keep_alive_seconds`: -1 forever, 0 park), and
    the body without the field. None for a request of another kind, or one that asks
    nothing. KeepAliveError for a value that is not a duration."""
    if path.split("?", 1)[0].rstrip("/") not in PATHS:
        return None, body
    found: object = headers.get(HEADER)
    if FIELD.encode() in body:
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        if isinstance(request, dict) and FIELD in request:
            found = found if found is not None else request[FIELD]
            body = without_fields(body, (FIELD,))
    if found is None:
        return None, body
    if not isinstance(found, str | int | float):
        raise KeepAliveError(f"keep_alive {found!r} is not a duration")
    return keep_alive_seconds(found), body
