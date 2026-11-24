"""Forwarding one HTTP request to an engine and its reply back: what the router
does for every instance.

Standard library only: the HTTP side of the router, kept apart from the services
it calls, so it can be read and tested on its own.

- **Rewrites.** A request's body passes as it came, except for three POSTs. `size` is
  dropped from an image generation (sd-server obeys it over the instance's own `-W/-H`) and
  from an edit's multipart form. An edit's picture is kept from the text encoder where it
  has no vision (`without_references`). `speed` is dropped from a speech request when the
  voice refuses runtime style controls (PocketTTS answers 500 to any `speed`).
- **Framing.** The forwarder frames every message itself: hop-by-hop fields, and any field
  the `Connection` header names, never pass. A reply without a length (server-sent events,
  a chunked body) is passed on chunk by chunk as it arrives.
"""

from __future__ import annotations

import http.client
import json
import re
import socketserver
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from http.client import HTTPResponse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from makenomistakes.loading_notes import CHAT_COMPLETIONS, without_notes

GENERATIONS = "/v1/images/generations"
EDITS = "/v1/images/edits"  # multipart: an image, a prompt, and a `size` the engine would obey
SPEECH = "/v1/audio/speech"  # JSON: `speed` (or `speaking_rate`) some voices refuse
# A request may wait for a whole image, then for the next one in the engine's queue.
UPSTREAM_TIMEOUT_SECONDS = 3600.0
STREAM_CHUNK = 65536
REPLY_TAIL = 4096  # the end of a reply the router's meter reads (llama.cpp's timings)
# Hop-by-hop fields (RFC 9110 7.6.1) never pass a proxy; the forwarder frames each message
# itself (Content-Length) and http.client names the upstream host, so the client's go too.
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_REFRAMED = frozenset({"content-length", "host"})


def dropped(headers: Message) -> frozenset[str]:
    """The fields not to forward: hop-by-hop ones, and any the `Connection` field names."""
    named = {
        token.strip().lower()
        for value in headers.get_all("Connection") or []
        for token in value.split(",")
    }
    return _HOP_BY_HOP | _REFRAMED | named


# What the router reads of a request's body before it refuses it (413), unread: JSON with
# pictures in base64 (Ollama's API) fits in 16 MiB; the OpenAI routes also carry uploads
# (a picture to edit, a clip to transcribe: OpenAI's own limit for audio is 25 MB), so 64 MiB.
JSON_BODY_LIMIT = 16 << 20
UPLOAD_BODY_LIMIT = 64 << 20
# What the router reads and drops of a body it refuses (413) before it answers: a client
# still sending when the connection closes may see a reset instead of the answer.
DRAIN_LIMIT = 4 << 20
DRAIN_SECONDS = 2.0  # of silence: a client holding its body back is answered, not waited for


class TooLarge(ValueError):  # noqa: N818 - read as "the body is too large"
    """A body over the reader's limit, refused before it is read."""

    def __init__(self, limit: int) -> None:
        size = f"{limit >> 20} MiB" if limit >= 1 << 20 else f"{limit >> 10} KiB"
        super().__init__(f"the body is over {size}")
        self.limit = limit


@dataclass(frozen=True)
class Rewrites:
    """What the router changes in one instance's requests (`router.rewrites` reads it from
    the recipe); the defaults change nothing."""

    keep_size: bool = True  # a client's image `size` passes on (`client-size=on`)
    drop_speed: bool = False  # a voice that refuses any `speed` (`AudioCppArgs.drops_speed`)
    drop_references: bool = False  # an edit's picture is the starting image only


def rewrite(path: str, body: bytes, rules: Rewrites, content_type: str = "") -> bytes:
    """The body sent on: an image generation request without its `size` (anything that is
    not one, or not JSON, passes as it came), an edit request without its `size` part
    (then the engine answers at the picture's own size) and, with `drop_references`, with
    its picture as the starting image only; a speech request without its `speed` when the
    voice refuses one; a chat request without the router's loading notes in its history."""
    route = path.split("?", 1)[0].rstrip("/")
    if route == CHAT_COMPLETIONS:
        return without_notes(body)  # the router's own loading notes, sent back by a client
    if route == SPEECH:
        return without_fields(body, ("speed", "speaking_rate")) if rules.drop_speed else body
    if route == EDITS:
        if rules.drop_references:
            body = without_references(body, content_type)
        return body if rules.keep_size else drop_part(body, content_type, "size")
    if rules.keep_size or route != GENERATIONS:
        return body
    return without_fields(body, ("size",))


def with_model(body: bytes, asked: str, served: str) -> bytes:
    """A JSON object body whose `model` is `asked` names `served` instead; anything else
    passes as it came. vLLM answers 404 to any id but its own (clients name the
    instance)."""
    if asked == served or asked.encode() not in body:
        return body
    try:
        request = json.loads(body)
    except ValueError:
        return body
    if not isinstance(request, dict) or request.get("model") != asked:
        return body
    request["model"] = served
    return _dump(request)


def without_fields(body: bytes, fields: tuple[str, ...]) -> bytes:
    """A JSON object body minus `fields`; anything else passes as it came."""
    try:
        request = json.loads(body)
    except ValueError:
        return body
    if not isinstance(request, dict) or not any(f in request for f in fields):
        return body
    for field in fields:
        request.pop(field, None)
    return _dump(request)


def _dump(request: dict[str, object]) -> bytes:
    """A rewritten JSON body: non-ASCII as UTF-8 (a CJK prompt keeps its size), escaped only
    when it cannot be (a lone surrogate, which JSON's `\\ud800` allows)."""
    try:
        return json.dumps(request, ensure_ascii=False).encode()
    except UnicodeEncodeError:
        return json.dumps(request).encode()


_BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.I)
_NAME_RE = re.compile(rb'(?<![a-z])name="([^"]*)"', re.I)  # not `filename="…"`


def _multipart(body: bytes, content_type: str) -> tuple[bytes, list[bytes]] | None:
    """The delimiter and the parts of a multipart body, framed the way RFC 2046 frames it
    (the preamble and the closing piece are the first and last), or None."""
    boundary = _BOUNDARY_RE.search(content_type or "")
    if "multipart/form-data" not in (content_type or "").lower() or boundary is None:
        return None
    delimiter = b"--" + boundary.group(1).encode()
    pieces = body.split(delimiter)
    return (delimiter, pieces) if len(pieces) >= 3 else None


def _named(piece: bytes, name: str, exact: bool = False) -> bool:
    """Whether a part is the field `name` (each part is CRLF, headers, CRLF CRLF, content,
    CRLF): spelled exactly as sd-server's form reader looks it up, or in any case."""
    found = _NAME_RE.search(piece.split(b"\r\n\r\n", 1)[0])
    if found is None:
        return False
    sent = found.group(1).decode("latin-1")
    return sent == name if exact else sent.lower() == name


def drop_part(body: bytes, content_type: str, name: str) -> bytes:
    """A multipart body without the field `name`; as it came when it is not multipart, has
    no such field, or is not framed the way RFC 2046 frames it (then the engine judges)."""
    framed = _multipart(body, content_type)
    if framed is None:
        return body
    delimiter, pieces = framed
    kept = [pieces[0], *(p for p in pieces[1:-1] if not _named(p, name))]
    if len(kept) == 1:  # every part was the field: an empty multipart would be malformed
        return body
    return delimiter.join([*kept, pieces[-1]])


def _text_part(pieces: list[bytes], name: str, exact: bool = False) -> int | None:
    """The index of the first text field `name` among a multipart body's pieces, or None."""
    for index, piece in enumerate(pieces[1:-1], start=1):
        headers = piece.split(b"\r\n\r\n", 1)[0]
        if b"filename=" in headers.lower():
            continue  # an uploaded file, not a text field, whatever its name
        if _named(piece, name, exact) and b"\r\n\r\n" in piece:
            return index
    return None


def form_field(body: bytes, content_type: str, name: str) -> str | None:
    """A multipart text field's value (a transcription's or an edit's `model`), or None."""
    framed = _multipart(body, content_type)
    index = _text_part(framed[1], name) if framed is not None else None
    if framed is None or index is None:
        return None
    value = framed[1][index].split(b"\r\n\r\n", 1)[1].removesuffix(b"\r\n")
    return value.decode("utf-8", errors="replace").strip()


# sd-server reads the first tag's JSON over the request (one line: its `.` stops at a line
# break) and removes every tag from the prompt.
_EXTRA_ARGS_RE = re.compile(rb"<sd_cpp_extra_args>([^\r\n]*?)</sd_cpp_extra_args>")
NO_REFERENCES: dict[str, object] = {"ref_images": []}


def with_extra_args(prompt: bytes, extra: dict[str, object]) -> bytes:
    """`prompt`, as sent and never decoded, with `extra` in the `<sd_cpp_extra_args>` tag
    sd-server reads (the first): merged over the same keys into a JSON object; put in front
    of a tag whose JSON is not an object (sd-server takes `null` or `[]` and ignores it);
    appended when there is none. A tag that is not JSON is left as it is: the engine answers
    it 400 ("invalid sd_cpp_extra_args") before it generates anything."""
    found = _EXTRA_ARGS_RE.search(prompt)
    if found is None:
        return prompt + _extra_args_tag(extra)
    try:
        args = json.loads(found.group(1))
    except ValueError:  # not JSON (or not UTF-8): the engine's 400
        return prompt
    if not isinstance(args, dict):
        return prompt[: found.start()] + _extra_args_tag(extra) + prompt[found.start() :]
    return prompt[: found.start()] + _extra_args_tag({**args, **extra}) + prompt[found.end() :]


def _extra_args_tag(args: dict[str, object]) -> bytes:
    """One tag; ASCII JSON (`\\u` escapes), so no prompt is decoded and encoded again."""
    return b"<sd_cpp_extra_args>" + json.dumps(args).encode() + b"</sd_cpp_extra_args>"


def without_references(body: bytes, content_type: str) -> bytes:
    """An edit request whose picture is only the starting image. sd-server also makes the
    first picture a reference, which Z-Image's, Qwen-Image's and Krea 2's presets hand to
    the text encoder as vision input: with the text-only encoder their kits ship, the
    request segfaulted. `"ref_images": []` in the prompt's extra arguments
    clears the references after the pictures are read, leaving a plain repaint. A body that
    is not multipart, or has no prompt or an empty one (sd-server answers those 400 before
    it reads a picture), passes as it came."""
    framed = _multipart(body, content_type)
    if framed is None:
        return body
    delimiter, pieces = framed
    index = _text_part(pieces, "prompt", exact=True)  # the field sd-server reads
    if index is None:
        return body
    head, _, rest = pieces[index].partition(b"\r\n\r\n")
    value = rest.removesuffix(b"\r\n")
    if not value:
        return body
    pieces[index] = head + b"\r\n\r\n" + with_extra_args(value, NO_REFERENCES) + rest[len(value) :]
    return delimiter.join(pieces)


def error_body(message: str) -> dict[str, dict[str, str]]:
    """The error shape OpenAI clients read; `router_client.error_message` reads it back."""
    return {"error": {"message": message}}


def bodiless(status: int) -> bool:
    """1xx, 204 and 304 carry no body and no framing (RFC 9112 6.3)."""
    return status < 200 or status in (204, 304)


def streamed(answer: HTTPResponse) -> bool:
    """A reply whose end only the connection knows: server-sent events, or chunked."""
    if bodiless(answer.status):
        return False
    kind = (answer.getheader("Content-Type") or "").lower()
    return "text/event-stream" in kind or answer.getheader("Content-Length") is None


class QuickBindServer(ThreadingHTTPServer):
    """A threaded server that binds without HTTPServer's `getfqdn` lookup: a slow resolver
    could hold a unit's READY past its start timeout, and nothing here uses `server_name`."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = str(host), int(port)


class Forwarder(BaseHTTPRequestHandler):
    """The request side and the reply side of forwarding; subclasses decide where to."""

    protocol_version = "HTTP/1.1"
    responded = False  # a status line went out for this request (reset per request)
    reply_tail = b""  # and the end of its body, for the router's meter

    def send_response_only(self, code: int, message: str | None = None) -> None:
        self.responded = True
        super().send_response_only(code, message)

    def drain_body(self, limit: int = DRAIN_LIMIT) -> None:
        """Read and drop what a refused request declared, up to `limit` bytes, so its answer
        reaches a client that sends the whole body before reading: closing on unread bytes
        makes the kernel reset the connection, and the answer is lost with it. A body larger
        than `limit` (or chunked, whose size is not declared) may still see that reset: the
        connection closes after the answer either way. A client that asked to continue
        (`Expect: 100-continue`) has sent nothing yet, so nothing is read; one that goes quiet
        for DRAIN_SECONDS is answered then."""
        try:
            left = min(int(self.headers.get("Content-Length") or 0), limit)
        except ValueError:
            return
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            return
        if self.headers.get("Expect", "").lower() == "100-continue":
            return
        self.connection.settimeout(DRAIN_SECONDS)
        while left > 0:
            try:
                chunk = self.rfile.read(min(left, 1 << 16))
            except TimeoutError:
                return
            if not chunk:  # the client hung up mid-body
                return
            left -= len(chunk)

    def read_body(self, limit: int | None = None) -> bytes:
        """The request's body: chunked (trailer fields read and dropped) or by length.
        ValueError for a malformed size; `TooLarge` past `limit`, before it is read."""
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            chunks, total = [], 0
            while size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16):
                if size < 0:  # `read(-1)` reads to the end, past any limit
                    raise ValueError(f"negative chunk size {size}")
                total += size
                if limit is not None and total > limit:
                    raise TooLarge(limit)
                chunks.append(self.rfile.read(size))
                self.rfile.readline()
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass  # trailer fields, up to the blank line that ends the message
            return b"".join(chunks)
        length = int(self.headers.get("Content-Length") or 0)
        if length < 0:  # `read(-1)` would wait for the client to hang up
            raise ValueError(f"negative Content-Length {length}")
        if limit is not None and length > limit:
            raise TooLarge(limit)
        return self.rfile.read(length) if length else b""

    def ask(
        self, upstream: http.client.HTTPConnection, body: bytes, path: str | None = None
    ) -> HTTPResponse:
        """The request as it came (repeated fields kept), minus hop-by-hop fields, to `path`
        (the router's `/upstream/<name>/…` without its prefix; None: the request's own)."""
        upstream.putrequest(
            self.command, self.path if path is None else path, skip_accept_encoding=True
        )
        skip = dropped(self.headers)
        for key, value in self.headers.items():
            if key.lower() not in skip:
                upstream.putheader(key, value)
        if body or self.command in ("POST", "PUT", "PATCH"):
            upstream.putheader("Content-Length", str(len(body)))
        upstream.endheaders(body or None)
        return upstream.getresponse()

    def pass_on(
        self, answer: HTTPResponse, edit: Callable[[int, bytes], bytes] | None = None
    ) -> bool:
        """The engine's reply to the client: streamed when it has no length, else whole
        (`edit`ed first: the router's words on a context error). False when the engine cut
        it before it was read (nothing has been sent yet)."""
        skip = dropped(answer.msg)
        fields = [(k, v) for k, v in answer.getheaders() if k.lower() not in skip]
        self.reply_tail = b""
        if self.command != "HEAD" and streamed(answer):
            self._stream(answer, fields)
            return True
        try:
            data = answer.read()
        except (OSError, http.client.HTTPException):  # cut mid-reply
            return False
        if edit is not None:
            data = edit(answer.status, data)
        length = answer.getheader("Content-Length") if self.command == "HEAD" else None
        self.send_response_only(answer.status, answer.reason)  # the engine's Date, Server
        for key, value in fields:
            self.send_header(key, value)
        if not bodiless(answer.status):
            self.send_header("Content-Length", length or str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self.reply_tail = data[-REPLY_TAIL:]
        return True

    def _stream(self, answer: HTTPResponse, fields: list[tuple[str, str]]) -> None:
        """A reply without a length (server-sent events, a chunked body): passed on chunk by
        chunk as it arrives, so a chat model's tokens reach the client as they come."""
        chunked = self.open_stream(answer.status, answer.reason, fields)
        if self.stream_body(answer, chunked):
            self.end_stream(chunked)

    def open_stream(self, status: int, reason: str | None, fields: list[tuple[str, str]]) -> bool:
        """The head of a reply whose end only the connection knows; whether it is chunked
        (HTTP/1.0 has no chunks: the close ends it)."""
        chunked = self.request_version != "HTTP/1.0"
        self.send_response_only(status, reason)
        for key, value in fields:
            self.send_header(key, value)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        return chunked

    def write_piece(self, data: bytes, chunked: bool) -> None:
        """One piece of an open stream, sent at once."""
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data) if chunked else data)
        self.wfile.flush()
        self.reply_tail = (self.reply_tail + data)[-REPLY_TAIL:]

    def stream_body(self, answer: HTTPResponse, chunked: bool) -> bool:
        """The engine's body into an open stream as it arrives; False when either side cut
        it (the client then sees the stream end early)."""
        try:
            while chunk := answer.read1(STREAM_CHUNK):
                self.write_piece(chunk, chunked)
        except (OSError, http.client.HTTPException):
            self.close_connection = True
            return False
        return True

    def end_stream(self, chunked: bool) -> None:
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def preflight(self) -> None:
        """A CORS preflight, answered as the engines answer it (llama-server and vLLM allow
        any origin by default)."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods",
            self.headers.get("Access-Control-Request-Method") or "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            self.headers.get("Access-Control-Request-Headers") or "*",
        )
        self.end_headers()  # a 204 carries no Content-Length (RFC 9110 §8.6)

    def json_reply(
        self, status: int, value: object, cors: bool = False, close: bool = False
    ) -> None:
        """`close`: the connection ends after it (`Connection: close`, which also tells the
        server to stop reading it), for an answer sent before the request's body was read."""
        data = json.dumps(value).encode()
        self.send_response(status)
        if cors:  # a browser client reads these answers as it reads the engine's
            self.send_header("Access-Control-Allow-Origin", "*")
        if close:
            self.send_header("Connection", "close")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def text_reply(self, status: int, data: bytes, content_type: str) -> None:
        """A whole reply that is not JSON (the router's `/metrics`)."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Quiet: the engine logs every request already."""
