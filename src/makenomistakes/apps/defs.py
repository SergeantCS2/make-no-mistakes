"""The workspace apps `mnm apps` installs, each pinned: what an app is, how
it runs, and the exact build of it this mnm was tested against. A newer pin arrives only with
a newer mnm; `mnm apps update` moves to it and keeps the one before for `rollback`. Never a
moving tag: an image is named by its tag *and* the digest it had when it was read, a binary by
its release tag and the sha256 of its asset.

Read on 2026-10-22/23:
- OpenCode 1.19.4, the 1.x line (opencode.ai/docs, github.com/anomalyco/opencode/releases,
  tag v1.19.4 of 2026-10-22): `opencode-linux-x64.tar.gz` and, for a CPU without AVX2,
  `opencode-linux-x64-baseline.tar.gz` (the choice its own installer, opencode.ai/install,
  makes from /proc/cpuinfo); each holds one `opencode` executable. Both hashes were taken
  from the assets downloaded on 2026-10-23. 2.x replaced `serve --port --hostname` and the
  provider schema: not until its docs call it stable.
- LibreChat v0.8.7 (github.com/danny-avila/LibreChat, its docker-compose.yml at that tag,
  MIT): its own image from ghcr.io (the compose file names `…-dev:latest` only), the
  third-party images at the tags that compose file names, and the RAG API's lite image at
  its newest release tag (v0.9.0, 2026-07-31; v0.8.7's compose pulls `latest`). The digests
  are the multi-platform index each tag pointed at on 2026-10-23 (`docker buildx imagetools
  inspect`).
- OpenHands Agent Canvas 1.31.0 (github.com/OpenHands/OpenHands, release v1.31.0 of
  2026-10-22, MIT): `ghcr.io/openhands/agent-canvas:1.31.0`, the all-in-one image (agent
  server, automation, the web front end at /canvas on 8000); never the Docker socket.
- Open WebUI v0.12.3 (github.com/open-webui/open-webui, tag v0.12.3 of 2026-10-16, commit
  91ea4e3; the Open WebUI License: BSD-3 with a branding clause): the plain image
  `ghcr.io/open-webui/open-webui:v0.12.3` (not `:cuda` or `:ollama`: mnm's router serves the
  models), its index digest read with `docker buildx imagetools inspect` on 2026-10-23; port
  8080 in the container, 3000 on 127.0.0.1 as its README publishes it. Its settings are read
  from `backend/open_webui/config.py` and `env.py` at that tag and docs.openwebui.com/reference/
  env-configuration (read 2026-10-23, "up to date with v0.12.1"): `apps/open_webui.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes.apps.names import LABELS, App
from makenomistakes.consumers import origin
from makenomistakes.router_bind import LOOPBACK

OPENCODE_RELEASES = "https://github.com/anomalyco/opencode/releases/download"


class Kind(StrEnum):
    BINARY = "binary"  # a release binary run by a systemd user unit (OpenCode)
    COMPOSE = "compose"  # mnm's own compose project from pinned images (LibreChat, OpenHands,
    # Open WebUI)
    CONNECT = "connect"  # never installed: mnm writes its files and prints its card


class Asset(BaseModel):
    model_config = ConfigDict(frozen=True)

    target: Literal["x64", "x64-baseline", "arm64"]
    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BinaryPin(BaseModel):
    """A release binary: its tag and the sha256 of each asset mnm may download."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["binary"] = "binary"
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    assets: tuple[Asset, ...]

    def asset(self, target: str) -> Asset | None:
        return next((a for a in self.assets if a.target == target), None)

    def url(self, asset: Asset) -> str:
        return f"{OPENCODE_RELEASES}/v{self.version}/{asset.name}"


class ImagesPin(BaseModel):
    """A compose project's images by service, each `repo:tag@sha256:<index digest>` (a tag a
    person reads, the digest Docker runs)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["images"] = "images"
    version: str
    images: dict[str, str]


Pin = Annotated[BinaryPin | ImagesPin, Field(discriminator="kind")]


@dataclass(frozen=True)
class AppDef:
    app: App
    kind: Kind
    blurb: str  # what it is, one line
    licence: str
    port: int | None = None  # on 127.0.0.1 only (until `mnm apps expose` opens it)
    ui_path: str = "/"  # where a browser opens it
    health_path: str = "/"
    project: str | None = None  # the compose project's name
    pin: BinaryPin | ImagesPin | None = None

    @property
    def url(self) -> str | None:
        return origin(LOOPBACK, self.port) + self.ui_path if self.port else None

    @property
    def health_url(self) -> str | None:
        return origin(LOOPBACK, self.port) + self.health_path if self.port else None


OPENCODE_PIN = BinaryPin(
    version="1.19.4",
    assets=(
        Asset(
            target="x64",
            name="opencode-linux-x64.tar.gz",
            sha256="849bd6b3d959a1d2ef200ed1634a51c79a45ca236aa26665d75debd694805cce",
        ),
        Asset(
            target="x64-baseline",
            name="opencode-linux-x64-baseline.tar.gz",
            sha256="1697ef40558d378bd2f72efa8fafe1bed1f0e664f09247b9357556b95dbc7cbc",
        ),
        # The arm64 build (a DGX Spark, an Arm board), downloaded from the release on
        # 2026-10-28 and hashed here (its ELF header says AArch64); the x64 asset fetched the
        # same way hashed to the pin above, so the path is the one the pin was read from
        Asset(
            target="arm64",
            name="opencode-linux-arm64.tar.gz",
            sha256="8568230ccd847dc93e95b9f6894379077d3e014a16b48040743573e77c23e85f",
        ),
    ),
)

LIBRECHAT_PIN = ImagesPin(
    version="v0.8.7",
    images={
        "api": "ghcr.io/danny-avila/librechat:v0.8.7"
        "@sha256:c5db3331b845e1f289f8d04c0c77936c4bbe372f76730a804abc1c37e44d23a9",
        "mongodb": "mongo:8.0.20"
        "@sha256:098862b1339f031900ca66cf8fef799e616d6324fa41b9a263f2ec899552c1ef",
        "meilisearch": "getmeili/meilisearch:v1.35.1"
        "@sha256:8b57fc3c7f46535ddef3828df1538465ac19d892eb57c9a10da6df0880bd5856",
        "vectordb": "pgvector/pgvector:0.8.0-pg15-trixie"
        "@sha256:8809cfffff0082cf260c9ac752f1dd1afc77f6f0a55c4e6411321e78efc3d9a5",
        "rag_api": "ghcr.io/danny-avila/librechat-rag-api-dev-lite:v0.9.0"
        "@sha256:18a8dffa9316276b924bf3fd23bfb4544b0f68f8f6b14c048affb35c01721a35",
    },
)

OPENHANDS_PIN = ImagesPin(
    version="1.31.0",
    images={
        "canvas": "ghcr.io/openhands/agent-canvas:1.31.0"
        "@sha256:348253b254600817b627fc05030a135876a408e9988303232d41e9f89fae7d1a",
    },
)

OPEN_WEBUI_PIN = ImagesPin(
    version="v0.12.3",
    images={
        "webui": "ghcr.io/open-webui/open-webui:v0.12.3"
        "@sha256:a537bbe1e6b428820f52ed4a029d7fff97062e24d5cf32c365fba36fd0542def",
    },
)

DEFS: dict[App, AppDef] = {
    App.OPENCODE: AppDef(
        App.OPENCODE,
        Kind.BINARY,
        "a coding agent that acts on your repositories: sessions, MCP, LSP, a web UI and a TUI",
        "MIT",
        port=4096,
        health_path="/global/health",
        pin=OPENCODE_PIN,
    ),
    App.LIBRECHAT: AppDef(
        App.LIBRECHAT,
        Kind.COMPOSE,
        "a chat app with files (RAG), pictures, voice and mnm's MCP tools; accounts, "
        "registration off",
        "MIT",
        port=3080,
        health_path="/health",
        project="mnm-librechat",
        pin=LIBRECHAT_PIN,
    ),
    App.OPEN_WEBUI: AppDef(
        App.OPEN_WEBUI,
        Kind.COMPOSE,
        "a chat app with documents (RAG), pictures, voice and mnm's MCP tools; its first "
        "account is its admin, sign-up off after it",
        "the Open WebUI License (BSD-3 with a branding clause)",
        port=3000,
        health_path="/health",
        project="mnm-open-webui",
        pin=OPEN_WEBUI_PIN,
    ),
    App.OPENHANDS: AppDef(
        App.OPENHANDS,
        Kind.COMPOSE,
        "Agent Canvas, a web workspace whose agent edits, runs and tests code in the projects "
        "folder, inside its own container (no Docker socket)",
        "MIT",
        port=8000,
        ui_path="/canvas",
        health_path="/health",
        project="mnm-openhands",
        pin=OPENHANDS_PIN,
    ),
    App.GOOSE: AppDef(
        App.GOOSE,
        Kind.CONNECT,
        "a desktop and CLI agent, MCP-native (its Windows app is its documented route)",
        "Apache-2.0",
    ),
    App.ODYSSEUS: AppDef(
        App.ODYSSEUS,
        Kind.CONNECT,
        "the chat front end you installed yourself in ~/odysseus; mnm never installs it",
        "its own",
    ),
}

INSTALLABLE = tuple(app for app, found in DEFS.items() if found.kind is not Kind.CONNECT)
CONNECTABLE = tuple(app for app, found in DEFS.items() if found.kind is Kind.CONNECT)


CARD_ONLY = (
    "{label} is a card only: mnm neither installs nor connects it yet (its own file is not "
    "written); `mnm url --for {app}` prints the values to paste into it."
)


def card_only(app: App) -> str | None:
    """Why an app mnm only prints a card for (Hermes Agent) takes no `mnm apps` verb, or None
    for one it installs or connects (else `status hermes` crashes and the install and connect
    refusals send the user from one to the other)."""
    if app in DEFS:
        return None
    return CARD_ONLY.format(label=LABELS[app], app=app.value)
