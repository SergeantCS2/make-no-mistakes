"""The compose projects mnm renders for its Docker apps: typed, every image from the
pin (`defs.py`), every port on 127.0.0.1 (0.0.0.0 after `mnm apps expose`), every
container labelled `mnm.app` (what `uninstall.sh` finds them by), never the Docker socket,
`latest` or a privileged container.

LibreChat follows its v0.8.7 `docker-compose.yml` (api, mongodb, meilisearch, vectordb,
rag_api; the admin panel left out) with three changes: the data under
`MNM_APPS_DIR/librechat/` as the user's uid (Postgres' too: the image runs as any uid that
owns its data folder), the vector database's password generated (the upstream file hard-codes
`mypassword`; nothing outside the project reaches it either way), and `restart:
unless-stopped`, so `mnm apps stop` stays stopped across a reboot.

OpenHands is Agent Canvas as its docs' Docker option runs it (`-p 127.0.0.1:8000:8000`, the
projects folder at /projects, `host.docker.internal:host-gateway`), run as the user's uid with
its home a folder of the user's (`MNM_APPS_DIR/openhands/home`, `HOME=/oh`): the image's own
user (uid 10001) would write the user's repositories as a stranger, and a bind-mounted
`~/.openhands` it cannot write is the docs' "unable to open database file". As uid 1000 the
agent server, the settings API and /projects all work. Its session key comes from its `.env`
(never the command line); `AGENT_CANVAS_ALLOW_LAN_SESSION_KEY` is left off, so the page asks
for the key (`mnm apps status openhands` says where it is).
Telemetry off (`DO_NOT_TRACK`).

Every service gets `no-new-privileges`, and the RAG API reads its own `rag.env` (its
embeddings, `JWT_SECRET`, the vector database's password), not LibreChat's whole `.env` as
upstream's file gives it. It runs as its image's user, root: its Dockerfile (rag_api v0.9.0,
`Dockerfile.lite`) sets no USER and writes its uploads under `/app`, which only root may
write. `cap_drop: [ALL]` is left out: no service has been started without its capabilities
yet.

Open WebUI (v0.12.3) is one container, `mnm-open-webui`, as its README runs it (`-p
3000:8080`, `host.docker.internal:host-gateway`, its data at /app/backend/data), the data a
folder of the user's (`MNM_APPS_DIR/open-webui/data`) and the container run as the user's
uid: its image runs as root by default (Dockerfile `ARG UID=0`), and as uid 1000 it starts,
signs up, chats and writes `webui.db` and its uploads as the user. The one cost is about
twenty `Permission denied` ERROR lines at each start as uid 1000: it copies its front end's
favicons, splash, loader and the like over the identical ones in
`/app/backend/open_webui/static`, which only root may write (config.py, each copy in a try).
The bind mount hides the models its image bakes under `/app/backend/data/cache` (a local
embedding model and whisper): with the desk's `memory` and `ear` wired they are never asked
for; without, Open WebUI downloads its own the first time. Its secrets (`WEBUI_SECRET_KEY`,
the router's key) come from its `.env` (0600)."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from makenomistakes.apps.card import Doc
from makenomistakes.apps.card import render as render_doc
from makenomistakes.apps.defs import ImagesPin
from makenomistakes.apps.names import CONTAINERS, App

LABEL = "mnm.app"
HOST_GATEWAY = "host.docker.internal:host-gateway"
COMPOSE_FILE = "compose.yaml"
ENV_FILE = ".env"
RAG_ENV_FILE = "rag.env"  # the RAG API's lines only
RAG_PREFIXES = ("RAG_", "EMBEDDINGS_")
RAG_SECRETS = frozenset({"JWT_SECRET", "POSTGRES_PASSWORD"})
NO_NEW_PRIVILEGES = ["no-new-privileges:true"]
OPENHANDS_HOME = "/oh"


class Service(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str
    container_name: str | None = None
    command: list[str] | None = None
    user: str | None = None
    restart: str = "unless-stopped"
    environment: dict[str, str] = Field(default_factory=dict)
    env_file: list[str] = Field(default_factory=list)
    volumes: list[str] = Field(default_factory=list)
    ports: list[str] = Field(default_factory=list)
    extra_hosts: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    labels: dict[str, str] = Field(default_factory=dict)
    security_opt: list[str] = Field(default_factory=lambda: list(NO_NEW_PRIVILEGES))


class Compose(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    services: dict[str, Service]


def value(compose: Compose) -> Doc:
    """The file's content as compose reads it; empty lists and maps left out."""
    found = compose.model_dump(exclude_none=True)
    for service in found["services"].values():
        for key in [k for k, v in service.items() if v in ([], {})]:
            del service[key]
    return found


def render(compose: Compose) -> str:
    return render_doc("yaml", value(compose))


def owner() -> str:
    """The uid:gid the containers run as, so their files are the user's."""
    return f"{os.getuid()}:{os.getgid()}"


def published(port: int, host: str = "127.0.0.1", inside: int | None = None) -> str:
    """The port as Docker publishes it: 127.0.0.1, or 0.0.0.0 once exposed; `inside`
    when the container listens on another port (Open WebUI's 8080)."""
    return f"{host}:{port}:{inside or port}"


LIBRECHAT_FOLDERS = ("data-node", "meili_data", "pgdata", "uploads", "images", "logs", "skill")


def librechat(
    pin: ImagesPin, folder: Path, port: int, user: str, host: str = "127.0.0.1"
) -> Compose:
    def at(name: str) -> str:
        return str(folder / name)

    label = {LABEL: App.LIBRECHAT.value}
    images = pin.images
    return Compose(
        name="mnm-librechat",
        services={
            "api": Service(
                image=images["api"],
                user=user,
                ports=[published(port, host)],
                extra_hosts=[HOST_GATEWAY],
                depends_on=["mongodb", "rag_api"],
                environment={
                    "HOST": "0.0.0.0",  # inside its container; Docker publishes 127.0.0.1 only
                    "MONGO_URI": "mongodb://mongodb:27017/LibreChat",
                    "MEILI_HOST": "http://meilisearch:7700",
                    "RAG_PORT": "8000",
                    "RAG_API_URL": "http://rag_api:8000",
                },
                volumes=[
                    f"{at(ENV_FILE)}:/app/.env:ro",
                    f"{at('librechat.yaml')}:/app/librechat.yaml:ro",
                    f"{at('images')}:/app/client/public/images",
                    f"{at('uploads')}:/app/uploads",
                    f"{at('logs')}:/app/logs",
                    f"{at('skill')}:/app/skill",
                ],
                labels=label,
            ),
            "mongodb": Service(
                image=images["mongodb"],
                user=user,
                command=["mongod", "--noauth"],  # reached only inside the project's network
                volumes=[f"{at('data-node')}:/data/db"],
                labels=label,
            ),
            "meilisearch": Service(
                image=images["meilisearch"],
                user=user,
                environment={
                    "MEILI_HOST": "http://meilisearch:7700",
                    "MEILI_NO_ANALYTICS": "true",
                    "MEILI_MASTER_KEY": "${MEILI_MASTER_KEY}",
                },
                volumes=[f"{at('meili_data')}:/meili_data"],
                labels=label,
            ),
            "vectordb": Service(
                image=images["vectordb"],
                user=user,
                environment={
                    "POSTGRES_DB": "mydatabase",
                    "POSTGRES_USER": "myuser",
                    "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD}",
                },
                volumes=[f"{at('pgdata')}:/var/lib/postgresql/data"],
                labels=label,
            ),
            "rag_api": Service(
                image=images["rag_api"],
                environment={  # LibreChat v0.8.7's rag.yml's database lines
                    "DB_HOST": "vectordb",
                    "DB_PORT": "5432",
                    "RAG_PORT": "8000",
                    "POSTGRES_DB": "mydatabase",
                    "POSTGRES_USER": "myuser",
                },
                env_file=[at(RAG_ENV_FILE)],  # its embeddings, JWT_SECRET, the database's password
                extra_hosts=[HOST_GATEWAY],
                depends_on=["vectordb"],
                labels=label,
            ),
        },
    )


def openhands(
    pin: ImagesPin, folder: Path, projects: Path, port: int, user: str, host: str = "127.0.0.1"
) -> Compose:
    return Compose(
        name="mnm-openhands",
        services={
            "canvas": Service(
                image=pin.images["canvas"],
                container_name=CONTAINERS[App.OPENHANDS],
                user=user,
                ports=[published(port, host)],
                extra_hosts=[HOST_GATEWAY],
                environment={
                    "HOME": OPENHANDS_HOME,
                    "DO_NOT_TRACK": "1",
                    "VITE_DO_NOT_TRACK": "1",
                    "AGENT_CANVAS_DISABLE_TELEMETRY": "true",
                },
                env_file=[str(folder / ENV_FILE)],  # its session key
                volumes=[f"{folder / 'home'}:{OPENHANDS_HOME}", f"{projects}:/projects"],
                labels={LABEL: App.OPENHANDS.value},
            )
        },
    )


OPEN_WEBUI_PORT = 8080  # its image's PORT
OPEN_WEBUI_DATA = "/app/backend/data"


def open_webui(
    pin: ImagesPin,
    folder: Path,
    port: int,
    user: str,
    environment: dict[str, str],
    host: str = "127.0.0.1",
) -> Compose:
    """`environment`: its settings without a secret (`apps/open_webui.py`); the `.env` beside
    the compose file holds the rest."""
    return Compose(
        name="mnm-open-webui",
        services={
            "webui": Service(
                image=pin.images["webui"],
                container_name=CONTAINERS[App.OPEN_WEBUI],
                user=user,
                ports=[published(port, host, OPEN_WEBUI_PORT)],
                extra_hosts=[HOST_GATEWAY],
                environment=environment,
                env_file=[str(folder / ENV_FILE)],
                volumes=[f"{folder / 'data'}:{OPEN_WEBUI_DATA}"],
                labels={LABEL: App.OPEN_WEBUI.value},
            )
        },
    )
