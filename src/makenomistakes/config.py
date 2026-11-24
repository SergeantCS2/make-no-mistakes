"""Runtime settings. Values come from the environment or an optional secrets file.

Nothing here is ever written back to disk by the application, and the repo never
contains a filled-in copy. See `.env.example` for the variable names.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from pydantic import AliasChoices, Field, PrivateAttr, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from makenomistakes.quality import FLOOR_MAX, FLOOR_MIN

SECRETS_FILE = Path("~/.config/makenomistakes/secrets.env").expanduser()
# Folders nothing of mnm's may own or be chowned to (the same list guards uninstall.sh).
_SYSTEM_FOLDERS = frozenset(
    [
        "/",
        "/bin",
        "/boot",
        "/data",
        "/dev",
        "/etc",
        "/home",
        "/lib",
        "/lib64",
        "/mnt",
        "/opt",
        "/proc",
        "/root",
        "/run",
        "/sbin",
        "/srv",
        "/sys",
        "/tmp",
        "/usr",
        "/var",
    ]
)
_SYSTEM_TREES = ("/etc/", "/usr/", "/var/")


def is_system_folder(path: Path) -> bool:
    """`/`, `/home`, `/usr/local`, `/mnt/c` …: a folder the user must never be told to
    `chown`."""
    real = str(Path(os.path.abspath(path)))
    if real in _SYSTEM_FOLDERS or real.startswith(_SYSTEM_TREES):
        return True
    return real.startswith("/mnt/") and real.count("/") == 2  # a Windows drive under WSL


# Each token's names, environment first (the order pydantic-settings reads them in).
TOKEN_NAMES = {
    "hf_token": ("HF_TOKEN", "MNM_HF_TOKEN"),
    "civitai_token": ("CIVITAI_TOKEN", "MNM_CIVITAI_TOKEN"),
    "api_key": ("MNM_API_KEY",),  # the router's: off when unset
    "ntfy_token": ("MNM_NTFY_TOKEN",),  # an ntfy topic's access token, Bearer
}
# The workspace apps' own credentials mnm generates at `mnm apps install`,
# kept in the secrets file only: OpenCode's server password (HTTP basic auth, user `opencode`)
# and Agent Canvas's session key (its agent server's `OH_SESSION_API_KEYS_0`).
APP_SECRET_NAMES = ("MNM_OPENCODE_PASSWORD", "MNM_OPENHANDS_KEY")
_DEFAULT = object()  # "the secrets file at its usual place", looked up at each read


def fresh_token(
    names: tuple[str, ...], env_files: tuple[Path, ...], environment: bool = True
) -> str | None:
    """A token as the environment, then the files (the last one that names it wins), say
    now; None when unset or empty. Names match in any case, as pydantic-settings matches
    them at startup (`hf_token=` in the file counts). `environment=False`: the files only."""
    wanted = [name.casefold() for name in names]
    sources: list[dict[str, str | None]] = []
    for env_file in reversed(env_files):
        if env_file.is_file():
            try:
                sources.append(dict(dotenv_values(env_file)))
            except (OSError, UnicodeDecodeError):
                continue
    shell: dict[str, str | None] = dict(os.environ) if environment else {}
    for values in [shell, *sources]:
        folded = {key.casefold(): value for key, value in values.items()}
        for name in wanted:
            if name in folded and folded[name] is not None:
                return folded[name] or None
    return None


class Settings(BaseSettings):
    """Process-wide configuration, read at startup. The tokens are read again from the
    environment and the secrets file each time they are asked for (so a key added while the
    web UI runs is seen), unless the caller passed them in."""

    model_config = SettingsConfigDict(
        env_prefix="MNM_",
        env_file=SECRETS_FILE if SECRETS_FILE.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    models_dir: Path = Field(default=Path("/data/models"))
    state_dir: Path = Field(default=Path("~/.local/state/makenomistakes"))
    # What Runs, pipelines and workflows produce, one folder per run
    outputs_dir: Path = Field(default=Path("/data/outputs"))
    # The workspace apps' data, one folder per app (LibreChat's databases
    # and uploads, OpenHands' home), and the one folder of the user's code the apps see.
    apps_dir: Path = Field(default=Path("/data/apps"))
    projects_dir: Path = Field(default=Path("~/src"))
    # Where `mnm backup --auto` writes and rotates its archives (7 daily, 4 weekly)
    backups_dir: Path = Field(default=Path("/data/backups"))
    port: int = Field(default=7800, ge=1024, le=65535)
    router_port: int = Field(default=7801, ge=1024, le=65535)  # one URL for every model
    # Where the router listens: 0.0.0.0, `local` (this machine and its containers), or
    # addresses; `router_bind` reads it.
    router_host: str = Field(default="0.0.0.0")
    # A second listener for Ollama's API (11434 is the port Ollama clients assume),
    # same addresses and key as the router; unset: only the router's own port serves `/api`.
    ollama_port: int | None = Field(default=None, ge=1024, le=65535)
    # Where another machine opens mnm's web UI (`http://192.168.1.10:7800`),
    # for the links MCP's run_status hands Odysseus there; empty: this machine's localhost.
    web_url: str = Field(default="")
    # On a client install only, the box's router (`http://192.168.1.10:7801`,
    # `https://box.tailnet.ts.net`): what `mnm url`, `mnm doctor --from lan` and `mnm router
    # manifest` ask; empty on the box itself, where mnm asks 127.0.0.1.
    router_url: str = Field(default="")
    # Names besides localhost, host.docker.internal and the router's own addresses that it
    # answers to (a Tailscale or LAN DNS name), commas between
    router_hosts: str = Field(default="")
    # The NVIDIA card (nvidia-smi's index) mnm plans for, reads and runs every engine on; 0,
    # the first, unless a box with two cards says otherwise; or several (`0,1`) or `all`
    # (`hardware.gpu_select`).
    gpu: str = Field(default="0")
    # The backend the engines run on, `cuda12`, `cuda13`, `rocm`, `vulkan` or
    # `cpu`; empty: mnm decides from the cards (`backend.decide`), which also refuses one the
    # cards cannot run. Written by setup's question on an AMD box.
    backend: str = Field(default="")
    hf_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(*TOKEN_NAMES["hf_token"])
    )
    civitai_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(*TOKEN_NAMES["civitai_token"])
    )
    api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(*TOKEN_NAMES["api_key"])
    )
    # Notifications: an ntfy topic's URL (`https://ntfy.sh/<topic>`, or a server of your own),
    # empty for none; a Windows toast under WSL, `on`/`off`, empty: on under WSL.
    ntfy_url: str = Field(default="")
    ntfy_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices(*TOKEN_NAMES["ntfy_token"])
    )
    toast: Literal["on", "off", ""] = ""
    # The decode speed (tok/s) a quality
    # tune never goes under: a chat or vision instance's, and a coding one's
    floor_chat: float = Field(default=15.0, ge=FLOOR_MIN, le=FLOOR_MAX)
    floor_agent: float = Field(default=20.0, ge=FLOOR_MIN, le=FLOOR_MAX)

    @field_validator(
        "models_dir", "state_dir", "outputs_dir", "apps_dir", "projects_dir", "backups_dir"
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        """`MNM_MODELS_DIR=` (empty) is Path("."): setup would chown the folder the command ran
        in and downloads would land there. Only absolute paths (after `~`) are accepted."""
        if not value.expanduser().is_absolute():
            raise ValueError(f"must be an absolute path, got {str(value)!r}")
        return value

    @field_validator("gpu", mode="before")
    @classmethod
    def _gpu(cls, value: object) -> str:
        from makenomistakes.hardware.gpu_select import parse_choice  # noqa: PLC0415

        text = str(value).strip()
        choice = parse_choice(text)  # a ValueError names what MNM_GPU may be
        return "all" if choice.every else ",".join(str(i) for i in choice.indices)

    @field_validator("backend", mode="before")
    @classmethod
    def _backend(cls, value: object) -> str:
        from makenomistakes.backend import parse_backend  # noqa: PLC0415

        found = parse_backend(str(value))  # a ValueError names what MNM_BACKEND may be
        return found.value if found is not None else ""

    @field_validator("web_url")
    @classmethod
    def _web_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if value and not re.fullmatch(r"https?://[^/\s?#]+", value):
            raise ValueError(
                f"MNM_WEB_URL must be an origin like http://192.168.1.10:7800, got {value!r}"
            )
        return value

    @field_validator("router_url")
    @classmethod
    def _router_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if value and not re.fullmatch(r"https?://[^/\s?#@]+", value):
            raise ValueError(
                "MNM_ROUTER_URL must be the box's router as an origin like "
                f"http://192.168.1.10:7801 (no path, no user), got {value!r}"
            )
        return value

    @field_validator("ollama_port", mode="before")
    @classmethod
    def _ollama_port(cls, value: object) -> object:
        """`MNM_OLLAMA_PORT=` (empty, as `mnm setup` might leave it) is off, not an error."""
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("ntfy_url")
    @classmethod
    def _ntfy_url(cls, value: str) -> str:
        value = value.strip()
        if value and not re.fullmatch(r"https?://[^/\s?#]+/[^\s?#]+", value):
            raise ValueError(
                f"MNM_NTFY_URL must be an ntfy topic's URL like https://ntfy.sh/my-topic, got "
                f"{value!r}"
            )
        return value

    @field_validator("toast", mode="before")
    @classmethod
    def _toast(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("router_host")
    @classmethod
    def _router_host(cls, value: str) -> str:
        from makenomistakes.router_bind import check_setting  # noqa: PLC0415 - no cycle at import

        return check_setting(value)

    def expanded_models_dir(self) -> Path:
        return self.models_dir.expanduser()

    def expanded_state_dir(self) -> Path:
        return self.state_dir.expanduser()

    def expanded_outputs_dir(self) -> Path:
        return self.outputs_dir.expanduser()

    def expanded_apps_dir(self) -> Path:
        return self.apps_dir.expanduser()

    def expanded_projects_dir(self) -> Path:
        return self.projects_dir.expanduser()

    def expanded_backups_dir(self) -> Path:
        return self.backups_dir.expanduser()

    _pinned: frozenset[str] = PrivateAttr(default=frozenset())
    _source: object = PrivateAttr(default=_DEFAULT)

    def __init__(self, **values: Any) -> None:
        super().__init__(**values)
        self._pinned = frozenset(
            field for field, names in TOKEN_NAMES.items() if {field, *names} & values.keys()
        )
        self._source = values.get("_env_file", _DEFAULT)

    def hf_token_value(self) -> str | None:
        """None when unset or empty (`HF_TOKEN=` parses as an empty secret)."""
        return self._token("hf_token")

    def civitai_token_value(self) -> str | None:
        return self._token("civitai_token")

    def api_key_value(self) -> str | None:
        """The key the router asks every request for; None leaves the router open.
        From the secrets file only, never a shell's environment: the router's service has
        none of a shell's variables, so a terminal's `mnm router`, the cards, `mnm try` and
        the service all agree on one key. Read per call: a new key applies without a restart."""
        if "api_key" in self._pinned:
            return self._token("api_key")
        return fresh_token(TOKEN_NAMES["api_key"], self._env_files(), environment=False)

    def ntfy_token_value(self) -> str | None:
        """The ntfy topic's token, from the secrets file only, like the router's key:
        every process that notifies (a terminal, the web UI's service) reads the same one."""
        if "ntfy_token" in self._pinned:
            return self._token("ntfy_token")
        return fresh_token(TOKEN_NAMES["ntfy_token"], self._env_files(), environment=False)

    def app_secret(self, name: str) -> str | None:
        """One of `APP_SECRET_NAMES`, from the secrets file only (never a shell's environment),
        read per call like the router's key."""
        if name not in APP_SECRET_NAMES:
            raise ValueError(f"{name} is not an app's secret")
        return fresh_token((name,), self._env_files(), environment=False)

    def _token(self, field: str) -> str | None:
        if field in self._pinned:
            secret: SecretStr | None = getattr(self, field)
            return (secret.get_secret_value() or None) if secret else None
        return fresh_token(TOKEN_NAMES[field], self._env_files())

    def _env_files(self) -> tuple[Path, ...]:
        """The env files these settings were read from (pydantic takes one or a sequence)."""
        source = self._source
        if source is _DEFAULT:
            return (SECRETS_FILE,)
        if isinstance(source, str | Path):
            return (Path(source),)
        if isinstance(source, list | tuple):
            return tuple(Path(item) for item in source)
        return ()


_NOW: dict[str, tuple[tuple[object, ...], Settings]] = {}  # `settings_now`'s one kept


def settings_now() -> Settings:
    """`Settings()` for the paths a poll reads many times (the backend, the cards' ids and
    `MNM_GPU` on every verdict, diagnosis and render): built again only when
    the environment or the env file it reads changed, so a `mnm set` or a new secrets line
    applies at once. Treat it as read-only."""
    env_file = Settings.model_config.get("env_file")
    stamp: tuple[int, int] | None = None
    if isinstance(env_file, (str, Path)):
        try:
            info = Path(env_file).stat()
            stamp = (info.st_mtime_ns, info.st_size)
        except OSError:
            stamp = None
    key = (tuple(sorted(os.environ.items())), str(env_file), stamp)
    found = _NOW.get("now")
    if found is not None and found[0] == key:
        return found[1]
    made = Settings()
    _NOW["now"] = (key, made)
    return made
