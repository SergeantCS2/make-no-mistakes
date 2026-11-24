"""Open WebUI's accounts as mnm asks about them: whether its first account (the admin) exists,
read from `GET /api/config` without a session, whose `onboarding` is true only while no user
exists (v0.12.3 main.py `get_app_config`); asked inside its container with its own curl (its
image's HEALTHCHECK uses it), so the answer does not depend on where its port is published.
The install says "create your account now" while there is none; `mnm apps expose` refuses
until there is one."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, ValidationError

from makenomistakes.apps import compose
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import DEFS
from makenomistakes.apps.names import CONTAINERS, App

URL = f"http://localhost:{DEFS[App.OPEN_WEBUI].port or 3000}"
FIRST_ACCOUNT = (
    "Open WebUI makes no accounts from outside: open {url} and create your account now, the "
    "first one is the admin (sign-up is off for everyone after it; the admin adds others in its "
    "Admin Panel → Users)."
)
# Inside the container: its own port on its own loopback, never one of the router's doors.
CONFIG = ["curl", "-sS", "-m", "20", f"http://127.0.0.1:{compose.OPEN_WEBUI_PORT}/api/config"]


class _Config(BaseModel):
    """`GET /api/config` without a session: `onboarding` only while no user exists."""

    model_config = ConfigDict(extra="ignore")

    onboarding: bool = False


def first_account() -> str:
    return FIRST_ACCOUNT.format(url=URL)


def has_account(ctx: AppContext) -> bool | str:
    """Whether an account exists (its admin), else why it could not be asked."""
    argv = ["docker", "exec", CONTAINERS[App.OPEN_WEBUI], *CONFIG]
    code, out, err = ctx.runner.run(argv, timeout=60.0)
    if code != 0:
        said = (err or out).strip().splitlines()[-1:] or [f"exit {code}"]
        return f"Open WebUI did not answer inside its container ({said[0]})"
    try:
        return not _Config.model_validate_json(out).onboarding
    except ValidationError:
        return f"Open WebUI's /api/config is not what v0.12 answers ({out.strip()[:120]})"
