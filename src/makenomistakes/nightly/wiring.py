"""The nightly as this box runs it: the real steps, the router's gate, the actions'
doers and the notifier, for the CLI (`mnm nightly run`, the timer) and the web's Run now."""

from __future__ import annotations

from makenomistakes.config import Settings
from makenomistakes.instances import InstanceService
from makenomistakes.journal import Note
from makenomistakes.nightly import settings as nightly_settings
from makenomistakes.nightly.actions import Doers
from makenomistakes.nightly.fresh import for_box, hub_answers
from makenomistakes.nightly.gate import gate_for
from makenomistakes.nightly.report import NightlyReport, NightlyStore
from makenomistakes.nightly.run import NightlyRun, Options, Skip, running_lock
from makenomistakes.nightly.steps import managed_apps, real_steps, restart_app
from makenomistakes.notify import notifier
from makenomistakes.router_client import router_tune
from makenomistakes.successors import cached

BUSY = "a nightly is running already (`mnm nightly ls` shows it once it ends)"


class NightlyBusyError(RuntimeError):
    """Another run holds the nightly's lock."""


def nightly_for(settings: Settings, say: Note) -> NightlyRun:
    service = InstanceService(settings)
    steps = real_steps(service, say, for_box(service, cached(hub_answers(settings))))
    return NightlyRun(
        steps=steps,
        gate=gate_for(settings),
        store=NightlyStore(settings.expanded_state_dir()),
        doers=Doers(
            retune=lambda name: router_tune(settings, name, b'{"by": "nightly"}'),
            backup=steps.backup,
            restart=lambda app: restart_app(service, app),
            managed=lambda: managed_apps(service),
        ),
        channels=notifier(settings).send,
        say=say,
    )


def run_nightly(
    settings: Settings,
    say: Note,
    now: bool = False,
    skip: frozenset[Skip] = frozenset(),
    wait: bool = True,
) -> NightlyReport:
    """One run under the lock; NightlyBusyError when another holds it. The user's `clean=`
    is read here (a file that cannot be read turns the clean step off and says so)."""
    state = settings.expanded_state_dir()
    try:
        clean = nightly_settings.read(state).clean_age
    except nightly_settings.NightlySettingsError as exc:
        say(f"The clean step is off tonight: {exc}")
        clean = None
    options = Options(now=now, skip=skip, clean=clean, wait=wait)
    with running_lock(state).attempt() as mine:
        if not mine:
            raise NightlyBusyError(BUSY)
        return nightly_for(settings, say).run(options)
