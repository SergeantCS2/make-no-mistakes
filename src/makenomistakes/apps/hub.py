"""`mnm apps`: install, start, stop, update, roll back and uninstall the
workspace apps mnm runs, each through its driver (`OpenCodeApp`, or `ComposeApp` with
LibreChat's, OpenHands' or Open WebUI's spec), and the ledger (`ledger.py`) that says which pin
runs and which one came before. Every change holds the apps folder's lock; a status only reads.

What `update` and `rollback` mean: the pin a newer mnm carries (`defs.py`) replaces the one in
the ledger, which is kept as `previous`; `rollback` swaps them back. The pin before
`previous` is pruned (its binary folder, its images) when an update drops it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from makenomistakes.apps import exposure, librechat_app, open_webui_app, openhands_app
from makenomistakes.apps import ledger as ledgers
from makenomistakes.apps.compose_app import ComposeApp
from makenomistakes.apps.context import AppContext
from makenomistakes.apps.defs import DEFS, AppDef, Kind, Pin, card_only
from makenomistakes.apps.folder import IN_APPS_FOLDER, ready
from makenomistakes.apps.ledger import AppLedger, AppsError
from makenomistakes.apps.names import LABELS, App
from makenomistakes.apps.opencode_app import OpenCodeApp

GONE = ("removed", "stopped", "deleted")  # how a driver's `remove` lines say it took something


class Driver(Protocol):
    app: App

    def install(self, pin: Pin) -> None: ...
    def write_config(self) -> None: ...
    def switch(self, pin: Pin) -> None: ...
    def start(self, pin: Pin) -> None: ...
    def stop(self) -> None: ...
    def running(self) -> bool | None: ...
    def logs(self, tail: int) -> str: ...
    def remove(self, keep_data: bool) -> list[str]: ...
    def prune(self, dropped: Pin) -> list[str]: ...
    def notes(self) -> list[str]: ...


DRIVERS: dict[App, Callable[[AppContext], Driver]] = {
    App.OPENCODE: OpenCodeApp,
    App.LIBRECHAT: lambda ctx: ComposeApp(ctx, librechat_app.SPEC),
    App.OPENHANDS: lambda ctx: ComposeApp(ctx, openhands_app.SPEC),
    App.OPEN_WEBUI: lambda ctx: ComposeApp(ctx, open_webui_app.SPEC),
}


@dataclass(frozen=True)
class AppStatus:
    definition: AppDef
    installed: AppLedger | None = None
    running: bool | None = None  # None: not installed, or nothing could say
    notes: list[str] = field(default_factory=list)
    error: str | None = None  # the ledger cannot be read, or Docker/systemd did not answer

    @property
    def app(self) -> App:
        return self.definition.app

    @property
    def label(self) -> str:
        return LABELS[self.app]

    @property
    def pinned(self) -> str | None:
        return self.definition.pin.version if self.definition.pin else None

    @property
    def version(self) -> str | None:
        return self.installed.pin.version if self.installed else None

    @property
    def behind(self) -> bool:
        """This mnm carries a newer pin than the one installed (`mnm apps update`)."""
        pin = self.definition.pin
        return self.installed is not None and pin is not None and self.installed.pin != pin

    @property
    def state(self) -> str:
        if self.error:
            return "unknown"
        if self.definition.kind is Kind.CONNECT:
            return "connect only"
        if self.installed is None:
            return "not installed"
        return {True: "running", False: "stopped", None: "unknown"}[self.running]


class AppsService:
    def __init__(self, ctx: AppContext) -> None:
        self.ctx = ctx

    def driver(self, app: App) -> Driver:
        if (why := card_only(app)) is not None:
            raise AppsError(why)
        found = DRIVERS.get(app)
        if found is None:  # every app in INSTALLABLE has a driver
            raise AppsError(
                f"mnm does not install {LABELS[app]}: `mnm apps connect {app.value}` writes its "
                "files and prints its card."
            )
        return found(self.ctx)

    def ledger(self, app: App) -> AppLedger | None:
        return ledgers.read(self.ctx.state_dir, app)

    def _installed(self, app: App) -> AppLedger:
        found = self.ledger(app)
        if found is None:
            raise AppsError(f"{LABELS[app]} is not installed: `mnm apps install {app.value}`.")
        return found

    # --- reading ---------------------------------------------------------------------

    def status(self, app: App) -> AppStatus:
        if (why := card_only(app)) is not None:
            raise AppsError(why)
        definition = DEFS[app]
        if definition.kind is Kind.CONNECT:
            return AppStatus(definition)
        try:
            found = self.ledger(app)
        except AppsError as exc:
            return AppStatus(definition, error=str(exc))
        if found is None:
            return AppStatus(definition)
        driver = self.driver(app)
        return AppStatus(definition, found, driver.running(), driver.notes())

    def statuses(self) -> list[AppStatus]:
        return [self.status(app) for app in DEFS]

    def logs(self, app: App, tail: int = 200) -> str:
        self._installed(app)
        return self.driver(app).logs(tail)

    # --- changing --------------------------------------------------------------------

    def install(self, app: App) -> AppLedger:
        """The pinned build, its config, started and answering. An installed app is brought
        to this mnm's config again (its pin is kept: `update` moves it). The apps folder is
        checked first (`folder.ready`): an app that keeps its data there is refused before
        anything is pulled when the folder cannot be made or written."""
        if app in IN_APPS_FOLDER:
            ready(self.ctx.apps_dir)
        driver = self.driver(app)
        pin = _pin(app)
        with ledgers.locked(self.ctx.state_dir):
            before = self.ledger(app)
            if before is not None:
                self.ctx.say(f"{LABELS[app]} {before.pin.version} is installed: its config again.")
                driver.write_config()
                driver.switch(before.pin)  # its own steps again (OpenHands' settings)
                return before
            driver.install(pin)
            done = AppLedger(app=app, pin=pin)
            ledgers.save(self.ctx.state_dir, done)
        return done

    def start(self, app: App) -> None:
        found = self._installed(app)
        with ledgers.locked(self.ctx.state_dir):
            self.driver(app).start(found.pin)

    def stop(self, app: App) -> None:
        self._installed(app)
        with ledgers.locked(self.ctx.state_dir):
            self.driver(app).stop()

    def update(self, app: App) -> AppLedger | None:
        """The pin this mnm carries, the installed one kept for rollback; None when the
        installed one is already it."""
        driver = self.driver(app)
        pin = _pin(app)
        with ledgers.locked(self.ctx.state_dir):
            found = self._installed(app)
            if found.pin == pin:
                return None
            driver.switch(pin)
            dropped = found.previous
            done = found.model_copy(
                update={"pin": pin, "previous": found.pin, "changed_at": datetime.now(UTC)}
            )
            ledgers.save(self.ctx.state_dir, done)
            if dropped is not None and dropped not in (pin, found.pin):
                for gone in driver.prune(dropped):
                    self.ctx.say(f"removed {gone} (two pins back)")
        return done

    def rollback(self, app: App) -> AppLedger:
        driver = self.driver(app)
        with ledgers.locked(self.ctx.state_dir):
            found = self._installed(app)
            if found.previous is None:
                raise AppsError(
                    f"{LABELS[app]} has no earlier version to go back to: only an update keeps one."
                )
            driver.switch(found.previous)
            done = found.model_copy(
                update={
                    "pin": found.previous,
                    "previous": found.pin,
                    "changed_at": datetime.now(UTC),
                }
            )
            ledgers.save(self.ctx.state_dir, done)
        return done

    def may_expose(self, app: App) -> bool:
        """Whether opening `app` would change anything (AppsError when it is refused): the CLI
        asks its plain-http question only then, and `expose` checks again."""
        with ledgers.locked(self.ctx.state_dir):
            found = self._installed(app)
            if (why := exposure.refusal(self.ctx, app)) is not None:
                raise AppsError(f"{LABELS[app]} stays on 127.0.0.1: {why}")
            return not found.exposed

    def expose(self, app: App, opened: bool) -> list[str]:
        """On every interface (`opened`) or back on 127.0.0.1 (`exposure.py`): the
        ledger says which, the unit or compose file is written again and the app restarted.
        A restart that fails puts the ledger back as it was (and the app with it)."""
        driver = self.driver(app)
        with ledgers.locked(self.ctx.state_dir):
            found = self._installed(app)
            if opened and (why := exposure.refusal(self.ctx, app)) is not None:
                raise AppsError(f"{LABELS[app]} stays on 127.0.0.1: {why}")
            if found.exposed == opened:
                return exposure.notes(app, opened)
            ledgers.save(self.ctx.state_dir, found.model_copy(update={"exposed": opened}))
            try:
                driver.start(found.pin)
            except AppsError:
                ledgers.save(self.ctx.state_dir, found)
                if opened:  # back on loopback, as the ledger says again (said if not)
                    try:
                        driver.start(found.pin)
                    except AppsError as again:
                        self.ctx.say(f"and back on 127.0.0.1 it did not start: {again}")
                raise
        return exposure.notes(app, opened)

    def uninstall(self, app: App, keep_data: bool) -> list[str]:
        driver = self.driver(app)
        with ledgers.locked(self.ctx.state_dir):
            found = self.ledger(app)
            done = driver.remove(keep_data)  # what a failed install left too
            ledgers.forget(self.ctx.state_dir, app)
        if found is None:
            if not any(line.startswith(GONE) for line in done):
                raise AppsError(f"{LABELS[app]} is not installed: nothing to remove.")
            done.insert(0, f"{LABELS[app]} was not installed; removed what an attempt left")
        return done


def _pin(app: App) -> Pin:
    pin = DEFS[app].pin
    if pin is None:
        raise AppsError(f"{LABELS[app]} has no pin in this mnm.")
    return pin
