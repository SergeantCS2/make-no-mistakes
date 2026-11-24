"""Each instance's last `mnm try --tools` verdict, for its page's Tools line and
`mnm show`: `<state>/tools/<name>.json` (a `JsonFolder`), written whole after every tool
test (the CLI's, Test it's, `mnm try --all`'s). It names the settings it was taken with
(`measurements.settings_key`), so the line says when the settings changed since. A file that
cannot be read counts as no verdict: a hint, never a reason a page fails."""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from makenomistakes.json_folder import BadNameError, JsonFolder
from makenomistakes.local_time import local
from makenomistakes.measurements import settings_key
from makenomistakes.recipes import Recipe
from makenomistakes.tool_check import ToolCheck, Verdict, summary_of

FOLDER = "tools"


class ToolVerdict(BaseModel):
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verdict: Verdict
    detail: str  # `ToolCheck.detail`: the steps counted, those failed or not supported
    steps: list[str]  # each step's line, as `mnm try --tools` printed it
    settings: str  # `settings_key` of the recipe it ran with

    @property
    def summary(self) -> str:
        return summary_of(self.verdict, self.detail)

    def when(self, recipe: Recipe) -> str:
        """`10-23 14:02` on the box's clock, and whether the settings changed since."""
        older = "" if self.settings == settings_key(recipe) else ", with earlier settings"
        return f"{local(self.at):%m-%d %H:%M}{older}"


class ToolVerdicts:
    def __init__(self, state_dir: Path) -> None:
        self.files = JsonFolder(state_dir / FOLDER)

    def save(self, recipe: Recipe, check: ToolCheck) -> ToolVerdict:
        """Record `check` as the instance's verdict; one that cannot be written is dropped
        (the test still printed it)."""
        verdict = ToolVerdict(
            verdict=check.verdict,
            detail=check.detail,
            steps=[step.line for step in check.steps],
            settings=settings_key(recipe),
        )
        with contextlib.suppress(OSError, BadNameError), self.files.locked():
            self.files.write(recipe.name, verdict.model_dump_json(indent=2) + "\n")
        return verdict

    def get(self, name: str) -> ToolVerdict | None:
        try:
            return ToolVerdict.model_validate_json(self.files.read(name))
        except (OSError, BadNameError, ValidationError):
            return None

    def forget(self, name: str) -> None:
        """`mnm rm`: another model saved under the name later starts untested."""
        with contextlib.suppress(OSError, BadNameError), self.files.locked():
            self.files.unlink(name)
