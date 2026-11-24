"""The desk's roles as model ids: role -> the member instance's name. A leaf
module (no imports of mnm's own), so `routing` can take it without a cycle; `desk` reads it
from the files."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Aliases:
    desk: str | None = None  # the default set's name
    members: dict[str, str] = field(default_factory=dict)
    shadowed: tuple[str, ...] = ()  # roles an instance is named after: never an alias
    problem: str | None = None  # why there are none although a desk is chosen

    def instance(self, model: str) -> str | None:
        """The instance an alias names, or None when `model` is no alias."""
        return self.members.get(model)

    def of(self, instance: str) -> list[str]:
        """The aliases that name `instance`, in role order."""
        return [role for role, name in self.members.items() if name == instance]


NO_ALIASES = Aliases()
