"""What one `mnm add` adds, said once: as the command line `mnm what` prints, and as the
Add page's link whose form starts on it (`web/add_routes`), so the two never say different
things. A leaf module: the grade builds one, the web reads one back from a link."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

from makenomistakes.hub.modality import Modality


@dataclass(frozen=True)
class AddIntent:
    """A reference and what `mnm add` is told with it: the agent profile (`--for-agents`:
    type coding, the agent's context), a quant (`--quant`), a name (`--name`)."""

    reference: str
    agents: bool = False
    quant: str | None = None
    name: str | None = None

    def cli(self) -> str:
        parts = ["mnm add", self.reference]
        if self.agents:
            parts.append("--for-agents")
        if self.quant is not None:
            parts += ["--quant", self.quant]
        if self.name is not None:
            parts += ["--name", self.name]
        return " ".join(parts)

    def href(self) -> str:
        """The Add page's plan of it, its form preset (`/add`'s `agents`, `quant`, `name`)."""
        query = [("reference", self.reference), ("plan", "1")]
        if self.agents:
            query += [("type", Modality.CODING.value), ("agents", "1")]
        if self.quant is not None:
            query.append(("quant", self.quant))
        if self.name is not None:
            query.append(("name", self.name))
        return f"/add?{urlencode(query)}"

    @classmethod
    def from_link(cls, reference: str, agents: bool, quant: str, name: str) -> AddIntent:
        """The link's fields as `/add` reads them (empty is unset)."""
        return cls(reference, agents, quant or None, name or None)
