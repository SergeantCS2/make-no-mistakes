"""The agent profile: a chat model added for coding agents, editors and agent apps.

An agent sends its own instructions and tool list with every request, so the one setting
that matters is the context one request gets (llama.cpp's `ctx ÷ parallel`). Each client's
floor is in `FLOORS`, with where it comes from (read 2026-10-23 unless said otherwise):

- Claude Code: its first request was about 15,750 tokens (measured against a CPU
  llama-server), about 19,300 in 2.1.309 with its 31 tools (counted on Qwen3's
  template); a working session needs more: 32,768.
- Codex CLI: about 6,200 tokens first (measured likewise): 16,384.
- OpenCode: what the user sets as the model's `limit.context` in opencode.json
  (opencode.ai/docs/providers, the llama.cpp section; the card writes the instance's);
  its docs suggest 16k to 32k for reliable tool calls; its first request was 7,338 tokens
  (1.19.4 with its ten tools): 32,768.
- Qwen Code: a Gemini CLI descendant with a long system prompt and tool list
  (github.com/QwenLM/qwen-code; qwenlm.github.io/qwen-code-docs, Authentication); not
  stated there; its first request was about 11,400 tokens (0.26.0, six tools, counted on
  Qwen3's template): 32,768.
- OpenHands: "at least 22000", 32,768 recommended (docs.openhands.dev/openhands/usage/
  llms/local-llms): 32,768.
- Hermes Agent: at least 64,000 tokens, "rejected at startup" below it
  (hermes-agent.nousresearch.com/docs/getting-started/quickstart, llama.cpp's
  `--ctx-size 65536`): 65,536, one request at a time.
- Goose: not stated (goose-docs.ai, custom providers take a `context_limit` per model):
  32,768.
- Aider, Cline, Zed, llama.vscode: as their cards had them, not measured.

`mnm add --for-agents` plans for the busiest of the clients it serves (`AGENT_CTX`): every
floor but Hermes Agent's, an always-on assistant rather than a repo workspace,
whose 64k would take Qwen3.8-27B off a 12 GiB card. `mnm add --for <client>` plans for one
floor, Hermes included. The planner always runs one request at a time (`-np 1`), so the
whole context is each request's; a hand-set `parallel` divides it, which the cards say.
Tools are on for every chat recipe (`--jinja`); the type becomes `coding`, which groups the
instance with its kind and leads with the agent cards.
"""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.hub.facts import CHAT_LIKE, RETYPABLE, Modality
from makenomistakes.plan_types import Plan, PlanError


@dataclass(frozen=True)
class Floor:
    """The context one request must have for a client, and why."""

    label: str
    tokens: int
    why: str
    refuses_below: bool = False  # the client will not start on less (Hermes Agent)
    busiest: bool = True  # counted in `--for-agents`' own floor (`AGENT_CTX`)


CLAUDE_CODE_FIRST = 19_300  # Claude Code 2.1.309's first request, tokens
FLOORS: dict[str, Floor] = {  # `mnm add --for <key>`; the card clients' keys are Consumers'
    "claude-code": Floor(
        "Claude Code",
        32_768,
        f"its first request is about {CLAUDE_CODE_FIRST:,} tokens, measured, before the "
        "project's CLAUDE.md, its memory and any file it reads",
    ),
    "codex": Floor(
        "Codex CLI",
        16_384,
        "its first request is about 6,200 tokens, measured, before any file it reads",
    ),
    "opencode": Floor(
        "OpenCode",
        32_768,
        "its first request is about 7,300 tokens, measured; the card writes the instance's "
        "context as the model's limit.context",
    ),
    "qwen-code": Floor(
        "Qwen Code",
        32_768,
        "its first request is about 11,400 tokens, measured, before any file it reads",
    ),
    "openhands": Floor(
        "OpenHands",
        32_768,
        "its docs ask for at least 22,000 tokens and recommend 32,768",
    ),
    "hermes": Floor(
        "Hermes Agent",
        65_536,
        "it refuses a model with less than 64,000 tokens at startup",
        refuses_below=True,
        busiest=False,
    ),
    "goose": Floor(
        "Goose",
        32_768,
        "its docs state no minimum; its prompt and extensions' tools are long",
    ),
    "aider": Floor(
        "Aider",
        8_192,
        "its prompts are the repo map and the files you add; not measured here",
    ),
    "cline": Floor(
        "Cline",
        32_768,
        "its system prompt and tool definitions are long; not measured here",
    ),
    "zed": Floor("Zed", 16_384, "its agent's prompt and tools; not measured here"),
    "llama-vscode": Floor(
        "llama.vscode",
        4_096,
        "a completion sends the code around the cursor, a few thousand tokens",
    ),
}

AGENT_CTX = max(floor.tokens for floor in FLOORS.values() if floor.busiest)
BUSIEST = Floor(  # `--for-agents` without a client: every floor counted in AGENT_CTX
    "The busiest agent client",
    AGENT_CTX,
    "Claude Code, OpenCode, Qwen Code, OpenHands, Goose and Cline each want that much",
)


def floor_for(client: str | None) -> Floor:
    """`--for <client>`'s floor, or the busiest one (`--for-agents`). An unknown name is a
    PlanError naming the ones there are."""
    if client is None:
        return BUSIEST
    found = FLOORS.get(client.strip().lower())
    if found is None:
        raise PlanError(f"No agent client called {client!r}; there are {', '.join(FLOORS)}.")
    return found


def requested_type(model_type: Modality | None, for_agents: bool) -> Modality | None:
    """The type `mnm add` plans as: `--type` when given, else coding for `--for-agents`."""
    return model_type or (Modality.CODING if for_agents else None)


def add_page_hints(plan: Plan) -> tuple[list[Modality], int | None]:
    """What the web add page says under a plan: the other types the model can be added as,
    and the agent context when it is a chat-shaped model."""
    others = sorted(RETYPABLE.get(plan.facts.modality, frozenset()))
    return others, agent_context(plan, None) if plan.facts.modality in CHAT_LIKE else None


def agent_context(plan: Plan, ctx: int | None, client: str | None = None) -> int:
    """The context an agent instance plans with: the user's own when given, else the
    client's floor (the busiest client's without one), or the model's native context when
    that is smaller. A client that refuses less than its floor (Hermes Agent, at its own
    startup) refuses the model here, `--ctx` or not (`PlanError`), before anything is saved."""
    floor = floor_for(client)
    native = plan.header.context_length if plan.header is not None else None
    if native is None and plan.hf_config is not None:  # a vLLM plan
        native = plan.hf_config.text.max_position_embeddings
    planned = ctx if ctx is not None else min(floor.tokens, native or floor.tokens)
    if floor.refuses_below and planned < floor.tokens:
        why = (
            f"--ctx {ctx:,} is less"
            if ctx is not None
            else f"this model's own context is {native:,}; pick a model trained for "
            f"{floor.tokens:,} or more (Qwen3.6 and Qwen3.8 take 262,144)"
        )
        raise PlanError(
            f"{floor.label} needs {floor.tokens:,} tokens a request ({floor.why}): {why}."
        )
    return planned


def serves(floor: Floor, have: int, how: str) -> str:
    """The floor line, under `mnm add`'s plan and on every agent card: what the client needs
    a request, what this instance serves (`how`: `ctx 32,768 ÷ parallel 1`, or vLLM's
    `max_model_len`), and whether that is enough."""
    said = f"{floor.label} needs {floor.tokens:,} tokens a request; this instance serves {have:,}"
    enough = "enough" if have >= floor.tokens else "too few"
    return f"{said} ({how}): {enough} ({floor.why})."
