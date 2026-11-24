"""The roles a set's member plays (`mnm sets create desk coder=qwen36 chat=qwen3-8b …`), and
the types that play each. The roles of the default set (`mnm sets default`,
the *desk*) are also model ids clients name (`desk.Aliases`), so a role word is never a new
instance's name (`InstanceService.check_name`). `workflows` re-exports both tables."""

from __future__ import annotations

from makenomistakes.hub.facts import Modality

# The role's name is what the user types, and what an app names as its model.
ROLES: dict[str, frozenset[Modality]] = {
    "chat": frozenset({Modality.CHAT, Modality.CODING}),
    "coder": frozenset({Modality.CODING, Modality.CHAT}),
    "vision": frozenset({Modality.VISION_CHAT}),
    "image": frozenset({Modality.IMAGE_GEN}),
    # an editor (Kontext) edits by instruction; any image model repaints and inpaints
    "edit": frozenset({Modality.IMAGE_EDIT, Modality.IMAGE_GEN}),
    "video": frozenset({Modality.VIDEO}),
    "voice": frozenset({Modality.TEXT_TO_SPEECH}),
    "ear": frozenset({Modality.SPEECH_TO_TEXT}),
    "memory": frozenset({Modality.EMBEDDING}),
    "rerank": frozenset({Modality.RERANK}),
}
ROLE_ORDER = tuple(ROLES)
# The roles an app chats with (the OpenAI chat API), in the order an app's list offers them:
# the coder first, since the workspace apps are agents.
CHAT_ROLES = ("coder", "chat", "vision")
