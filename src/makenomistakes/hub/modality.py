"""The model types and the groups the rest of the system reads them
by: one engine, one fit model, one controls panel, one smoke test, one card family each.
Kept apart from `hub/facts.py`, which re-exports every name."""

from __future__ import annotations

from enum import StrEnum


class Modality(StrEnum):
    """The model type: one engine, one fit model, one controls panel, one
    smoke test, one card family each. The saved field is `modality`; `label` is what the
    user reads in `mnm ls`, the web and the cards."""

    CHAT = "chat"
    CODING = "coding"  # a chat model used by coding agents and editors: context, tools, FIM
    VISION_CHAT = "vision_chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"
    IMAGE_GEN = "image_gen"
    IMAGE_EDIT = "image_edit"  # image + text to image: edit-trained models (Kontext, Fill …)
    VIDEO = "video"  # text or image to video (Wan, LingBot, HunyuanVideo, LTX-2): `vid_gen`
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    # The studio: audio.cpp's `/v1/tasks/run` (not an OpenAI route), one type per verb
    SONG = "song"  # tags and lyrics to a song (ACE-Step, HeartMuLa …)
    STEMS = "stems"  # a mix split into stems (the RoFormers, HTDemucs)
    VOICE_CONVERT = "voice_convert"  # a vocal sung again in another voice (RVC, Seed-VC …)
    MUSIC_SCORE = "music_score"  # audio to a score (MuScriptor, SheetSage2)
    UNKNOWN = "unknown"

    @property
    def label(self) -> str:
        return TYPE_LABELS[self]

    @property
    def a_label(self) -> str:
        """The label with its article, for sentences: "an image", "a chat"."""
        label = TYPE_LABELS[self]
        return f"{'an' if label[:1] in 'aeiou' else 'a'} {label}"


TYPE_LABELS: dict[Modality, str] = {
    Modality.CHAT: "chat",
    Modality.CODING: "coding",
    Modality.VISION_CHAT: "vision",
    Modality.EMBEDDING: "embedding",
    Modality.RERANK: "rerank",
    Modality.IMAGE_GEN: "image",
    Modality.IMAGE_EDIT: "image edit",
    Modality.VIDEO: "video",
    Modality.SPEECH_TO_TEXT: "speech-to-text",
    Modality.TEXT_TO_SPEECH: "text-to-speech",
    Modality.SONG: "song",
    Modality.STEMS: "stems",
    Modality.VOICE_CONVERT: "voice conversion",
    Modality.MUSIC_SCORE: "music score",
    Modality.UNKNOWN: "unknown",
}
TYPE_ORDER: tuple[Modality, ...] = tuple(TYPE_LABELS)  # how lists group: language first
# Chat-shaped: the chat API, tools, thinking; one profile ladder; the LLM endpoint in Odysseus.
CHAT_LIKE = frozenset({Modality.CHAT, Modality.CODING, Modality.VISION_CHAT})
EMBEDDING_MODALITIES = frozenset({Modality.EMBEDDING, Modality.RERANK})  # whole-input batches
LLAMACPP_MODALITIES = CHAT_LIKE | EMBEDDING_MODALITIES
# Diffusion models on stable-diffusion.cpp: image weights, kits and families apply to all.
SDCPP_MODALITIES = frozenset({Modality.IMAGE_GEN, Modality.IMAGE_EDIT, Modality.VIDEO})
VLLM_MODALITIES = CHAT_LIKE  # safetensors-only chat, coding and vision models
# The studio's types: audio.cpp's task route, never listed to OpenAI or Ollama clients
# (`/v1/models`, `/api/tags`: no chat client can use `/v1/tasks/run`).
MUSIC_MODALITIES = frozenset(
    {Modality.SONG, Modality.STEMS, Modality.VOICE_CONVERT, Modality.MUSIC_SCORE}
)
SERVABLE_MODALITIES = (
    LLAMACPP_MODALITIES
    | SDCPP_MODALITIES
    | MUSIC_MODALITIES
    | {
        Modality.SPEECH_TO_TEXT,
        Modality.TEXT_TO_SPEECH,
    }
)
# The types a plan may be retyped to (`mnm add --type`): the same engine and endpoint,
# so only the cards, the grouping and the agent profile change.
RETYPABLE: dict[Modality, frozenset[Modality]] = {
    Modality.CHAT: frozenset({Modality.CODING}),
    Modality.CODING: frozenset({Modality.CHAT}),
    # Coding models such as Qwen3.6 and Qwen3.8 ship a vision projector; added as
    # coding or chat they run text-only (`plan.retyped` drops it: 1.4 GiB of the card an
    # agent does not use). Added as they are, they keep their vision cards.
    Modality.VISION_CHAT: frozenset({Modality.CODING, Modality.CHAT}),
}
