"""whisper-server's flags (speech-to-text), out of `engine_args` to make room there for
llama.cpp's prompt cache."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, Field, field_validator

from makenomistakes.engine_args_languages import WHISPER_LANGUAGES


class WhisperArgs(BaseModel):
    """Flags for whisper-server (speech-to-text)."""

    engine: Literal["whispercpp"] = "whispercpp"
    # Settings that leave what it holds on the GPU as planned: changing only these keeps the
    # saved estimate (`Recipe.with_args`; the buffers follow the size family).
    MEMORY_NEUTRAL: ClassVar[frozenset[str]] = frozenset({"language", "threads", "convert"})
    language: str = "auto"  # `auto` or one of WHISPER_LANGUAGES
    threads: int | None = Field(default=None, ge=1)
    convert: bool = (
        True  # ffmpeg in the image turns webm/mp3 uploads into WAV (Odysseus sends webm)
    )
    extra: list[str] = Field(default_factory=list)

    @field_validator("language", mode="before")
    @classmethod
    def _lower(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("language")
    @classmethod
    def _known(cls, value: str) -> str:
        if value != "auto" and value not in WHISPER_LANGUAGES:
            raise ValueError(
                f"'{value}' is not a whisper language; use auto or a code such as en, de, ja, "
                "zh, yue (ISO 639-1 where one exists)"
            )
        return value

    @property
    def summary(self) -> str:
        return f"language {self.language}"

    def to_cli(self) -> list[str]:
        argv = ["-l", self.language]
        if self.threads is not None:
            argv += ["-t", str(self.threads)]
        if self.convert:
            argv.append("--convert")
        return argv + list(self.extra)
