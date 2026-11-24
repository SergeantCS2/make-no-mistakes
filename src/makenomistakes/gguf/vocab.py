"""A GGUF's vocabulary, read whole from a local file: what decides whether one model can
draft for another (speculative decoding).

`read_header` stops before the tokenizer arrays; the begin and end ids and the add flags sit
after them in every real file, and the token texts are the arrays themselves. This reader
walks the whole metadata section (a few megabytes for a 150,000-token vocabulary) and keeps
only the tokenizer's keys. It is for files on this machine, read once when a draft is set.
"""

from __future__ import annotations

from pydantic import BaseModel

from makenomistakes.gguf.reader import (
    GGUF_MAGIC,
    SUPPORTED_VERSIONS,
    ByteSource,
    GgufError,
    Scalar,
    ValueType,
    _check_count,
    _Cursor,
    _read_value,
    _skip_array,
    _value_type,
)

TOKENS_KEY = "tokenizer.ggml.tokens"
# llama.cpp's `common_speculative_are_compatible` (common/speculative.cpp): token texts are
# compared from this id on, and the vocabularies may differ in size by at most this much.
CHECK_START_TOKEN_ID = 5
MAX_SIZE_DIFFERENCE = 128


class Vocab(BaseModel):
    """The tokenizer keys llama.cpp compares between a target and its draft."""

    model: str | None  # `tokenizer.ggml.model`: llama (SPM), gpt2 (BPE), bert (WPM) …
    tokens: tuple[str, ...]
    bos: int | None = None
    eos: int | None = None
    add_bos: bool | None = None
    add_eos: bool | None = None


def read_vocab(source: ByteSource) -> Vocab:
    """Every tokenizer key of a GGUF's metadata; GgufError for a file that is not one."""
    cursor = _Cursor(source)
    if cursor.read(4) != GGUF_MAGIC:
        raise GgufError("not a GGUF file (bad magic)")
    if cursor.u32() not in SUPPORTED_VERSIONS:
        raise GgufError("unsupported GGUF version")
    cursor.u64()  # tensor count
    kv_count = cursor.u64()
    values: dict[str, Scalar] = {}
    tokens: tuple[str, ...] = ()
    for _ in range(kv_count):
        key = cursor.string()
        value_type = _value_type(cursor.u32())
        if key == TOKENS_KEY and value_type is ValueType.ARRAY:
            tokens = _strings(cursor)
            continue
        value = _read_value(cursor, value_type)
        if value is not None and key.startswith("tokenizer.ggml."):
            values[key] = value
    return Vocab(
        model=_text(values.get("tokenizer.ggml.model")),
        tokens=tokens,
        bos=_whole(values.get("tokenizer.ggml.bos_token_id")),
        eos=_whole(values.get("tokenizer.ggml.eos_token_id")),
        add_bos=_flag(values.get("tokenizer.ggml.add_bos_token")),
        add_eos=_flag(values.get("tokenizer.ggml.add_eos_token")),
    )


def incompatible(target: Vocab, draft: Vocab) -> str | None:
    """Why `draft` cannot draft for `target`, in words, or None when it can: llama.cpp's own
    rule, which refuses to start the server otherwise ("the target and draft vocabs are not
    compatible")."""
    if target.model != draft.model:
        return (
            f"the vocabularies are of different kinds ({target.model or 'unknown'} against "
            f"{draft.model or 'unknown'})"
        )
    for edge, add in (("begin", "add_bos"), ("end", "add_eos")):
        wanted, offered = getattr(target, add), getattr(draft, add)
        token = "bos" if edge == "begin" else "eos"
        same_id = getattr(target, token) == getattr(draft, token)
        if wanted != offered or (wanted and not same_id):
            return f"their {edge}-of-text tokens differ ({token} id or whether it is added)"
    size, other = len(target.tokens), len(draft.tokens)
    if abs(size - other) > MAX_SIZE_DIFFERENCE:
        return (
            f"the vocabularies differ in size by {abs(size - other):,} tokens ({size:,} against "
            f"{other:,}; at most {MAX_SIZE_DIFFERENCE} may differ)"
        )
    for token_id in range(CHECK_START_TOKEN_ID, min(size, other)):
        if target.tokens[token_id] != draft.tokens[token_id]:
            return (
                f"token {token_id:,} differs ({target.tokens[token_id]!r} against "
                f"{draft.tokens[token_id]!r})"
            )
    return None


TEMPLATE_KEY = "tokenizer.chat_template"


def read_chat_template(source: ByteSource) -> str | None:
    """The GGUF's own chat template (`tokenizer.chat_template`, which llama-server runs with
    `--jinja`), or None when the file carries none. It sits after the token arrays, which are
    skipped, not read (for the manifest: whether the template declares tools)."""
    cursor = _Cursor(source)
    if cursor.read(4) != GGUF_MAGIC:
        raise GgufError("not a GGUF file (bad magic)")
    if cursor.u32() not in SUPPORTED_VERSIONS:
        raise GgufError("unsupported GGUF version")
    cursor.u64()  # tensor count
    for _ in range(cursor.u64()):
        key = cursor.string()
        value = _read_value(cursor, _value_type(cursor.u32()))
        if key == TEMPLATE_KEY:
            return _text(value)
    return None


def _strings(cursor: _Cursor) -> tuple[str, ...]:
    element_type = _value_type(cursor.u32())
    count = cursor.u64()
    if element_type is not ValueType.STRING:
        _skip_array(cursor, element_type, count)
        return ()
    _check_count(cursor, element_type, count)
    return tuple(cursor.string() for _ in range(count))


def _text(value: Scalar | None) -> str | None:
    return value if isinstance(value, str) else None


def _whole(value: Scalar | None) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _flag(value: Scalar | None) -> bool | None:
    return value if isinstance(value, bool) else None
