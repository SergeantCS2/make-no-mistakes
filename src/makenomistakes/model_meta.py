"""What a model's own files say about it, for the router's listings: the context it was
trained for (`/v1/models`' `meta.n_ctx_train`), its size on disk, its architecture and size
label (the Ollama API's `/api/tags` and `/api/show`).

A GGUF header is read up to its tokenizer arrays (a few MB of the file) and a vLLM model's
config.json whole, so each instance's answer is kept while its files stay the same (the same
paths, the first file's size and modification time): Odysseus asks `/v1/models` before every
message to a local endpoint (`src/model_context.py:246-266` in 0.20.19 never caches one)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from makenomistakes.engine_args import LlamaCppArgs
from makenomistakes.engine_args_vllm import VllmArgs
from makenomistakes.gguf.reader import FileSource, GgufError
from makenomistakes.gguf.vocab import read_chat_template
from makenomistakes.hub.facts import CHAT_LIKE, Engine
from makenomistakes.model_files import read_model_files
from makenomistakes.recipes import Recipe

if TYPE_CHECKING:
    from makenomistakes.measurements import Args

FORMATS = {Engine.LLAMACPP: "gguf", Engine.VLLM: "safetensors"}


@dataclass(frozen=True)
class ModelMeta:
    trained_ctx: int | None = None  # the GGUF's `<arch>.context_length`, config.json's window
    size_bytes: int = 0  # the weights and the projector on disk
    architecture: str | None = None  # `general.architecture`, or config.json's `model_type`
    size_label: str | None = None  # `general.size_label` ("8B"), when the file carries one
    embedding_length: int | None = None  # `<arch>.embedding_length`, config.json's hidden_size
    parameter_count: int | None = None  # `general.parameter_count`, when the file carries one


def serving_context(recipe: Recipe) -> int | None:
    """The context one request gets (`context_of`, of the recipe's args)."""
    return context_of(recipe.args)


def context_of(args: Args) -> int | None:
    """The context one request gets with these args: llama-server splits `-c` across its
    slots (what its own `/slots` reports as `n_ctx`); vLLM gives every request up to
    `--max-model-len`. None for an engine without a context (pictures, clips, voices,
    whisper). A reading's args too (the router's queue: the context a request held, against
    this)."""
    if isinstance(args, VllmArgs):
        return args.max_model_len
    if isinstance(args, LlamaCppArgs):
        return args.ctx // max(args.parallel, 1)
    return None


def serving_slots(recipe: Recipe) -> int:
    """The slots one context is split into (`serving_context`): llama-server's `-np`; one for
    every other engine, vLLM's included, which gives each request its whole context. What an
    engine serves at once is `at_once`."""
    args = recipe.args
    return max(args.parallel, 1) if isinstance(args, LlamaCppArgs) else 1


def at_once(args: Args) -> int:
    """The requests an engine serves at once with these args (the router's queue):
    llama-server's slots, vLLM's `--max-num-seqs`; sd-server, whisper-server and audio.cpp's
    server one, each running one request at a time behind one lock (read in their sources;
    mnm never passes audio.cpp's experimental `--parallel-jobs`)."""
    if isinstance(args, LlamaCppArgs):
        return max(args.parallel, 1)
    if isinstance(args, VllmArgs):
        return max(args.max_num_seqs, 1)
    return 1


def read_meta(recipe: Recipe, models_dir: Path) -> ModelMeta:
    """The files' facts, or an empty `ModelMeta` when they cannot be read (not downloaded,
    a file gone): a listing never fails for it."""
    files = read_model_files(recipe, models_dir)
    if files is None:
        return ModelMeta()
    size = files.weights_bytes + files.projector_bytes
    if files.header is not None:
        header = files.header
        return ModelMeta(
            header.context_length,
            size,
            header.architecture,
            header.size_label,
            header.embedding_length,
            header.parameter_count,
        )
    if files.hf_config is not None:
        config = files.hf_config
        trained = config.text.max_position_embeddings
        return ModelMeta(
            trained, size, config.model_type or None, embedding_length=config.text.hidden_size
        )
    return ModelMeta(size_bytes=size)


class MetaCache:
    """`read_meta` per instance, read again only when its files change. Thread-safe: the
    router's request threads share one."""

    def __init__(
        self,
        models_dir: Callable[[], Path],
        read: Callable[[Recipe, Path], ModelMeta] = read_meta,
    ) -> None:
        self._models_dir = models_dir
        self._read = read
        self._lock = threading.Lock()
        self._kept: dict[str, tuple[object, ModelMeta]] = {}
        self._tools: dict[str, tuple[object, bool | None]] = {}

    def get(self, recipe: Recipe) -> ModelMeta:
        if recipe.engine not in FORMATS or not recipe.downloaded:
            return ModelMeta()  # nothing a listing reads, and no file to look at
        models_dir = self._models_dir()
        key = _files_key(recipe, models_dir)
        if key is None:
            return ModelMeta()
        with self._lock:
            kept = self._kept.get(recipe.name)
        if kept is not None and kept[0] == key:
            return kept[1]
        meta = self._read(recipe, models_dir)
        with self._lock:
            self._kept[recipe.name] = (key, meta)
        return meta

    def tools(self, recipe: Recipe) -> bool | None:
        """Whether the model's template declares tools (`declares_tools`), read once per set
        of files and apart from `get`: the whole metadata is walked for it (a fraction of a
        second), which a model list asked before every message must not wait for."""
        models_dir = self._models_dir()
        key = _files_key(recipe, models_dir) if recipe.downloaded else None
        if key is None:  # nothing on disk to read, or no file to key a reading on
            return declares_tools(recipe, models_dir)
        with self._lock:
            kept = self._tools.get(recipe.name)
        if kept is not None and kept[0] == key:
            return kept[1]
        found = declares_tools(recipe, models_dir)
        with self._lock:
            self._tools[recipe.name] = (key, found)
        return found


def declares_tools(recipe: Recipe, models_dir: Path) -> bool | None:
    """Tool calls as the model's own template declares them (the Hub's test at add time,
    `facts.supports_tools`, run on the file the engine reads): a llama.cpp GGUF's
    `tokenizer.chat_template` names `tools`; vLLM calls tools only through a parser, which
    mnm picks by family (`VllmArgs.tool_parser`). None: no chat template is involved (a picture,
    a voice, an embedding), or the file cannot be read."""
    args = recipe.args
    if isinstance(args, VllmArgs):
        return args.tool_parser is not None
    if not isinstance(args, LlamaCppArgs) or recipe.modality not in CHAT_LIKE:
        return None
    first = recipe.local_paths.get(recipe.files[0]) if recipe.files else None
    if first is None:
        return None
    try:
        template = read_chat_template(FileSource(models_dir / first))
    except (OSError, GgufError):
        return None
    return template is not None and "tools" in template


def _files_key(recipe: Recipe, models_dir: Path) -> object | None:
    """What says the files are the ones read before; None when the first is not there."""
    first = recipe.local_paths.get(recipe.files[0]) if recipe.files else None
    if first is None:
        return None
    try:
        found = (models_dir / first).stat()
    except OSError:
        return None
    return (tuple(sorted(recipe.local_paths.items())), found.st_size, found.st_mtime_ns)
