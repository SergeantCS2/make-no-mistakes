"""Map engine log output to a cause and a remedy a non-expert can act on.

Patterns come from llama.cpp's own failure messages and from the failure table Odysseus
built up over its issue tracker. They match failure text only: a healthy log prints
`clip_model_loader`, `n_ctx = 4096`, `11401 MiB free` and must produce nothing.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from makenomistakes.backend import Backend
from makenomistakes.diagnose_backends import BACKEND_RULES
from makenomistakes.diagnose_rules import (
    ALL_ENGINES,
    CUDA_ONLY,
    DOWNLOAD_ONLY,
    OOM_PATTERN,
    Cause,
    Diagnosis,
    Rule,
    make_rule,
)
from makenomistakes.engine_support import MERGES, UPDATE
from makenomistakes.hub.facts import Engine
from makenomistakes.log_notes import note_for

__all__ = ["ALL_ENGINES", "Cause", "Diagnosis", "LogLine", "LogReview", "diagnose", "review_log"]


_REDOWNLOAD = (
    "Run `mnm start` again (downloads resume). If it repeats, `mnm rm` and `mnm add` to "
    "re-download."
)

RULES: tuple[Rule, ...] = (
    *BACKEND_RULES,  # ROCm's and Vulkan's own lines first, each scoped to its backend
    make_rule(
        Cause.NO_GPU_RUNTIME,
        r"could not select device driver|nvidia-container-cli|unknown or invalid runtime name",
        "Docker cannot see the GPU (NVIDIA container runtime missing).",
        "Run `mnm setup` (or `mnm doctor` for the manual steps): it installs the NVIDIA "
        "Container Toolkit and registers the runtime with Docker.",
        backends=CUDA_ONLY,
    ),
    make_rule(
        Cause.CUDA_DRIVER,
        r"CUDA driver version is insufficient|no CUDA-capable device|cudaErrorNoDevice|"
        r"failed to initialize CUDA|unable to find cudart|cudaErrorInsufficientDriver",
        "The NVIDIA driver is missing or older than the engine image needs.",
        "Under WSL: update the NVIDIA Windows driver, run `wsl --shutdown`, then try again "
        "(never install a Linux NVIDIA driver inside WSL). On native Linux: `mnm setup` "
        "installs or updates the driver; reboot afterwards.",
        backends=CUDA_ONLY,
    ),
    make_rule(
        Cause.OUT_OF_MEMORY,
        OOM_PATTERN,
        "Out of GPU memory while loading.",
        "Lower the context (`mnm set <name> ctx=8192`; 2048 or 1024 for an embedding or rerank "
        "instance), on chat and vision also `kv=q8_0`, or pick a smaller quant (`mnm rm <name>` "
        "then `mnm add <ref> --profile speed`), or start it when more of the card is free.",
        engines=frozenset({Engine.LLAMACPP}),
        # llama.cpp ends every failed load with "failed to load model": not a corrupt file.
        supersedes=frozenset({Cause.MISSING_FILE}),
    ),
    make_rule(
        Cause.OUT_OF_MEMORY,
        OOM_PATTERN,
        "Out of GPU memory while loading.",
        "Lower the resolution (`mnm set <name> width=768 height=768`), turn on `vae-tiling` or "
        "`offload` (weights stream from RAM), stop another instance, or start it when more of "
        "the card is free.",
        engines=frozenset({Engine.SDCPP}),
        supersedes=frozenset({Cause.SD_LOAD_FAILED, Cause.MISSING_FILE}),  # same start ends so
    ),
    make_rule(
        Cause.OUT_OF_MEMORY,
        OOM_PATTERN,
        "Out of GPU memory while loading.",
        "Stop another instance (`mnm ls` shows what holds VRAM) or pick a smaller size: "
        "`mnm rm <name>` then `mnm add <ref> --profile speed`.",
        engines=frozenset({Engine.WHISPERCPP}),
        supersedes=frozenset({Cause.MISSING_FILE}),
    ),
    make_rule(
        Cause.OUT_OF_MEMORY,
        OOM_PATTERN,
        "Out of GPU memory while loading.",
        "Stop another instance (`mnm ls` shows what holds VRAM) or pick the 8-bit file: "
        "`mnm rm <name>` then `mnm add <ref> --profile balanced`.",
        engines=frozenset({Engine.AUDIOCPP}),  # ggml's CUDA wording; not yet seen on a card
        supersedes=frozenset({Cause.MISSING_FILE, Cause.VOICE_LOAD_FAILED}),
    ),
    make_rule(  # vLLM checks the free VRAM against its share before it loads anything
        Cause.OUT_OF_MEMORY,
        r"Free memory on device .* is less than desired GPU memory utilization",
        "Less of the card is free than the share vLLM was asked to take.",
        "Stop another instance (`mnm start <name> --stop-others`), start it when more of the "
        "card is free, or lower the context so the next start asks for less (`mnm set <name> "
        "ctx=8192`, or `kv=fp8`).",
        engines=frozenset({Engine.VLLM}),
    ),
    make_rule(  # the share holds the weights but not one full-length sequence of KV cache
        Cause.CTX_TOO_LARGE,
        r"larger than the maximum number of tokens that can be stored in KV cache|"
        r"KV cache is needed, which is larger than the available KV cache memory|"
        r"No available memory for the cache blocks",
        "vLLM's share of the card cannot hold the KV cache for the context it was given.",
        "Lower the context (`mnm set <name> ctx=8192`) or the KV precision (`kv=fp8`); the "
        "next start sizes the share for it.",
        engines=frozenset({Engine.VLLM}),
        supersedes=frozenset({Cause.OUT_OF_MEMORY}),
    ),
    make_rule(  # a context past what the model was trained for (config.json's positions)
        Cause.CTX_TOO_LARGE,
        r"User-specified max_model_len .* is greater than the derived max_model_len",
        "The context is longer than the model was trained for.",
        "Lower it to the model's own (`mnm show <name>` names it): `mnm set <name> ctx=…`.",
        engines=frozenset({Engine.VLLM}),
    ),
    make_rule(
        Cause.OUT_OF_MEMORY,
        r"torch\.OutOfMemoryError|CUDA out of memory",
        "Out of GPU memory while vLLM loaded the model or captured its CUDA graphs.",
        "Turn on eager mode (`mnm set <name> eager=on`, about 0.5 GiB less), lower the context, "
        "or use a smaller (AWQ or FP8) build of the model.",
        engines=frozenset({Engine.VLLM}),
    ),
    make_rule(  # the same start also ends in `new_sd_ctx_t failed`, which would blame the kit
        Cause.SD_WRONG_FAMILY,
        r"tensor '[^']+' (?:has wrong shape in|not in) model (?:metadata|file)",
        "The weights do not match the model stable-diffusion.cpp built from them: its "
        "`Version:` line names the family it read them as, and their tensors belong to "
        "another one (or a companion file, such as the VAE, does).",
        "Compare that line, near the top of `mnm logs <name> --tail 0`, with the family "
        "`mnm show <name>` names. If they differ, "
        "send `mnm logs <name> --review`: the engine needs a flag mnm does not pass yet, and "
        "the model's single-file checkpoint (one .safetensors) is the way round it until then. "
        "If they match, a companion file belongs to another family.",
        engines=frozenset({Engine.SDCPP}),
        supersedes=frozenset({Cause.SD_LOAD_FAILED}),
    ),
    make_rule(  # logged at info level, right before the engine aborts
        Cause.LORA_NOT_APPLIED,
        r"\(0 / [1-9]\d*\) LoRA tensors have been applied",
        "None of a LoRA's tensors reached the model: it was trained for another model, or "
        "the engine tried to merge it into weights it converts while loading.",
        "`mnm lora ls <name>` shows the LoRAs; remove the one the log names with `mnm lora rm "
        "<name> <lora>` and start again. If its family matches, send `mnm logs <name> "
        "--review`: mnm already asks the engine to apply LoRAs at run time on weights it "
        "converts, so this one is a new case.",
        engines=frozenset({Engine.SDCPP}),
        # the abort's backtrace names a source file it cannot open (`accept4.c: No such file`)
        supersedes=frozenset({Cause.SD_LOAD_FAILED, Cause.MISSING_FILE}),
    ),
    make_rule(
        Cause.SD_LOAD_FAILED,
        r"new_sd_ctx_t failed|failed to load (?:diffusion|vae|clip|t5|llm)|"
        r"unknown model type|could not determine model|unsupported model",
        "stable-diffusion.cpp could not load one of the kit's files.",
        "Pull a newer engine build (`mnm engines update sdcpp`), then `mnm start` again; if it "
        "repeats, the companion file in hub/kits.py no longer matches this model family. Send "
        "`mnm logs <name> --review`.",
        engines=frozenset({Engine.SDCPP}),
    ),
    make_rule(
        Cause.PORT_TAKEN,
        # The engine's own words, and Docker's when another container publishes the port.
        r"address already in use|bind\(\) failed|port \d+ is already|port is already allocated",
        "The port is taken by another process.",
        "`mnm ls` shows whether an instance owns it (stop that one). Otherwise another program "
        "holds it: stop it, or `mnm rm <name>` and `mnm add` again, which skips ports that are "
        "in use.",
    ),
    make_rule(
        Cause.KV_NEEDS_FLASH_ATTN,
        r"quantized V cache requires flash.?attn",
        "A quantised KV cache (q8_0, q4_0) runs only with flash attention, which is off.",
        "`mnm set <name> flash-attn=auto` (or `kv=f16` to keep flash attention off).",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(  # An architecture whose merge is known names it (`engine_support.MERGES`)
        Cause.IMAGE_TOO_OLD,
        r"unknown model architecture: '?deepseek41\b",
        f"This llama.cpp image is older than DeepSeek-V4.1 (arch `deepseek41`), which "
        f"{MERGES['deepseek41'].words} brought.",
        f"{UPDATE[0].upper()}{UPDATE[1:]}, then `mnm start` again (the official images follow "
        "master by hours to days; `mnm engines rollback llamacpp` undoes an update).",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(  # Unsloth's MXFP8 Engram tables, which a build before 2026-10-29 refused
        Cause.QUANT_DROPPED,
        r"has invalid ggml type 43\b",
        "This quant's Engram table is MXFP8 (ggml type 43 in this file), which llama.cpp does "
        "not read.",
        "Add a quant that loads (UD-IQ1_S, the UD-IQ2 ones or UD-Q2_K_XL: `mnm fit` lists "
        "them); unsloth re-uploads these quants.",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(
        Cause.IMAGE_TOO_OLD,
        r"unknown model architecture|unknown architecture|unsupported model type|"
        r"invalid magic characters|unknown \(magic",
        "This llama.cpp image is older than the model format.",
        "Pull a newer engine build: `mnm engines update llamacpp`, then `mnm start` again "
        "(`mnm engines rollback llamacpp` undoes it if another model breaks).",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(
        Cause.IMAGE_TOO_OLD,
        r"error: invalid argument: --(?:fit|moe-cache-mib|spec-[a-z-]+)\b|"
        r"unknown speculative type: draft-mtp",
        "This llama.cpp image is older than a flag the recipe passes (`--fit off`, "
        "`--moe-cache-mib`, `--spec-type draft-mtp`).",
        "Pull a newer engine build: `mnm engines update llamacpp`, then `mnm start` again "
        "(`mnm engines rollback llamacpp` undoes it if another model breaks).",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(
        Cause.CTX_TOO_LARGE,
        r"n_ctx_per_seq \(\d+\) > n_ctx_train|possible training context overflow|"
        r"requested n_ctx .* exceeds|context length .* exceeds the trained",
        "The context is larger than the model was trained for.",
        "Set it at or below the native context shown by `mnm fit`: `mnm set <name> ctx=...`.",
    ),
    make_rule(
        Cause.PROJECTOR_FAILED,
        r"failed to load (?:the )?(?:mmproj|projector|vision|clip)|"
        r"(?:mmproj|projector)[^\n]*(?:not found|no such file|failed)",
        "The vision projector did not load.",
        "Make sure the projector file downloaded (`mnm start` resumes downloads); "
        "the mmproj must come from the same repo as the weights.",
        engines=frozenset({Engine.LLAMACPP}),
    ),
    make_rule(
        Cause.MISSING_FILE,
        r"failed to load model|error loading model|no such file or directory|"
        r"cannot open|failed to open gguf|failed to read",
        "The model file is missing or corrupt.",
        _REDOWNLOAD,
    ),
    make_rule(
        Cause.MISSING_FILE,
        # whisper prints "failed to initialize whisper context" for every load failure, out
        # of memory included, so only its own file messages count.
        r"failed to open '|invalid model data \(bad magic\)",
        "The model file is missing or corrupt.",
        _REDOWNLOAD,
        engines=frozenset({Engine.WHISPERCPP}),
    ),
    make_rule(
        Cause.MISSING_FILE,
        # audio.cpp reads a file's format from its name and refuses a link or a file without
        # the extension (`unsupported tensor source format: <path>`, the HF cache's blobs).
        r"failed to open (?:tensor source|binary file|safetensors file|text file)|"
        r"unsupported tensor source format|no model file|model path does not exist",
        "The voice package's file is missing, or not a plain file under voices/.",
        _REDOWNLOAD,
        engines=frozenset({Engine.AUDIOCPP}),
    ),
    make_rule(
        Cause.VOICE_NEEDS_ESPEAK,
        # Each request's answer and its `http.error` line: the official image has no
        # eSpeak-ng, and the server starts healthy without it.
        r"Could not load eSpeak-ng",
        "audio.cpp's official image has no eSpeak-ng, which Kokoro needs to turn text into "
        "phonemes, so every request fails.",
        "Use PocketTTS or Qwen3-TTS (`mnm add audio-cpp/audio.cpp-gguf/PocketTTS-GGUF/english "
        "--name pocket`); `mnm rm <name> --files` removes this instance and its download.",
        engines=frozenset({Engine.AUDIOCPP}),
    ),
    make_rule(
        Cause.STYLE_MISMATCH,
        # Audio.cpp checks a style's LoRA against the DiT before it merges anything
        # (`weight_adapter.cpp`); one shaped for another model stops the load here
        r"audiocpp_server failed: tensor shape mismatch for decoder\.",
        "The style set on this instance does not fit its model: the adapter's tensors are "
        "shaped for another one (trained elsewhere, or on another ACE-Step).",
        "`mnm set <name> style=` clears it; `mnm studio styles show <style>` says what it was "
        "trained on.",
        engines=frozenset({Engine.AUDIOCPP}),
        supersedes=frozenset({Cause.VOICE_LOAD_FAILED}),
    ),
    make_rule(
        Cause.VOICE_LOAD_FAILED,
        # `audiocpp_server failed: <what>` is the one fatal line (app/server/main.cpp); the
        # cause is in <what>, so the file and memory rules above explain it when they match.
        r"audiocpp_server failed:",
        "The voice engine could not load this package.",
        "Read the line after `audiocpp_server failed:` in `mnm logs <name>`.",
        engines=frozenset({Engine.AUDIOCPP}),
    ),
    make_rule(
        Cause.HUB_REFUSED,
        # An engine never talks to the Hub, and a healthy 400-line server log easily contains
        # a bare 401 or 403 (slot ids, token counts), so this needs HTTP-client wording.
        r"\b40[13] client error|\bgated\b|\bunauthorized\b|\bforbidden\b|"
        r"access to model .* is restricted|cannot access gated repo",
        "The Hub refused access (gated or private repo).",
        "Accept the model's terms on huggingface.co and set HF_TOKEN in "
        "~/.config/makenomistakes/secrets.env.",
        engines=DOWNLOAD_ONLY,
    ),
)


def diagnose(
    log: str, engine: Engine | None = None, backend: Backend | None = None
) -> list[Diagnosis]:
    """Distinct diagnoses in rule order, most specific first.

    `engine` scopes the rules to that engine's log; `None` (a download or Docker error, not
    an engine log) runs every rule, including the `DOWNLOAD_ONLY` ones. `backend` (the
    instance's image's) scopes them to its backend's lines; `None` runs every backend's.
    """
    found: dict[Cause, Diagnosis] = {}  # one per cause: the first matching rule's wording
    explained: set[Cause] = set()
    for rule in RULES:
        if rule.code in found or (engine is not None and engine not in rule.engines):
            continue
        if backend is not None and backend not in rule.backends:
            continue
        if rule.pattern.search(log):
            found[rule.code] = Diagnosis(cause=rule.cause, remedy=rule.remedy, cause_code=rule.code)
            explained |= rule.supersedes
    return [d for code, d in found.items() if code not in explained]


class LogLine(BaseModel):
    """One warning or error the engine printed, with what it means when we know."""

    text: str
    note: str | None = None
    harmless: bool = False  # a known line that needs nothing from anyone


class LogReview(BaseModel):
    """Everything in an engine log a user should look at: nothing is silently dropped."""

    diagnoses: list[Diagnosis] = Field(default_factory=list)
    warnings: list[LogLine] = Field(default_factory=list)
    errors: list[LogLine] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.diagnoses or self.warnings or self.errors)

    @property
    def notable_warnings(self) -> list[LogLine]:
        return [line for line in self.warnings if not line.harmless]

    @property
    def harmless_warnings(self) -> list[LogLine]:
        return [line for line in self.warnings if line.harmless]


# llama.cpp (2025+) prefixes every line with `<uptime> <level> `: `0.01.589.670 W srv ...`.
_LEVELLED_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+ ([WE]) (.*)$")
# stable-diffusion.cpp (recorded 2026-10-10): `[W] message --- file.cpp:123`, `[E] ...`.
_SDCPP_RE = re.compile(r"^\[([WE])\] (.*?)(?: --- \S+:\d+)?\s*$")
# Lines printed before the logger is up, older builds and whisper.cpp: `warning: ...`,
# `x: error: ...`, `llama_model_load: error loading model: ...`, `operator(): WARNING: ...`.
_CLASSIC_RE = re.compile(
    r"^(?:[\w ()]+:\s*)?(?:(warning|warn|error)\s*:|(error) loading)\s*(.+)$", re.IGNORECASE
)
# audio.cpp (recorded 2026-10-15): `[SERVER_HTTP_DEBUG] http.error <message>` per failed
# request, `audiocpp_server failed: <what>` once when the server dies; `[TRACE …]` and
# `[TIMING …]` lines are its progress, not warnings.
_AUDIOCPP_RE = re.compile(r"^(?:\[SERVER_HTTP_DEBUG\] http\.error|audiocpp_server failed:) (.+)$")


# vLLM (recorded 2026-10-21): `(EngineCore pid=53) WARNING 10-22 00:18:29 [file.py:47] text`,
# the process tag optional. One line per weight converted while loading: one entry.
_VLLM_RE = re.compile(
    r"^(?:\([\w ]+ pid=\d+\) )?(WARNING|ERROR) \d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)? \[[^\]]+\] (.+)$"
)
_VLLM_WEIGHT = re.compile(r"(Attempted to load weight )\S+( with dtype)")


def review_log(log: str, engine: Engine | None = None, backend: Backend | None = None) -> LogReview:
    """Diagnoses plus every distinct warning and error line, each with a note when known."""
    warnings: list[LogLine] = []
    errors: list[LogLine] = []
    seen: set[str] = set()
    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        level, text = _classify(line)
        if level is None or text in seen:
            continue
        seen.add(text)
        note = note_for(text, engine, backend.value if backend is not None else None)
        entry = LogLine(
            text=text,
            note=note.text if note else None,
            harmless=note.harmless if note else False,
        )
        (errors if level == "E" else warnings).append(entry)
    return LogReview(diagnoses=diagnose(log, engine, backend), warnings=warnings, errors=errors)


def _classify(line: str) -> tuple[str | None, str]:
    levelled = _LEVELLED_RE.match(line)
    if levelled:
        return levelled.group(1), levelled.group(2).strip()
    sdcpp = _SDCPP_RE.match(line)
    if sdcpp:
        return sdcpp.group(1), sdcpp.group(2).strip()
    vllm = _VLLM_RE.match(line)
    if vllm:
        return vllm.group(1)[0], _VLLM_WEIGHT.sub(r"\1…\2", vllm.group(2).strip())
    classic = _CLASSIC_RE.match(line)
    if classic:
        word = (classic.group(1) or classic.group(2)).lower()
        return "E" if word == "error" else "W", line
    audiocpp = _AUDIOCPP_RE.match(line)
    if audiocpp:
        return "E", audiocpp.group(1).strip()
    return None, line
