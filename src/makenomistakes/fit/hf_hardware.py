"""Hugging Face's hardware table (https://huggingface.co/hardware?view=table), the reference
list of parts the device table covers: an export of 2026-10-27 kept under
`fit/data/`, read once at import and typed here.

Its `tflops` is each vendor's headline figure, not one convention (NVIDIA GeForce rows are fp16
without tensor cores, T4 and MI100 with them, AMD RDNA rows the matrix peak): a device row
says which it takes. It carries no memory bandwidth: each row names the vendor page it read."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DATA = Path(__file__).parent / "data" / "hf-hardware-2026-10-27.json"
POPULAR_USERS = 1000  # the table covers every GPU registered by at least this many users


class HfPart(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    sku: tuple[str, str, str]  # kind ("GPU", "CPU", "Apple Silicon"), brand, name
    users: int = Field(alias="uniqueUsers")
    brand: str
    memory_gb: tuple[float, ...] = Field(default=(), alias="memoryGb")
    # None for a part the table lists without its specs
    tflops: float | None = None
    power_w: int | None = Field(default=None, alias="powerW")
    msrp_usd: int | None = Field(default=None, alias="msrpUsd")
    year: int | None = Field(default=None, alias="releaseYear")

    @property
    def kind(self) -> str:
        return self.sku[0]

    @property
    def name(self) -> str:
        return self.sku[2]


def parse(text: str) -> tuple[HfPart, ...]:
    return tuple(HfPart.model_validate(row) for row in json.loads(text))


@cache
def parts() -> tuple[HfPart, ...]:
    return parse(DATA.read_text(encoding="utf-8"))


def popular_gpus(users: int = POPULAR_USERS) -> tuple[HfPart, ...]:
    """Every GPU at least `users` people registered, most registered first."""
    found = [p for p in parts() if p.kind == "GPU" and p.users >= users]
    return tuple(sorted(found, key=lambda p: -p.users))


def part(name: str) -> HfPart | None:
    return next((p for p in parts() if p.kind == "GPU" and p.name == name), None)
