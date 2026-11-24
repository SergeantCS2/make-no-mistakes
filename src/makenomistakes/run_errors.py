"""The runs' errors (a leaf: `runs` raises them and re-exports them)."""

from __future__ import annotations


class RunError(RuntimeError):
    """A run problem worded for the user (the CLI and the web print it alike)."""


class RunDamagedError(RunError):
    """A run record this mnm cannot make sense of whatever its version: not JSON, not a
    record, or naming another run, under a run id's name. Never a newer mnm's (that one
    `mnm update` reads), so `mnm outputs rm <id>` may remove it (`RunStore.remove`)."""
