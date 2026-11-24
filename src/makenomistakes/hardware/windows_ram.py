"""The PC's own RAM under WSL, asked of Windows (so the doctor's video RAM remedy never
asks `.wslconfig` for more than the PC has)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from makenomistakes.hardware.probe import CommandRunner

WINDOWS_RAM = [
    "powershell.exe",
    "-NoProfile",
    "-NonInteractive",
    "-Command",
    "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
]


def windows_ram(runner: CommandRunner) -> int | None:
    """The PC's own RAM in bytes, asked of Windows through WSL's interop; None when it does
    not answer (interop off, no PowerShell)."""
    code, out, _err = runner.run(WINDOWS_RAM, timeout=15)
    text = out.strip()
    return int(text) if code == 0 and text.isdigit() and int(text) > 0 else None
