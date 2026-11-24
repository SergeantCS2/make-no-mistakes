"""The CPU families mnm knows the memory of: which RAM platform a CPU sits on, read from
its model name as `/proc/cpuinfo` and lscpu give it ("AMD EPYC 7532 32-Core Processor",
"13th Gen Intel(R) Core(TM) i9-13900K"). Each row names the Hugging Face hardware table's
families it covers (`export`, the table's `sku[2]` names; every family there with 1,000 users
or more has a row),
the platform it sits on (`ram_platforms.Kind`), its channels per socket, the RAM kind and the
rated speed, and whether its L3 groups are two a CCD (Zen 1 and 2: two core complexes a die).

The channels and speeds are the vendors' datasheets (AMD's EPYC 7002/7003/9004/9005 and
Threadripper pages, Intel ARK's Xeon and Core pages), read on 2026-10-26 and 2026-10-28. A
CPU no row matches is "unknown": the RTX 4070 machine's constants price it, said as such."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    """The RAM platforms (`ram_platforms.ROWS` holds their figures)."""

    WSL = "wsl"
    DESKTOP_DDR4 = "desktop-ddr4"
    DESKTOP_DDR5_ONE_CCD = "desktop-ddr5-one-ccd"
    DESKTOP_DDR5 = "desktop-ddr5"
    WORKSTATION_DDR4 = "workstation-ddr4"  # 4 channels: TR 1000-3000, Xeon E5, Xeon W-2200
    WORKSTATION_DDR5 = "workstation-ddr5"  # 4 channels: TR 7000/9000, Xeon W-2400
    SERVER_DDR4 = "server-ddr4"  # 6-8 channels: EPYC 7001-7003, TR Pro 3000/5000, Xeon 1st-3rd
    SERVER_DDR5 = "server-ddr5"  # 6-12 channels: EPYC 8004/9004/9005, TR Pro 7000+, Xeon 4th+
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CpuFamily:
    label: str  # what the doctor says: "EPYC 7002 (Rome)"
    pattern: re.Pattern[str]
    kind: Kind
    channels: int  # per socket
    ram: str  # DDR4, DDR5; "DDR4 or DDR5" where the board decides (Intel 12th-14th Gen)
    mts: int  # the platform's rated speed
    export: tuple[str, ...] = ()  # the Hugging Face table's families it covers
    zen: bool = False  # AMD Zen: the CCD rule applies (each CCD reads through one link)
    two_per_ccd: bool = False  # Zen 1/2: two L3 groups (core complexes) a CCD
    expert_gbps: float | None = None  # a family's own default where the row's does not fit


def _p(text: str) -> re.Pattern[str]:
    return re.compile(text, re.IGNORECASE)


_RYZEN = r"Ryzen\s+[3579]\s+(?:PRO\s+)?"
# Most specific first: Threadripper before Ryzen, Ryzen AI before Ryzen, EPYC 4004 before EPYC.
FAMILIES: tuple[CpuFamily, ...] = (
    # --- AMD servers and workstations ---
    CpuFamily("EPYC 4004/4005 (AM5)", _p(r"EPYC\s+4\d{3}"), Kind.DESKTOP_DDR5, 2, "DDR5", 5200,
              zen=True),
    CpuFamily("EPYC 7001 (Naples)", _p(r"EPYC\s+7\w\w1\b"), Kind.SERVER_DDR4, 8, "DDR4", 2666,
              ("EPYC 1st Generation Zen (Naples)",), zen=True, two_per_ccd=True),
    CpuFamily("EPYC 7002 (Rome)", _p(r"EPYC\s+7\w\w2\b"), Kind.SERVER_DDR4, 8, "DDR4", 3200,
              ("EPYC 2th Generation Zen 2 (Rome)",), zen=True, two_per_ccd=True),
    CpuFamily("EPYC 7003 (Milan)", _p(r"EPYC\s+7\w\w3"), Kind.SERVER_DDR4, 8, "DDR4", 3200,
              ("EPYC 3th Generation Zen 3 (Milan)",), zen=True),
    CpuFamily("EPYC 8004 (Siena)", _p(r"EPYC\s+8\w\w4"), Kind.SERVER_DDR5, 6, "DDR5", 4800,
              zen=True),
    CpuFamily("EPYC 9004 (Genoa)", _p(r"EPYC\s+9\w\w4"), Kind.SERVER_DDR5, 12, "DDR5", 4800,
              ("EPYC 4th Generation Zen 4 (Genoa)",), zen=True),
    CpuFamily("EPYC 9005 (Turin)", _p(r"EPYC\s+9\w\w5"), Kind.SERVER_DDR5, 12, "DDR5", 6000,
              ("EPYC 5th Generation Zen 5 (Turin)",), zen=True),
    CpuFamily("Threadripper PRO 3000WX", _p(r"Threadripper\s+PRO\s+39\d5WX"), Kind.SERVER_DDR4,
              8, "DDR4", 3200, zen=True, two_per_ccd=True),
    CpuFamily("Threadripper PRO 5000WX", _p(r"Threadripper\s+PRO\s+59\d5WX"), Kind.SERVER_DDR4,
              8, "DDR4", 3200, ("Ryzen Threadripper Zen 3 5000 (Chagall)",), zen=True),
    CpuFamily("Threadripper PRO 7000WX", _p(r"Threadripper\s+PRO\s+79\d5WX"), Kind.SERVER_DDR5,
              8, "DDR5", 5200, zen=True),
    CpuFamily("Threadripper PRO 9000WX", _p(r"Threadripper\s+PRO\s+99\d5WX"), Kind.SERVER_DDR5,
              8, "DDR5", 6400, zen=True),
    CpuFamily("Threadripper 1000/2000", _p(r"Threadripper\s+[12]9\d0"), Kind.WORKSTATION_DDR4, 4,
              "DDR4", 2933, ("Ryzen Threadripper Zen 1000 (Whitehaven)",), zen=True,
              two_per_ccd=True),
    CpuFamily("Threadripper 3000 (TRX40)", _p(r"Threadripper\s+39\d0X"), Kind.WORKSTATION_DDR4,
              4, "DDR4", 3200, ("Ryzen Threadripper Zen 2 3000 (Castle Peak)",), zen=True,
              two_per_ccd=True),
    CpuFamily("Threadripper 7000 (TRX50)", _p(r"Threadripper\s+79\d0X"), Kind.WORKSTATION_DDR5,
              4, "DDR5", 5200, ("Ryzen Threadripper Zen 4 7000 (Storm Peak)",), zen=True),
    CpuFamily("Threadripper 9000 (TRX50)", _p(r"Threadripper\s+99\d0X"), Kind.WORKSTATION_DDR5,
              4, "DDR5", 6400, ("Ryzen Threadripper Zen 5 9000 (Shimada Peak)",), zen=True),
    # --- AMD desktops and laptops ---
    CpuFamily("Ryzen AI 300 (Zen 5)", _p(r"Ryzen\s+AI\s+[579]\s+(?:HX\s+)?\d{3}"),
              Kind.DESKTOP_DDR5, 2, "DDR5", 5600,
              ("Ryzen AI 300 (Ryzen AI 9 HX)", "Ryzen AI 300 (Ryzen AI 7)",
               "Ryzen AI 300 (Ryzen AI 9)", "Ryzen AI 300 (Ryzen AI 5)"), zen=True),
    CpuFamily("Ryzen 9000 (Zen 5)", _p(_RYZEN + r"9\d{3}"), Kind.DESKTOP_DDR5, 2, "DDR5", 5600,
              ("Ryzen Zen 5 9000 (Ryzen 9)", "Ryzen Zen 5 9000 (Ryzen 7)",
               "Ryzen Zen 5 9000 (Ryzen 5)"), zen=True),
    CpuFamily("Ryzen 7000/8000 (Zen 4)", _p(_RYZEN + r"[78]\d{3}"), Kind.DESKTOP_DDR5, 2, "DDR5",
              5200, ("Ryzen Zen 4 7000 (Ryzen 7)", "Ryzen Zen 4 7000 (Ryzen 9)",
                     "Ryzen Zen 4 7000 (Ryzen 5)"), zen=True),
    CpuFamily("Ryzen 5000 (Zen 3)", _p(_RYZEN + r"5\d{3}"), Kind.DESKTOP_DDR4, 2, "DDR4", 3200,
              ("Ryzen Zen 3 5000 (Ryzen 7)", "Ryzen Zen 3 5000 (Ryzen 5)",
               "Ryzen Zen 3 5000 (Ryzen 9)"), zen=True),
    CpuFamily("Ryzen 3000/4000 (Zen 2)", _p(r"Ryzen\s+[3579]\s+(?:PRO\s+)?[34]\d{3}"),
              Kind.DESKTOP_DDR4, 2, "DDR4", 3200,
              ("Ryzen Zen 2 3000 (Ryzen 5)", "Ryzen Zen 2 3000 (Ryzen 7)",
               "Ryzen Zen 2 3000 (Ryzen 9)", "Ryzen Zen 2 3000 (Ryzen 3)", "Ryzen 7 3800X (16)"),
              zen=True, two_per_ccd=True),
    # --- Intel servers and workstations ---
    CpuFamily("Xeon 6 (Granite Rapids)", _p(r"Xeon\(R\)\s+(?:6\s+)?69\d\d[PE]"), Kind.SERVER_DDR5,
              12, "DDR5", 6400),
    CpuFamily("Xeon Scalable 5th Gen (Emerald Rapids)",
              _p(r"Xeon.*(?:Platinum|Gold|Silver|Bronze)\s+\d5\d\d"), Kind.SERVER_DDR5, 8, "DDR5",
              5600),
    CpuFamily("Xeon Scalable 4th Gen (Sapphire Rapids)",
              _p(r"Xeon.*(?:Platinum|Gold|Silver|Bronze)\s+\d4\d\d|Xeon.*\bw[579]-3\d{3}"),
              Kind.SERVER_DDR5, 8, "DDR5", 4800, ("Xeon 4th Generation (Sapphire Rapids)",)),
    CpuFamily("Xeon W-2400 (Sapphire Rapids)", _p(r"Xeon.*\bw[3579]-2\d{3}"),
              Kind.WORKSTATION_DDR5, 4, "DDR5", 4800),
    CpuFamily("Xeon Scalable 3rd Gen (Ice Lake)",
              _p(r"Xeon.*(?:Platinum|Gold|Silver)\s+\d3\d\d"), Kind.SERVER_DDR4, 8, "DDR4", 3200,
              ("Xeon 3th Generation (Ice Lake)",), expert_gbps=50.0),
    CpuFamily("Xeon Scalable 1st/2nd Gen (Skylake, Cascade Lake)",
              _p(r"Xeon.*(?:Platinum|Gold|Silver|Bronze)\s+\d[12]\d\d|Xeon.*\bW-3[12]\d\d"),
              Kind.SERVER_DDR4, 6, "DDR4", 2933, ("Xeon 2th Generation (Cascade Lake)",),
              expert_gbps=35.0),
    CpuFamily("Xeon W-2100/2200", _p(r"Xeon.*\bW-2[12]\d\d"), Kind.WORKSTATION_DDR4, 4, "DDR4",
              2666),
    CpuFamily("Xeon E5 v3/v4 (Haswell, Broadwell)", _p(r"Xeon.*E5-\d{4}\w?\s+v[34]"),
              Kind.WORKSTATION_DDR4, 4, "DDR4", 2400,
              ("Xeon E5v4 (Broadwell)", "Xeon E5v3 (Haswell)"), expert_gbps=20.0),
    CpuFamily("Xeon E5 v2 (Ivy Bridge)", _p(r"Xeon.*E5-\d{4}\w?\s+v2"), Kind.WORKSTATION_DDR4,
              4, "DDR3", 1866, ("Xeon E5v2 (Ivy Bridge)",), expert_gbps=15.0),
    # --- Intel desktops and laptops ---
    CpuFamily("Core Ultra (series 1 and 2)", _p(r"Core\(TM\)\s+Ultra\s+[3579]\s+\d{3}"),
              Kind.DESKTOP_DDR5, 2, "DDR5", 5600,
              ("Intel Core Ultra 7 265KF", "Intel Core Ultra 9 275HX", "Intel Core Ultra 7 255HX")),
    CpuFamily("Core 12th-14th Gen", _p(r"Core\(TM\)\s+i[3579]-1[234]\d{2,3}"), Kind.DESKTOP_DDR5, 2,
              "DDR4 or DDR5", 5600,
              tuple(f"Intel Core {g}th Generation ({i})" for g in (12, 13, 14)
                    for i in ("i3", "i5", "i7", "i9"))),
    CpuFamily("Core 10th/11th Gen", _p(r"Core\(TM\)\s+i[3579]-1[01]\d{2,3}"), Kind.DESKTOP_DDR4,
              2, "DDR4", 2933,
              tuple(f"Intel Core {g}th Generation ({i})" for g in (10, 11)
                    for i in ("i3", "i5", "i7", "i9"))),
    CpuFamily("Core 6th-9th Gen", _p(r"Core\(TM\)\s+i[3579]-[6-9]\d{3}"), Kind.DESKTOP_DDR4, 2,
              "DDR4", 2666),
)  # fmt: skip


def family_of(model: str | None) -> CpuFamily | None:
    """The first row whose pattern finds the model name; None for a CPU no row knows."""
    if not model:
        return None
    return next((f for f in FAMILIES if f.pattern.search(model)), None)


def ccds(family: CpuFamily | None, l3_groups: int | None, sockets: int | None) -> int | None:
    """A Zen CPU's CCDs per socket from its L3 groups (two a CCD on Zen 1 and 2); None where
    the CCD rule does not apply (Intel) or the groups were not read."""
    if family is None or not family.zen or not l3_groups:
        return None
    return ccds_of_groups(family.label, max(l3_groups // max(sockets or 1, 1), 1), family)


def ccds_of_groups(model: str | None, groups: int, family: CpuFamily | None = None) -> int:
    """A CPU's CCDs from its L3 groups: half of them on Zen 1 and 2 (two core complexes a CCD,
    each with its own L3: an EPYC 7532 lists 16), else as many. `family` when already found."""
    family = family or family_of(model)
    if family is not None and family.two_per_ccd:
        return max(groups // 2, 1)
    return groups
