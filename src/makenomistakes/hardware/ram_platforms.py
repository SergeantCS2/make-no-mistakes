"""The RAM platform is a row like a card: what the box's RAM reads at, and what
the planner prices RAM-held weights at until this box measures. One row per platform, found
from the CPU's model name (`cpu_families`), its L3 groups (the CCD rule), the sockets, WSL or
native, and the DIMMs where they were read (`ram_slots`).

Each row carries its peak per socket (channels x MT/s x 8 bytes, capped by the CCDs on a Zen
part: each CCD reads through one fabric link, about 41 GB/s on Zen 2/3 with DDR4 (the 3955WX's
2 CCDs read 82) and 64 on Zen 4/5 (one AM5 CCD near 64)), the expert rate a plan uses until
this box's readings teach one (defaults, "estimated" until measured: 22 GB/s
for a DDR4 desktop, 28 for a one-CCD DDR5 Zen, 45 for a two-CCD or Intel DDR5 desktop, 70/50/35
for an 8-channel DDR4 EPYC or Threadripper PRO with 8+/4/2 CCDs, 120 for 8-12 channels of DDR5;
the 4-channel workstation rows are the same share of their peak), the share of the multi-process
probe RAM-held experts read at, the rate whole RAM-held blocks are read at without a
probe (the peak x `LLAMA_RAM_SHARE`), what the system keeps beside the models, and how many
lanes a card gets. Two sockets read at one socket's figure: mainline llama.cpp decodes slower on
two sockets than on one.

WSL keeps the RTX 4070 machine's figures, which its readings were fitted to (`fit.cpu_rates`:
30 GB/s with the per-expert touch cost, 55 GB/s for whole blocks): a figure of 12 GB/s for
WSL is the same readings priced without the touch cost. A CPU no row knows on a native box
keeps them too, said as "unknown"."""

from __future__ import annotations

from dataclasses import dataclass

from makenomistakes.fit.cpu_rates import CPU_BANDWIDTH, CPU_EXPERT_RATE, LLAMA_RAM_SHARE
from makenomistakes.hardware.cpu_families import CpuFamily, Kind, ccds, family_of
from makenomistakes.hardware.fingerprint import MachineFingerprint

GIB = 2**30
GB = 10**9
REPORT = "vendor and review figures"
# What one CCD's link reads (the CCD rule): Zen 2/3 on DDR4 (the 3955WX's
# two CCDs read 82 GB/s), Zen 4/5 on DDR5 (one AM5 CCD near 64, one Genoa CCD just under 70).
CCD_LINK_DDR4 = 41.0
CCD_LINK_DDR5 = 64.0
SERVER_MARGIN = 8 * GIB  # a server's kernel, page cache and BMC agents: more than WSL's 4 [est]
DESKTOP_MARGIN = 4 * GIB


@dataclass(frozen=True)
class RamRow:
    kind: Kind
    label: str
    expert_gbps: float  # the default until measured [est]
    share: float  # of the every-core probe RAM-held experts read at
    lanes: str
    margin_bytes: int = DESKTOP_MARGIN
    layer_gbps: float | None = None  # None: the peak x LLAMA_RAM_SHARE
    channels: int = 2
    ram: str = "DDR5"
    mts: int = 5600
    server: bool = False
    source: str = REPORT


ROWS: dict[Kind, RamRow] = {
    r.kind: r
    for r in (
        RamRow(Kind.WSL, "WSL", CPU_EXPERT_RATE / GB, 0.5, "[N/A] under WSL",
               layer_gbps=CPU_BANDWIDTH / GB,
               source="an RTX 4070 machine's readings under WSL (`fit.cpu_rates`)"),
        RamRow(Kind.DESKTOP_DDR4, "a desktop with 2 channels of DDR4", 22.0, 0.45,
               "x16, or x8 + x8; a third card on the chipset at x4", ram="DDR4", mts=3200),
        RamRow(Kind.DESKTOP_DDR5_ONE_CCD, "a one-CCD Zen desktop with 2 channels of DDR5", 28.0,
               0.45, "x16, or x8 + x8; a third card on the chipset at x4"),
        RamRow(Kind.DESKTOP_DDR5, "a desktop with 2 channels of DDR5", 45.0, 0.5,
               "x16, or x8 + x8; a third card on the chipset at x4"),
        RamRow(Kind.WORKSTATION_DDR4, "a workstation with 4 channels of DDR4", 35.0, 0.4,
               "x16 or x8 each for 3-4 cards", channels=4, ram="DDR4", mts=3200),
        RamRow(Kind.WORKSTATION_DDR5, "a workstation with 4 channels of DDR5", 60.0, 0.35,
               "x16 Gen5 or Gen4 each for 3-4 cards", channels=4, mts=5200),
        RamRow(Kind.SERVER_DDR4, "a server board with 8 channels of DDR4", 70.0, 0.4,
               "x16 Gen4 each", SERVER_MARGIN, channels=8, ram="DDR4", mts=3200, server=True),
        RamRow(Kind.SERVER_DDR5, "a server board with 8-12 channels of DDR5", 120.0, 0.35,
               "x16 Gen5 each", SERVER_MARGIN, channels=12, mts=4800, server=True),
        RamRow(Kind.UNKNOWN, "a platform mnm does not know", CPU_EXPERT_RATE / GB, 0.5,
               "unknown", layer_gbps=CPU_BANDWIDTH / GB,
               source="an RTX 4070 machine's figures: no row matches this CPU"),
    )
}  # fmt: skip
# The CCD rule on an 8-channel DDR4 Zen part: 70 GB/s with 8 CCDs or more,
# 50 with 4 to 7, 35 with 2 or 3
CCD_TIERS = ((8, 70.0), (4, 50.0), (1, 35.0))


@dataclass(frozen=True)
class RamPlatform:
    """The row this box sits on, with what was read of it."""

    row: RamRow
    family: CpuFamily | None
    cpu: str  # `EPYC 7532`
    ccds: int | None  # per socket; None off Zen or unread
    channels: int  # the socket's (the family's, else the row's)
    populated: int | None  # the channels a DIMM was read in (all sockets), None unread
    ram: str  # DDR4, DDR5, "DDR4 or DDR5"
    mts: int
    sockets: int
    numa_nodes: int

    @property
    def kind(self) -> Kind:
        return self.row.kind

    @property
    def used_channels(self) -> int:
        """The channels a DIMM was read in (dmidecode names them the same on each socket),
        else every channel the socket has."""
        if self.populated is None:
            return self.channels
        return min(max(self.populated, 1), self.channels)

    @property
    def _link(self) -> float:
        return CCD_LINK_DDR5 if self.ram == "DDR5" else CCD_LINK_DDR4

    @property
    def peak_gbps(self) -> float:
        """One socket's peak: its channels with a DIMM at the speed read (else the family's),
        capped by the CCDs' links on Zen."""
        peak = self.used_channels * self.mts * 8 / 1000
        return min(peak, self.ccds * self._link) if self.ccds else peak

    @property
    def ccd_limited(self) -> bool:
        """The CCDs read less than the channels give (a 4-CCD EPYC, a 2-CCD TR PRO); a one-CCD
        desktop has its own row."""
        if not self.ccds or self.kind is Kind.DESKTOP_DDR5_ONE_CCD:
            return False
        return self.ccds * self._link < self.channels * self.mts * 8 / 1000

    @property
    def expert_gbps(self) -> float:
        """The expert rate until measured: the family's own, the CCD tier on an 8-channel DDR4
        Zen, else the row's; lowered by empty channels, by RAM slower than the family's rated
        speed, and (outside the tiers and the one-CCD row, which count them) by the CCDs."""
        tiered = self.kind is Kind.SERVER_DDR4 and self.ccds is not None
        if self.family is not None and self.family.expert_gbps is not None:
            base = self.family.expert_gbps
        elif tiered and self.ccds is not None:
            base = next(rate for least, rate in CCD_TIERS if self.ccds >= least)
        else:
            base = self.row.expert_gbps
        if self.kind in (Kind.WSL, Kind.UNKNOWN):
            return base
        scale = self.used_channels / self.channels
        rated = self.family.mts if self.family is not None else self.row.mts
        scale *= min(self.mts / rated, 1.0)
        if not tiered and self.ccd_limited and self.ccds:
            scale *= self.ccds * self._link / (self.channels * self.mts * 8 / 1000)
        return round(base * min(scale, 1.0), 1)

    @property
    def layer_gbps(self) -> float:
        """Whole RAM-held blocks without a probe: the row's fixed figure, else the peak x
        `LLAMA_RAM_SHARE`."""
        if self.row.layer_gbps is not None:
            return self.row.layer_gbps
        return round(self.peak_gbps * LLAMA_RAM_SHARE, 1)

    @property
    def margin_bytes(self) -> int:
        return self.row.margin_bytes

    @property
    def ccd_words(self) -> str:
        if not self.ccds:
            return ""
        return f" with {self.ccds} CCD{'s' if self.ccds != 1 else ''}"

    def words(self) -> str:
        """`EPYC 7002 (Rome): 8 channels of DDR4-3200 a socket, 204.8 GB/s peak, EPYC 7532
        with 8 CCDs`."""
        if self.kind is Kind.WSL:
            return f"WSL ({self.cpu}): the RAM .wslconfig gives the VM"
        if self.kind is Kind.UNKNOWN:
            return f"{self.cpu}: a CPU mnm has no RAM row for"
        name = self.family.label if self.family else self.row.label
        two = "; 2 sockets: priced as one" if self.sockets > 1 else ""
        return (
            f"{name}: {self.channels} channels of {self.ram}-{self.mts} a socket, "
            f"{self.peak_gbps:.0f} GB/s peak ({self.cpu}{self.ccd_words}){two}"
        )

    def priced_words(self) -> str:
        """`priced at 50 GB/s for an EPYC 7532 with 4 CCDs until this box measures`."""
        return (
            f"priced at {self.expert_gbps:g} GB/s for {_a(self.cpu)}{self.ccd_words} until this "
            f"box measures (estimated from {self.row.source})"
        )


def _a(cpu: str) -> str:
    return f"an {cpu}" if cpu[:1].upper() in "AEIOU8" else f"a {cpu}"


def platform_of(box: MachineFingerprint) -> RamPlatform:
    """The row `box` sits on: WSL by its platform, else its CPU's family (a one-CCD Zen DDR5
    desktop apart), else unknown."""
    cpu = box.cpu
    family = family_of(cpu.model)
    sockets = cpu.sockets or 1
    found_ccds = ccds(family, cpu.l3_groups, sockets)
    if box.platform.kind == "wsl":
        kind = Kind.WSL
    elif family is None:
        kind = Kind.UNKNOWN
    elif family.kind is Kind.DESKTOP_DDR5 and family.zen and found_ccds == 1:
        kind = Kind.DESKTOP_DDR5_ONE_CCD
    else:
        kind = family.kind
    # LPDDR5 is DDR5 to the CCD rule
    ram = (box.ram.kind or "").removeprefix("LP") or (family.ram if family else ROWS[kind].ram)
    if family is not None and family.ram == "DDR4 or DDR5" and box.ram.kind is None:
        ram = "DDR4 or DDR5"
    if kind is Kind.DESKTOP_DDR5 and ram == "DDR4":  # an Intel 12th-14th Gen board of DDR4
        kind = Kind.DESKTOP_DDR4
    if ram == "DDR4 or DDR5":  # unread: priced as the slower, said as such
        kind = Kind.DESKTOP_DDR4
    if ram == "DDR4 or DDR5" or family is None:
        mts = box.ram.speed_mts or ROWS[kind].mts
    else:
        mts = box.ram.speed_mts or family.mts
    return RamPlatform(
        row=ROWS[kind],
        family=family,
        cpu=cpu.short,
        ccds=found_ccds,
        channels=family.channels if family else ROWS[kind].channels,
        populated=box.ram.channels,
        ram=ram,
        mts=mts,
        sockets=sockets,
        numa_nodes=cpu.numa_nodes or 1,
    )


def cpu_note(rate: float, probe_gbps: float | None, platform: str) -> str:
    """How a CPU-backend plan reads its weights: `102 GB/s read a token: EPYC 7002 …,
    its peak x 0.5 until this box measures`; the probe's figure once measured."""
    if probe_gbps:
        return (
            f"{rate / 1e9:.0f} GB/s read a token: this box's probe ({probe_gbps:g} GB/s) x "
            f"{LLAMA_RAM_SHARE:g}, capped by the cores; `mnm try --speed` and `mnm tune` measure it"
        )
    return (
        f"{rate / 1e9:.0f} GB/s read a token: {platform}; its peak x {LLAMA_RAM_SHARE:g} until "
        "this box measures (`mnm doctor --deep`, then `mnm try --speed`) [est]"
    )
