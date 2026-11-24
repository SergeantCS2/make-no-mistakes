"""The device table's rows: every GPU in Hugging Face's hardware table registered by at
least 1,000 users (`fit.hf_hardware`, an export of 2026-10-27) and a hand-picked list of
other cards, with the bandwidth from the vendor's page each row names; the families' and
vendors' defaults are `fit.devices`.

`tflops` is the export's headline figure, except where its convention is not the GeForce
rows' fp16 without tensor cores (the 4070's 29, which the picture model is fitted to): AMD
Radeon rows carry the export's figure halved (its matrix or dual-issue fp16 peak, twice the
fp32-like rate), T4 and MI100 their fp16 without the matrix units. Every row borrows the
4070's efficiencies, launch term and context until a box with that card reads its own
(`DeviceRow.with_learned`); only the 4070's row is measured (`fit.device_row.RTX_4070`)."""

from __future__ import annotations

from functools import partial

from makenomistakes.fit.device_row import RTX_4070, DeviceRow
from makenomistakes.fit.hf_hardware import part
from makenomistakes.hardware.fingerprint import MemoryKind, Vendor

GB = 10**9
NV = "https://www.nvidia.com/en-us"
GEFORCE = f"{NV}/geforce/graphics-cards"
LAPTOPS = f"{NV}/geforce/laptops"
DC = f"{NV}/data-center"
PRO = f"{NV}/products/workstations"
AMD = "https://www.amd.com/en/products/graphics"
INTEL = "https://www.intel.com/content/www/us/en/products/sku"


# The GB10's decode efficiencies from published Spark readings:
# Qwen3.6-27B Q4_K_M at 11.85 tok/s x 16.7 GB is about 198 of 273 GB/s (0.72, dense);
# gpt-oss-120b MXFP4 at 58.7 tok/s x about 3 GB a token is about 176 (0.64, MoE). The codebook
# i-quants keep the 4070's ratio to its dense figure (0.50 / 0.91). Until a Spark's own
# readings refit them (`device_learning`).
GB10_EFFICIENCIES: dict[str, object] = {
    "efficiency": 0.72,
    "moe_efficiency": 0.64,
    "iq_grid_efficiency": round(0.72 * 0.50 / 0.91, 2),
    "efficiency_source": "third-party Spark benches (nabe2030; llama.cpp's benches/dgx-spark): "
    "0.72 dense, 0.64 MoE",
}


def _row(  # noqa: PLR0913 - one row's every figure, by keyword where it differs
    hf: str,
    match: tuple[str, ...],
    gbps: float,
    page: str,
    *,
    family: str,
    compute: str | None = None,
    gfx: str | None = None,
    vendor: Vendor = Vendor.NVIDIA,
    memory_gb: int | None = None,
    memory: MemoryKind = MemoryKind.DISCRETE,
    tflops: float | None = None,
    halve: bool = False,
    name: str | None = None,
    source: str | None = None,
    pci_ids: tuple[int, ...] = (),
    backend_tflops: dict[str, float] | None = None,
) -> DeviceRow:
    found = part(hf)
    headline = found.tflops if found is not None and found.tflops is not None else 25.0
    figure = tflops if tflops is not None else headline / 2 if halve else headline
    size = f" {memory_gb} GB" if memory_gb else ""
    return DeviceRow(
        match=match,
        vendor=vendor,
        family=family,
        memory=memory,
        bandwidth=int(gbps * GB),
        tflops=figure,
        source=source or f"marketing: {gbps:g} GB/s from {page}",
        name=(name or hf) + size,
        compute=compute,
        gfx=gfx,
        memory_gb=memory_gb,
        hf_name=hf if found is not None else None,
        pci_ids=pci_ids,
        backend_tflops=backend_tflops or {},
    )


_blackwell = partial(_row, family="blackwell", compute="12.0")
_ada = partial(_row, family="ada", compute="8.9")
_ampere = partial(_row, family="ampere", compute="8.6")
_turing = partial(_row, family="turing", compute="7.5")
_pascal = partial(_row, family="pascal", compute="6.1")
_radeon = partial(_row, vendor=Vendor.AMD, halve=True)
# The R9700's compute as llama.cpp reaches it (a 27B's prompt at about 1,150 tok/s x 54 GFLOP
# a token on HIP, two thirds of that on Vulkan), never the 191 TFLOPS
# matrix figure; the RX 9070 scaled to its 56 CUs. Keys are `backend.Backend` values.
R9700_TFLOPS = {"rocm": 62.0, "vulkan": 43.0}
R9070_TFLOPS = {"rocm": 54.3, "vulkan": 37.6}


def _estimated(gbps: int, basis: str) -> str:
    return (
        f"estimated from published readings ({basis}); {gbps} GB/s from the 256-bit 20 Gbps GDDR6"
    )


S30, S40, S50 = f"{GEFORCE}/30-series", f"{GEFORCE}/40-series", f"{GEFORCE}/50-series"
L30, L40, L50 = f"{LAPTOPS}/30-series", f"{LAPTOPS}/40-series", f"{LAPTOPS}/50-series"
S20, S16, S10 = f"{GEFORCE}/20-series", f"{GEFORCE}/16-series", f"{GEFORCE}/10-series"

ROWS: tuple[DeviceRow, ...] = (
    # GeForce 50 (Blackwell, compute 12.0)
    _blackwell("RTX 5090", ("rtx 5090",), 1792, f"{S50}/rtx-5090/"),
    _blackwell("RTX 5080", ("rtx 5080",), 960, f"{S50}/rtx-5080/"),
    _blackwell("RTX 5070 Ti", ("rtx 5070 ti",), 896, f"{S50}/rtx-5070-family/"),
    _blackwell("RTX 5070", ("rtx 5070",), 672, f"{S50}/rtx-5070-family/"),
    _blackwell("RTX 5060 Ti", ("rtx 5060 ti",), 448, f"{S50}/rtx-5060-family/", memory_gb=16),
    _blackwell("RTX 5060 Ti", ("rtx 5060 ti",), 448, f"{S50}/rtx-5060-family/", memory_gb=8),
    _blackwell("RTX 5060", ("rtx 5060",), 448, f"{S50}/rtx-5060-family/"),
    _blackwell("RTX 5090 Mobile", ("rtx 5090 laptop", "rtx 5090 mobile"), 896, f"{L50}/"),
    _blackwell("RTX 5080 Mobile", ("rtx 5080 laptop", "rtx 5080 mobile"), 896, f"{L50}/"),
    _blackwell("RTX 5070 Ti Mobile", ("rtx 5070 ti laptop", "rtx 5070 ti mobile"), 672, f"{L50}/"),
    _blackwell("RTX 5070 Mobile", ("rtx 5070 laptop", "rtx 5070 mobile"), 384, f"{L50}/"),
    _blackwell("RTX 5060 Mobile", ("rtx 5060 laptop", "rtx 5060 mobile"), 384, f"{L50}/"),
    _blackwell("RTX PRO 6000 WS", ("rtx pro 6000",), 1792, f"{PRO}/rtx-pro-6000/"),
    _blackwell(
        "RTX PRO 6000 Max-Q",
        ("rtx pro 6000 blackwell max q", "rtx pro 6000 max q"),
        1792,
        f"{PRO}/rtx-pro-6000-max-q/",
    ),
    _row(
        "GB10",
        ("gb10",),
        273,
        "https://docs.nvidia.com/dgx/dgx-spark/hardware.html (128 GB LPDDR5x, 256-bit)",
        family="gb10",
        compute="12.1",
        memory=MemoryKind.UNIFIED,
        name="DGX Spark GB10",
    ).model_copy(update=GB10_EFFICIENCIES),
    _row("B200", ("b200",), 8000, f"{DC}/dgx-b200/", family="blackwell-dc", compute="10.0"),
    _row("B300", ("b300",), 8000, f"{DC}/dgx-b300/", family="blackwell-dc", compute="10.3"),
    # GeForce 40 and Ada pro/datacenter (compute 8.9); the 4070 is `device_row.RTX_4070`
    _ada("RTX 4090", ("rtx 4090",), 1008, f"{S40}/rtx-4090/"),
    _ada("RTX 4080 SUPER", ("rtx 4080 super",), 736, f"{S40}/rtx-4080-family/"),
    _ada("RTX 4080", ("rtx 4080",), 716.8, f"{S40}/rtx-4080-family/"),
    _ada("RTX 4070 Ti Super", ("rtx 4070 ti super",), 672, f"{S40}/rtx-4070-family/"),
    _ada("RTX 4070 Ti", ("rtx 4070 ti",), 504, f"{S40}/rtx-4070-family/"),
    _ada("RTX 4070 Super", ("rtx 4070 super",), 504, f"{S40}/rtx-4070-family/"),
    _ada("RTX 4060 Ti", ("rtx 4060 ti",), 288, f"{S40}/rtx-4060-4060ti/", memory_gb=8),
    _ada("RTX 4060 Ti", ("rtx 4060 ti",), 288, f"{S40}/rtx-4060-4060ti/", memory_gb=16),
    _ada("RTX 4060", ("rtx 4060",), 272, f"{S40}/rtx-4060-4060ti/"),
    _ada("RTX 4070 Laptop", ("rtx 4070 laptop",), 256, f"{L40}/"),
    _ada("RTX 4070 Mobile", ("rtx 4070 mobile",), 256, f"{L40}/"),
    _ada("RTX 4060 Laptop", ("rtx 4060 laptop", "rtx 4060 mobile"), 256, f"{L40}/"),
    _ada("RTX 4050 Laptop", ("rtx 4050 laptop", "rtx 4050 mobile"), 192, f"{L40}/"),
    _ada("RTX 6000 Ada", ("rtx 6000 ada",), 960, f"{PRO}/rtx-6000/"),
    _ada("RTX 4000 Ada", ("rtx 4000 ada",), 360, f"{PRO}/rtx-4000/"),
    _ada("RTX 2000 Ada", ("rtx 2000 ada",), 224, f"{PRO}/rtx-2000/"),
    _ada("L40s", ("l40s",), 864, f"{DC}/l40s/"),
    _ada("L4", ("l4",), 300, f"{DC}/l4/"),
    # GeForce 30 and Ampere pro/datacenter (compute 8.6, 8.0, 8.7)
    _ampere("RTX 3090 Ti", ("rtx 3090 ti",), 1008, f"{S30}/rtx-3090-3090ti/"),
    _ampere("RTX 3090", ("rtx 3090",), 936, f"{S30}/rtx-3090-3090ti/"),
    _ampere("RTX 3080 Ti", ("rtx 3080 ti",), 912, f"{S30}/rtx-3080-3080ti/"),
    _ampere("RTX 3080", ("rtx 3080",), 760, f"{S30}/rtx-3080-3080ti/", memory_gb=10),
    _ampere("RTX 3080", ("rtx 3080",), 912, f"{S30}/rtx-3080-3080ti/", memory_gb=12),
    _ampere("RTX 3070 Ti", ("rtx 3070 ti",), 608, f"{S30}/rtx-3070-3070ti/"),
    _ampere("RTX 3070", ("rtx 3070",), 448, f"{S30}/rtx-3070-3070ti/"),
    _ampere("RTX 3060 Ti", ("rtx 3060 ti",), 448, f"{S30}/rtx-3060-3060ti/"),
    _ampere("RTX 3060", ("rtx 3060",), 360, f"{S30}/rtx-3060-3060ti/", memory_gb=12),
    _ampere("RTX 3060", ("rtx 3060",), 240, f"{S30}/rtx-3060-3060ti/", memory_gb=8),
    _ampere("RTX 3080 Mobile", ("rtx 3080 laptop", "rtx 3080 mobile"), 448, f"{L30}/"),
    _ampere("RTX 3070 Ti Mobile", ("rtx 3070 ti laptop", "rtx 3070 ti mobile"), 448, f"{L30}/"),
    _ampere("RTX 3060 Mobile", ("rtx 3060 laptop", "rtx 3060 mobile"), 336, f"{L30}/"),
    _ampere("RTX 3050 Mobile", ("rtx 3050 laptop", "rtx 3050 mobile"), 192, f"{L30}/"),
    _ampere("RTX A6000", ("rtx a6000",), 768, f"{PRO}/rtx-a6000/"),
    _ampere("RTX A4000", ("rtx a4000",), 448, f"{PRO}/rtx-a4000/"),
    _ampere("RTX A2000", ("rtx a2000",), 288, f"{PRO}/rtx-a2000/"),
    _row(
        "A100", ("a100 sxm4",), 2039, f"{DC}/a100/", family="ampere-dc", compute="8.0", memory_gb=80
    ),
    _row("A100", ("a100",), 1935, f"{DC}/a100/", family="ampere-dc", compute="8.0", memory_gb=80),
    _row("A100", ("a100",), 1555, f"{DC}/a100/", family="ampere-dc", compute="8.0", memory_gb=40),
    _row(
        "Jetson Orin Nano 8GB",
        ("orin nano", "orin"),
        68,
        f"{NV}/autonomous-machines/embedded-systems/jetson-orin/",
        family="orin",
        compute="8.7",
        memory=MemoryKind.UNIFIED,
        name="Jetson Orin Nano",
    ),
    # Hopper (compute 9.0)
    _row("H100", ("h100",), 3350, f"{DC}/h100/", family="hopper", compute="9.0", name="H100 SXM"),
    _row(
        "H100",
        ("h100 pcie",),
        2000,
        f"{DC}/h100/",
        family="hopper",
        compute="9.0",
        name="H100 PCIe",
    ),
    _row(
        "H100", ("h100 nvl",), 3900, f"{DC}/h100/", family="hopper", compute="9.0", name="H100 NVL"
    ),
    _row("H200", ("h200",), 4800, f"{DC}/h200/", family="hopper", compute="9.0"),
    # Turing (compute 7.5): the 2080 Ti's 22 GB memory mod is the same row, its VRAM read
    _turing("RTX 2080 Ti", ("rtx 2080 ti",), 616, f"{S20}/rtx-2080-ti/"),
    _turing("RTX 2080 SUPER", ("rtx 2080 super",), 496, f"{S20}/rtx-2080-super/"),
    _turing("RTX 2080", ("rtx 2080",), 448, f"{S20}/rtx-2080/"),
    _turing("RTX 2070 SUPER", ("rtx 2070 super",), 448, f"{S20}/rtx-2070-super/"),
    _turing("RTX 2070", ("rtx 2070",), 448, f"{S20}/rtx-2070/"),
    _turing("RTX 2060 SUPER", ("rtx 2060 super",), 448, f"{S20}/rtx-2060-super/"),
    _turing("RTX 2060", ("rtx 2060",), 336, f"{S20}/rtx-2060/"),
    _turing(
        "RTX 2060 Mobile",
        ("rtx 2060 laptop", "rtx 2060 mobile", "rtx 2060 with max q"),
        336,
        f"{LAPTOPS}/20-series/",
    ),
    _turing("GTX 1660", ("gtx 1660",), 192, f"{S16}/gtx-1660/"),
    _turing(
        "GTX 1650 Mobile",
        ("gtx 1650 laptop", "gtx 1650 mobile", "gtx 1650 with max q"),
        128,
        f"{LAPTOPS}/16-series/",
    ),
    _turing("T4", ("t4",), 320, f"{DC}/tesla-t4/", tflops=8.1),  # fp32 8.1; 65 with tensor cores
    # Volta and Pascal: llama.cpp's CUDA 12 image by PTX only (`hardware.cards`' floors)
    _row("V100", ("v100",), 900, f"{DC}/v100/", family="volta", compute="7.0"),
    _row("P100", ("p100",), 732, f"{DC}/tesla-p100/", family="pascal-dc", compute="6.0"),
    _pascal("P40", ("p40",), 346, f"{DC}/tesla-p40/"),
    _pascal("GTX 1080 Ti", ("gtx 1080 ti",), 484, f"{S10}/gtx-1080-ti/"),
    _pascal("GTX 1070 Ti", ("gtx 1070 ti",), 256, f"{S10}/gtx-1070-ti/"),
    _pascal("GTX 1070", ("gtx 1070",), 256, f"{S10}/gtx-1070/"),
    _pascal("GTX 1060", ("gtx 1060",), 192, f"{S10}/gtx-1060/"),
    _pascal("GTX 1050 Ti", ("gtx 1050 ti",), 112, f"{S10}/gtx-1050/"),
    # AMD (matched by name, else by gfx target)
    _radeon(
        "RX 9070 XT",
        ("rx 9070 xt",),
        640,
        f"{AMD}/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html",
        family="rdna4",
        gfx="gfx1201",
        pci_ids=(0x7550,),
        tflops=R9700_TFLOPS["vulkan"],
        backend_tflops=R9700_TFLOPS,
        source=_estimated(640, "the R9700's readings: the same Navi 48 with 64 CUs"),
    ),
    _radeon(
        "RX 9070",
        ("rx 9070",),
        640,
        f"{AMD}/desktops/radeon/9000-series/amd-radeon-rx-9070.html",
        family="rdna4",
        gfx="gfx1201",
        pci_ids=(0x7550,),
        tflops=R9070_TFLOPS["vulkan"],
        backend_tflops=R9070_TFLOPS,
        source=_estimated(640, "the R9700's readings scaled to 56 of its 64 CUs"),
    ),
    _radeon(
        "R9700 PRO",
        ("r9700",),
        640,
        f"{AMD}/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html",
        family="rdna4",
        gfx="gfx1201",
        name="Radeon AI PRO R9700",
        pci_ids=(0x7551,),
        tflops=R9700_TFLOPS["vulkan"],
        backend_tflops=R9700_TFLOPS,
        source=_estimated(640, "llama.cpp readings on one and two R9700s"),
    ),
    _radeon(
        "RX 9060 XT",
        ("rx 9060 xt",),
        322,
        f"{AMD}/desktops/radeon/9000-series/amd-radeon-rx-9060xt.html",
        family="rdna4",
        gfx="gfx1200",
    ),
    _radeon(
        "RX 7900 XTX",
        ("rx 7900 xtx",),
        960,
        f"{AMD}/desktops/radeon/7000-series/amd-radeon-rx-7900xtx.html",
        family="rdna3",
        gfx="gfx1100",
        pci_ids=(0x744C,),  # the RX 7900 XT shares it: amd-smi or lspci names that one
    ),
    _radeon(
        "RX 7900 XT",
        ("rx 7900 xt",),
        800,
        f"{AMD}/desktops/radeon/7000-series/amd-radeon-rx-7900xt.html",
        family="rdna3",
        gfx="gfx1100",
    ),
    _radeon(
        "RX 7800 XT",
        ("rx 7800 xt",),
        624,
        f"{AMD}/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html",
        family="rdna3",
        gfx="gfx1101",
    ),
    _radeon(
        "RX 7600 XT",
        ("rx 7600 xt",),
        288,
        f"{AMD}/desktops/radeon/7000-series/amd-radeon-rx-7600-xt.html",
        family="rdna3",
        gfx="gfx1102",
    ),
    _radeon(
        "RX 6950 XT",
        ("rx 6950 xt",),
        576,
        f"{AMD}/desktops/radeon/6000-series/amd-radeon-rx-6950-xt.html",
        family="rdna2",
        gfx="gfx1030",
    ),
    _radeon(
        "RX 6800",
        ("rx 6800",),
        512,
        f"{AMD}/desktops/radeon/6000-series/amd-radeon-rx-6800.html",
        family="rdna2",
        gfx="gfx1030",
    ),
    _radeon(
        "RX 6700 XT",
        ("rx 6700 xt",),
        384,
        f"{AMD}/desktops/radeon/6000-series/amd-radeon-rx-6700-xt.html",
        family="rdna2",
        gfx="gfx1031",
    ),
    _radeon(
        "RX 6600 XT",
        ("rx 6600 xt",),
        256,
        f"{AMD}/desktops/radeon/6000-series/amd-radeon-rx-6600-xt.html",
        family="rdna2",
        gfx="gfx1032",
    ),
    _radeon(
        "RX 6600",
        ("rx 6600",),
        224,
        f"{AMD}/desktops/radeon/6000-series/amd-radeon-rx-6600.html",
        family="rdna2",
        gfx="gfx1032",
    ),
    _radeon(
        "RX 5500 XT",
        ("rx 5500 xt",),
        224,
        f"{AMD}/desktops/radeon/5000-series/amd-radeon-rx-5500-xt.html",
        family="rdna1",
        gfx="gfx1012",
    ),
    _radeon(
        "Radeon 780M",
        ("780m",),
        89.6,
        "the machine's dual-channel DDR5-5600 (AMD's Ryzen 7040/8040 pages)",
        family="rdna3",
        gfx="gfx1103",
        memory=MemoryKind.CARVE_OUT,
        pci_ids=(0x15BF,),
    ),
    _radeon(
        "Radeon 890M",
        ("890m",),
        89.6,
        "",
        family="rdna3.5",
        gfx="gfx1150",
        memory=MemoryKind.CARVE_OUT,
        pci_ids=(0x150E,),
        name="Radeon 890M (Ryzen AI 9 HX 370)",
        source="estimated: dual-channel DDR5-5600 SODIMMs, 89.6 GB/s (no published reading found)",
    ),
    _radeon(
        "Ryzen AI Max+ 395",
        ("8060s", "8050s", "ryzen ai max", "strix halo"),
        215,
        "https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/"
        "amd-ryzen-ai-max-plus-395.html (256-bit LPDDR5X-8000)",
        family="strix-halo",
        gfx="gfx1151",
        memory=MemoryKind.CARVE_OUT,
        name="Strix Halo (Ryzen AI Max+ 395)",
        pci_ids=(0x1586,),
        source="measured by others: about 215 GB/s of the 256 GB/s the 256-bit LPDDR5X-8000 "
        "gives (datahardware.ai)",
    ),
    _row(
        "MI100",
        ("mi100",),
        1229,
        "https://www.amd.com/en/products/accelerators/instinct/mi100.html",
        family="cdna1",
        gfx="gfx908",
        vendor=Vendor.AMD,
        tflops=46.1,
        name="Instinct MI100",
    ),
    _row(
        "MI50",
        ("mi50",),
        1024,
        "https://www.amd.com/en/support/downloads/drivers.html/accelerators/instinct/instinct-mi-series/instinct-mi50-32gb.html",
        family="gcn5",
        gfx="gfx906",
        vendor=Vendor.AMD,
        memory_gb=32,
        name="Instinct MI50",
        pci_ids=(0x66A1,),
    ),
    # Intel Arc (nothing reads Intel cards yet)
    _row(
        "Arc B580",
        ("arc b580",),
        456,
        f"{INTEL}/241598/intel-arc-b580-graphics/specifications.html",
        family="xe2",
        vendor=Vendor.INTEL,
    ),
    _row(
        "Arc Pro B70",
        ("arc pro b70",),
        608,
        "https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/arc/workstations/b-series/overview.html",
        family="xe2",
        vendor=Vendor.INTEL,
    ),
    _row(
        "Arc A770",
        ("arc a770",),
        560,
        f"{INTEL}/229151/intel-arc-a770-graphics-16gb/specifications.html",
        family="xe-hpg",
        vendor=Vendor.INTEL,
        memory_gb=16,
    ),
    _row(
        "Arc A770",
        ("arc a770",),
        512,
        f"{INTEL}/229151/intel-arc-a770-graphics-16gb/specifications.html",
        family="xe-hpg",
        vendor=Vendor.INTEL,
        memory_gb=8,
    ),
    RTX_4070,
)
