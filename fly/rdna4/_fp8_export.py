"""Shared machinery for the RDNA4 fp8 AOT export scripts.

The per-op export_*.py scripts own their kernel-specific C wrapper template; everything
else — the per-arch compile loop, HSACO extraction from the FlyDSL MLIR, the artifacts
directory layout, and the embed-friendly `<prefix>_Module_LoadData` loader — lives here so
all four ops stay consistent.

RDNA4 has no single "generic" code object that compiles through FlyDSL's MLIR (the ROCDL
chipset parser rejects gfx12-generic) and per-chip objects don't cross-load, so each kernel
is exported once per concrete RDNA4 arch. fluke_fp8_select (src/fused_hip.cpp) picks the
matching embedded HSACO by the device's gcnArchName at runtime.
"""
import os
import re

# Concrete RDNA4 arches we ship. Add a chip here + re-run the exports to support it.
RDNA4_ARCHS = ["gfx1200", "gfx1201"]

# fluke repo root = .../fluke (this file is fluke/fly/rdna4/_fp8_export.py)
_FLUKE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def artifacts_dir_for(arch: str) -> str:
    """artifacts/<arch>/ under the fluke repo root (git-ignored build output).

    Flat per-arch layout, matching the CUDA side's artifacts/sm80/: one dir per concrete
    GPU arch (sm80 for CUDA; gfx1200/gfx1201/... for HIP)."""
    return os.path.join(_FLUKE_ROOT, "artifacts", arch)


def set_compile_arch(arch: str) -> None:
    """Force FlyDSL to compile for `arch` (overrides the auto-detected local GPU) and enable
    compile-only mode so the export never *launches* the kernel — the launcher call triggers
    compilation (populating _mem_cache) and returns without a GPU dispatch. This lets us export
    a non-local arch (e.g. gfx1200 from a gfx1201 host) without hipErrorNoBinaryForGpu poisoning
    the HIP context for the next kernel's compile."""
    from flydsl.utils import env
    env.compile.arch = arch
    env.compile.compile_only = True


# ── HSACO extraction from the FlyDSL MLIR IR text ─────────────────────────────
# (These are duplicated verbatim from the original fly/export_*.py scripts; centralised
#  here so every op decodes the module the same way.)

def decode_mlir_bin(ir_text: str, start: int) -> bytes:
    marker = 'bin = "'
    pos = ir_text.find(marker, start)
    if pos == -1:
        raise ValueError("'bin = \"' not found in IR text after position %d" % start)
    i = pos + len(marker)
    result = bytearray()
    while i < len(ir_text):
        c = ir_text[i]
        if c == '\\':
            nxt = ir_text[i + 1]
            if nxt == '\\':
                result.append(ord('\\')); i += 2
            elif nxt == '"':
                result.append(ord('"')); i += 2
            else:
                result.append(int(ir_text[i + 1: i + 3], 16)); i += 3
        elif c == '"':
            break
        else:
            result.append(ord(c)); i += 1
    return bytes(result)


def extract_hsaco(ir_text: str, prefer_no_wave64: bool = True) -> bytes:
    anchor = 0
    if prefer_no_wave64:
        found = ir_text.find("no_wave64")
        if found == -1:
            print("Warning: no_wave64 variant not found, falling back to first object")
        else:
            anchor = found
    hsaco = decode_mlir_bin(ir_text, anchor)
    if hsaco[:4] != b"\x7fELF":
        raise RuntimeError(f"Expected ELF magic, got: {hsaco[:8].hex()}")
    return hsaco


def find_kernel_name(ir_text: str, default: str) -> str:
    m = re.search(r'#gpu\.kernel_metadata<"([^"]+)"', ir_text)
    return m.group(1) if m else default


# ── Shared C loader fragment (embed-friendly) ─────────────────────────────────

def module_loaddata_fn(prefix: str, kernel_name: str) -> str:
    """Return the `<prefix>_Module_LoadData` C function that loads the kernel from an
    in-memory HSACO image (bytes bundled into libfluke.a via objcopy), rather than a file
    path. Paired with the `<prefix>_Module_Load(path)` the templates already emit."""
    return f"""
/* Load the kernel from an in-memory HSACO image (bundled into libfluke.a). */
static inline int {prefix}_Module_LoadData({prefix}_Module_t *m, const void *image) {{
    hipError_t err = hipModuleLoadData(&m->module, image);
    if (err != hipSuccess) {{
        fprintf(stderr, "{prefix}: hipModuleLoadData: %s\\n", hipGetErrorString(err));
        return (int)err;
    }}
    err = hipModuleGetFunction(&m->func, m->module, "{kernel_name}");
    if (err != hipSuccess) {{
        fprintf(stderr, "{prefix}: hipModuleGetFunction: %s\\n", hipGetErrorString(err));
        (void)hipModuleUnload(m->module);
        return (int)err;
    }}
    return 0;
}}
"""


def write_artifacts(artifacts_dir: str, name: str, hsaco: bytes, header_text: str) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    hsaco_path = os.path.join(artifacts_dir, f"{name}.hsaco")
    header_path = os.path.join(artifacts_dir, f"{name}.h")
    with open(hsaco_path, "wb") as f:
        f.write(hsaco)
    with open(header_path, "w") as f:
        f.write(header_text)
    print(f"  wrote {len(hsaco):,} bytes -> {hsaco_path}")
    print(f"  wrote header       -> {header_path}")
