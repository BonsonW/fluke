"""Shared, op-agnostic test/export helpers for the FlyDSL (AMD) fp8 kernels.

Mirror of cute/common.py for the RDNA/CDNA side. Kernel IMPLEMENTATIONS live under
fly/<arch>/<op>/ (the FlyDSL kernel + its export script). The per-op TESTS live at
fly/test_<op>.py; each picks an arch (matching the current GPU, or --arch), imports that
arch's implementation module, runs it (JIT-compiled, or from an AOT-exported .hsaco), and
compares the output against a naive torch reference and/or the pure HIP C kernel.

Add an arch = drop its implementation under fly/<newarch>/ and add its gfx prefix to
ARCH_BY_GFX. The fused fp8 artifacts are compiled to the gfx12-generic code object, so one
artifact set covers every RDNA4 chip (gfx1200/gfx1201/...).
"""
import ctypes
import importlib
import os
import subprocess
import sys

import torch

FLY_DIR = os.path.dirname(os.path.abspath(__file__))    # fluke/fly
ROOT = os.path.dirname(FLY_DIR)                          # fluke repo root

# Code-object target the fused artifacts are exported for (covers all RDNA4).
GENERIC_ARCH = "gfx12-generic"

# Map a device gfx-arch prefix -> arch subdir under fly/. Extend as archs are added.
ARCH_BY_GFX = {
    "gfx120": "rdna4",   # RDNA4: gfx1200 (RX 9070/XT), gfx1201 (R9700), ...
}


def get_gfx_arch():
    """Return the current GPU's gfx arch string (e.g. 'gfx1201'), or '' if none.

    Works under a ROCm build of torch (which presents as torch.cuda.*)."""
    if not torch.cuda.is_available():
        return ""
    name = torch.cuda.get_device_properties(0).gcnArchName  # e.g. 'gfx1201:sramecc+:xnack-'
    return name.split(":", 1)[0]


def detect_arch():
    gfx = get_gfx_arch()
    for prefix, arch in ARCH_BY_GFX.items():
        if gfx.startswith(prefix):
            return arch
    raise SystemExit(f"No arch mapping for GPU gfx-arch {gfx!r}. "
                     f"Add its prefix to ARCH_BY_GFX (fly/common.py) or pass --arch.")


def import_impl(arch, subdir, module):
    """Import `module` from fly/<arch>/<subdir>. Also puts the shared base dirs
    (fly/<arch>/gemm, fly/<arch>/quantize) and fly/<arch> on the path so cross-op imports
    resolve from any op subdir."""
    for p in (os.path.join(FLY_DIR, arch, subdir),
              os.path.join(FLY_DIR, arch, "gemm"),
              os.path.join(FLY_DIR, arch, "quantize"),
              os.path.join(FLY_DIR, arch)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(module)


def report(name, got, exp):
    """Print + return max abs error between two tensors (also reports rel)."""
    abs_err = (got.float() - exp.float()).abs()
    rel_err = abs_err / exp.float().abs().clamp(min=1e-6)
    print(f"  {name:16s}  max_abs={abs_err.max().item():.6f}  "
          f"mean_abs={abs_err.mean().item():.8f}  max_rel={rel_err.max().item():.6f}")
    return abs_err.max().item()


# ── fluke fp8 C ABI via ctypes (the same ABI slorado links) ───────────────────
# Mirrors fluke_lib.py, but for the fused fp8 kernels in libfluke_fp8.so (built by
# `make fp8_shared rocm=1`). Tests call fluke_fp8_select + the launch wrappers directly on
# torch device pointers to exercise the embedded-HSACO load + arch dispatch path end-to-end.

_FP8_SO = os.path.join(ROOT, "lib", "libfluke_fp8.so")


class fluke_dims_t(ctypes.Structure):
    _fields_ = [("d_model", ctypes.c_int), ("dim_feedforward", ctypes.c_int),
                ("nhead", ctypes.c_int), ("head_dim", ctypes.c_int), ("max_seq", ctypes.c_int)]


def build_fp8_shared(rebuild=False):
    """Build lib/libfluke_fp8.so for the current GPU (make fp8_shared rocm=1 ROCM_ARCH=...)."""
    if rebuild or not os.path.isfile(_FP8_SO):
        gfx = get_gfx_arch() or "gfx1201"
        print(f">> building libfluke_fp8.so (make fp8_shared, --offload-arch={gfx})")
        subprocess.run(["make", "fp8_shared", "rocm=1", f"ROCM_ARCH=--offload-arch={gfx}"],
                       cwd=ROOT, check=True)


def load_fp8_lib(rebuild=False):
    """Return the libfluke_fp8.so CDLL with argtypes set (building it first if needed)."""
    build_fp8_shared(rebuild)
    lib = ctypes.CDLL(_FP8_SO)
    P, I = ctypes.c_void_p, ctypes.c_int
    lib.fluke_fp8_select.restype = P
    lib.fluke_fp8_select.argtypes = [I, fluke_dims_t]
    lib.fluke_qkv_rotary_fp8_gpu.restype = I
    lib.fluke_qkv_rotary_fp8_gpu.argtypes = [P, P, P, P, P, P, P, P, I, I]
    lib.fluke_gated_mlp_fp8_gpu.restype = I
    lib.fluke_gated_mlp_fp8_gpu.argtypes = [P, P, P, P, P, P, P, P, I]
    return lib
