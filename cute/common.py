"""Shared, op-agnostic test helpers for the CuTe (NVIDIA) kernels.

Layout: kernel IMPLEMENTATIONS live under cute/<arch>/... (the DSL kernel + its
export script). The per-op TESTS live at cute/test_<op>.py; each picks an arch
(matching the current GPU, or --arch), imports that arch's implementation module,
runs it (JIT-compiled, or from an AOT-exported .o), and compares the output against
the naive torch reference and/or the pure CUDA C kernel.

Add an arch = drop its implementation under cute/<newarch>/ and add the compute
capability to ARCH_BY_CC. (A HIP/fly mirror would live under fly/; not built yet.)
"""
import importlib
import os
import sys

import torch

CUTE_DIR = os.path.dirname(os.path.abspath(__file__))   # fluke/cute
ROOT = os.path.dirname(CUTE_DIR)                         # fluke repo root

# Map GPU compute capability -> arch subdir under cute/. Extend as archs are added.
ARCH_BY_CC = {
    (8, 0): "ampere", (8, 6): "ampere", (8, 7): "ampere",
    (8, 9): "ada",
    (9, 0): "hopper",
    (10, 0): "blackwell", (12, 0): "blackwell",
}


def detect_arch():
    cc = torch.cuda.get_device_capability()
    arch = ARCH_BY_CC.get(cc)
    if arch is None:
        raise SystemExit(f"No arch mapping for compute capability {cc}. "
                         f"Add it to ARCH_BY_CC (cute/common.py) or pass --arch.")
    return arch


def import_impl(arch, subdir, module):
    """Import `module` from cute/<arch>/<subdir> (also puts cute/<arch> on the path
    so the module's shared base-class imports, e.g. gemm_i8_quant, resolve)."""
    for p in (os.path.join(CUTE_DIR, arch, subdir), os.path.join(CUTE_DIR, arch)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(module)


def quantize_tensor(t, dim=-1):
    """Symmetric per-row (dim=-1) int8 quantization. Returns (int8, dequant scale)."""
    qm = 127
    fr = t.abs().amax(dim=dim).clamp_min(1e-8)
    qs = qm / fr
    ti = (t * qs.unsqueeze(dim)).round().clamp(-qm, qm).to(torch.int8)
    return ti, qs.to(torch.float32).reciprocal()


def report(name, got, exp):
    """Print + return max abs error between two tensors (also reports rel)."""
    abs_err = (got - exp).abs()
    rel_err = abs_err / exp.abs().clamp(min=1e-6)
    print(f"  {name:16s}  max_abs={abs_err.max().item():.6f}  "
          f"mean_abs={abs_err.mean().item():.8f}  max_rel={rel_err.max().item():.6f}")
    return abs_err.max().item()
