#!/usr/bin/env python3
"""Export the dual INT8 GEMM + SiLU kernel to C headers (AOT).

Edit CONFIGS / the tiling knobs below and run:
    <venv>/bin/python cute/ampere/dual_gemm_silu/export_dual_gemm_i8_silu.py

Output (in the top-level artifacts/):
    gemm_i8_dual_silu_N{N}_K{K}.h  (+ .o)
    out[M, N] = silu(A@B_gate^T) * (A@B_up^T), fp16 output.  N = inter dim, K = model dim.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                    # this dir (dual_gemm_i8_silu)
sys.path.insert(0, os.path.dirname(_HERE))          # parent cute/ampere
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gemm"))  # shared base classes (gemm_i8_quant, gemm_f16)
from dual_gemm_i8_silu import export_dual_gemm_i8_silu

# ── export configuration ──────────────────────────────────────────────────────
# Each entry emits one C header/object pair. M sizes the symbolic trace only (M is
# dynamic at runtime). N = inter/hidden dim (gate & up projection width), K = model dim.
CONFIGS = [
    dict(M=256, N=2048, K=512),
]

# Tiling / scheduling knobs. Dual accumulator (gate + up) doubles register pressure.
# bN=64 with the NARROW atom (2,2,1) wins at production scale (large M): it halves the
# A/weight L2 re-reads (kernel is L2-traffic bound), reaching ~46% of peak vs ~38% at
# bN=32 — ~+10% at M>=2048, and it stacks with the coalesced+padded smem-staged epilogue.
# bN=32 only wins for M<=1024 (decode); production is M~1M. Constraint: bN >= atom_N*mmaN*2
# (=64 for atom_N=2 with the N-doubled permutation). Do NOT pair bN=64 with atom (2,4,1) —
# the wide atom loses (~30% of peak). See autotune_dual.py / the round-129 profiling.
BM = 128
BN = 64
NUM_STAGES = 3
ATOM_LAYOUT = (2, 2, 1)

# Exported .h/.o live in the top-level artifacts/<arch>/ dir (bundled into libfluke.a per
# arch). Ampere compiles to sm80 cubins (also run on sm86/sm89). _HERE = cute/ampere/dual_gemm_silu.
ARCH = "sm80"
_FLUKE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
ARTIFACTS_DIR = os.path.join(_FLUKE_ROOT, "artifacts", ARCH)


def _export_one(cfg, artifacts_dir):
    name = f"gemm_i8_dual_silu_N{cfg['N']}_K{cfg['K']}"
    print(f"\n[M={cfg['M']} N={cfg['N']} K={cfg['K']}]  -> {name}.h")
    export_dual_gemm_i8_silu(
        atom_layout_mnk=ATOM_LAYOUT,
        file_path=artifacts_dir,
        file_name=name,
        function_prefix=name,
        bm=BM, bn=BN, num_stages=NUM_STAGES,
        m_size=cfg["M"], n_size=cfg["N"], k_size=cfg["K"],
    )


if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    for cfg in CONFIGS:
        _export_one(cfg, ARTIFACTS_DIR)
    print(f"\nAll exports complete -> {ARTIFACTS_DIR}/")
