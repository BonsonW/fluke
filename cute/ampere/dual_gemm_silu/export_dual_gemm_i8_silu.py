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
sys.path.insert(0, os.path.dirname(_HERE))   # parent cute/ampere (gemm_i8_quant base class)
from dual_gemm_i8_silu import export_dual_gemm_i8_silu

# ── export configuration ──────────────────────────────────────────────────────
# Each entry emits one C header/object pair. M sizes the symbolic trace only (M is
# dynamic at runtime). N = inter/hidden dim (gate & up projection width), K = model dim.
CONFIGS = [
    dict(M=256, N=2048, K=512),
]

# Tiling / scheduling knobs. Dual accumulator (gate + up) doubles register pressure,
# so bN is small (32 wins at K=512); constraint: bN >= atom_N * mmaN * 2.
BM = 128
BN = 32
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
