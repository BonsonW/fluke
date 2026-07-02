#!/usr/bin/env python3
"""Export the factored-LSTM layer kernels to C headers (AOT).

The factored-LSTM layer runs two kernels (see slorado CRFModel.cpp):
  1. int8 down-projection  hh_down = h_int8 @ dn_int8^T   (recurrent per-step, and the
     input-projection precompute) -- the plain INT8 GEMM (gemm_i8_quant, int8 -> f16).
  2. fused factored-LSTM step (factored_lstm_i8): two f16 up-projections into 4 gate
     accumulators + gates + cell update + int8 hidden output (fixed scale 1/127).

Edit CONFIG / the tiling knobs below and run:
    <venv>/bin/python cute/ampere/factored_lstm/export_factored_lstm_i8.py

Output (in the top-level artifacts/<arch>/):
    factored_lstm_i8_H{H}_Khh{K_hh}_R{R}.h  (+ .o)   fused step (11-arg wrapper)
    down_proj_i8_R{R}_K{H}.h                (+ .o)   int8 down-proj (5-arg GEMM wrapper)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                    # this dir (factored_lstm_i8)
sys.path.insert(0, os.path.dirname(_HERE))          # parent cute/ampere
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gemm"))  # shared base classes (gemm_i8_quant, gemm_f16)

import cutlass
from factored_lstm_i8 import export_factored_lstm_i8
from gemm_i8_quant import export_tensor_op_gemm_i8

# ── export configuration ──────────────────────────────────────────────────────
# One LSTM layer shape. B (M) sizes the symbolic trace only (dynamic at runtime).
# H = hidden dim, K_hh = hidden rank, R = input rank; Kc = K_hh + R (merged contraction).
CONFIG = dict(H=1024, K_hh=128, R=128, B=128)

# Tiling / scheduling knobs.
# Fused step: 4 gate accumulators (i/f/g/o) -> keep bN small. K-merged f16 up-proj.
LSTM_BM, LSTM_BN, LSTM_BK, LSTM_STAGES = 64, 32, 32, 3
LSTM_ATOM = (2, 2, 1)
# int8 down-proj: N=R=128 (one N tile), small-M friendly bm.
DP_BM, DP_BN, DP_STAGES = 64, 128, 3
DP_ATOM = (2, 2, 1)

# Exported .h/.o live in the top-level artifacts/<arch>/ (bundled into libfluke per arch).
# Ampere compiles to sm80 cubins (also run on sm86/sm89). _HERE = cute/ampere/factored_lstm.
ARCH = "sm80"
_FLUKE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
ARTIFACTS_DIR = os.path.join(_FLUKE_ROOT, "artifacts", ARCH)


def _export_step(cfg, artifacts_dir):
    H, K_hh, R = cfg["H"], cfg["K_hh"], cfg["R"]
    name = f"factored_lstm_i8_H{H}_Khh{K_hh}_R{R}"
    print(f"\n[fused step  H={H} K_hh={K_hh} R={R}]  -> {name}.h")
    export_factored_lstm_i8(
        atom_layout_mnk=LSTM_ATOM,
        file_path=artifacts_dir, file_name=name, function_prefix=name,
        H=H, K_hh=K_hh, R=R,
        bm=LSTM_BM, bn=LSTM_BN, bk=LSTM_BK, num_stages=LSTM_STAGES, b_size=cfg["B"],
    )


def _export_down_proj_i8(cfg, artifacts_dir):
    H, R = cfg["H"], cfg["R"]
    name = f"down_proj_i8_R{R}_K{H}"
    print(f"\n[int8 down-proj  N=R={R} K=H={H}]  -> {name}.h")
    export_tensor_op_gemm_i8(
        a_dtype=cutlass.Int8, b_dtype=cutlass.Int8,
        c_dtype=cutlass.Float16, acc_dtype=cutlass.Int32,
        atom_layout_mnk=DP_ATOM,
        file_path=artifacts_dir, file_name=name, function_prefix=name,
        use_k32=True, bm=DP_BM, bn=min(DP_BN, R), num_stages=DP_STAGES,
        m_size=cfg["B"], n_size=R, k_size=H,
    )


def _export_all(cfg, artifacts_dir):
    _export_step(cfg, artifacts_dir)
    _export_down_proj_i8(cfg, artifacts_dir)


if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    _export_all(CONFIG, ARTIFACTS_DIR)
    print(f"\nAll exports complete -> {ARTIFACTS_DIR}/")
