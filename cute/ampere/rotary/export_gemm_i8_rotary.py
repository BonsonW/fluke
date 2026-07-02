#!/usr/bin/env python3
"""Export the fused INT8 GEMM + rotary kernel to C headers (AOT).

Edit CONFIGS / the tiling knobs below and run:
    <venv>/bin/python cute/ampere/rotary/export_gemm_i8_rotary.py

Output (in ./artifacts/):
    gemm_i8_rotary_N{N}_K{K}_H{nhead}D{head_dim}R{rotary_dim}S{seqlen}.h  (+ .o)
    (N = 3 * nhead * head_dim, the concatenated QKV width)
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                    # this dir (gemm_i8_rotary)
sys.path.insert(0, os.path.dirname(_HERE))          # parent cute/ampere
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gemm"))  # shared base classes (gemm_i8_quant, gemm_f16)
from gemm_i8_rotary import export_gemm_i8_rotary

# ── export configuration ──────────────────────────────────────────────────────
# Each entry emits one C header/object pair. M sizes the symbolic trace only (M is
# dynamic at runtime); seqlen sizes the baked sin/cos table = the MAX supported
# sequence length (the actual seqlen is a runtime scalar arg, any value in [1, seqlen]).
CONFIGS = [
    dict(M=256, K=512, nhead=8, head_dim=64, rotary_dim=64, seqlen=2048),
]

# Tiling / scheduling knobs for the exported kernel.
BM = 128
BN = 256              # N tile (must be a multiple of head_dim)
NUM_STAGES = 3
ATOM_LAYOUT = (2, 4, 1)

# Exported .h/.o live in the top-level artifacts/<arch>/ dir (bundled into libfluke.a per
# arch). Ampere compiles to sm80 cubins (also run on sm86/sm89). _HERE = cute/ampere/rotary.
ARCH = "sm80"
_FLUKE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
ARTIFACTS_DIR = os.path.join(_FLUKE_ROOT, "artifacts", ARCH)


def _export_one(cfg, artifacts_dir):
    N = 3 * cfg["nhead"] * cfg["head_dim"]
    name = (f"gemm_i8_rotary_N{N}_K{cfg['K']}"
            f"_H{cfg['nhead']}D{cfg['head_dim']}R{cfg['rotary_dim']}S{cfg['seqlen']}")
    print(f"\n[M={cfg['M']} N={N} K={cfg['K']}]  nhead={cfg['nhead']} "
          f"head_dim={cfg['head_dim']} rotary_dim={cfg['rotary_dim']} seqlen={cfg['seqlen']}  -> {name}.h")
    export_gemm_i8_rotary(
        nhead=cfg["nhead"], head_dim=cfg["head_dim"],
        rotary_dim=cfg["rotary_dim"], seqlen=cfg["seqlen"],
        atom_layout_mnk=ATOM_LAYOUT,
        file_path=artifacts_dir,
        file_name=name,
        function_prefix=name,
        bm=BM, bn=BN, num_stages=NUM_STAGES,
        m_size=cfg["M"], k_size=cfg["K"],
    )


if __name__ == "__main__":
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    for cfg in CONFIGS:
        _export_one(cfg, ARTIFACTS_DIR)
    print(f"\nAll exports complete -> {ARTIFACTS_DIR}/")
