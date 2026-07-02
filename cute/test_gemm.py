"""Test the basic quantized INT8 GEMM against an fp16 torch GEMM.

Two-way check (there is no pure-CUDA GEMM counterpart in src/ — the GEMMs are
DSL-only):
  ref  - torch: dequantize the int8 inputs to fp16, matmul (fp32 accumulate), fp16 out.
  cute - the arch's INT8 GEMM (cute/<arch>/gemm/gemm_i8_quant.py, TensorOpGemmI8), which
         int32-accumulates then applies the per-token/per-channel dequant scales.

    <venv>/bin/python cute/test_gemm.py                 # auto-detect arch
    <venv>/bin/python cute/test_gemm.py --arch ampere   # force an arch

Exit 0 on PASS, 1 otherwise. Needs a CUDA torch.
"""
import argparse
import os
import sys
import types

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cute/ (common helpers)
import common

ABS_TOL = 0.05
ATOM_LAYOUT, NUM_STAGES, USE_K32 = (2, 2, 1), 3, True
BM, BN, BK = 128, 128, 64


def run_jit(kern, A_int8, B_int8, scale_a, scale_b, d):
    """Compile + run TensorOpGemmI8; return the dequantized (M, N) result (fp32, cuda)."""
    M, K, N, L = d.M, d.K, d.N, d.L
    M_pad = ((M + BM - 1) // BM) * BM
    N_pad = ((N + BN - 1) // BN) * BN
    K_pad = ((K + BK - 1) // BK) * BK
    mA, a_t = kern.create_and_permute_tensor(L, M_pad, K_pad, False, cutlass.Int8)
    mB, b_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mC, c_t = kern.create_and_permute_tensor(L, M_pad, N_pad, False, cutlass.Float16)
    a_t[:M, :K, 0] = A_int8.cuda()
    b_t[:N, :K, 0] = B_int8.cuda()
    for tt, r, c in [(a_t, M, K), (b_t, N, K)]:
        if tt.shape[0] > r: tt[r:, :, :] = 0
        if tt.shape[1] > c: tt[:, c:, :] = 0
    sa = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); sa[:M, 0] = scale_a.cuda()
    sb = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); sb[:N, 0] = scale_b.cuda()
    mSA, mSB = from_dlpack(sa, assumed_align=16), from_dlpack(sb, assumed_align=16)

    gemm = kern.TensorOpGemmI8(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        ATOM_LAYOUT, USE_K32, BM, bn=BN, num_stages=NUM_STAGES)
    compiled = cute.compile(gemm, mA, mB, mC, mSA, mSB)
    compiled(mA, mB, mC, mSA, mSB)
    torch.cuda.synchronize()
    return c_t[:M, :N, 0].float()


def main():
    ap = argparse.ArgumentParser(description="INT8 GEMM vs fp16 torch GEMM.")
    ap.add_argument("--arch", default=None, help="arch under cute/ (default: auto-detect)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "gemm", "gemm_i8_quant")

    d = types.SimpleNamespace()
    d.M, d.N, d.K, d.L = 256, 768, 512, 1
    print(f"Problem: M={d.M}, N={d.N}, K={d.K}\n")

    # ── inputs: per-token A (M,K), per-channel B (N,K) ────────────────────────
    torch.manual_seed(0)
    A_int8, scale_a = common.quantize_tensor(torch.randn(d.M, d.K) * 0.1, dim=-1)
    B_int8, scale_b = common.quantize_tensor(torch.randn(d.N, d.K) * 0.1, dim=-1)

    # ── (1) cute: run the INT8 GEMM ───────────────────────────────────────────
    print(f"Running INT8 GEMM from cute/{arch}/gemm/gemm_i8_quant.py ...")
    C_cute = run_jit(kern, A_int8, B_int8, scale_a, scale_b, d)

    # ── (2) ref: fp16 torch GEMM of the dequantized inputs ────────────────────
    with torch.inference_mode():
        A_dq = (A_int8.float().cuda() * scale_a.cuda()[:, None]).to(torch.float16)
        B_dq = (B_int8.float().cuda() * scale_b.cuda()[:, None]).to(torch.float16)
        C_ref = (A_dq @ B_dq.T).float()        # (M, N), fp16 inputs -> fp32 accumulate

    print("\n=== INT8 GEMM vs fp16 torch GEMM (max abs error) ===")
    ok = common.report("cute vs ref", C_cute, C_ref) < ABS_TOL
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
