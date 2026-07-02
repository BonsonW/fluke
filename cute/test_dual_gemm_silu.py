"""Dual-GEMM+SiLU test: run the CuTe fused kernel and check it.

Computes out[M,N] = silu(A@B_gate^T) * (A@B_up^T), fp16.

  --impl jit   run the DSL kernel in-process via cute.compile           (default)
  --impl aot   load the AOT-exported .o (export_to_c) via load_module and run it

  --ref torch  naive torch: dequant GEMMs, silu(gate)*up                 (default: both)
  --ref cuda   torch GEMMs then the pure CUDA C kernel fluke_silu_mul_gpu (src/nn_cuda.c)
  --ref both

The DSL kernel implementation lives under cute/<arch>/dual_gemm_silu/. This file owns
the test: inputs, references, running the chosen implementation, and the comparison.
The tiling config is taken from the export module so jit/aot/export can't drift.

    <venv>/bin/python cute/test_dual_gemm_silu.py
    <venv>/bin/python cute/test_dual_gemm_silu.py --impl aot --ref torch

Exit 0 on PASS, 1 otherwise. Needs a CUDA torch + CUDA toolkit.
"""
import argparse
import os
import sys
import types

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cute/ (common)
import common
sys.path.insert(0, common.ROOT)                                  # fluke root (fluke_lib)
import fluke_lib

ABS_TOL = 0.05


# ── build the padded CuTe descriptors the kernel consumes ─────────────────────
def build_tensors(kern, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d, bm, bn, bk):
    M, K, N, L = d.M, d.K, d.N, d.L
    M_pad = ((M + bm - 1) // bm) * bm
    N_pad = ((N + bn - 1) // bn) * bn
    K_pad = ((K + bk - 1) // bk) * bk
    mA, a_t = kern.create_and_permute_tensor(L, M_pad, K_pad, False, cutlass.Int8)
    mBg, bg_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mBu, bu_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mC, c_t = kern.create_and_permute_tensor(L, M_pad, N_pad, False, cutlass.Float16)
    a_t[:M, :K, 0] = A_i8.cuda()
    bg_t[:N, :K, 0] = Bg_i8.cuda()
    bu_t[:N, :K, 0] = Bu_i8.cuda()
    for tt, r, c in [(a_t, M, K), (bg_t, N, K), (bu_t, N, K)]:
        if tt.shape[0] > r: tt[r:, :, :] = 0
        if tt.shape[1] > c: tt[:, c:, :] = 0
    sca = torch.zeros(M_pad, L, dtype=torch.float32, device='cuda'); sca[:M, 0] = sa.cuda()
    scg = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scg[:N, 0] = sbg.cuda()
    scu = torch.zeros(N_pad, L, dtype=torch.float32, device='cuda'); scu[:N, 0] = sbu.cuda()
    return (mA, mBg, mBu, mC,
            from_dlpack(sca, assumed_align=16), from_dlpack(scg, assumed_align=16),
            from_dlpack(scu, assumed_align=16), c_t)


def run_jit(kern, exp, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d):
    mA, mBg, mBu, mC, mSA, mSBg, mSBu, c_t = build_tensors(
        kern, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d, exp.BM, exp.BN, 64)
    gemm = kern.TensorOpDualGemmI8Silu(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        exp.ATOM_LAYOUT, True, exp.BM, bn=exp.BN, num_stages=exp.NUM_STAGES)
    compiled = cute.compile(gemm, mA, mBg, mBu, mC, mSA, mSBg, mSBu)
    compiled(mA, mBg, mBu, mC, mSA, mSBg, mSBu)
    torch.cuda.synchronize()
    return c_t[:d.M, :d.N, 0].float()


def run_aot(kern, exp, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d):
    name = f"gemm_i8_dual_silu_N{d.N}_K{d.K}"
    obj = os.path.join(exp.ARTIFACTS_DIR, f"{name}.o")
    if not os.path.isfile(obj):
        print(f"Artifact missing, exporting {name} ...")
        os.makedirs(exp.ARTIFACTS_DIR, exist_ok=True)
        exp._export_one(exp.CONFIGS[0], exp.ARTIFACTS_DIR)
    mA, mBg, mBu, mC, mSA, mSBg, mSBu, c_t = build_tensors(
        kern, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d, exp.BM, exp.BN, 64)
    module = cute.runtime.load_module(obj)
    getattr(module, name)(mA, mBg, mBu, mC, mSA, mSBg, mSBu)
    torch.cuda.synchronize()
    return c_t[:d.M, :d.N, 0].float()


def torch_reference(A_i8, Bg_i8, Bu_i8, sa, sbg, sbu):
    """Naive torch: dequant GEMMs, silu(gate) * up, cast to fp16."""
    A = A_i8.float().cuda() * sa.cuda()[:, None]
    Bg = Bg_i8.float().cuda() * sbg.cuda()[:, None]
    Bu = Bu_i8.float().cuda() * sbu.cuda()[:, None]
    gate, up = A @ Bg.T, A @ Bu.T
    return (gate * torch.sigmoid(gate) * up).to(torch.float16).float()


def cuda_reference(A_i8, Bg_i8, Bu_i8, sa, sbg, sbu):
    """torch GEMMs (fp16), then the pure CUDA C fluke_silu_mul_gpu on [up | gate]."""
    lib = fluke_lib.load()
    A = A_i8.float().cuda() * sa.cuda()[:, None]
    Bg = Bg_i8.float().cuda() * sbg.cuda()[:, None]
    Bu = Bu_i8.float().cuda() * sbu.cuda()[:, None]
    gate = (A @ Bg.T).to(torch.float16)
    up = (A @ Bu.T).to(torch.float16)
    # silu_mul does out = silu(second half) * first half, so pass [up | gate].
    in2n = torch.cat([up, gate], dim=1).contiguous()
    out = torch.empty_like(gate)
    lib.fluke_silu_mul_gpu(in2n.data_ptr(), out.data_ptr(), in2n.size(0), out.size(1))
    torch.cuda.synchronize()
    return out.float()


def main():
    ap = argparse.ArgumentParser(description="Dual-GEMM+SiLU test (pick impl + reference).")
    ap.add_argument("--arch", default=None, help="arch under cute/ (default: auto-detect)")
    ap.add_argument("--impl", choices=["jit", "aot"], default="jit")
    ap.add_argument("--ref", choices=["torch", "cuda", "both"], default="both")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  impl: {args.impl}  ref: {args.ref}  "
          f"(GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "dual_gemm_silu", "dual_gemm_i8_silu")
    exp = common.import_impl(arch, "dual_gemm_silu", "export_dual_gemm_i8_silu")

    # dims: AOT is pinned to the exported M/N (scale descriptors carry no dynamic shape);
    # JIT is flexible/small.
    d = types.SimpleNamespace(K=512, L=1)
    if args.impl == "aot":
        d.M, d.N, d.K = exp.CONFIGS[0]["M"], exp.CONFIGS[0]["N"], exp.CONFIGS[0]["K"]
    else:
        d.M, d.N = 64, 512
    print(f"Problem: M={d.M}, N={d.N}, K={d.K}  (out = silu(A@Bg^T) * (A@Bu^T))\n")

    # ── inputs: per-token A (M,K), per-channel Bgate/Bup (N,K) ────────────────
    torch.manual_seed(0)
    A_i8, sa = common.quantize_tensor(torch.randn(d.M, d.K) * 0.1, dim=-1)
    Bg_i8, sbg = common.quantize_tensor(torch.randn(d.N, d.K) * 0.1, dim=-1)
    Bu_i8, sbu = common.quantize_tensor(torch.randn(d.N, d.K) * 0.1, dim=-1)

    print(f"Running {args.impl} implementation from cute/{arch}/dual_gemm_silu/ ...")
    run = run_aot if args.impl == "aot" else run_jit
    C_dsl = run(kern, exp, A_i8, Bg_i8, Bu_i8, sa, sbg, sbu, d)

    print(f"\n=== {args.impl} kernel vs reference(s) (max abs error) ===")
    worst = 0.0
    if args.ref in ("torch", "both"):
        worst = max(worst, common.report("vs torch", C_dsl, torch_reference(A_i8, Bg_i8, Bu_i8, sa, sbg, sbu)))
    if args.ref in ("cuda", "both"):
        worst = max(worst, common.report("vs cuda C", C_dsl, cuda_reference(A_i8, Bg_i8, Bu_i8, sa, sbg, sbu)))

    ok = worst < ABS_TOL
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
