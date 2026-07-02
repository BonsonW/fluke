"""Rotary test: run the CuTe fused INT8 GEMM + rotary kernel and check it.

Picks an IMPLEMENTATION to exercise and one or more REFERENCES to check against:

  --impl jit   run the DSL kernel in-process via cute.compile           (default)
  --impl aot   load the AOT-exported .o (export_to_c) via load_module and run it

  --ref torch  naive torch: dequant(int8 GEMM) fp32, rotate-half, fp16   (default: both)
  --ref cuda   the pure CUDA C kernel fluke_rotary_emb_gpu (src/nn_cuda.c)
  --ref both

The DSL kernel implementation lives under cute/<arch>/rotary/ (imported for the arch
matching the current GPU; override with --arch). This file owns only the test: input
generation, the references, running the chosen implementation, and the comparison.

    <venv>/bin/python cute/test_rotary.py                    # jit, vs torch+cuda
    <venv>/bin/python cute/test_rotary.py --impl aot         # exported .o, vs torch+cuda
    <venv>/bin/python cute/test_rotary.py --arch ampere --ref torch

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

# JIT tiling config (the AOT path uses whatever the export script baked in).
JIT_ATOM_LAYOUT, JIT_NUM_STAGES, JIT_USE_K32 = (2, 2, 1), 3, True
JIT_BM, JIT_BN, JIT_BK = 128, 128, 64


# ── build the padded CuTe descriptors the kernel consumes ─────────────────────
def build_tensors(kern, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d, bm, bn, bk):
    M, K, N, L = d.M, d.K, d.N, d.L
    M_pad = ((M + bm - 1) // bm) * bm
    N_pad = ((N + bn - 1) // bn) * bn
    K_pad = ((K + bk - 1) // bk) * bk
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
    return (from_dlpack(sa, assumed_align=16), from_dlpack(sb, assumed_align=16),
            from_dlpack(sin_buf, assumed_align=16), from_dlpack(cos_buf, assumed_align=16),
            mA, mB, mC, c_t)


def run_jit(kern, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d):
    mSA, mSB, mSin, mCos, mA, mB, mC, c_t = build_tensors(
        kern, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d, JIT_BM, JIT_BN, JIT_BK)
    gemm = kern.TensorOpGemmI8Rotary(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        JIT_ATOM_LAYOUT, JIT_USE_K32, JIT_BM, bn=JIT_BN, num_stages=JIT_NUM_STAGES,
        nhead=d.nhead, head_dim=d.head_dim, rotary_dim=d.rotary_dim, seqlen=d.seqlen)
    seqlen_arg = cutlass.Int32(d.seqlen)
    compiled = cute.compile(gemm, mA, mB, mC, mSA, mSB, mSin, mCos, seqlen_arg)
    compiled(mA, mB, mC, mSA, mSB, mSin, mCos, seqlen_arg)
    torch.cuda.synchronize()
    return c_t[:d.M, :d.N, 0].float()


def run_aot(kern, exp, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d):
    name = (f"gemm_i8_rotary_N{d.N}_K{d.K}"
            f"_H{d.nhead}D{d.head_dim}R{d.rotary_dim}S{exp.CONFIGS[0]['seqlen']}")
    obj = os.path.join(exp.ARTIFACTS_DIR, f"{name}.o")
    if not os.path.isfile(obj):
        print(f"Artifact missing, exporting {name} ...")
        os.makedirs(exp.ARTIFACTS_DIR, exist_ok=True)
        exp._export_one(exp.CONFIGS[0], exp.ARTIFACTS_DIR)
    mSA, mSB, mSin, mCos, mA, mB, mC, c_t = build_tensors(
        kern, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d, exp.BM, exp.BN, 64)
    module = cute.runtime.load_module(obj)
    getattr(module, name)(mA, mB, mC, mSA, mSB, mSin, mCos, d.seqlen)
    torch.cuda.synchronize()
    return c_t[:d.M, :d.N, 0].float()


def torch_reference(C_gemm, sin_buf, cos_buf, d):
    """Naive torch: rotate-half on Q/K of the fp32 dequant GEMM; cast to fp16."""
    out = C_gemm.clone()
    seq_idx = torch.arange(d.M, device=C_gemm.device) % d.seqlen
    cos_row, sin_row = cos_buf[seq_idx], sin_buf[seq_idx]
    for chunk_start in (0, d.nhead * d.head_dim):
        for h in range(d.nhead):
            h0 = chunk_start + h * d.head_dim
            x0 = out[:, h0:h0 + d.sincos_width].clone()
            x1 = out[:, h0 + d.sincos_width:h0 + d.rotary_dim].clone()
            out[:, h0:h0 + d.sincos_width] = x0 * cos_row - x1 * sin_row
            out[:, h0 + d.sincos_width:h0 + d.rotary_dim] = x0 * sin_row + x1 * cos_row
    return out.to(torch.float16).float()


def cuda_reference(C_gemm, sin_buf, cos_buf, d):
    """Pure CUDA C kernel (libfluke.so via ctypes): rotate Q/K chunks in place."""
    lib = fluke_lib.load()
    C = C_gemm.to(torch.float16)
    q_end = d.nhead * d.head_dim
    for chunk_start in (0, q_end):
        chunk = C[:, chunk_start:chunk_start + q_end].contiguous().view(
            d.batch_size, d.sequence_len, d.nhead, d.head_dim)
        lib.fluke_rotary_emb_gpu(
            chunk.data_ptr(), sin_buf.data_ptr(), cos_buf.data_ptr(),
            d.batch_size, d.sequence_len, d.nhead, d.head_dim, d.sincos_width,
            chunk.stride(0), chunk.stride(1), chunk.stride(2))
        C[:, chunk_start:chunk_start + q_end] = chunk.view(d.M, q_end)
    torch.cuda.synchronize()
    return C.float()


def main():
    ap = argparse.ArgumentParser(description="Rotary test (pick impl + reference).")
    ap.add_argument("--arch", default=None, help="arch under cute/ (default: auto-detect)")
    ap.add_argument("--impl", choices=["jit", "aot"], default="jit")
    ap.add_argument("--ref", choices=["torch", "cuda", "both"], default="both")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  impl: {args.impl}  ref: {args.ref}  "
          f"(GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "rotary", "gemm_i8_rotary")
    exp = common.import_impl(arch, "rotary", "export_gemm_i8_rotary") if args.impl == "aot" else None

    # dims: JIT is flexible/small; AOT is pinned to the exported M (scale/sin/cos carry
    # no dynamic shape). Both keep M = batch_size * sequence_len for the CUDA reference.
    d = types.SimpleNamespace()
    d.nhead, d.head_dim, d.rotary_dim, d.K, d.L = 8, 64, 64, 512, 1
    if args.impl == "aot":
        d.M = exp.CONFIGS[0]["M"]                 # 256
        d.batch_size, d.sequence_len = 1, d.M
    else:
        d.batch_size, d.sequence_len = 4, 16
        d.M = d.batch_size * d.sequence_len       # 64
    d.N = 3 * d.nhead * d.head_dim
    d.seqlen = d.sequence_len
    d.sincos_width = d.rotary_dim // 2

    print(f"Problem: M={d.M} (batch={d.batch_size} x seq={d.sequence_len}), N={d.N}, K={d.K}; "
          f"nhead={d.nhead} head_dim={d.head_dim} rotary_dim={d.rotary_dim}\n")

    # ── inputs ────────────────────────────────────────────────────────────────
    torch.manual_seed(0)
    A_int8, scale_a = common.quantize_tensor(torch.randn(d.M, d.K) * 0.1, dim=-1)
    B_int8, scale_b = common.quantize_tensor(torch.randn(d.N, d.K) * 0.1, dim=-1)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d.sincos_width).float() / d.head_dim * 2))
    freqs = torch.outer(torch.arange(d.seqlen).float(), inv_freq)
    sin_buf = freqs.sin().contiguous().cuda()
    cos_buf = freqs.cos().contiguous().cuda()

    # ── run the chosen implementation ─────────────────────────────────────────
    print(f"Running {args.impl} implementation from cute/{arch}/rotary/ ...")
    if args.impl == "aot":
        C_dsl = run_aot(kern, exp, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d)
    else:
        C_dsl = run_jit(kern, A_int8, B_int8, scale_a, scale_b, sin_buf, cos_buf, d)

    # ── shared: fp32 dequant GEMM (the pre-rotary QKV both references start from) ─
    with torch.inference_mode():
        A_dq = A_int8.float().cuda() * scale_a.cuda()[:, None]
        B_dq = B_int8.float().cuda() * scale_b.cuda()[:, None]
        C_gemm = A_dq @ B_dq.T

    # ── compare against the requested reference(s) ────────────────────────────
    print(f"\n=== {args.impl} kernel vs reference(s) (max abs error) ===")
    worst = 0.0
    if args.ref in ("torch", "both"):
        worst = max(worst, common.report("vs torch", C_dsl, torch_reference(C_gemm, sin_buf, cos_buf, d)))
    if args.ref in ("cuda", "both"):
        worst = max(worst, common.report("vs cuda C", C_dsl, cuda_reference(C_gemm, sin_buf, cos_buf, d)))

    ok = worst < ABS_TOL
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
