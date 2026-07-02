"""Benchmark the fused dual INT8 GEMM + SiLU kernel (cute/<arch>/dual_gemm_silu/).

out[M,N] = silu(A@B_gate^T) * (A@B_up^T). Two GEMMs (gate + up) share the A load, so
the compute is 2 * (2*M*N*K). Compiles the kernel with the exported tiling config,
warms the GPU clocks, then times with CUDA events and reports TOPS + bandwidth.

    <venv>/bin/python cute/bench_dual_gemm_silu.py
    <venv>/bin/python cute/bench_dual_gemm_silu.py --M 131072 --N 2048 --K 512

ALWAYS let it warm the GPU first — a cold clock badly skews the numbers.
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

# A100 reference peaks (adjust for other GPUs).
PEAK_TOPS = 624.0
PEAK_BW_GBS = 2000.0


def build(kern, d, bm, bn, bk):
    M, K, N, L = d.M, d.K, d.N, d.L
    M_pad = ((M + bm - 1) // bm) * bm
    N_pad = ((N + bn - 1) // bn) * bn
    K_pad = ((K + bk - 1) // bk) * bk
    mA, a_t = kern.create_and_permute_tensor(L, M_pad, K_pad, False, cutlass.Int8)
    mBg, bg_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mBu, bu_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mC, c_t = kern.create_and_permute_tensor(L, M_pad, N_pad, False, cutlass.Float16)
    for tt, r, c in [(a_t, M, K), (bg_t, N, K), (bu_t, N, K)]:
        if tt.shape[0] > r: tt[r:, :, :] = 0
        if tt.shape[1] > c: tt[:, c:, :] = 0
    sca = from_dlpack(torch.ones(M_pad, L, dtype=torch.float32, device='cuda'), assumed_align=16)
    scg = from_dlpack(torch.ones(N_pad, L, dtype=torch.float32, device='cuda'), assumed_align=16)
    scu = from_dlpack(torch.ones(N_pad, L, dtype=torch.float32, device='cuda'), assumed_align=16)
    return mA, mBg, mBu, mC, sca, scg, scu


def warm_gpu(call, seconds=1.5):
    """Ramp the GPU clocks so timing reflects steady-state, not boost/idle."""
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(20):
            call()
        stop.record(); torch.cuda.synchronize()
        if start.elapsed_time(stop) >= seconds * 1000:
            break


def main():
    ap = argparse.ArgumentParser(description="Benchmark the fused dual INT8 GEMM + SiLU.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, default=16384, help="tokens")
    ap.add_argument("--N", type=int, default=2048, help="inter/hidden dim")
    ap.add_argument("--K", type=int, default=512, help="model dim")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "dual_gemm_silu", "dual_gemm_i8_silu")
    exp = common.import_impl(arch, "dual_gemm_silu", "export_dual_gemm_i8_silu")
    d = types.SimpleNamespace(M=args.M, N=args.N, K=args.K, L=1)
    mA, mBg, mBu, mC, mSA, mSBg, mSBu = build(kern, d, exp.BM, exp.BN, 64)

    gemm = kern.TensorOpDualGemmI8Silu(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        exp.ATOM_LAYOUT, True, exp.BM, bn=exp.BN, num_stages=exp.NUM_STAGES)
    print(f"Compiling dual INT8 GEMM + SiLU  tile={exp.BM}x{exp.BN}x64  atom={exp.ATOM_LAYOUT} ...")
    compiled = cute.compile(gemm, mA, mBg, mBu, mC, mSA, mSBg, mSBu)

    print("Warming GPU clocks ...")
    warm_gpu(lambda: compiled(mA, mBg, mBu, mC, mSA, mSBg, mSBu))

    avg_us = cute.testing.benchmark(
        compiled,
        kernel_arguments=cute.testing.JitArguments(mA, mBg, mBu, mC, mSA, mSBg, mSBu),
        warmup_iterations=args.warmup, iterations=args.iters)

    M, N, K = d.M, d.N, d.K
    total_ops = 2 * (2 * M * N * K)                         # two GEMMs (gate + up)
    total_bytes = M * K + 2 * N * K + M * N * 2             # int8 A + int8 Bg,Bu + fp16 out
    s = avg_us * 1e-6
    tops = total_ops / s / 1e12
    gbs = total_bytes / s / 1e9
    print(f"\n=== dual INT8 GEMM+SiLU  M={M} N={N} K={K} ===")
    print(f"  time:      {avg_us:.2f} us")
    print(f"  compute:   {tops:.1f} TOPS  ({tops / PEAK_TOPS * 100:.1f}% of {PEAK_TOPS} peak)")
    print(f"  bandwidth: {gbs:.1f} GB/s  ({gbs / PEAK_BW_GBS * 100:.1f}% of {PEAK_BW_GBS} peak)")


if __name__ == "__main__":
    main()
