"""Benchmark the basic quantized INT8 GEMM (cute/<arch>/gemm_i8_quant.py).

Compiles TensorOpGemmI8 for the given shape, warms the GPU clocks, then times it
with CUDA events and reports TOPS + effective bandwidth. Arch is auto-detected
(override with --arch).

    <venv>/bin/python cute/bench_gemm.py
    <venv>/bin/python cute/bench_gemm.py --M 8192 --N 8192 --K 8192

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
ATOM_LAYOUT, NUM_STAGES, USE_K32 = (2, 2, 1), 3, True
BM, BN, BK = 128, 128, 64


def build(kern, d):
    M, K, N, L = d.M, d.K, d.N, d.L
    M_pad = ((M + BM - 1) // BM) * BM
    N_pad = ((N + BN - 1) // BN) * BN
    K_pad = ((K + BK - 1) // BK) * BK
    mA, a_t = kern.create_and_permute_tensor(L, M_pad, K_pad, False, cutlass.Int8)
    mB, b_t = kern.create_and_permute_tensor(L, N_pad, K_pad, False, cutlass.Int8)
    mC, c_t = kern.create_and_permute_tensor(L, M_pad, N_pad, False, cutlass.Float16)
    for tt, r, c in [(a_t, M, K), (b_t, N, K)]:
        if tt.shape[0] > r: tt[r:, :, :] = 0
        if tt.shape[1] > c: tt[:, c:, :] = 0
    sa = torch.ones(M_pad, L, dtype=torch.float32, device='cuda')
    sb = torch.ones(N_pad, L, dtype=torch.float32, device='cuda')
    return mA, mB, mC, from_dlpack(sa, assumed_align=16), from_dlpack(sb, assumed_align=16)


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
    ap = argparse.ArgumentParser(description="Benchmark the INT8 GEMM.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "", "gemm_i8_quant")
    d = types.SimpleNamespace(M=args.M, N=args.N, K=args.K, L=1)
    mA, mB, mC, mSA, mSB = build(kern, d)

    gemm = kern.TensorOpGemmI8(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        ATOM_LAYOUT, USE_K32, BM, bn=BN, num_stages=NUM_STAGES)
    print("Compiling INT8 GEMM ...")
    compiled = cute.compile(gemm, mA, mB, mC, mSA, mSB)

    print("Warming GPU clocks ...")
    warm_gpu(lambda: compiled(mA, mB, mC, mSA, mSB))

    avg_us = cute.testing.benchmark(
        compiled,
        kernel_arguments=cute.testing.JitArguments(mA, mB, mC, mSA, mSB),
        warmup_iterations=args.warmup, iterations=args.iters)

    M, N, K = d.M, d.N, d.K
    total_ops = 2 * M * N * K
    total_bytes = M * K + N * K + M * N * 2                 # int8 A + int8 B + fp16 C
    s = avg_us * 1e-6
    tops = total_ops / s / 1e12
    gbs = total_bytes / s / 1e9
    print(f"\n=== INT8 GEMM  M={M} N={N} K={K} ===")
    print(f"  time:      {avg_us:.2f} us")
    print(f"  compute:   {tops:.1f} TOPS  ({tops / PEAK_TOPS * 100:.1f}% of {PEAK_TOPS} peak)")
    print(f"  bandwidth: {gbs:.1f} GB/s  ({gbs / PEAK_BW_GBS * 100:.1f}% of {PEAK_BW_GBS} peak)")
    print(f"  intensity: {total_ops / total_bytes:.1f} op/byte")


if __name__ == "__main__":
    main()
