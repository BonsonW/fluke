"""Benchmark the fused INT8 GEMM + rotary kernel (cute/<arch>/rotary/).

Compiles TensorOpGemmI8Rotary for the given QKV-projection shape, warms the GPU
clocks, then times it with CUDA events and reports TOPS + effective bandwidth. The
rotary epilogue is latency-hidden behind the GEMM, so numbers should track the plain
INT8 GEMM at the same shape. Arch is auto-detected (override with --arch).

    <venv>/bin/python cute/bench_rotary.py
    <venv>/bin/python cute/bench_rotary.py --M 8192 --K 2048 --nhead 16

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
# Matches the shipped/exported rotary config (export_gemm_i8_rotary.py): atom (2,2,1)
# with bN=128. Paired with the coalesced smem-staged store, this narrow atom beats the
# old (2,4,1)+bN=256 by ~15% at M>=16k (242 vs 210 TOPS): half the threads/block spawn
# ~2x more, smaller CTAs that balance better across the 108 SMs and raise SM-pipe
# utilization. bN=128 keeps the rotary companion column (sincos_width away) in-register
# under this atom (cols_per_mma_n = atom_N*16 = 32 divides sincos_width=32). Do NOT use
# (2,2,1)+bN=256: it over-allocates accumulator registers and spills. See bench numbers
# in the coalesced-store profiling round.
ATOM_LAYOUT, NUM_STAGES, USE_K32 = (2, 2, 1), 3, True
BM, BN, BK = 128, 128, 64   # BN must be a multiple of head_dim


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
    sin = torch.zeros(d.seqlen, d.sincos_width, dtype=torch.float32, device='cuda')
    cos = torch.ones(d.seqlen, d.sincos_width, dtype=torch.float32, device='cuda')
    return (mA, mB, mC, from_dlpack(sa, assumed_align=16), from_dlpack(sb, assumed_align=16),
            from_dlpack(sin, assumed_align=16), from_dlpack(cos, assumed_align=16))


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
    ap = argparse.ArgumentParser(description="Benchmark the fused INT8 GEMM + rotary.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, default=4096, help="tokens (batch*seq)")
    ap.add_argument("--K", type=int, default=512, help="input dim")
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--rotary-dim", type=int, default=64)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "rotary", "gemm_i8_rotary")
    d = types.SimpleNamespace(
        M=args.M, K=args.K, nhead=args.nhead, head_dim=args.head_dim,
        rotary_dim=args.rotary_dim, seqlen=args.seqlen, L=1)
    d.N = 3 * d.nhead * d.head_dim
    d.sincos_width = d.rotary_dim // 2
    mA, mB, mC, mSA, mSB, mSin, mCos = build(kern, d)

    gemm = kern.TensorOpGemmI8Rotary(
        cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
        ATOM_LAYOUT, USE_K32, BM, bn=BN, num_stages=NUM_STAGES,
        nhead=d.nhead, head_dim=d.head_dim, rotary_dim=d.rotary_dim, seqlen=d.seqlen)
    seqlen_arg = cutlass.Int32(d.seqlen)
    print("Compiling fused INT8 GEMM + rotary ...")
    compiled = cute.compile(gemm, mA, mB, mC, mSA, mSB, mSin, mCos, seqlen_arg)

    print("Warming GPU clocks ...")
    warm_gpu(lambda: compiled(mA, mB, mC, mSA, mSB, mSin, mCos, seqlen_arg))

    avg_us = cute.testing.benchmark(
        compiled,
        kernel_arguments=cute.testing.JitArguments(mA, mB, mC, mSA, mSB, mSin, mCos, seqlen_arg),
        warmup_iterations=args.warmup, iterations=args.iters)

    M, N, K = d.M, d.N, d.K
    total_ops = 2 * M * N * K                               # GEMM dominates; rotary is cheap
    total_bytes = M * K + N * K + M * N * 2                 # int8 A + int8 B + fp16 C
    s = avg_us * 1e-6
    tops = total_ops / s / 1e12
    gbs = total_bytes / s / 1e9
    print(f"\n=== fused INT8 GEMM+rotary  M={M} N={N} K={K}  (nhead={d.nhead} head_dim={d.head_dim}) ===")
    print(f"  time:      {avg_us:.2f} us")
    print(f"  compute:   {tops:.1f} TOPS  ({tops / PEAK_TOPS * 100:.1f}% of {PEAK_TOPS} peak)")
    print(f"  bandwidth: {gbs:.1f} GB/s  ({gbs / PEAK_BW_GBS * 100:.1f}% of {PEAK_BW_GBS} peak)")


if __name__ == "__main__":
    main()
