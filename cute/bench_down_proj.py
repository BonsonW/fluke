"""Benchmark the INT8 down-projection GEMM (cute/<arch>/factored_lstm/ gemm_i8_quant).

hh_down[M,R] = (a_i8[M,H] * scale_a[M]) @ (w_i8[R,H] * scale_b[R])^T  -> f16

Two fLSTM call sites, same kernel at different M:
  - recurrent hh down-proj: M = B (tokens per step), called T times per layer
  - ih precompute for int8-input layers: M = T*B (one big GEMM)

Compiles with the exported tiling config, warms the GPU clocks, times with
cute.testing.benchmark, and reports latency + effective TOPS vs A100 INT8 peak.

    <venv>/bin/python cute/bench_down_proj.py
    <venv>/bin/python cute/bench_down_proj.py --M 256 16384

ALWAYS let it warm the GPU first — a cold clock badly skews the numbers.
"""
import argparse
import os
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cute/ (common)
import common

PEAK_TOPS = 624.0   # A100 INT8 tensor-core peak


def build_down_proj(kern, M, H, R, bm):
    bN, bK = 128, 64
    M_pad = ((M + bm - 1) // bm) * bm
    N_pad = ((R + bN - 1) // bN) * bN
    K_pad = ((H + bK - 1) // bK) * bK
    mA, _ = kern.create_and_permute_tensor(1, M_pad, K_pad, False, cutlass.Int8)
    mB, _ = kern.create_and_permute_tensor(1, N_pad, K_pad, False, cutlass.Int8)
    mC, _ = kern.create_and_permute_tensor(1, M_pad, N_pad, False, cutlass.Float16)
    sa = from_dlpack(torch.full((M_pad, 1), 1.0 / 127, dtype=torch.float32, device='cuda'), assumed_align=16)
    sb = from_dlpack(torch.ones(N_pad, 1, dtype=torch.float32, device='cuda'), assumed_align=16)
    return mA, mB, mC, sa, sb


def main():
    ap = argparse.ArgumentParser(description="Benchmark the INT8 down-projection GEMM.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, nargs="+", default=[256],
                    help="rows to bench (per-step B, and/or T*B for the ih precompute)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    dp_kern = common.import_impl(arch, "factored_lstm", "gemm_i8_quant")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, R = exp.CONFIG["H"], exp.CONFIG["R"]

    print(f"down-proj i8 GEMM  N=R={R} K=H={H}  "
          f"(bm={exp.DP_BM} atom={exp.DP_ATOM} stages={exp.DP_STAGES})\n")
    for M in args.M:
        dp_args = build_down_proj(dp_kern, M, H, R, exp.DP_BM)
        dp = dp_kern.TensorOpGemmI8(cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
                                    exp.DP_ATOM, True, exp.DP_BM, bn=min(128, R),
                                    num_stages=exp.DP_STAGES)
        dp_c = cute.compile(dp, *dp_args)

        print(f"[M={M}] warming GPU clocks ...")
        common.warm_gpu(lambda: dp_c(*dp_args))

        us_dp = cute.testing.benchmark(dp_c, kernel_arguments=cute.testing.JitArguments(*dp_args),
                                       warmup_iterations=args.warmup, iterations=args.iters)
        tops = (2 * M * R * H) / (us_dp * 1e-6) / 1e12
        print(f"  M={M:>7}: {us_dp:6.2f} us   {tops:5.1f} TOPS int8 "
              f"({tops / PEAK_TOPS * 100:.1f}% of {PEAK_TOPS} peak)\n")


if __name__ == "__main__":
    main()
