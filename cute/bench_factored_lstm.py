"""Benchmark the INT8 factored-LSTM per-step pipeline (cute/<arch>/factored_lstm/).

Per timestep the layer runs two kernels: the recurrent int8 down-projection
(hh_down = h_int8 @ dn_int8^T, the plain INT8 GEMM) and the fused factored-LSTM step
(two f16 up-projections into 4 gate accumulators + gates + cell update + int8 out).
Compiles both with the exported tiling config, warms the GPU clocks, times each with
cute.testing.benchmark, and reports per-step latency + effective TOPS vs A100 peak.

    <venv>/bin/python cute/bench_factored_lstm.py
    <venv>/bin/python cute/bench_factored_lstm.py --B 512

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

PEAK_TOPS = 624.0   # A100 INT8 tensor-core peak


def warm_gpu(call, seconds=1.5):
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    while True:
        for _ in range(20):
            call()
        stop.record(); torch.cuda.synchronize()
        if start.elapsed_time(stop) >= seconds * 1000:
            break


def build_down_proj(kern, B, H, R, bm):
    bN, bK = 128, 64
    M_pad = ((B + bm - 1) // bm) * bm
    N_pad = ((R + bN - 1) // bN) * bN
    K_pad = ((H + bK - 1) // bK) * bK
    mA, _ = kern.create_and_permute_tensor(1, M_pad, K_pad, False, cutlass.Int8)
    mB, _ = kern.create_and_permute_tensor(1, N_pad, K_pad, False, cutlass.Int8)
    mC, _ = kern.create_and_permute_tensor(1, M_pad, N_pad, False, cutlass.Float16)
    sa = from_dlpack(torch.full((M_pad, 1), 1.0 / 127, dtype=torch.float32, device='cuda'), assumed_align=16)
    sb = from_dlpack(torch.ones(N_pad, 1, dtype=torch.float32, device='cuda'), assumed_align=16)
    return mA, mB, mC, sa, sb


def build_step(H, K_hh, R, B, bm, bn, bk):
    Kc = K_hh + R
    M_pad = ((B + bm - 1) // bm) * bm
    K_pad = ((Kc + bk - 1) // bk) * bk
    N_pad = ((H + bn - 1) // bn) * bn

    def f16t(rows, cols):
        t = torch.zeros(rows, cols, 1, dtype=torch.float16, device='cuda')
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8))

    mA = f16t(M_pad, K_pad)
    mB = [f16t(N_pad, K_pad) for _ in range(4)]
    bias = [from_dlpack(torch.zeros(N_pad, 1, dtype=torch.float32, device='cuda').contiguous(),
                        assumed_align=16) for _ in range(4)]
    mC = from_dlpack(torch.zeros(M_pad, H, 1, dtype=torch.float32, device='cuda'),
                     assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mH = from_dlpack(torch.zeros(M_pad, H, 1, dtype=torch.int8, device='cuda'),
                     assumed_align=16).mark_layout_dynamic(leading_dim=1)
    return (mA, mB[0], mB[1], mB[2], mB[3], bias[0], bias[1], bias[2], bias[3], mC, mH)


def main():
    ap = argparse.ArgumentParser(description="Benchmark the INT8 factored-LSTM per-step pipeline.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, default=256, help="batch (tokens per step)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    dp_kern = common.import_impl(arch, "factored_lstm", "gemm_i8_quant")
    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, K_hh, R = exp.CONFIG["H"], exp.CONFIG["K_hh"], exp.CONFIG["R"]
    B = args.B

    # int8 down-projection (recurrent hh_down)
    dp_args = build_down_proj(dp_kern, B, H, R, exp.DP_BM)
    dp = dp_kern.TensorOpGemmI8(cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
                                exp.DP_ATOM, True, exp.DP_BM, bn=min(128, R), num_stages=exp.DP_STAGES)
    dp_c = cute.compile(dp, *dp_args)

    # fused LSTM step
    step_args = build_step(H, K_hh, R, B, exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK)
    step = kern.TensorOpFactoredLstmI8(cutlass.Float16, cutlass.Int8, cutlass.Float32,
                                       exp.LSTM_ATOM, bm=exp.LSTM_BM, bn=exp.LSTM_BN,
                                       bk=exp.LSTM_BK, num_stages=exp.LSTM_STAGES)
    step_c = cute.compile(step, *step_args)

    print("Warming GPU clocks ...")
    warm_gpu(lambda: (dp_c(*dp_args), step_c(*step_args)))

    us_dp = cute.testing.benchmark(dp_c, kernel_arguments=cute.testing.JitArguments(*dp_args),
                                   warmup_iterations=args.warmup, iterations=args.iters)
    us_step = cute.testing.benchmark(step_c, kernel_arguments=cute.testing.JitArguments(*step_args),
                                     warmup_iterations=args.warmup, iterations=args.iters)
    us_total = us_dp + us_step
    FH = 4 * H
    flops = 2 * B * H * K_hh + 2 * B * (K_hh + R) * FH   # down-proj + merged up-proj
    s = us_total * 1e-6
    tops = flops / s / 1e12
    print(f"\n=== INT8 factored-LSTM per-step  B={B} H={H} K_hh={K_hh} R={R} ===")
    print(f"  down_proj (int8, hh_down): {us_dp:6.2f} us")
    print(f"  factored_lstm step        : {us_step:6.2f} us")
    print(f"  TOTAL (dp + step)         : {us_total:6.2f} us")
    print(f"  compute: {tops:.1f} TOPS  ({tops / PEAK_TOPS * 100:.1f}% of {PEAK_TOPS} peak)")


if __name__ == "__main__":
    main()
