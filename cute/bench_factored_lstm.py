"""Benchmark the fused factored-LSTM step kernel (cute/<arch>/factored_lstm/).

One timestep of the fused step: two f16 up-projections (K-merged, K = K_hh + R) into
4 gate accumulators + gates + cell update + int8 hidden output. The recurrent int8
down-projection is a separate kernel — bench it with cute/bench_down_proj.py.

Compiles with the exported tiling config, warms the GPU clocks, times with
cute.testing.benchmark, and reports latency + effective TFLOPS vs the A100 f16
tensor-core peak.

    <venv>/bin/python cute/bench_factored_lstm.py
    <venv>/bin/python cute/bench_factored_lstm.py --B 512

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

PEAK_TFLOPS = 312.0   # A100 F16 tensor-core peak (f16 up-projection GEMM)


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
    ap = argparse.ArgumentParser(description="Benchmark the fused factored-LSTM step kernel.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, default=256, help="batch (tokens per step)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, K_hh, R = exp.CONFIG["H"], exp.CONFIG["K_hh"], exp.CONFIG["R"]
    B = args.B

    step_args = build_step(H, K_hh, R, B, exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK)
    step = kern.TensorOpFactoredLstmI8(cutlass.Float16, cutlass.Int8, cutlass.Float32,
                                       exp.LSTM_ATOM, bm=exp.LSTM_BM, bn=exp.LSTM_BN,
                                       bk=exp.LSTM_BK, num_stages=exp.LSTM_STAGES)
    step_c = cute.compile(step, *step_args)

    print("Warming GPU clocks ...")
    common.warm_gpu(lambda: step_c(*step_args))

    us_step = cute.testing.benchmark(step_c, kernel_arguments=cute.testing.JitArguments(*step_args),
                                     warmup_iterations=args.warmup, iterations=args.iters)
    FH = 4 * H
    flops = 2 * B * (K_hh + R) * FH   # merged up-proj into 4 gate accumulators
    tflops = flops / (us_step * 1e-6) / 1e12
    print(f"\n=== fused factored-LSTM step  B={B} H={H} K_hh={K_hh} R={R} "
          f"(bm={exp.LSTM_BM} bn={exp.LSTM_BN} bk={exp.LSTM_BK} "
          f"stages={exp.LSTM_STAGES} atom={exp.LSTM_ATOM}) ===")
    print(f"  factored_lstm step: {us_step:6.2f} us")
    print(f"  compute: {tflops:.1f} TFLOPS f16  ({tflops / PEAK_TFLOPS * 100:.1f}% of {PEAK_TFLOPS} peak)")


if __name__ == "__main__":
    main()
