"""Benchmark: fused single-launch factored-LSTM step vs the two-kernel pipeline.

Per timestep the unfused pipeline runs the int8 down-projection GEMM then the fused
gate/step kernel. The fused kernel (factored_lstm_fused_i8.py) does both in one
launch with split-K producers + overlapped consumers. This bench times both on warm
clocks at the same shapes.

    <venv>/bin/python cute/bench_factored_lstm_fused.py
    <venv>/bin/python cute/bench_factored_lstm_fused.py --B 512

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
import bench_factored_lstm as bstep
import bench_down_proj as bdp
import test_factored_lstm_fused as tff

PEAK_TFLOPS = 312.0


def main():
    ap = argparse.ArgumentParser(description="Fused vs two-kernel factored-LSTM step.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)})")

    dp_kern = common.import_impl(arch, "factored_lstm", "gemm_i8_quant")
    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_i8")
    fkern = common.import_impl(arch, "factored_lstm", "factored_lstm_fused_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, K_hh, R = exp.CONFIG["H"], exp.CONFIG["K_hh"], exp.CONFIG["R"]
    bm, bn, bk = exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK

    print(f"H={H} K_hh={K_hh} R={R}  step cfg bm={bm} bn={bn} bk={bk} "
          f"stages={exp.LSTM_STAGES} atom={exp.LSTM_ATOM}\n")
    print(f"{'B':>5} | {'down_proj':>9} | {'step':>7} | {'pair':>7} | {'fused':>7} | {'speedup':>7}")
    print("-" * 60)

    warmed = False
    for B in args.B:
        # unfused pair
        dp_args = bdp.build_down_proj(dp_kern, B, H, R, exp.DP_BM)
        dp = dp_kern.TensorOpGemmI8(cutlass.Int8, cutlass.Int8, cutlass.Float16,
                                    cutlass.Int32, exp.DP_ATOM, True, exp.DP_BM,
                                    bn=min(128, R), num_stages=exp.DP_STAGES)
        dp_c = cute.compile(dp, *dp_args)
        step_args = bstep.build_step(H, K_hh, R, B, bm, bn, bk)
        step = kern.TensorOpFactoredLstmI8(cutlass.Float16, cutlass.Int8, cutlass.Float32,
                                           exp.LSTM_ATOM, bm=bm, bn=bn, bk=bk,
                                           num_stages=exp.LSTM_STAGES)
        step_c = cute.compile(step, *step_args)

        # fused
        inp = tff.make_inputs(B, H, K_hh, R, "cuda")
        f_args, *_ = tff.build_tensors(inp, H, K_hh, R, B, bm, bn, bk)
        fused = fkern.TensorOpFactoredLstmFusedI8(
            cutlass.Float16, cutlass.Int8, cutlass.Float32, exp.LSTM_ATOM,
            H=H, K_hh=K_hh, R=R, bm=bm, bn=bn, bk=bk, num_stages=exp.LSTM_STAGES)
        fused_c = cute.compile(fused, *f_args)

        if not warmed:
            print("(warming GPU clocks ...)")
            common.warm_gpu(lambda: (dp_c(*dp_args), step_c(*step_args), fused_c(*f_args)))
            warmed = True

        us_dp = cute.testing.benchmark(dp_c, kernel_arguments=cute.testing.JitArguments(*dp_args),
                                       warmup_iterations=args.warmup, iterations=args.iters)
        us_st = cute.testing.benchmark(step_c, kernel_arguments=cute.testing.JitArguments(*step_args),
                                       warmup_iterations=args.warmup, iterations=args.iters)
        us_fu = cute.testing.benchmark(fused_c, kernel_arguments=cute.testing.JitArguments(*f_args),
                                       warmup_iterations=args.warmup, iterations=args.iters)
        pair = us_dp + us_st
        print(f"{B:>5} | {us_dp:>6.2f} us | {us_st:>4.2f} us | {pair:>4.2f} us | "
              f"{us_fu:>4.2f} us | {pair / us_fu:>6.2f}x")


if __name__ == "__main__":
    main()
