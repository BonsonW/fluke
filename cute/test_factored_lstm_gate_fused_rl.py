"""Register-local INT8-gate factored-LSTM step test + bench.

Tests factored_lstm_gate_fused_rl.TensorOpFactoredLstmGateFusedI8: the interleaved
single-accumulator gate GEMM with a REGISTER-LOCAL fused LSTM epilogue (atom_N=1,
contiguous 32-wide warp N-tile -> all 4 gates of a channel in one thread's fragment,
no int32 acc smem exchange).  This is the fastest DSL port of fused_cutlass.cu
(30.2us gate); winner bm=128 bn=64 bk=64 st=3 atom=(4,1,1) -> ~33.4us at N=2048.

    <venv>/bin/python cute/test_factored_lstm_gate_fused_rl.py            # correctness
    <venv>/bin/python cute/test_factored_lstm_gate_fused_rl.py --bench    # + sweep
"""
import argparse
import os
import sys

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from test_factored_lstm import make_inputs, torch_reference
from test_factored_lstm_gate_i8 import int8_recompute_reference
# reuse the sibling's tensor builder (identical arg layout / interleave)
from test_factored_lstm_gate_fused_i8 import build_tensors

OUT_SCALE = 1.0 / 127.0
DEF_BM, DEF_BN, DEF_BK, DEF_STAGES = 128, 64, 64, 3    # sweep winner @ N=2048 (33.4us)
DEF_ATOM = (4, 1, 1)
CONFIG = dict(H=1024, K_hh=128, R=128)


def run_case(mod, inp, H, K_hh, R, B, bm, bn, bk, stages, atom):
    kern_args, c_t, h_t, deq = build_tensors(mod, inp, H, K_hh, R, B, bm, bn, bk)
    lstm = mod.TensorOpFactoredLstmGateFusedI8(
        cutlass.Int8, cutlass.Int8, cutlass.Int32, atom, bm=bm, bn=bn, bk=bk, num_stages=stages)
    compiled = cute.compile(lstm, *kern_args)
    compiled(*kern_args)
    torch.cuda.synchronize()
    h = h_t[:B, :, 0].float() * OUT_SCALE
    c = c_t[:B, :, 0].clone()
    return h, c, deq, compiled, kern_args


def bench(compiled, kern_args, iters=1000):
    call = lambda: compiled(*kern_args)
    common.warm_gpu(call, seconds=2.5)
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    stop.record(); torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    mod = common.import_impl(arch, "factored_lstm", "factored_lstm_gate_fused_rl")
    H, K_hh, R = CONFIG["H"], CONFIG["K_hh"], CONFIG["R"]
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)})")
    print(f"Shape: H={H} Kc={K_hh+R}  interleaved GEMM N=4H={4*H}  register-local epilogue "
          f"default {DEF_BM}x{DEF_BN}x{DEF_BK} st={DEF_STAGES} atom={DEF_ATOM}\n")

    worst_k = 0.0
    acc_signals = {}
    for B in (128, 256, 2048):
        inp = make_inputs(B, H, K_hh, R, "cuda")
        h, c, deq, compiled, kargs = run_case(mod, inp, H, K_hh, R, B,
                                              DEF_BM, DEF_BN, DEF_BK, DEF_STAGES, DEF_ATOM)
        h_i8ref, c_i8ref = int8_recompute_reference(inp, deq, H)
        h_fp16, c_fp16 = torch_reference(inp, H)
        print(f"=== B={B}: kernel vs int8-recompute (correctness) ===")
        e_h = common.report("h vs int8recompute", h, h_i8ref)
        e_c = common.report("c vs int8recompute", c, c_i8ref)
        print(f"--- B={B}: kernel vs fp16 (quantization cost) ---")
        a_h = common.report("h vs fp16", h, h_fp16)
        a_c = common.report("c vs fp16", c, c_fp16)
        acc_signals[B] = (a_h, a_c)
        worst_k = max(worst_k, e_h, e_c)
        print()

    ok = worst_k < (OUT_SCALE + 1e-3)
    print(f"Correctness (kernel vs int8-recompute): worst max_abs={worst_k:.6f} "
          f"(gate {OUT_SCALE + 1e-3:.5f})  -> {'PASS' if ok else 'FAIL'}")
    print("\n*** ACCURACY SIGNAL (h vs fp16) ***")
    for B, (a_h, a_c) in acc_signals.items():
        print(f"    B={B:5d}  h max_abs={a_h:.6f}   c max_abs={a_c:.6f}")

    if args.bench:
        print("\n=== per-step us @ N=2048 (vs C++ gate 30.2us; fused_i8 smem-staged 43us) ===")
        B = 2048
        inp = make_inputs(B, H, K_hh, R, "cuda")
        best = None
        for (bm, bn, bk, stages, atom) in [
            (128, 64, 64, 3, (4, 1, 1)),
            (128, 64, 64, 4, (4, 1, 1)),
            (128, 128, 64, 3, (4, 1, 1)),
            (256, 64, 64, 3, (4, 1, 1)),
            (64, 64, 64, 3, (2, 1, 1)),
            (64, 128, 64, 3, (2, 1, 1)),
        ]:
            try:
                kern_args, c_t, h_t, deq = build_tensors(mod, inp, H, K_hh, R, B, bm, bn, bk)
                lstm = mod.TensorOpFactoredLstmGateFusedI8(
                    cutlass.Int8, cutlass.Int8, cutlass.Int32, atom,
                    bm=bm, bn=bn, bk=bk, num_stages=stages)
                compiled = cute.compile(lstm, *kern_args)
                us = bench(compiled, kern_args)
                tag = f"bm={bm} bn={bn} bk={bk} st={stages} atom={atom}"
                print(f"  {tag:44s}  {us:7.2f} us")
                if best is None or us < best[0]:
                    best = (us, tag)
            except Exception as e:
                print(f"  bm={bm} bn={bn} bk={bk} st={stages} atom={atom}: FAILED "
                      f"{type(e).__name__}: {str(e)[:100]}")
        if best:
            print(f"\nWinning config: {best[1]}  ->  {best[0]:.2f} us "
                  f"(C++ gate 30.2us => {best[0]/30.2:.2f}x)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
