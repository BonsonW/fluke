"""Fused-interleaved INT8-gate factored-LSTM step test + bench.

Tests factored_lstm_gate_fused_i8.TensorOpFactoredLstmGateFusedI8: ONE interleaved
gate GEMM (bN=128, single accumulator -- vs the 4-accumulator bN=32 gate_i8 kernel)
+ smem-staged fused LSTM epilogue.  Design proven in CUDA/CUTLASS (fused_cutlass.cu,
gate 30us at N=2048); this is the CuTe DSL port.

    <venv>/bin/python cute/test_factored_lstm_gate_fused_i8.py            # correctness
    <venv>/bin/python cute/test_factored_lstm_gate_fused_i8.py --bench    # + sweep
"""
import argparse
import os
import sys

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from test_factored_lstm import make_inputs, torch_reference
from test_factored_lstm_gate_i8 import int8_recompute_reference

OUT_SCALE = 1.0 / 127.0
DEF_BM, DEF_BN, DEF_BK, DEF_STAGES = 64, 64, 32, 3    # sweep winner @ N=2048 (48.3us)
DEF_ATOM = (2, 2, 1)
CONFIG = dict(H=1024, K_hh=128, R=128)


def _ceil(a, b):
    return ((a + b - 1) // b) * b


def build_tensors(mod, inp, H, K_hh, R, B, bm, bn, bk, device="cuda"):
    Kc = K_hh + R
    M_pad, K_pad = _ceil(B, bm), _ceil(Kc, bk)

    def i8_tensor(rows, cols, fill_i8):
        t = torch.zeros(rows, cols, 1, dtype=torch.int8, device=device)
        t[:fill_i8.shape[0], :fill_i8.shape[1], 0] = fill_i8
        ct = (from_dlpack(t, assumed_align=16)
              .mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=16))
        return ct, t

    def col_f32(rows, fill):
        t = torch.zeros(rows, 1, dtype=torch.float32, device=device)
        t[:fill.shape[0], 0] = fill
        return from_dlpack(t.contiguous(), assumed_align=16), t

    A = torch.cat([inp["hh_down"], inp["x_down"]], dim=1)
    A_i8, scale_a = common.quantize_tensor(A, dim=-1)
    mA, _ = i8_tensor(M_pad, K_pad, A_i8)
    A_deq = A_i8.float() * scale_a.unsqueeze(1)

    W_i8, mScaleW, mBias, W_deq = [], [], [], []
    for g in range(4):
        Wg = torch.cat([inp["up_hh"][g * H:(g + 1) * H], inp["up_ih"][g * H:(g + 1) * H]], dim=1)
        Wg_i8, scale_w = common.quantize_tensor(Wg, dim=-1)
        W_i8.append(Wg_i8)
        csw, _ = col_f32(H, scale_w); mScaleW.append(csw)
        bias = (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]
        cb, _ = col_f32(H, bias); mBias.append(cb)
        W_deq.append(Wg_i8.float() * scale_w.unsqueeze(1))

    Bint = mod.interleave_gate_weights(W_i8)                # [4H, Kc] int8 interleaved
    mBint, _ = i8_tensor(4 * H, K_pad, Bint)

    msa, _ = col_f32(M_pad, scale_a)
    c_t = torch.zeros(M_pad, H, 1, dtype=torch.float32, device=device)
    c_t[:B, :, 0] = inp["c"]
    # fully static layout: the fused kernel's cell-tile cp.async prefetch needs a
    # provable 16B tile alignment (dynamic leading stride defeats the proof).
    mC = from_dlpack(c_t, assumed_align=16)
    h_t = torch.zeros(M_pad, H, 1, dtype=torch.int8, device=device)
    mH = from_dlpack(h_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    kern_args = (mA, mBint, msa,
                 mScaleW[0], mScaleW[1], mScaleW[2], mScaleW[3],
                 mBias[0], mBias[1], mBias[2], mBias[3], mC, mH)
    return kern_args, c_t, h_t, (A_deq, W_deq)


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


def bench(compiled, kern_args, iters=300):
    call = lambda: compiled(*kern_args)
    common.warm_gpu(call, seconds=2.0)
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
    mod = common.import_impl(arch, "factored_lstm", "factored_lstm_gate_fused_i8")
    H, K_hh, R = CONFIG["H"], CONFIG["K_hh"], CONFIG["R"]
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)})")
    print(f"Shape: H={H} Kc={K_hh+R}  ONE interleaved GEMM N=4H={4*H}  default tile "
          f"{DEF_BM}x{DEF_BN}x{DEF_BK} st={DEF_STAGES} atom={DEF_ATOM}\n")

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
        print("\n=== per-step us @ N=2048 (vs gate_i8 4-acc ~44us, CUDA-fused 33us) ===")
        B = 2048
        inp = make_inputs(B, H, K_hh, R, "cuda")
        best = None
        for (bm, bn, bk, stages, atom) in [
            (128, 128, 64, 3, (2, 2, 1)),
            (128, 128, 64, 4, (2, 2, 1)),
            (128, 128, 32, 3, (2, 2, 1)),
            (128, 256, 64, 3, (2, 4, 1)),
            (256, 128, 64, 3, (4, 2, 1)),
            (128, 128, 64, 3, (2, 4, 1)),
            (64, 128, 64, 4, (2, 2, 1)),
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
                  f"(4-acc gate_i8 ~44us => {44.0/best[0]:.2f}x)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
