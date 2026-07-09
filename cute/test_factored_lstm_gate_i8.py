"""INT8-gate factored-LSTM step test + bench.

Forks test_factored_lstm.py. The gate GEMM is INT8 here (TensorOpFactoredLstmGateI8):
  A = [hh_down|x_down] [B,Kc] quantized per-row -> int8 + scale_a[B]
  W_g = [up_hh_g|up_ih_g] [H,Kc] quantized per-channel -> int8 + scale_w[g][H]
  acc_g = A_int8 @ W_g_int8^T (int32); raw_g = acc_g*scale_a[m]*scale_w[g][n] + bias_g
  i,f,o = clamp(0.2*raw+0.5,0,1); g = clamp(raw,-1,1); c'=f*c+i*g; h=o*tanh(c') -> int8 @ 1/127

Two references:
  (1) int8-recompute  -- dequant int8 A/B back to f32, gate in f32. The kernel must
      match this TIGHTLY (proves kernel correctness; ~1 int8 quantum).
  (2) fp16 torch_reference -- the ORIGINAL full-precision gate. kernel-vs-fp16 is the
      QUANTIZATION cost of int8-gating (the accuracy-gate signal).

    <venv>/bin/python cute/test_factored_lstm_gate_i8.py            # correctness (small N + N=2048)
    <venv>/bin/python cute/test_factored_lstm_gate_i8.py --bench    # + per-step us + bn/bk/stages sweep
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
from test_factored_lstm import make_inputs, torch_reference

OUT_SCALE = 1.0 / 127.0

# Default int8-gate tile (mirrors the fp16 step default; int8 MMA K=32 so bK multiple of 32).
DEF_BM, DEF_BN, DEF_BK, DEF_STAGES = 64, 32, 64, 3
DEF_ATOM = (2, 2, 1)
CONFIG = dict(H=1024, K_hh=128, R=128)


def _ceil(a, b):
    return ((a + b - 1) // b) * b


def build_tensors_i8(inp, H, K_hh, R, B, bm, bn, bk, device="cuda"):
    """Quantize + pad the merged descriptors the int8-gate kernel consumes.
    Returns kernel tensors plus the dequantized-int8 fp32 operands for the recompute ref."""
    Kc = K_hh + R
    M_pad, K_pad, N_pad = _ceil(B, bm), _ceil(Kc, bk), _ceil(H, bn)

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

    A = torch.cat([inp["hh_down"], inp["x_down"]], dim=1)             # [B, Kc]
    A_i8, scale_a = common.quantize_tensor(A, dim=-1)                 # int8 [B,Kc], f32 [B]
    mA, _ = i8_tensor(M_pad, K_pad, A_i8)
    A_deq = A_i8.float() * scale_a.unsqueeze(1)                       # dequantized A (fp32)

    mB, mScaleW, mBias, W_deq, keep = [], [], [], [], []
    for g in range(4):
        Wg = torch.cat([inp["up_hh"][g * H:(g + 1) * H], inp["up_ih"][g * H:(g + 1) * H]], dim=1)
        Wg_i8, scale_w = common.quantize_tensor(Wg, dim=-1)          # int8 [H,Kc], f32 [H]
        ct, t = i8_tensor(N_pad, K_pad, Wg_i8); mB.append(ct); keep.append(t)
        csw, _ = col_f32(N_pad, scale_w); mScaleW.append(csw)
        bias = (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]
        cb, _ = col_f32(N_pad, bias); mBias.append(cb)
        W_deq.append(Wg_i8.float() * scale_w.unsqueeze(1))

    msa, _ = col_f32(M_pad, scale_a)

    c_t = torch.zeros(M_pad, H, 1, dtype=torch.float32, device=device)
    c_t[:B, :, 0] = inp["c"]
    mC = from_dlpack(c_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    h_t = torch.zeros(M_pad, H, 1, dtype=torch.int8, device=device)
    mH = from_dlpack(h_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    kern_args = (mA, mB[0], mB[1], mB[2], mB[3],
                 msa, mScaleW[0], mScaleW[1], mScaleW[2], mScaleW[3],
                 mBias[0], mBias[1], mBias[2], mBias[3], mC, mH)
    return kern_args, c_t, h_t, (A_deq, W_deq)


def int8_recompute_reference(inp, deq, H):
    """Exact int8 recompute: dequantized A/W matmul in fp32, then LSTM epilogue.
    This is what the kernel must match tightly."""
    A_deq, W_deq = deq
    gates = []
    for g in range(4):
        bias = (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]
        gates.append(A_deq @ W_deq[g].t() + bias)
    i, f, g_, o = gates
    i = (i * 0.2 + 0.5).clamp(0, 1)
    f = (f * 0.2 + 0.5).clamp(0, 1)
    o = (o * 0.2 + 0.5).clamp(0, 1)
    g_ = g_.clamp(-1, 1)
    c_new = f * inp["c"] + i * g_
    return o * c_new.tanh(), c_new


def make_kernel(exp_mod, bm, bn, bk, stages, atom):
    return exp_mod.TensorOpFactoredLstmGateI8(
        cutlass.Int8, cutlass.Int8, cutlass.Int32, atom,
        bm=bm, bn=bn, bk=bk, num_stages=stages)


def run_case(mod, inp, H, K_hh, R, B, bm, bn, bk, stages, atom):
    kern_args, c_t, h_t, deq = build_tensors_i8(inp, H, K_hh, R, B, bm, bn, bk)
    lstm = make_kernel(mod, bm, bn, bk, stages, atom)
    compiled = cute.compile(lstm, *kern_args)
    compiled(*kern_args)
    torch.cuda.synchronize()
    h = h_t[:B, :, 0].float() * OUT_SCALE
    c = c_t[:B, :, 0].clone()
    return h, c, deq, compiled, kern_args


def bench(compiled, kern_args, iters=200):
    call = lambda: compiled(*kern_args)
    common.warm_gpu(call, seconds=2.0)
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        call()
    stop.record(); torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters * 1000.0   # us/step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=None)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    mod = common.import_impl(arch, "factored_lstm", "factored_lstm_gate_i8")
    H, K_hh, R = CONFIG["H"], CONFIG["K_hh"], CONFIG["R"]
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)})")
    print(f"Shape: H={H} K_hh={K_hh} R={R}  Kc={K_hh+R}  tile default "
          f"{DEF_BM}x{DEF_BN}x{DEF_BK} stages={DEF_STAGES} atom={DEF_ATOM}\n")

    # ---- correctness: small N, then N=2048 ----
    worst_k = 0.0
    acc_signals = {}
    for B in (64, 256, 2048):
        inp = make_inputs(B, H, K_hh, R, "cuda")
        h, c, deq, compiled, kargs = run_case(mod, inp, H, K_hh, R, B,
                                              DEF_BM, DEF_BN, DEF_BK, DEF_STAGES, DEF_ATOM)
        h_i8ref, c_i8ref = int8_recompute_reference(inp, deq, H)
        h_fp16, c_fp16 = torch_reference(inp, H)
        print(f"=== B={B}: kernel vs int8-recompute (correctness, want ~{OUT_SCALE:.5f}) ===")
        e_h = common.report("h vs int8recompute", h, h_i8ref)
        e_c = common.report("c vs int8recompute", c, c_i8ref)
        print(f"--- B={B}: kernel vs fp16 (QUANTIZATION cost of int8-gating) ---")
        a_h = common.report("h vs fp16", h, h_fp16)
        a_c = common.report("c vs fp16", c, c_fp16)
        acc_signals[B] = (a_h, a_c)
        worst_k = max(worst_k, e_h, e_c)
        print()

    # kernel-vs-int8recompute correctness gate: ~1 int8 quantum on h (1/127) + fp slack
    ok = worst_k < (OUT_SCALE + 1e-3)
    print(f"Correctness (kernel vs int8-recompute): worst max_abs={worst_k:.6f} "
          f"(gate {OUT_SCALE + 1e-3:.5f})  -> {'PASS' if ok else 'FAIL'}")
    print("\n*** ACCURACY SIGNAL (int8-gate h vs fp16 h, h in [-1,1]) ***")
    for B, (a_h, a_c) in acc_signals.items():
        print(f"    B={B:5d}  h max_abs={a_h:.6f}   c max_abs={a_c:.6f}")

    if args.bench:
        print("\n=== per-step us @ N=2048 (int8 gate vs ~50us fp16) — sweep ===")
        B = 2048
        inp = make_inputs(B, H, K_hh, R, "cuda")
        configs = []
        for bn in (32, 64):
            for bk in (32, 64):
                for stages in (3, 4):
                    configs.append((DEF_BM, bn, bk, stages, DEF_ATOM))
        best = None
        for (bm, bn, bk, stages, atom) in configs:
            try:
                kern_args, c_t, h_t, deq = build_tensors_i8(inp, H, K_hh, R, B, bm, bn, bk)
                lstm = make_kernel(mod, bm, bn, bk, stages, atom)
                compiled = cute.compile(lstm, *kern_args)
                us = bench(compiled, kern_args)
                tag = f"bm={bm} bn={bn} bk={bk} st={stages} atom={atom}"
                print(f"  {tag:44s}  {us:7.2f} us")
                if best is None or us < best[0]:
                    best = (us, tag)
            except Exception as e:
                print(f"  bm={bm} bn={bn} bk={bk} st={stages}: FAILED {type(e).__name__}: {str(e)[:80]}")
        if best:
            print(f"\nWinning config: {best[1]}  ->  {best[0]:.2f} us/step "
                  f"(fp16 ~50us => {50.0/best[0]:.2f}x)")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
