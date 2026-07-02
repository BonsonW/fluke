"""Test the fused fp8/f16 factored-LSTM step (RDNA4) against a torch reference.

DSL-only (no pure-C ref, like cute/test_factored_lstm.py). Runs the FlyDSL kernel in-process
(--impl jit) and diffs the updated cell state (f32) and quantized int8 hidden output vs torch.

    <venv>/bin/python fly/test_factored_lstm.py

  gates[B,4H] = A @ W_gate^T + bias   (A = [hh_down|x_down] f16, W_gate = [up_hh|up_ih] f16)
  i,f,o = clamp(0.2*gate+0.5, 0, 1);  g = clamp(gate, -1, 1)
  c_new = f*c + i*g;   h = o*tanh(c_new)  ->  int8 @ 1/127

Requires a ROCm torch + RDNA4 GPU.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fly"))
import common

H, K_hh, R = 1024, 128, 128
Kc = K_hh + R


def reference(A, Bs, biases, cell0):
    gate = [(A.float() @ Bs[k].float().t()) + biases[k] for k in range(4)]
    i_a = (0.2 * gate[0] + 0.5).clamp(0, 1)
    f_a = (0.2 * gate[1] + 0.5).clamp(0, 1)
    o_a = (0.2 * gate[3] + 0.5).clamp(0, 1)
    g_a = gate[2].clamp(-1, 1)
    c_new = f_a * cell0 + i_a * g_a
    h = o_a * torch.tanh(c_new)
    h_i8 = torch.round(h * 127).clamp(-127, 127).to(torch.int8)
    return c_new, h_i8


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["jit"], default="jit")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, default=128)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    arch = args.arch or common.detect_arch()
    common.import_impl(arch, "factored_lstm", "rdna_fp8_factored_lstm")
    from rdna_fp8_factored_lstm import compile_fp8_factored_lstm

    M = args.M
    torch.manual_seed(0)
    A = (torch.randn(M, Kc, device="cuda") * 0.3).to(torch.float16)
    Bs = [(torch.randn(H, Kc, device="cuda") * 0.3).to(torch.float16) for _ in range(4)]
    biases = [torch.randn(H, device="cuda") * 0.1 for _ in range(4)]
    cell0 = torch.randn(M, H, device="cuda") * 0.5

    cell = cell0.clone()
    hout = torch.zeros(M, H, dtype=torch.int8, device="cuda")
    launch = compile_fp8_factored_lstm(H=H, K_hh=K_hh, R=R)
    launch(hout, cell, A, *Bs, *biases, M, torch.cuda.current_stream())
    torch.cuda.synchronize()

    c_ref, h_ref = reference(A, Bs, biases, cell0)
    cerr = common.report("lstm cell", cell, c_ref)
    hmax = (hout.float() - h_ref.float()).abs().max().item()
    hmis = ((hout.float() - h_ref.float()).abs() > 1).float().mean().item()
    print(f"  lstm h_int8       max_int_diff={hmax:.0f}  frac(|diff|>1)={hmis:.4f}")
    # cell must match closely (f16 GEMM); int8 h within +-1 level (f16 vs f32 rounding at ties)
    ok = cerr < 1e-2 and hmax <= 1 and hmis == 0.0
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
