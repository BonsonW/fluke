"""Factored-LSTM step test: run the CuTe fused kernel and check it.

Computes one LSTM timestep (K-merged): with A = [hh_down | x_down] and per-gate weight
W_g = [up_hh_g | up_ih_g],
    gates = A @ W_g^T + bias_g   (g in i,f,g,o)
    i,f,o = clamp(0.2*gate+0.5, 0, 1);  g = clamp(gate, -1, 1)
    c_new = f*c + i*g;  h = o*tanh(c_new) -> int8 (fixed scale 1/127)

  --impl jit   run the DSL kernel in-process via cute.compile           (default)
  --impl aot   load the AOT-exported .o (export_to_c) via load_module and run it
  --ref torch  naive torch reference (only reference; the LSTM step is DSL-only, no
               pure-CUDA C counterpart in src/)

The DSL kernel lives under cute/<arch>/factored_lstm/. This file owns the test: inputs,
reference, running the chosen implementation, comparison. Tiling comes from the export
module so jit/aot/export can't drift.

    <venv>/bin/python cute/test_factored_lstm.py
    <venv>/bin/python cute/test_factored_lstm.py --impl aot

Exit 0 on PASS, 1 otherwise. Needs a CUDA torch + CUDA toolkit.
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

ABS_TOL = 0.08          # matches the RDNA factored-LSTM reference tolerance
OUT_SCALE = 1.0 / 127.0  # fixed int8 hidden-output scale (h in [-1,1])


def make_inputs(B, H, K_hh, R, device, seed=42):
    torch.manual_seed(seed)
    FH = 4 * H
    return dict(
        hh_down=torch.randn(B, K_hh, device=device) * 0.3,
        x_down=torch.randn(B, R, device=device) * 0.3,
        up_hh=torch.randn(FH, K_hh, device=device) * 0.1,
        up_ih=torch.randn(FH, R, device=device) * 0.1,
        bhh=torch.randn(FH, device=device) * 0.05,
        bih=torch.randn(FH, device=device) * 0.05,
        c=torch.randn(B, H, device=device) * 0.1,
    )


def torch_reference(inp, H):
    """Naive torch: gates -> sighard/clamp -> cell update -> h = o*tanh(c_new)."""
    gates = (inp["hh_down"].float() @ inp["up_hh"].float().t()
             + inp["x_down"].float() @ inp["up_ih"].float().t()
             + inp["bhh"] + inp["bih"])
    i, f, g, o = gates.chunk(4, dim=1)
    i = (i * 0.2 + 0.5).clamp(0, 1)
    f = (f * 0.2 + 0.5).clamp(0, 1)
    o = (o * 0.2 + 0.5).clamp(0, 1)
    g = g.clamp(-1, 1)
    c_new = f * inp["c"] + i * g
    return o * c_new.tanh(), c_new


def build_tensors(inp, H, K_hh, R, B, bm, bn, bk, device="cuda"):
    """Build the padded merged descriptors the kernel consumes.
    A = [hh_down | x_down] [B, Kc];  W_g = [up_hh_g | up_ih_g] [H, Kc]; bias_g = bhh_g+bih_g."""
    Kc = K_hh + R
    f16 = cutlass.Float16
    M_pad = ((B + bm - 1) // bm) * bm
    K_pad = ((Kc + bk - 1) // bk) * bk
    N_pad = ((H + bn - 1) // bn) * bn

    def f16_tensor(rows, cols, fill):
        t = torch.zeros(rows, cols, 1, dtype=torch.float16, device=device)
        t[:fill.shape[0], :fill.shape[1], 0] = fill.to(torch.float16)
        ct = (from_dlpack(t, assumed_align=16)
              .mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8))
        return ct, t

    A = torch.cat([inp["hh_down"], inp["x_down"]], dim=1)          # [B, Kc]
    mA, _ = f16_tensor(M_pad, K_pad, A)
    mB, bias_dl = [], []
    for g in range(4):
        Wg = torch.cat([inp["up_hh"][g * H:(g + 1) * H], inp["up_ih"][g * H:(g + 1) * H]], dim=1)
        mB.append(f16_tensor(N_pad, K_pad, Wg)[0])
        bt = torch.zeros(N_pad, 1, dtype=torch.float32, device=device)
        bt[:H, 0] = (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]
        bias_dl.append(from_dlpack(bt.contiguous(), assumed_align=16))

    c_t = torch.zeros(M_pad, H, 1, dtype=torch.float32, device=device)
    c_t[:B, :, 0] = inp["c"]
    mC = from_dlpack(c_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    h_t = torch.zeros(M_pad, H, 1, dtype=torch.int8, device=device)
    mH = from_dlpack(h_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    return (mA, mB, bias_dl, mC, mH, c_t, h_t)


def run_jit(kern, exp, inp, d):
    mA, mB, bias, mC, mH, c_t, h_t = build_tensors(
        inp, d.H, d.K_hh, d.R, d.B, exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK)
    lstm = kern.TensorOpFactoredLstmI8(
        cutlass.Float16, cutlass.Int8, cutlass.Float32, exp.LSTM_ATOM,
        bm=exp.LSTM_BM, bn=exp.LSTM_BN, bk=exp.LSTM_BK, num_stages=exp.LSTM_STAGES)
    args = (mA, mB[0], mB[1], mB[2], mB[3], bias[0], bias[1], bias[2], bias[3], mC, mH)
    compiled = cute.compile(lstm, *args)
    compiled(*args)
    torch.cuda.synchronize()
    return h_t[:d.B, :, 0].float() * OUT_SCALE, c_t[:d.B, :, 0]


def run_aot(kern, exp, inp, d):
    name = f"factored_lstm_i8_H{d.H}_Khh{d.K_hh}_R{d.R}"
    obj = os.path.join(exp.ARTIFACTS_DIR, f"{name}.o")
    if not os.path.isfile(obj):
        print(f"Artifact missing, exporting {name} ...")
        os.makedirs(exp.ARTIFACTS_DIR, exist_ok=True)
        exp._export_step(exp.CONFIG, exp.ARTIFACTS_DIR)
    mA, mB, bias, mC, mH, c_t, h_t = build_tensors(
        inp, d.H, d.K_hh, d.R, d.B, exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK)
    module = cute.runtime.load_module(obj)
    getattr(module, name)(mA, mB[0], mB[1], mB[2], mB[3], bias[0], bias[1], bias[2], bias[3], mC, mH)
    torch.cuda.synchronize()
    return h_t[:d.B, :, 0].float() * OUT_SCALE, c_t[:d.B, :, 0]


def main():
    ap = argparse.ArgumentParser(description="Factored-LSTM step test (pick impl).")
    ap.add_argument("--arch", default=None, help="arch under cute/ (default: auto-detect)")
    ap.add_argument("--impl", choices=["jit", "aot"], default="jit")
    ap.add_argument("--ref", choices=["torch"], default="torch")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  impl: {args.impl}  ref: {args.ref}  "
          f"(GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")

    d = types.SimpleNamespace(H=exp.CONFIG["H"], K_hh=exp.CONFIG["K_hh"], R=exp.CONFIG["R"])
    d.B = exp.CONFIG["B"] if args.impl == "aot" else 256   # AOT pinned to exported B; JIT flexible
    print(f"Problem: B={d.B}, H={d.H}, K_hh={d.K_hh}, R={d.R}  "
          f"(h = o*tanh(f*c + i*g), int8 out scale 1/127)\n")

    inp = make_inputs(d.B, d.H, d.K_hh, d.R, "cuda")
    print(f"Running {args.impl} implementation from cute/{arch}/factored_lstm/ ...")
    run = run_aot if args.impl == "aot" else run_jit
    h_dsl, c_dsl = run(kern, exp, inp, d)

    h_ref, c_ref = torch_reference(inp, d.H)
    print(f"\n=== {args.impl} kernel vs torch reference (max abs error) ===")
    worst = max(common.report("h vs torch", h_dsl, h_ref),
                common.report("c vs torch", c_dsl, c_ref))
    ok = worst < ABS_TOL
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
