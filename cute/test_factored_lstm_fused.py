"""Fused factored-LSTM step test: single-launch down-proj + step kernel vs torch.

The fused kernel (cute/<arch>/factored_lstm/factored_lstm_fused_i8.py) computes in ONE
launch what lstm_model runs as two kernels per timestep:

    hh_down = (h_i8 @ W_dn_i8^T) * comb_scale      (int8 GEMM, split-K producers)
    gates   = [hh_down | x_down] @ W_g^T + bias    (f16 GEMM x4, two-pass consumer)
    c' = f*c + i*g;  h' = o*tanh(c') -> int8 (1/127)

The kernel is launched several times back-to-back (with c restored between runs) to
verify the flag/workspace self-cleaning: a missing reset shows up as a hang or as
garbage on the second run, not the first.

    <venv>/bin/python cute/test_factored_lstm_fused.py
    <venv>/bin/python cute/test_factored_lstm_fused.py --B 512

Exit 0 on PASS, 1 otherwise.
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

ABS_TOL = 0.08           # matches test_factored_lstm.py
OUT_SCALE = 1.0 / 127.0


def make_inputs(B, H, K_hh, R, device, seed=42):
    torch.manual_seed(seed)
    FH = 4 * H
    h_prev = torch.rand(B, H, device=device) * 2 - 1            # in [-1, 1]
    h_i8 = (h_prev * 127).round().clamp(-127, 127).to(torch.int8)
    w_dn = torch.randn(K_hh, H, device=device) * 0.02
    w_dn_i8, w_dn_scale = common.quantize_tensor(w_dn, dim=-1)  # per-out-channel
    comb_scale = (w_dn_scale * OUT_SCALE).float()               # fold 1/127 (host-side)
    return dict(
        h_i8=h_i8,
        w_dn_i8=w_dn_i8,
        comb_scale=comb_scale,
        x_down=torch.randn(B, R, device=device) * 0.3,
        up_hh=torch.randn(FH, K_hh, device=device) * 0.1,
        up_ih=torch.randn(FH, R, device=device) * 0.1,
        bhh=torch.randn(FH, device=device) * 0.05,
        bih=torch.randn(FH, device=device) * 0.05,
        c=torch.randn(B, H, device=device) * 0.1,
    )


def torch_reference(inp, H):
    hh_down = (inp["h_i8"].float() @ inp["w_dn_i8"].float().t()) * inp["comb_scale"][None, :]
    gates = (hh_down @ inp["up_hh"].float().t()
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
    """Padded merged descriptors for the fused kernel. All row tensors get M_pad rows."""
    Kc = K_hh + R
    M_pad = ((B + bm - 1) // bm) * bm
    grid_m = M_pad // bm

    def dyn_f16(rows, cols, fill=None):
        t = torch.zeros(rows, cols, 1, dtype=torch.float16, device=device)
        if fill is not None:
            t[:fill.shape[0], :fill.shape[1], 0] = fill.to(torch.float16)
        ct = (from_dlpack(t, assumed_align=16)
              .mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8))
        return ct, t

    def dyn_i8(rows, cols, fill=None):
        t = torch.zeros(rows, cols, 1, dtype=torch.int8, device=device)
        if fill is not None:
            t[:fill.shape[0], :fill.shape[1], 0] = fill
        ct = (from_dlpack(t, assumed_align=16)
              .mark_layout_dynamic(leading_dim=1)
              .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=16))
        return ct, t

    mHp, _ = dyn_i8(M_pad, H, inp["h_i8"])
    mWdn, _ = dyn_i8(K_hh, H, inp["w_dn_i8"])
    sc = torch.zeros(K_hh, 1, dtype=torch.float32, device=device)
    sc[:, 0] = inp["comb_scale"]
    mSc = from_dlpack(sc.contiguous(), assumed_align=16)
    mX, _ = dyn_f16(M_pad, R, inp["x_down"])

    mB, bias_dl = [], []
    for g in range(4):
        Wg = torch.cat([inp["up_hh"][g * H:(g + 1) * H], inp["up_ih"][g * H:(g + 1) * H]], dim=1)
        mB.append(dyn_f16(H, Kc, Wg)[0])
        bt = torch.zeros(H, 1, dtype=torch.float32, device=device)
        bt[:, 0] = (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]
        bias_dl.append(from_dlpack(bt.contiguous(), assumed_align=16))

    c_t = torch.zeros(M_pad, H, 1, dtype=torch.float32, device=device)
    c_t[:B, :, 0] = inp["c"]
    mC = from_dlpack(c_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    h_t = torch.zeros(M_pad, H, 1, dtype=torch.int8, device=device)
    mH = from_dlpack(h_t, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # mHH feeds the pass-2 cp.async pipeline: needs the same divisibility marking as
    # the other f16 operands or the 128-bit copy alignment can't be inferred.
    mHH, hh_t = dyn_f16(M_pad, K_hh)
    flags_t = torch.zeros(grid_m * 4, dtype=torch.int32, device=device)      # MUST be zero
    mFlags = from_dlpack(flags_t, assumed_align=16)

    args = (mHp, mWdn, mSc, mX, mB[0], mB[1], mB[2], mB[3],
            bias_dl[0], bias_dl[1], bias_dl[2], bias_dl[3], mC, mH, mHH, mFlags)
    return args, c_t, h_t, flags_t, hh_t


def main():
    ap = argparse.ArgumentParser(description="Fused factored-LSTM step test.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3, help="back-to-back launches (flag reuse)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_fused_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, K_hh, R = exp.CONFIG["H"], exp.CONFIG["K_hh"], exp.CONFIG["R"]
    B = args.B
    bm, bn, bk = exp.LSTM_BM, exp.LSTM_BN, exp.LSTM_BK
    print(f"Problem: B={B}, H={H}, K_hh={K_hh}, R={R}  (fused single-launch step)\n")

    inp = make_inputs(B, H, K_hh, R, "cuda")
    h_ref, c_ref = torch_reference(inp, H)

    tensors, c_t, h_t, flags_t, hh_t = build_tensors(inp, H, K_hh, R, B, bm, bn, bk)
    lstm = kern.TensorOpFactoredLstmFusedI8(
        cutlass.Float16, cutlass.Int8, cutlass.Float32, exp.LSTM_ATOM,
        H=H, K_hh=K_hh, R=R, bm=bm, bn=bn, bk=bk, num_stages=exp.LSTM_STAGES)
    print("Compiling fused kernel ...")
    compiled = cute.compile(lstm, *tensors)

    c0 = c_t.clone()
    ok = True
    for run in range(args.runs):
        c_t.copy_(c0)          # restore cell state (kernel updates in place)
        h_t.zero_()
        compiled(*tensors)
        torch.cuda.synchronize()
        h_dsl = h_t[:B, :, 0].float() * OUT_SCALE
        c_dsl = c_t[:B, :, 0]
        print(f"--- run {run + 1}/{args.runs} ---")
        worst = max(common.report("h vs torch", h_dsl, h_ref),
                    common.report("c vs torch", c_dsl, c_ref))
        if worst >= ABS_TOL:
            ok = False
        # self-cleaning invariant: flags must be back to zero after every launch
        fl_max = flags_t.abs().max().item()
        if fl_max != 0:
            print(f"  CLEANUP FAIL: |flags|max={fl_max}")
            ok = False

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
