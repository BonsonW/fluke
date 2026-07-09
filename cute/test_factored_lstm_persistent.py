"""Persistent full-recurrence factored-LSTM test.

Runs the ENTIRE T-step fLSTM recurrence in ONE kernel launch (cute/<arch>/
factored_lstm/factored_lstm_persistent_i8.py) and compares the written hh_all ring
and the final cell against a torch reference that loops the same recurrence.

    <venv>/bin/python cute/test_factored_lstm_persistent.py --B 64 --T 16
    <venv>/bin/python cute/test_factored_lstm_persistent.py --B 256 --T 256
    <venv>/bin/python cute/test_factored_lstm_persistent.py --B 2048 --T 2048 --bench
    <venv>/bin/python cute/test_factored_lstm_persistent.py --B 256 --T 256 --reverse

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

ABS_TOL = 0.10
OUT_SCALE = 1.0 / 127.0

H, K_hh, R = 1024, 128, 128     # hac v6
Kc = K_hh + R


def make_inputs(N, T, device, seed=42):
    torch.manual_seed(seed)
    FH = 4 * H
    w_dn = torch.randn(K_hh, H, device=device) * 0.02
    w_dn_i8, w_dn_scale = common.quantize_tensor(w_dn, dim=-1)      # per-out-channel
    comb_scale = (w_dn_scale * OUT_SCALE).float()
    return dict(
        x_down=torch.randn(T, N, R, device=device) * 0.3,
        w_dn_i8=w_dn_i8,
        comb_scale=comb_scale,
        up_hh=torch.randn(FH, K_hh, device=device) * 0.1,
        up_ih=torch.randn(FH, R, device=device) * 0.1,
        bhh=torch.randn(FH, device=device) * 0.05,
        bih=torch.randn(FH, device=device) * 0.05,
    )


def _gate_w(inp, g):
    return torch.cat([inp["up_hh"][g * H:(g + 1) * H], inp["up_ih"][g * H:(g + 1) * H]], dim=1)  # [H,Kc]


def _gate_bias(inp, g):
    return (inp["bhh"] + inp["bih"])[g * H:(g + 1) * H]  # [H]


def torch_reference(inp, N, T, reverse, n_cmp=None):
    """Loop the recurrence in the same order as the kernel; return (hh_all[T+1,n,H] int8, cell[n,H]).

    Batch rows are independent, so we only reference the first n_cmp rows (keeps the
    huge N=T=2048 ring out of memory)."""
    dev = inp["x_down"].device
    n = N if n_cmp is None else min(n_cmp, N)
    Wg = [_gate_w(inp, g).float() for g in range(4)]
    Bg = [_gate_bias(inp, g).float() for g in range(4)]
    w_dn_i8 = inp["w_dn_i8"].float()
    comb = inp["comb_scale"][None, :]

    hh_all = torch.zeros(T + 1, n, H, dtype=torch.int8, device=dev)
    h_i8 = torch.zeros(n, H, dtype=torch.int8, device=dev)
    c = torch.zeros(n, H, dtype=torch.float32, device=dev)

    order = range(T - 1, -1, -1) if reverse else range(T)
    for tt in order:
        wslot = tt if reverse else tt + 1
        hh_down = (h_i8.float() @ w_dn_i8.t()) * comb                # [n,K_hh]
        A = torch.cat([hh_down, inp["x_down"][tt, :n].float()], dim=1)   # [n,Kc]
        gi = A @ Wg[0].t() + Bg[0]
        gf = A @ Wg[1].t() + Bg[1]
        gg = A @ Wg[2].t() + Bg[2]
        go = A @ Wg[3].t() + Bg[3]
        i = (gi * 0.2 + 0.5).clamp(0, 1)
        f = (gf * 0.2 + 0.5).clamp(0, 1)
        o = (go * 0.2 + 0.5).clamp(0, 1)
        g = gg.clamp(-1, 1)
        c = f * c + i * g
        h = o * c.tanh()
        h_i8 = (h * 127).round().clamp(-127, 127).to(torch.int8)
        hh_all[wslot] = h_i8
    return hh_all, c


def build_tensors(inp, N, H_, K_hh_, R_, T, reverse, GX=8, bM_group=128, i8gate=False, device="cuda"):
    """Pack the cute tensors the kernel consumes. Returns (tensors_tuple, hh_all_torch, cell_torch).

    i8gate: microbench — pass int8 activations (x_down, hh_down ring) and int8 gate weights
    (const scalar A-scale, no per-row scale tensors / no in-kernel amax)."""
    def dl(t):
        return from_dlpack(t.contiguous(), assumed_align=16)

    mW_dn = dl(inp["w_dn_i8"])                                      # [K_hh,H]
    comb = inp["comb_scale"].reshape(K_hh_, 1).contiguous()
    mComb = dl(comb)
    mBias = [dl(_gate_bias(inp, g).reshape(H_, 1).contiguous()) for g in range(4)]

    if i8gate:
        adt = torch.int8
        mX = dl((inp["x_down"] * 127).round().clamp(-127, 127).to(torch.int8))   # [T,N,R] int8
        mW = [dl(common.quantize_tensor(_gate_w(inp, g), dim=-1)[0]) for g in range(4)]  # int8
        scratch_hh_t = torch.zeros(0, dtype=torch.int8, device=device)            # sized below
    else:
        mX = dl(inp["x_down"].to(torch.float16))                   # [T,N,R] f16
        mW = [dl(_gate_w(inp, g).to(torch.float16)) for g in range(4)]

    hh_all_t = torch.zeros(T + 1, N, H_, dtype=torch.int8, device=device)
    cell_t = torch.zeros(N, H_, dtype=torch.float32, device=device)
    mHH_all = dl(hh_all_t)
    mCell = dl(cell_t)

    GY = (N + bM_group - 1) // bM_group
    scratch_t = torch.zeros(2 * GY * GX * bM_group * K_hh_, dtype=torch.int32, device=device)
    flags_t = torch.zeros(GY, dtype=torch.int32, device=device)     # MUST be zero
    hh_dt = torch.int8 if i8gate else torch.float16
    scratch_hh_t = torch.zeros(GY * bM_group * K_hh_, dtype=hh_dt, device=device)
    mScratch = dl(scratch_t)
    mFlags = dl(flags_t)
    mScratchHH = dl(scratch_hh_t)
    mScaleW = dl(torch.ones(4 * H_, 1, dtype=torch.float32, device=device))   # unused

    tensors = (mX, mW_dn, mComb, mW[0], mW[1], mW[2], mW[3],
               mBias[0], mBias[1], mBias[2], mBias[3], mHH_all, mCell, mScratch, mFlags,
               mScratchHH, mScaleW)
    return tensors, hh_all_t, cell_t


def main():
    ap = argparse.ArgumentParser(description="Persistent full-recurrence factored-LSTM test.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--atom_n", type=int, default=8)
    ap.add_argument("--gx", type=int, default=16)          # HX=H/GX; larger GX => resident B fits regs
    ap.add_argument("--bm_group", type=int, default=384)
    ap.add_argument("--bm_reg", type=int, default=16)
    ap.add_argument("--bN", type=int, default=128)
    ap.add_argument("--bk_g", type=int, default=16)
    ap.add_argument("--bk_dp", type=int, default=32)
    ap.add_argument("--stages_g", type=int, default=3)
    ap.add_argument("--stages_dp", type=int, default=3)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--int8_gate", action="store_true")
    ap.add_argument("--variant", default="full", choices=["full", "core", "core_ar", "core_epi", "core_i8"])
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)}, cc {torch.cuda.get_device_capability()})")

    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_persistent_i8")
    # N padded to a multiple of bm_group so grid.y groups tile the batch exactly
    N = ((args.B + args.bm_group - 1) // args.bm_group) * args.bm_group
    T = args.T
    print(f"Problem: B={args.B} (N={N}), H={H}, K_hh={K_hh}, R={R}, T={T}, reverse={args.reverse}, "
          f"int8_gate={args.int8_gate}, GX={args.gx} bM_group={args.bm_group} bm_reg={args.bm_reg} "
          f"atom_n={args.atom_n}\n")

    inp = make_inputs(N, T, "cuda")
    tensors, hh_all_t, cell_t = build_tensors(inp, N, H, K_hh, R, T, args.reverse,
                                              GX=args.gx, bM_group=args.bm_group,
                                              i8gate=(args.variant == "core_i8"))

    lstm = kern.TensorOpFactoredLstmPersistentI8(
        cutlass.Float16, cutlass.Int8, cutlass.Float32, (1, args.atom_n, 1),
        H=H, K_hh=K_hh, R=R, T=T, bm=kern.PERSIST_BM,
        GX=args.gx, bM_group=args.bm_group, bm_reg=args.bm_reg, bN=args.bN,
        bK_g=args.bk_g, bK_dp=args.bk_dp, stages_g=args.stages_g, stages_dp=args.stages_dp,
        reverse=args.reverse, int8_gate=args.int8_gate, variant=args.variant)
    print(f"Compiling persistent kernel (variant={args.variant}) ...")
    compiled = cute.compile(lstm, *tensors)
    print("Compiled. Running ...")

    compiled(*tensors)
    torch.cuda.synchronize()

    if args.variant == "full":
        n_cmp = min(args.B, 64)   # rows are independent; comparing a subset keeps the ring small
        hh_ref, c_ref = torch_reference(inp, N, T, args.reverse, n_cmp=n_cmp)
        hh_dsl = hh_all_t[:, :n_cmp].float() * OUT_SCALE
        hh_ref_f = hh_ref.float() * OUT_SCALE
        worst = max(common.report("hh_all vs torch", hh_dsl, hh_ref_f),
                    common.report("cell   vs torch", cell_t[:n_cmp], c_ref))
        ok = worst < ABS_TOL
        print("\nPASS" if ok else "\nFAIL")
    else:
        ok = True
        print(f"\n(variant={args.variant}: correctness skipped — decomposition timing only)")

    if args.bench:
        common.warm_gpu(lambda: compiled(*tensors))
        start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
        iters = 20
        start.record()
        for _ in range(iters):
            compiled(*tensors)
        stop.record(); torch.cuda.synchronize()
        total_us = start.elapsed_time(stop) * 1000.0 / iters
        print(f"\n=== BENCH  N={N} T={T} ===")
        print(f"  per-launch: {total_us:.2f} us   per-step: {total_us / T:.3f} us/step")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
