"""Autotune the fused factored-LSTM step kernel (cute/<arch>/factored_lstm/).

Mirrors cute/autotune_dual.py in the old repo: config list -> validity filter ->
compile -> correctness check vs the torch reference -> cute.testing.benchmark ->
sorted report. Inputs/reference/tensor building are reused from test_factored_lstm.py
so the autotune can't drift from the test.

Two-phase sweep to keep compile time sane:
  1. full config list at the primary batch (--B, default 256)
  2. top --top configs re-evaluated at the other batches (--Bs, default 128 512 1024)
The shipped config should be best/near-best across the whole range (robust winner,
same method as the dual-GEMM bN=32 decision).

Config validity: bM % (atom_M*16) == 0; bN % (atom_N*8) == 0; and bN must cover one
full MMA-N permutation tile, bN >= atom_N*8*2 — violating that hangs compilation
(established on the dual-GEMM kernel). smem = stages*bK*2B*(bM + 4*bN) capped at 96KB.

Run:
    pyvenv/bin/python cute/autotune_factored_lstm.py
    pyvenv/bin/python cute/autotune_factored_lstm.py --B 512 --Bs 256 1024
"""
import argparse
import os
import sys

import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cute/ (common, test_*)
import common
import test_factored_lstm as tf

PEAK_TFLOPS = 312.0   # A100 F16 tensor-core peak
SMEM_CAP = 96 * 1024


# (bm, bn, bk, atom_mnk, stages)
DEFAULT_CONFIGS = [
    (64,  32, 32, (2, 2, 1), 3),   # shipped default
    (64,  32, 32, (2, 2, 1), 4),
    (64,  32, 64, (2, 2, 1), 3),
    (64,  32, 64, (2, 2, 1), 4),
    (32,  32, 32, (2, 2, 1), 3),   # more CTAs in M
    (32,  32, 32, (2, 2, 1), 4),
    (32,  32, 64, (2, 2, 1), 3),
    (32,  16, 32, (2, 1, 1), 3),   # more CTAs in N (atom_N=1 allows bn=16)
    (32,  16, 64, (2, 1, 1), 3),
    (64,  16, 32, (2, 1, 1), 3),
    (64,  16, 64, (2, 1, 1), 3),
    (64,  16, 32, (2, 1, 1), 4),
    (64,  16, 32, (4, 1, 1), 3),   # tall atom, 128 threads on 16-wide N
    (64,  32, 32, (4, 1, 1), 3),
    (64,  32, 64, (4, 1, 1), 3),
    (32,  32, 32, (1, 2, 1), 3),   # 2 warps
    (32,  16, 32, (1, 1, 1), 3),   # 1 warp, max CTAs
    (64,  64, 32, (2, 2, 1), 3),   # wider N tiles (fewer CTAs, more reuse)
    (64,  64, 32, (2, 4, 1), 3),
    (128, 32, 32, (2, 2, 1), 3),   # taller M tiles (better at large B)
    (128, 32, 64, (4, 2, 1), 3),
    (128, 64, 32, (2, 2, 1), 3),
]


def _valid_config(bm, bn, bk, atom_mnk, stages):
    atom_m, atom_n, atom_k = atom_mnk
    mma_m, mma_n = 16, 8
    if atom_k != 1 or stages < 3:
        return False, "atom_K/stages"
    if bm % (atom_m * mma_m) != 0:
        return False, "bM % atom_M*16"
    if bn % (atom_n * mma_n) != 0:
        return False, "bN % atom_N*8"
    if bn < atom_n * mma_n * 2:
        return False, "bN < atom_N*mmaN*2 (compile hang)"
    if bk % 16 != 0:
        return False, "bK % 16"
    # atom_M=4 with bn=16 compiles but computes garbage (err ~1.26 on the A100 sweep,
    # 2026-07; same shape passes with atom_M<=2). Excluded pending investigation.
    if atom_m >= 4 and bn < 32:
        return False, "atom_M>=4 with bN<32 miscomputes"
    smem = stages * bk * 2 * (bm + 4 * bn)
    if smem > SMEM_CAP:
        return False, f"smem {smem // 1024}KB > {SMEM_CAP // 1024}KB"
    return True, ""


def bench_config(kern, inp, h_ref, c_ref, B, H, K_hh, R, cfg, iters, warmup, check=True):
    """Compile one config, check correctness against the torch reference, benchmark.
    Returns (us, err) or raises."""
    bm, bn, bk, atom_mnk, stages = cfg
    mA, mB, bias, mC, mH, c_t, h_t = tf.build_tensors(inp, H, K_hh, R, B, bm, bn, bk)
    lstm = kern.TensorOpFactoredLstmI8(
        cutlass.Float16, cutlass.Int8, cutlass.Float32, atom_mnk,
        bm=bm, bn=bn, bk=bk, num_stages=stages)
    args = (mA, mB[0], mB[1], mB[2], mB[3], bias[0], bias[1], bias[2], bias[3], mC, mH)
    compiled = cute.compile(lstm, *args)

    err = float("nan")
    if check:
        # first run only: the kernel updates c in place, so the reference holds
        # exactly once — check before benchmarking iterates the recurrence.
        compiled(*args)
        torch.cuda.synchronize()
        h_err = (h_t[:B, :, 0].float() * tf.OUT_SCALE - h_ref).abs().max().item()
        c_err = (c_t[:B, :, 0] - c_ref).abs().max().item()
        err = max(h_err, c_err)
        if err >= tf.ABS_TOL:
            raise RuntimeError(f"correctness FAIL err={err:.4f}")

    us = cute.testing.benchmark(compiled, kernel_arguments=cute.testing.JitArguments(*args),
                                warmup_iterations=warmup, iterations=iters)
    return us, err, compiled, args


def sweep(kern, B, H, K_hh, R, configs, iters, warmup, warmed, check=True):
    inp = tf.make_inputs(B, H, K_hh, R, "cuda")
    h_ref, c_ref = tf.torch_reference(inp, H)
    flops = 2 * B * (K_hh + R) * 4 * H
    results = []
    for cfg in configs:
        bm, bn, bk, atom_mnk, stages = cfg
        ctas = ((B + bm - 1) // bm) * ((H + bn - 1) // bn)
        tag = (f"bm={bm:3d} bn={bn:3d} bk={bk:2d} atom={str(atom_mnk):9s} "
               f"stages={stages} ctas={ctas:4d}")
        ok, why = _valid_config(*cfg)
        if not ok:
            print(f"  {tag}  SKIP ({why})")
            continue
        try:
            us, err, compiled, args = bench_config(
                kern, inp, h_ref, c_ref, B, H, K_hh, R, cfg, iters, warmup, check)
            if not warmed[0]:
                common.warm_gpu(lambda: compiled(*args))
                warmed[0] = True
                us = cute.testing.benchmark(
                    compiled, kernel_arguments=cute.testing.JitArguments(*args),
                    warmup_iterations=warmup, iterations=iters)
            tflops = flops / (us * 1e-6) / 1e12
            print(f"  {tag}  {us:7.2f}us  {tflops:5.1f} TFLOPS "
                  f"({tflops / PEAK_TFLOPS * 100:4.1f}%)  err={err:.4f}")
            results.append((us, cfg))
        except Exception as e:
            print(f"  {tag}  FAILED: {str(e)[:90]}")
    results.sort(key=lambda r: r[0])
    return results


def main():
    ap = argparse.ArgumentParser(description="Autotune the fused factored-LSTM step kernel.")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--B", type=int, default=256, help="primary batch for the full sweep")
    ap.add_argument("--Bs", type=int, nargs="*", default=[128, 512, 1024],
                    help="extra batches for the top-config cross-check")
    ap.add_argument("--top", type=int, default=6, help="configs carried into phase 2")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    arch = args.arch or common.detect_arch()
    print(f"Arch: {arch}  (GPU {torch.cuda.get_device_name(0)})")

    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_i8")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    H, K_hh, R = exp.CONFIG["H"], exp.CONFIG["K_hh"], exp.CONFIG["R"]
    check = not args.no_check
    warmed = [False]   # warm on the first compiled config, then stay hot

    print(f"\n########## phase 1: full sweep  B={args.B} H={H} Kc={K_hh + R} ##########")
    res = sweep(kern, args.B, H, K_hh, R, DEFAULT_CONFIGS, args.iters, args.warmup,
                warmed, check)
    if not res:
        sys.exit("no config survived phase 1")
    top = [cfg for _, cfg in res[:args.top]]

    per_b = {args.B: {cfg: us for us, cfg in res}}
    for B in args.Bs:
        print(f"\n########## phase 2: top {len(top)}  B={B} ##########")
        res_b = sweep(kern, B, H, K_hh, R, top, args.iters, args.warmup, warmed, check)
        per_b[B] = {cfg: us for us, cfg in res_b}

    all_b = sorted(per_b.keys())
    print(f"\n=== cross-B summary (us; rank vs best per B) ===")
    header = "  ".join(f"B={b:<5d}" for b in all_b)
    print(f"{'config':45s}  {header}")
    for cfg in top:
        bm, bn, bk, atom_mnk, stages = cfg
        tag = f"bm={bm} bn={bn} bk={bk} atom={atom_mnk} stages={stages}"
        cells = []
        for b in all_b:
            us = per_b[b].get(cfg)
            if us is None:
                cells.append("   --  ")
            else:
                best = min(per_b[b].values())
                cells.append(f"{us:6.2f}{'*' if us <= best * 1.02 else ' '}")
        print(f"{tag:45s}  {'  '.join(cells)}")
    print("(* = within 2% of best at that B)")


if __name__ == "__main__":
    main()
