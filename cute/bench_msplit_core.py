"""Bench/ncu driver for the M-split CORE proxy (go/no-go microbench)."""
import argparse, os, sys
import torch
import cutlass
import cutlass.cute as cute

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import test_factored_lstm_persistent as H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2048)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--gx", type=int, default=16)
    ap.add_argument("--bm_group", type=int, default=512)
    ap.add_argument("--variant", default="core")
    ap.add_argument("--min_blocks", type=int, default=1)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    arch = common.detect_arch()
    kern = common.import_impl(arch, "factored_lstm", "factored_lstm_msplit_i8")
    N = ((args.B + args.bm_group - 1) // args.bm_group) * args.bm_group
    print(f"Arch {arch}  B={args.B} N={N} T={args.T} GX={args.gx} bM_group={args.bm_group} "
          f"HX={H.H//args.gx} nsub={args.bm_group//128} variant={args.variant}")
    inp = H.make_inputs(N, args.T, "cuda")
    tensors, hh_all_t, cell_t = H.build_tensors(inp, N, H.H, H.K_hh, H.R, args.T, False,
                                                GX=args.gx, bM_group=args.bm_group, i8gate=True)
    if args.variant in ("core_ar", "full", "ar1", "ar2", "ar3"):
        from cutlass.cute.runtime import from_dlpack
        GY = N // args.bm_group
        Kc = H.K_hh + H.R
        scratch = torch.zeros(2 * GY * args.gx * args.bm_group * H.K_hh, dtype=torch.int32, device="cuda")
        flags = torch.zeros(GY, dtype=torch.int32, device="cuda")
        scratchHH = torch.zeros(GY * args.bm_group * Kc, dtype=torch.int8, device="cuda")
        tensors = list(tensors)
        tensors[13] = from_dlpack(scratch, assumed_align=16)
        tensors[14] = from_dlpack(flags, assumed_align=16)
        tensors[15] = from_dlpack(scratchHH, assumed_align=16)
        tensors = tuple(tensors)
    lstm = kern.TensorOpFactoredLstmMsplitI8(
        cutlass.Int8, cutlass.Int8, cutlass.Int32,
        H=H.H, K_hh=H.K_hh, R=H.R, T=args.T,
        GX=args.gx, bM_group=args.bm_group, variant=args.variant, min_blocks=args.min_blocks)
    print("Compiling msplit CORE ...")
    compiled = cute.compile(lstm, *tensors)
    print("Compiled. Running once ...")
    compiled(*tensors)
    torch.cuda.synchronize()
    print("Ran OK.")
    if args.bench:
        common.warm_gpu(lambda: compiled(*tensors))
        start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
        iters = 20
        start.record()
        for _ in range(iters):
            compiled(*tensors)
        stop.record(); torch.cuda.synchronize()
        total_us = start.elapsed_time(stop) * 1000.0 / iters
        print(f"\n=== BENCH N={N} T={args.T} ===")
        print(f"  per-launch: {total_us:.2f} us   per-step: {total_us/args.T:.3f} us/step")


if __name__ == "__main__":
    main()
