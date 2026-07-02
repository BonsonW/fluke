"""Torch-builtin baseline for the INT8 down-projection, benched vs the custom CuTe kernel.

The factored-LSTM down-projection is
    hh_down[B, R] (f16) = (h_int8[B, H] @ dn_int8[R, H]^T) * scale_a[B] * scale_b[R]

`torch._scaled_mm` is the obvious built-in, but it is fp8-only and needs compute
capability >= 8.9 (Ada/Hopper) -- it raises on Ampere (A100, sm80). The int8 equivalent
that DOES run on Ampere is `torch._int_mm` (int8 x int8 -> int32 via cuBLASLt IMMA); we
apply the per-token / per-channel dequant scales by hand afterward.

This tests that path against a torch reference and benchmarks it HEAD-TO-HEAD against the
custom fused cute/ampere/factored_lstm down_proj kernel (same CUDA-event timer, same warm
clocks), so you can see whether the builtin is "just as fast". The custom kernel fuses the
dequant into its epilogue (one launch); the _int_mm path is a separate int32 GEMM + an
elementwise scale/cast, so expect it to be somewhat slower.

    <venv>/bin/python cute/int_mm_down_proj.py            # test + bench (B=256, 512)
    <venv>/bin/python cute/int_mm_down_proj.py --B 512
    <venv>/bin/python cute/int_mm_down_proj.py --no-cute  # torch builtin only

Exit 0 on PASS, 1 otherwise. Needs a CUDA torch (and cutlass for the --cute comparison).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cute/ (common)

H = 1024          # contraction (hidden dim)
R = 128           # output rank
OUT_SCALE = 1.0 / 127.0   # fixed hidden-output scale (h in [-1,1]); recurrent path
PEAK_TOPS = 624.0         # A100 INT8 tensor-core peak


def quantize_per_channel(t, dim=-1):
    qmax = 127
    fr = t.abs().amax(dim=dim).clamp_min(1e-8)
    qs = qmax / fr
    ti = (t * qs.unsqueeze(dim)).round().clamp(-qmax, qmax).to(torch.int8)
    return ti, qs.to(torch.float32).reciprocal()


def down_proj_int_mm(h_int8, dn_int8_t, scale_a, scale_b):
    """hh_down[B,R] f16 = (h_int8 @ dn_int8^T) * scale_a[B] * scale_b[R], via torch._int_mm.
    dn_int8_t is dn^T [H,R] contiguous (K-major), prepared once by the caller."""
    acc = torch._int_mm(h_int8, dn_int8_t)                     # int32 [B, R]
    return (acc.float() * scale_a[:, None] * scale_b[None, :]).to(torch.float16)


def make_inputs(B, device="cuda", seed=0):
    torch.manual_seed(seed)
    h_prev = (torch.rand(B, H, device=device) * 2 - 1) * 0.5
    h_int8 = (h_prev / OUT_SCALE).round().clamp(-127, 127).to(torch.int8)
    scale_a = torch.full((B,), OUT_SCALE, dtype=torch.float32, device=device)
    dn = torch.randn(R, H, device=device) * 0.1
    dn_int8, scale_b = quantize_per_channel(dn, dim=-1)
    return h_int8, scale_a, dn, dn_int8, scale_b


def test(B, device="cuda"):
    h_int8, scale_a, dn, dn_int8, scale_b = make_inputs(B, device)
    out = down_proj_int_mm(h_int8, dn_int8.t().contiguous(), scale_a, scale_b)
    A_dq = h_int8.float() * scale_a[:, None]
    B_dq = dn_int8.float() * scale_b[:, None]
    ref_dq = (A_dq @ B_dq.T).to(torch.float16)
    ref_fp = (A_dq @ dn.T).to(torch.float16)
    e_dq = (out.float() - ref_dq.float()).abs().max().item()
    e_fp = (out.float() - ref_fp.float()).abs().max().item()
    ok = e_dq < 1e-2
    print(f"  B={B:5d}  M={B} N={R} K={H}   max|Δ| vs dequant ref: {e_dq:.6f}  "
          f"vs fp(dn): {e_fp:.6f}   {'PASS' if ok else 'FAIL'}")
    return ok


def time_call(call, warmup_s=1.5, iters=300):
    start = torch.cuda.Event(enable_timing=True); stop = torch.cuda.Event(enable_timing=True)
    # warm clocks
    start.record()
    while True:
        for _ in range(50):
            call()
        stop.record(); torch.cuda.synchronize()
        if start.elapsed_time(stop) >= warmup_s * 1000:
            break
    # measure
    start.record()
    for _ in range(iters):
        call()
    stop.record(); torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1e3 / iters   # us/call


def build_cute_down_proj(B, device="cuda"):
    """Compile the custom fused CuTe INT8 down_proj (TensorOpGemmI8) at the deployed config."""
    import common
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack
    arch = common.detect_arch()
    kern = common.import_impl(arch, "factored_lstm", "gemm_i8_quant")
    exp = common.import_impl(arch, "factored_lstm", "export_factored_lstm_i8")
    bm, bN, bK = exp.DP_BM, 128, 64
    M_pad = ((B + bm - 1) // bm) * bm
    N_pad = ((R + bN - 1) // bN) * bN
    K_pad = ((H + bK - 1) // bK) * bK
    mA, _ = kern.create_and_permute_tensor(1, M_pad, K_pad, False, cutlass.Int8)
    mB, _ = kern.create_and_permute_tensor(1, N_pad, K_pad, False, cutlass.Int8)
    mC, _ = kern.create_and_permute_tensor(1, M_pad, N_pad, False, cutlass.Float16)
    sa = from_dlpack(torch.full((M_pad, 1), OUT_SCALE, dtype=torch.float32, device=device), assumed_align=16)
    sb = from_dlpack(torch.ones(N_pad, 1, dtype=torch.float32, device=device), assumed_align=16)
    gemm = kern.TensorOpGemmI8(cutlass.Int8, cutlass.Int8, cutlass.Float16, cutlass.Int32,
                               exp.DP_ATOM, True, bm, bn=min(128, R), num_stages=exp.DP_STAGES)
    args = (mA, mB, mC, sa, sb)
    compiled = cute.compile(gemm, *args)
    return lambda: compiled(*args)


def bench(B, use_cute, iters, device="cuda"):
    h_int8, scale_a, dn, dn_int8, scale_b = make_inputs(B, device)
    dn_t = dn_int8.t().contiguous()
    # full torch path (GEMM + unfused dequant) and the _int_mm GEMM alone, to show where
    # the time goes: the IMMA GEMM is competitive; the eager dequant is the overhead.
    us_full = time_call(lambda: down_proj_int_mm(h_int8, dn_t, scale_a, scale_b), iters=iters)
    us_mm = time_call(lambda: torch._int_mm(h_int8, dn_t), iters=iters)
    flops = 2 * B * H * R
    print(f"  B={B:5d}   torch _int_mm+dequant: {us_full:7.2f} us   "
          f"(_int_mm alone {us_mm:6.2f} us, dequant ~{us_full - us_mm:5.2f} us)")
    if use_cute:
        try:
            us_cute = time_call(build_cute_down_proj(B, device), iters=iters)
            print(f"           cute fused (1 launch): {us_cute:7.2f} us   "
                  f"({flops/(us_cute*1e-6)/1e12:.1f} TOPS)   "
                  f"full torch/cute = {us_full/us_cute:.2f}x   "
                  f"(_int_mm alone/cute = {us_mm/us_cute:.2f}x)")
        except Exception as e:
            print(f"           cute: unavailable ({type(e).__name__}: {str(e)[:60]})")


def main():
    ap = argparse.ArgumentParser(description="torch._int_mm INT8 down-proj: test + bench vs custom CuTe kernel.")
    ap.add_argument("--B", type=int, default=None, help="batch (default: sweep 256, 512)")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--no-cute", action="store_true", help="skip the custom CuTe comparison")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available")
    cc = torch.cuda.get_device_capability()
    print(f"GPU {torch.cuda.get_device_name(0)}  cc {cc}")
    if cc < (8, 9):
        print("(torch._scaled_mm needs cc>=8.9; using int8 torch._int_mm, which runs on Ampere)")
    batches = [args.B] if args.B else [256, 512]

    print("\n=== correctness (torch._int_mm + dequant vs torch reference) ===")
    ok = all(test(B) for B in batches)

    print("\n=== benchmark (warm clocks, CUDA events; lower us is better) ===")
    for B in batches:
        bench(B, not args.no_cute, args.iters)

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
