"""Test the fused fp8 GEMM+rotary kernel (RDNA4) against a torch reference.

Two impls (both go through the SAME FlyDSL kernel, differently invoked):
  --impl abi  (default): the real fluke C ABI (fluke_fp8_select + fluke_qkv_rotary_fp8_gpu)
                         via ctypes on libfluke_fp8.so — exercises the embedded-HSACO load +
                         arch dispatch path that slorado uses.
  --impl jit           : the FlyDSL launcher compiled in-process (compile_fp8_gemm_rotary).

    <venv>/bin/python fly/test_rotary.py
    <venv>/bin/python fly/test_rotary.py --impl jit

Computes C[M,N] = rotary(A@B) with A fp8[M,K], B fp8[K,N] (per-token / per-channel scales),
rotary applied to the Q cols [0, nhead*head_dim) and K cols [nhead*head_dim, 2*nhead*head_dim);
V cols pass through. Diffs vs a torch fp32 reference. Requires a ROCm torch + RDNA4 GPU.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fluke root
sys.path.insert(0, os.path.join(ROOT, "fly"))
import common

# TxModel dims (must match the exported artifact + fused_hip.cpp baked dims).
D_MODEL, DIM_FF, NHEAD, HEAD_DIM, ROTARY_DIM = 512, 2048, 8, 64, 64
ROTARY_HALF = ROTARY_DIM // 2
N = 3 * NHEAD * HEAD_DIM     # 1536
K = D_MODEL                  # 512


def make_inputs(M, seqlen, seed=42):
    torch.manual_seed(seed)
    from rdna_fp8_gemm_rotary import preshuffle_b_fp8, fp8_quantize_per_token, fp8_quantize_per_channel
    A = torch.randn(M, K, device="cuda") * 0.1
    B = torch.randn(K, N, device="cuda") * 0.1
    A_fp8, sa = fp8_quantize_per_token(A)
    B_fp8, sb = fp8_quantize_per_channel(B)
    B_shuf = preshuffle_b_fp8(B_fp8).contiguous()
    theta = torch.arange(seqlen, device="cuda").float().unsqueeze(1) * 0.01
    rot = torch.arange(ROTARY_HALF, device="cuda").float().unsqueeze(0)
    sin_buf = torch.sin(theta + rot).contiguous()
    cos_buf = torch.cos(theta + rot).contiguous()
    return A_fp8, sa, B_fp8, sb, B_shuf, sin_buf, cos_buf


def reference(A_fp8, sa, B_fp8, sb, sin_buf, cos_buf, M, seqlen):
    C = ((A_fp8.float() * sa[:, None]) @ (B_fp8.float() * sb[None, :])).clone()
    rows = torch.arange(M, device="cuda")
    cr, sr = cos_buf[rows % seqlen], sin_buf[rows % seqlen]
    for chunk_start in (0, NHEAD * HEAD_DIM):          # Q, K (not V)
        for h in range(NHEAD):
            h0 = chunk_start + h * HEAD_DIM
            x0 = C[:, h0:h0 + ROTARY_HALF].clone()
            x1 = C[:, h0 + ROTARY_HALF:h0 + ROTARY_DIM].clone()
            C[:, h0:h0 + ROTARY_HALF] = x0 * cr - x1 * sr
            C[:, h0 + ROTARY_HALF:h0 + ROTARY_DIM] = x0 * sr + x1 * cr
    return C


def run_abi(A_fp8, sa, B_shuf, sb, sin_buf, cos_buf, M, seqlen):
    lib = common.load_fp8_lib()
    dims = common.fluke_dims_t(D_MODEL, DIM_FF, NHEAD, HEAD_DIM, seqlen)
    b = lib.fluke_fp8_select(0, dims)
    if not b:
        raise SystemExit("fluke_fp8_select returned NULL (unsupported arch or dim mismatch)")
    out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    rc = lib.fluke_qkv_rotary_fp8_gpu(b, out.data_ptr(), A_fp8.data_ptr(), B_shuf.data_ptr(),
                                      sa.data_ptr(), sb.data_ptr(),
                                      sin_buf.data_ptr(), cos_buf.data_ptr(), M, seqlen)
    torch.cuda.synchronize()
    if rc != 0:
        raise SystemExit(f"fluke_qkv_rotary_fp8_gpu returned {rc}")
    return out


def run_jit(A_fp8, sa, B_shuf, sb, sin_buf, cos_buf, M, seqlen):
    from rdna_fp8_gemm_rotary import compile_fp8_gemm_rotary
    launcher = compile_fp8_gemm_rotary(M=M, N=N, K=K, nhead=NHEAD, head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM)
    out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    launcher(out, A_fp8, B_shuf, sa, sb, sin_buf, cos_buf, torch.cuda.current_stream(), M, seqlen)
    torch.cuda.synchronize()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["abi", "jit"], default="abi")
    ap.add_argument("--arch", default=None, help="fly arch subdir (default: auto-detect)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    arch = args.arch or common.detect_arch()
    common.import_impl(arch, "rotary", "rdna_fp8_gemm_rotary")  # puts fly/<arch>/rotary on sys.path

    M, seqlen = args.batch * args.seqlen, args.seqlen
    A_fp8, sa, B_fp8, sb, B_shuf, sin_buf, cos_buf = make_inputs(M, seqlen)
    ref = reference(A_fp8, sa, B_fp8, sb, sin_buf, cos_buf, M, seqlen)

    runner = {"abi": run_abi, "jit": run_jit}[args.impl]
    out = runner(A_fp8, sa, B_shuf, sb, sin_buf, cos_buf, M, seqlen)

    err = common.report(f"rotary/{args.impl}", out, ref)
    tol = 5e-2  # fp8 e4m3 quant noise
    print("PASS" if err < tol else f"FAIL (max_abs={err:.4f} > tol={tol})")
    sys.exit(0 if err < tol else 1)
