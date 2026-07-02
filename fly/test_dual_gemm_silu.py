"""Test the fused fp8 dual-GEMM + SiLU kernel (RDNA4) against a torch reference.

  out[M,N] = silu(A@B_gate) * (A@B_up),  A fp8[M,K], B_gate/B_up fp8[K,N] (per-token / per-channel).

  --impl abi (default): real fluke C ABI (fluke_fp8_select + fluke_gated_mlp_fp8_gpu) via ctypes.
  --impl jit          : the FlyDSL launcher compiled in-process.

    <venv>/bin/python fly/test_dual_gemm_silu.py [--impl jit]

Requires a ROCm torch + RDNA4 GPU.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fly"))
import common

D_MODEL, DIM_FF, NHEAD, HEAD_DIM = 512, 2048, 8, 64
N = DIM_FF     # 2048
K = D_MODEL    # 512


def make_inputs(M, seed=1):
    torch.manual_seed(seed)
    from rdna_fp8_preshuffle_gemm import preshuffle_b_fp8, fp8_quantize_per_token, fp8_quantize_per_channel
    A = torch.randn(M, K, device="cuda") * 0.1
    Bg = torch.randn(K, N, device="cuda") * 0.1
    Bu = torch.randn(K, N, device="cuda") * 0.1
    A_fp8, sa = fp8_quantize_per_token(A)
    Bg_fp8, sbg = fp8_quantize_per_channel(Bg)
    Bu_fp8, sbu = fp8_quantize_per_channel(Bu)
    return (A_fp8, sa, Bg_fp8, sbg, Bu_fp8, sbu,
            preshuffle_b_fp8(Bg_fp8).contiguous(), preshuffle_b_fp8(Bu_fp8).contiguous())


def reference(A_fp8, sa, Bg_fp8, sbg, Bu_fp8, sbu):
    gate = (A_fp8.float() * sa[:, None]) @ (Bg_fp8.float() * sbg[None, :])
    up = (A_fp8.float() * sa[:, None]) @ (Bu_fp8.float() * sbu[None, :])
    return gate * torch.sigmoid(gate) * up


def run_abi(A_fp8, sa, Bgs, sbg, Bus, sbu, M):
    lib = common.load_fp8_lib()
    b = lib.fluke_fp8_select(0, common.fluke_dims_t(D_MODEL, DIM_FF, NHEAD, HEAD_DIM, 2048))
    if not b:
        raise SystemExit("fluke_fp8_select returned NULL (unsupported arch or dim mismatch)")
    out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    rc = lib.fluke_gated_mlp_fp8_gpu(b, out.data_ptr(), A_fp8.data_ptr(), Bgs.data_ptr(), Bus.data_ptr(),
                                     sa.data_ptr(), sbg.data_ptr(), sbu.data_ptr(), M)
    torch.cuda.synchronize()
    if rc != 0:
        raise SystemExit(f"fluke_gated_mlp_fp8_gpu returned {rc}")
    return out


def run_jit(A_fp8, sa, Bgs, sbg, Bus, sbu, M):
    from rdna_fp8_dual_gemm_silu import compile_fp8_dual_gemm_silu
    launcher = compile_fp8_dual_gemm_silu(M=M, N=N, K=K)
    out = torch.zeros(M, N, dtype=torch.float16, device="cuda")
    launcher(out, A_fp8, Bgs, Bus, sa, sbg, sbu, torch.cuda.current_stream(), M)
    torch.cuda.synchronize()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["abi", "jit"], default="abi")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--M", type=int, default=256)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    arch = args.arch or common.detect_arch()
    common.import_impl(arch, "dual_gemm_silu", "rdna_fp8_dual_gemm_silu")

    A_fp8, sa, Bg_fp8, sbg, Bu_fp8, sbu, Bgs, Bus = make_inputs(args.M)
    ref = reference(A_fp8, sa, Bg_fp8, sbg, Bu_fp8, sbu)

    runner = {"abi": run_abi, "jit": run_jit}[args.impl]
    out = runner(A_fp8, sa, Bgs, sbg, Bus, sbu, args.M)

    err = common.report(f"dual_silu/{args.impl}", out, ref)
    tol = 5e-2
    print("PASS" if err < tol else f"FAIL (max_abs={err:.4f} > tol={tol})")
    sys.exit(0 if err < tol else 1)
