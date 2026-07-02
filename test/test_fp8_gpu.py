"""Build-and-run test for the GPU fp8 kernels + the int8 dequant-transpose.

Builds lib/libfluke.so via `make` (fluke_lib) and calls the real fluke_rmsnorm_quant_fp8_gpu,
fluke_dequant_fp8_transpose_gpu, and fluke_dequant_int8_transpose_gpu through ctypes, diffing
against torch references -> PASS/FAIL. The fp8 path also validates fluke's *software* E4M3FN
conversion against PyTorch's kFloat8_e4m3fn.

    ./pyvenv/bin/python test/test_fp8_gpu.py

Requires a CUDA torch (with float8_e4m3fn) + a CUDA toolkit.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # fluke root (fluke_lib)
import fluke_lib

FP8 = torch.float8_e4m3fn


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")

    lib = fluke_lib.load()
    dev = "cuda"
    torch.manual_seed(0)
    ok = True

    # ── rmsnorm_quant_fp8: fused rmsnorm + fp8 quantize, in place ──────────────
    alpha, eps = 2.0, 1e-5
    n_tokens, hidden_dim = 512, 512
    in_ = torch.randn(n_tokens, hidden_dim, device=dev, dtype=torch.float16)
    weight = torch.randn(hidden_dim, device=dev, dtype=torch.float16)
    residual = (torch.randn(n_tokens, hidden_dim, device=dev) * 0.1).to(FP8)   # fp8 residual bytes
    res_scale = torch.rand(n_tokens, device=dev, dtype=torch.float32) * 0.02 + 0.001

    # torch reference (fp32 math; fp8 quantize via torch's kFloat8_e4m3fn)
    val = in_.float() + alpha * (residual.float() * res_scale[:, None])
    rms = torch.rsqrt(val.pow(2).mean(dim=-1, keepdim=True) + eps)
    normalized = val * rms * weight.float()
    amax = normalized.abs().amax(dim=-1, keepdim=True)
    fp8_scale_ref = torch.clamp(amax, min=1e-12) / 448.0                       # [n_tokens,1]
    q_ref = (normalized / fp8_scale_ref).clamp(-448, 448).to(FP8)
    recon_ref = q_ref.float() * fp8_scale_ref                                  # fp8-quantized normalized

    q_got = residual.clone()          # kernel overwrites residual (fp8) in place
    scale_got = res_scale.clone()
    lib.fluke_rmsnorm_quant_fp8_gpu(in_.data_ptr(), weight.data_ptr(), q_got.data_ptr(),
                                    scale_got.data_ptr(), n_tokens, hidden_dim, float(alpha), float(eps))
    torch.cuda.synchronize()
    recon_got = q_got.float() * scale_got[:, None]

    scale_diff = (scale_got - fp8_scale_ref.squeeze(-1)).abs().max().item()
    recon_diff = (recon_got - recon_ref).abs().max().item()                   # kernel-fp8 vs torch-fp8
    quant_err = (recon_got - normalized).abs().max().item()                   # fp8 rounding (informational)
    scale_tol, recon_tol = 1e-6, 1e-3
    print(f"[rmsnorm_quant_fp8]  scale max diff: {scale_diff:.3e} (tol {scale_tol:.0e})  "
          f"fp8 vs torch-fp8: {recon_diff:.3e} (tol {recon_tol:.0e})  fp8 quant err: {quant_err:.3e} (info)")
    ok &= (scale_diff < scale_tol) and (recon_diff < recon_tol)

    # ── dequant_fp8_transpose: fp8 [T,N,C] * scale -> f16 [N,T,C] ──────────────
    T, N, C = 33, 20, 512
    scale = 0.0473
    in_f8 = (torch.randn(T, N, C, device=dev) * 0.7).to(FP8)
    out = torch.empty(N, T, C, device=dev, dtype=torch.float16)
    lib.fluke_dequant_fp8_transpose_gpu(in_f8.data_ptr(), out.data_ptr(), T, N, C, float(scale))
    torch.cuda.synchronize()
    ref = (in_f8.float() * scale).transpose(0, 1).contiguous().to(torch.float16)
    d_fp8 = (out.float() - ref.float()).abs().max().item()
    print(f"[dequant_fp8_transpose]  max abs diff: {d_fp8:.3e} (tol 5e-3)")
    ok &= d_fp8 < 5e-3

    # ── dequant_int8_transpose: int8 [T,N,C] * scale -> f16 [N,T,C] ────────────
    in_i8 = torch.randint(-127, 128, (T, N, C), device=dev, dtype=torch.int8)
    out2 = torch.empty(N, T, C, device=dev, dtype=torch.float16)
    lib.fluke_dequant_int8_transpose_gpu(in_i8.data_ptr(), out2.data_ptr(), T, N, C, float(scale))
    torch.cuda.synchronize()
    ref2 = (in_i8.float() * scale).transpose(0, 1).contiguous().to(torch.float16)
    d_i8 = (out2.float() - ref2.float()).abs().max().item()
    print(f"[dequant_int8_transpose] max abs diff: {d_i8:.3e} (tol 5e-3)")
    ok &= d_i8 < 5e-3

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
