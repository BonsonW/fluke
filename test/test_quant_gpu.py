"""Build-and-run test for the standalone GPU quantize kernels.

Builds lib/libfluke.so via `make` (fluke_lib) and calls the real fluke_quant_int8_gpu and
fluke_quant_fp8_gpu through ctypes, diffing against torch references -> PASS/FAIL.

  * int8 mirrors quantize_tensor(x, dim=-1): scale = amax/128 (dequant multiplier), out =
    clamp(round(x/scale), -127, 127). Expected bit-exact vs the torch reference.
  * fp8 (E4M3FN): scale = clamp(amax,1e-12)/448, out = e4m3fn(clamp(x/scale, -448, 448)).
    Validated against PyTorch's kFloat8_e4m3fn (kernel-fp8 vs torch-fp8 must match exactly).

    ./pyvenv/bin/python test/test_quant_gpu.py

Requires a CUDA torch (with float8_e4m3fn) + a CUDA toolkit.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)   # fluke root (fluke_lib)
import fluke_lib

FP8 = torch.float8_e4m3fn

# Shapes exercised: even hidden_dim in {512, 1024, 2048} (kernel is half2-vectorized, <= 2048).
SHAPES = [(300, 512), (128, 1024), (64, 2048), (512, 512)]


def quantize_tensor_ref(x_f32):
    """Reference symmetric int8 per-token quant (quantize_tensor with dim=-1)."""
    fp_range = x_f32.abs().amax(-1)
    i_range = 128
    quant_scale = i_range / fp_range
    quant_max = i_range - 1
    x_quant = (x_f32 * quant_scale.unsqueeze(-1)).round().clip(-quant_max, quant_max)
    return x_quant.to(torch.int8), (1.0 / quant_scale).to(torch.float32)


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA torch not available; install torch with CUDA (see test/requirements.txt)")

    lib = fluke_lib.load()
    dev = "cuda"
    torch.manual_seed(0)
    ok = True

    for (N, C) in SHAPES:
        x = (torch.randn(N, C, device=dev, dtype=torch.float16) * 3.0)

        # ── quant_int8: bit-exact vs quantize_tensor ──────────────────────────
        out_i8 = torch.empty(N, C, device=dev, dtype=torch.int8)
        scale_i8 = torch.empty(N, device=dev, dtype=torch.float32)
        lib.fluke_quant_int8_gpu(x.data_ptr(), out_i8.data_ptr(), scale_i8.data_ptr(), N, C)
        torch.cuda.synchronize()

        q_ref, s_ref = quantize_tensor_ref(x.float())
        dq = (out_i8.int() - q_ref.int()).abs()
        ds = (scale_i8 - s_ref).abs().max().item()
        n_mismatch = int((dq != 0).sum().item())
        # round-to-nearest-even on both sides -> expect exact; allow rare off-by-1 near ties.
        i8_ok = (dq.max().item() <= 1) and (n_mismatch <= 2) and (ds < 1e-6)
        print(f"[quant_int8  N={N:<4} C={C:<4}] max|Δq|={dq.max().item()}  #Δ!=0={n_mismatch}/{N*C}  "
              f"max|Δscale|={ds:.2e}  ({'ok' if i8_ok else 'FAIL'})")
        ok &= i8_ok

        # ── quant_fp8: scale exact + dequant within one E4M3 step of the true value.
        # (fluke's software e4m3fn rounds half-up; torch casts half-to-even, so a bit-exact
        # kernel-vs-torch check would flag legitimate 1-ULP tie differences — coarse near ±448.
        # A correct quantizer's reconstruction is within one representable step; check that.)
        out_f8 = torch.empty(N, C, device=dev, dtype=torch.uint8)
        scale_f8 = torch.empty(N, device=dev, dtype=torch.float32)
        lib.fluke_quant_fp8_gpu(x.data_ptr(), out_f8.data_ptr(), scale_f8.data_ptr(), N, C)
        torch.cuda.synchronize()

        amax = x.float().abs().amax(-1)
        s_fp8_ref = torch.clamp(amax, min=1e-12) / 448.0
        recon_got = out_f8.view(FP8).float() * scale_f8[:, None]

        # local E4M3 step at the true normalized magnitude m=|x|/scale (3 mantissa bits ->
        # step 2^(e-3) for 2^e<=m; denormal step floor 2^-9). Rounding error <= one step.
        m = (x.float().abs() / scale_f8[:, None]).clamp(min=2.0 ** -6)
        e = torch.floor(torch.log2(m))
        step = torch.clamp(2.0 ** (e - 3.0), min=2.0 ** -9) * scale_f8[:, None]
        abs_err = (recon_got - x.float()).abs()
        worst_steps = (abs_err / step.clamp(min=1e-12)).max().item()   # <=1 for a valid quantizer

        ds8 = (scale_f8 - s_fp8_ref).abs().max().item()
        f8_ok = (ds8 < 1e-6) and (worst_steps <= 1.0 + 1e-3)
        print(f"[quant_fp8   N={N:<4} C={C:<4}] max|Δscale|={ds8:.2e}  worst_err={worst_steps:.3f} e4m3-steps  "
              f"max|Δdequant|={abs_err.max().item():.3e} (info)  ({'ok' if f8_ok else 'FAIL'})")
        ok &= f8_ok

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
