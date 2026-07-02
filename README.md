# fluke

Neural-net GPU kernels (quant + layers) meant to cover the gaps of torch.

Two parts:

1. **Pure C library** (`src/`, `include/fluke/`, `Makefile`) — hand-written CUDA/HIP
   and HIP kernels behind a small C ABI, Built as
   `lib/libfluke.a`.
2. **DSL kernels** (`cute/`, `fly/`) — fused kernels written in CuTe DSL (NVIDIA)
   and FlyDSL (AMD), organised by microarchitecture (`cute/ampere/`, `fly/rdna4/`,
   `fly/cdna3/`, …). Each has export scripts (AOT → C headers), a C-ABI host
   harness, and Python tests that compare the DSL output against a torch reference
   *and* against the pure C kernel from part 1.

## Layout

```
include/fluke/fluke.h            public C API (fluke_rotary_emb_cpu/gpu, fluke_*_i8/fp8_gpu, ...)
src/                             pure CPU/CUDA/HIP C kernels (flat cuda/hip split) + fused_{cuda,hip}.cpp
test/                            python tests for the pure C library
artifacts/<arch>/                AOT-exported kernels (gitignored, regenerate per arch — see below):
                                   sm80/      CUDA int8:  *.h + *.o     (CuTe)
                                   gfx1200/   RDNA4 fp8:  *.h + *.hsaco (FlyDSL)
                                   gfx1201/   RDNA4 fp8:  *.h + *.hsaco (FlyDSL)
cute/common.py                   CUDA test helpers (arch dispatch, quantize, report)
cute/test_<op>.py                per-op CUDA test: picks an impl + reference, compares
cute/<arch>/gemm/                the INT8 (gemm_i8_quant) + f16 (gemm_f16) GEMM bases, shared by all ops
cute/<arch>/<op>/                per-op CUDA implementation: CuTe kernel + export script
fly/common.py                    FlyDSL test helpers (gfx arch dispatch; fp8 C-ABI ctypes loader)
fly/test_<op>.py                 per-op HIP test: --impl abi (real C ABI) | jit (in-process)
fly/rdna4/<op>/                  per-op RDNA4 fp8 implementation: FlyDSL kernel + export script
fly/rdna4/_fp8_export.py         shared export machinery (per-arch loop, HSACO extract, embed loader)
```

The CUDA (`cute/`) and HIP (`fly/`) DSL trees mirror each other. The AMD side mirrors
`cute/ampere/`: `fly/rdna4/{rotary,dual_gemm_silu,quantize,gemm}/`. Both fuse the same ops but
RDNA4 uses **fp8** (e4m3, WMMA, f32-accumulate, amax/448 scale) where Ampere uses **int8**
(amax/127). RDNA4 weights are preshuffled to the WMMA B layout `[N/16,K/16,2,16,8]`.

## Test organization

**Implementations** live under `cute/<arch>/...` (the DSL kernel + its export script,
nothing else). **Tests** live at `cute/test_<op>.py` and own everything else: input
generation, the references, and the comparison. A test picks:

- an **implementation** to run — `--impl jit` (in-process `cute.compile`) or
  `--impl aot` (load the exported `.o` via `cute.runtime.load_module`);
- one or more **references** to check against — `--ref torch` (naive PyTorch),
  `--ref cuda` (the pure CUDA C kernel from `src/`), or `--ref both`.

The test imports the arch's implementation for the GPU it's running on via
`cute/common.py` (`import_impl`, `detect_arch`; override with `--arch`). Adding an arch
= drop its implementation under `cute/<newarch>/...` and add the compute capability to
`ARCH_BY_CC` in `common.py`. (A HIP/fly mirror would live under `fly/`; not built yet.)

The **`--ref cuda`** path (and the `test/*.py` pure-C-library tests) call the real C ABI
through `ctypes` on `lib/libfluke.so` — no inline C++/nvcc recompile. `fluke_lib.load()`
builds the shared lib on demand (`make shared` for the detected arch) and returns a
ctypes handle with argtypes set; this is the same C ABI slorado calls in production.

## The rotary embedding kernel (template)

The Ampere (sm80) rotary embedding kernel is the first kernel and the template
for how every other kernel is organised:

- Pure CUDA kernel: `src/nn_kernel_cuda.h` (`rotary_emb`) + launch wrapper
  `src/nn_cuda.c` (`fluke_rotary_emb_gpu`); CPU reference in `src/nn_cpu.c`.
- CuTe fused INT8 GEMM+rotary implementation: `cute/ampere/rotary/gemm_i8_rotary.py`
  (+ shared base class `cute/ampere/gemm/gemm_i8_quant.py`), exported by
  `cute/ampere/rotary/export_gemm_i8_rotary.py` (config inlined at the top) to `artifacts/`.
- Tests: `test/test_rotary_gpu.py` (pure GPU C kernel vs torch, standalone C-lib test),
  `cute/test_rotary.py` (the DSL kernel — jit or aot — vs torch and/or the pure CUDA C
  kernel), and `cute/test_gemm.py` (the base INT8 GEMM vs an fp16 torch GEMM).

The **dual INT8 GEMM + SiLU** kernel follows the same template: implementation in
`cute/ampere/dual_gemm_silu/` (`dual_gemm_i8_silu.py` + `export_dual_gemm_i8_silu.py`),
pure CUDA C epilogue `fluke_silu_mul_gpu` (`src/nn_cuda.c`), tested by
`cute/test_dual_gemm_silu.py` (`out = silu(A@Bg^T) * (A@Bu^T)`; jit/aot vs torch and/or
the pure CUDA C `silu_mul`).

The **factored LSTM** layer follows the same template but is two kernels (see slorado
`CRFModel.cpp`): the down-projection (both the recurrent `hh_down` per-step and the input
`x_down` precompute) is the plain INT8 GEMM (`gemm_i8_quant`, int8→f16); the fused step
(`cute/ampere/factored_lstm/factored_lstm_i8.py`, `TensorOpFactoredLstmI8`) does two f16
up-projections into four gate accumulators + gates + cell update + INT8 hidden output
(fixed scale 1/127). Faithful to the RDNA fp8 original, only the down-projection and the
h output are INT8; the up-projections stay f16 (via the plain f16 GEMM base
`cute/ampere/gemm/gemm_f16.py`, `TensorOpGemm`, which the fused step inherits). Both kernels
are exported by `cute/ampere/factored_lstm/export_factored_lstm_i8.py` (the fused step is
K-merged: A = [hh_down | x_down], per-gate weight W_g = [up_hh_g | up_ih_g]). Tested by
`cute/test_factored_lstm.py` (jit/aot vs torch; DSL-only, no pure-CUDA C ref).

**RMSNorm** is pure C/CUDA only (no CuTe DSL): `fluke_rmsnorm_gpu` and the fused
`fluke_rmsnorm_quant_int8_gpu` in `src/` (CUDA + HIP), with a fp32 CPU reference
`fluke_rmsnorm_cpu` for the non-quant kernel only. Tested by `test/test_rmsnorm_gpu.py`
(CUDA kernels vs a naive torch reference; `load_inline`).

The **fp8** variants are also pure C/CUDA+HIP: `fluke_rmsnorm_quant_fp8_gpu` (fused RMSNorm +
E4M3FN quantize) and `fluke_dequant_fp8_transpose_gpu` (dequant + transpose in one pass), plus an
int8 analogue `fluke_dequant_int8_transpose_gpu`. fp8 uses **software E4M3FN conversion**
(`e4m3fn_to_float`/`float_to_e4m3fn`, bit-exact vs PyTorch `float8_e4m3fn`), so it runs on CUDA too
(openfish shipped these HIP-only). Tested by `test/test_fp8_gpu.py`.

## Build

AOT artifacts are **not committed** — regenerate them for the target arch *before* `make`
(the build embeds/links them into `libfluke.a` and fails with a clear message if missing):

```
# CUDA (sm80): export the CuTe int8 artifacts -> artifacts/sm80/
<venv>/bin/python cute/ampere/rotary/export_gemm_i8_rotary.py
<venv>/bin/python cute/ampere/dual_gemm_silu/export_dual_gemm_i8_silu.py

# RDNA4 (gfx1200+gfx1201): export the FlyDSL fp8 artifacts -> artifacts/<gfxNNNN>/
<venv>/bin/python fly/rdna4/rotary/export_fp8_gemm_rotary.py
<venv>/bin/python fly/rdna4/dual_gemm_silu/export_fp8_dual_gemm_silu.py    # dual-silu + per-token quantize
<venv>/bin/python fly/rdna4/factored_lstm/export_fp8_factored_lstm.py     # fused step + fp8 down-proj
```

```
make cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"          # -> lib/libfluke.a
make shared cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"   # -> lib/libfluke.so (PIC, for tests)
make rocm=1 ROCM_ARCH="--offload-arch=gfx1201"                       # HIP backend (embeds fp8 HSACOs)
make fp8_shared rocm=1 ROCM_ARCH="--offload-arch=gfx1201"            # -> lib/libfluke_fp8.so (for fly tests)
make                                                                 # CPU-only
```

The RDNA4 fp8 fused kernels are compiled once per concrete arch (gfx1200, gfx1201) — a single
`gfx12-generic` object won't compile through FlyDSL's MLIR and per-chip objects don't cross-load,
so `fluke_fp8_select` picks the matching embedded HSACO by the device's `gcnArchName`.

(The Python tests build the needed `.so` automatically via `fluke_lib` / `fly.common` if missing;
`fluke_lib` auto-detects CUDA vs ROCm from the torch build.)

## Test

```
<venv>/bin/python test/test_rotary_gpu.py                    # pure GPU C rotary vs torch ref
<venv>/bin/python test/test_rmsnorm_gpu.py                   # pure GPU C rmsnorm (+quant int8) vs torch
<venv>/bin/python test/test_fp8_gpu.py                       # rmsnorm_quant_fp8 + dequant fp8/int8 transpose vs torch
<venv>/bin/python cute/test_rotary.py                         # DSL jit vs torch + pure CUDA
<venv>/bin/python cute/test_rotary.py --impl aot              # exported .o (load_module) instead
<venv>/bin/python cute/test_rotary.py --impl aot --ref torch  # pick impl + reference
<venv>/bin/python cute/test_gemm.py                           # INT8 GEMM vs fp16 torch GEMM
<venv>/bin/python cute/test_dual_gemm_silu.py                 # dual GEMM+SiLU: jit/aot vs torch + cuda C
<venv>/bin/python cute/test_factored_lstm.py                 # factored-LSTM step: jit/aot vs torch
<venv>/bin/python cute/ampere/rotary/export_gemm_i8_rotary.py # AOT export only (config inlined)
<venv>/bin/python cute/ampere/factored_lstm/export_factored_lstm_i8.py # AOT export (step + int8 down-proj)

# RDNA4 fp8 (HIP) — --impl abi exercises the real fluke_fp8_* C ABI (embedded HSACO + dispatch),
# --impl jit runs the FlyDSL kernel in-process. Both compared vs a torch reference.
<venv>/bin/python fly/test_rotary.py                          # fused fp8 GEMM+rotary   (abi|jit)
<venv>/bin/python fly/test_dual_gemm_silu.py                  # fused fp8 dual GEMM+SiLU (abi|jit)
<venv>/bin/python fly/test_factored_lstm.py                   # fused factored-LSTM step (jit; DSL-only)
```

## Benchmark

Perf harnesses (arch auto-detected; warm the GPU clocks then time with CUDA events,
reporting TOPS + effective bandwidth):

```
<venv>/bin/python cute/bench_gemm.py                          # INT8 GEMM  (--M --N --K)
<venv>/bin/python cute/bench_rotary.py                        # fused INT8 GEMM+rotary  (--M --K --nhead ...)
<venv>/bin/python cute/bench_dual_gemm_silu.py                # fused dual INT8 GEMM+SiLU  (--M --N --K)
<venv>/bin/python cute/bench_factored_lstm.py                # INT8 factored-LSTM per-step (down-proj + step)  (--B)
```
