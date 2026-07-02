# fluke

Neural-net GPU kernels (quant + layers) split out from
[openfish](https://github.com/warp9seq/openfish), which remains a pure decoding
library. Fluke is a refactor of the kernel work previously done in `gigaslop`.

Two parts:

1. **Pure C library** (`src/`, `include/fluke/`, `Makefile`) — hand-written CUDA
   and HIP kernels behind a small C ABI, mirroring openfish's structure. Built as
   `lib/libfluke.a`.
2. **DSL kernels** (`cute/`, `fly/`) — fused kernels written in CuTe DSL (NVIDIA)
   and FlyDSL (AMD), organised by microarchitecture (`cute/ampere/`, `fly/rdna4/`,
   `fly/cdna3/`, …). Each has export scripts (AOT → C headers), a C-ABI host
   harness, and Python tests that compare the DSL output against a torch reference
   *and* against the pure C kernel from part 1.

## Layout

```
include/fluke/fluke.h            public C API (fluke_rotary_emb_cpu/gpu, ...)
src/                             pure CPU/CUDA/HIP C kernels (flat cuda/hip split)
test/                            python tests for the pure C library (load_inline)
artifacts/                       AOT-exported .h/.o (top-level, shared; gitignored)
cute/common.py                   shared test helpers (arch dispatch, quantize, report)
cute/test_<op>.py                per-op test: picks an impl + reference, compares
cute/<arch>/gemm_i8_quant.py     the INT8 GEMM implementation (+ base for other kernels)
cute/<arch>/<op>/                per-op implementation: DSL kernel + export script
fly/<arch>/                      FlyDSL kernels per microarch (rdna4, cdna3, ...); stub
```

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
  (+ shared base class `cute/ampere/gemm_i8_quant.py`), exported by
  `cute/ampere/rotary/export_gemm_i8_rotary.py` (config inlined at the top) to `artifacts/`.
- Tests: `test/test_rotary_cuda.py` (pure CUDA vs torch, standalone C-lib test),
  `cute/test_rotary.py` (the DSL kernel — jit or aot — vs torch and/or the pure CUDA C
  kernel), and `cute/test_gemm.py` (the base INT8 GEMM vs an fp16 torch GEMM).

The **dual INT8 GEMM + SiLU** kernel follows the same template: implementation in
`cute/ampere/dual_gemm_silu/` (`dual_gemm_i8_silu.py` + `export_dual_gemm_i8_silu.py`),
pure CUDA C epilogue `fluke_silu_mul_gpu` (`src/nn_cuda.c`), tested by
`cute/test_dual_gemm_silu.py` (`out = silu(A@Bg^T) * (A@Bu^T)`; jit/aot vs torch and/or
the pure CUDA C `silu_mul`).

**RMSNorm** is pure C/CUDA only (no CuTe DSL): `fluke_rmsnorm_gpu` and the fused
`fluke_rmsnorm_quant_int8_gpu` in `src/` (CUDA + HIP), with a fp32 CPU reference
`fluke_rmsnorm_cpu` for the non-quant kernel only. Tested by `test/test_rmsnorm_cuda.py`
(CUDA kernels vs a naive torch reference; `load_inline`). The fp8 quant variant is not
ported — openfish leaves it unimplemented on CUDA.

## Build

```
make cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"          # -> lib/libfluke.a
make shared cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"   # -> lib/libfluke.so (PIC, for tests)
make rocm=1 ROCM_ARCH="--offload-arch=gfx1200"                       # HIP backend
make                                                                 # CPU-only
```

(The Python tests build `lib/libfluke.so` automatically via `fluke_lib` if it's missing.)

## Test

```
<venv>/bin/python test/test_rotary_cuda.py                    # pure CUDA rotary vs torch ref
<venv>/bin/python test/test_rmsnorm_cuda.py                   # pure CUDA rmsnorm (+quant int8) vs torch
<venv>/bin/python cute/test_rotary.py                         # DSL jit vs torch + pure CUDA
<venv>/bin/python cute/test_rotary.py --impl aot              # exported .o (load_module) instead
<venv>/bin/python cute/test_rotary.py --impl aot --ref torch  # pick impl + reference
<venv>/bin/python cute/test_gemm.py                           # INT8 GEMM vs fp16 torch GEMM
<venv>/bin/python cute/test_dual_gemm_silu.py                 # dual GEMM+SiLU: jit/aot vs torch + cuda C
<venv>/bin/python cute/ampere/rotary/export_gemm_i8_rotary.py # AOT export only (config inlined)
```

## Benchmark

Perf harnesses (arch auto-detected; warm the GPU clocks then time with CUDA events,
reporting TOPS + effective bandwidth):

```
<venv>/bin/python cute/bench_gemm.py                          # INT8 GEMM  (--M --N --K)
<venv>/bin/python cute/bench_rotary.py                        # fused INT8 GEMM+rotary  (--M --K --nhead ...)
<venv>/bin/python cute/bench_dual_gemm_silu.py                # fused dual INT8 GEMM+SiLU  (--M --N --K)
```
