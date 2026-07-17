// Per-arch fused-fp8 launch wrappers (HIP) — compiled ONCE PER ARCH from this single source.
//
// One translation unit per arch is required (each arch's generated header defines the same
// identifiers — fp8_gemm_rotary_wrapper, FP8_GEMM_ROTARY_THREADS_PER_BLOCK, ... — with different
// launch geometry, so two arches can't share a TU). The body is identical, so instead of a file
// per arch the Makefile compiles this source once per arch with:
//   -DFLUKE_FP8_ARCH_NAME=<gfxNNNN>          the arch tag (names the exported vtable)
//   -DFLUKE_FP8_ROTARY_HDR="artifacts/.../rotary.h"   that arch's generated headers
//   -DFLUKE_FP8_MLP_HDR="artifacts/.../mlp.h"
// Each object exports one external symbol, the vtable `fluke_fp8_ops_<arch>`, which
// src/fused_hip.c binds by gcnArchName. Add an arch = add it to the Makefile arch list + a
// g_archs[] row in fused_hip.c; no new source file. See src/fp8_ops.h.
//
// Pure HOST code (drives the HIP module/launch API only — no __global__ kernels of its own), so
// the Makefile compiles it with the host C compiler; building needs the headers, not the target GPU.
#include <fluke/fluke.h>

#if defined(HAVE_ROCM) && !defined(FLUKE_NO_FUSED)

#include <hip/hip_runtime.h>

#include "fp8_ops.h"
#include FLUKE_FP8_ROTARY_HDR   /* -> fp8_gemm_rotary_Module_t / _Module_LoadData / _wrapper */
#include FLUKE_FP8_MLP_HDR      /* -> fp8_dual_silu_Module_t   / _Module_LoadData / _wrapper */

#ifndef FLUKE_FP8_ARCH_NAME
#error "fp8_arch.c: define FLUKE_FP8_ARCH_NAME (e.g. gfx1200) — see the Makefile rule"
#endif

#define FLUKE_CAT_(a, b) a##b
#define FLUKE_CAT(a, b) FLUKE_CAT_(a, b)
#define FLUKE_OPS_SYM FLUKE_CAT(fluke_fp8_ops_, FLUKE_FP8_ARCH_NAME)

/* This arch's kernel modules, bound once by fp8_load() (via fluke_fp8_select). Only the arch
   matching the running device is ever selected, so file-local state is sufficient. */
static fp8_gemm_rotary_Module_t g_rotary;
static fp8_dual_silu_Module_t   g_mlp;

static int fp8_load(const void *rotary_image, const void *mlp_image) {
    if (fp8_gemm_rotary_Module_LoadData(&g_rotary, rotary_image) != 0) return -1;
    if (fp8_dual_silu_Module_LoadData(&g_mlp, mlp_image) != 0) return -1;
    return 0;
}

static int fp8_qkv_rotary(void *out, const void *a_fp8, const void *wqkv_fp8,
                          const void *scale_a, const void *scale_b,
                          const void *sin, const void *cos, int M, int seqlen) {
    return fp8_gemm_rotary_wrapper(
        &g_rotary, out, (void *)a_fp8, (void *)wqkv_fp8,
        (void *)scale_a, (void *)scale_b, (void *)sin, (void *)cos,
        (int32_t)M, (int32_t)seqlen, /*stream=*/0);
}

static int fp8_gated_mlp(void *out, const void *a_fp8, const void *gate_fp8, const void *up_fp8,
                         const void *scale_a, const void *scale_gate, const void *scale_up, int M) {
    return fp8_dual_silu_wrapper(
        &g_mlp, out, (void *)a_fp8, (void *)gate_fp8, (void *)up_fp8,
        (void *)scale_a, (void *)scale_gate, (void *)scale_up, (int32_t)M, /*stream=*/0);
}

/* A plain const file-scope object has external linkage in C; fused_hip.c references this symbol
   by its arch-qualified name. */
const struct fluke_fp8_ops FLUKE_OPS_SYM = {
    fp8_load, fp8_qkv_rotary, fp8_gated_mlp
};

#endif  /* HAVE_ROCM && !FLUKE_NO_FUSED */
