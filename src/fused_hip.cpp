// Fused FP8 DSL kernels (RDNA4) — arch dispatch + module load + launch plumbing.
//
// The AMD/HIP counterpart of src/fused_cuda.cpp. Implements the ATen-free C ABI declared
// in <fluke/fluke.h> (fluke_fp8_select, fluke_qkv_rotary_fp8_gpu, fluke_gated_mlp_fp8_gpu)
// over the AOT-exported FlyDSL kernels (artifacts/<gfxNNNN>/*.h + *.hsaco).
//
// Unlike the CUDA side (which statically links CuTe-generated cubin objects), the FlyDSL
// artifacts are standalone HSACO images. We embed the per-arch HSACO bytes into libfluke.a
// (see the Makefile embed rules -> fluke_fp8_<role>_<arch> symbols) and load them at runtime
// with hipModuleLoadData, so the archive stays self-contained (no runtime .hsaco file).
//
// RDNA4 code objects are ISA-specific and don't cross-load (a gfx1200 object won't run on
// gfx1201), and gfx12-generic won't compile through FlyDSL's MLIR — so we ship one HSACO per
// concrete arch and pick by the device's gcnArchName here. On a non-RDNA4 device (or a
// dims mismatch) fluke_fp8_select returns NULL and the caller keeps its fp16 path.

#include <fluke/fluke.h>

#include <stdio.h>
#include <string.h>

#if defined(HAVE_ROCM)

#include <hip/hip_runtime.h>

// The generated headers are arch-independent (same kernel name / dims / launch config for
// every RDNA4 arch); only the embedded HSACO bytes differ, and those are selected at runtime.
#include "artifacts/gfx1201/rdna_fp8_gemm_rotary_N1536_K512_TM64_TN256.h"
#include "artifacts/gfx1201/rdna_fp8_dual_gemm_silu_N2048_K512_TM32_TN256.h"

// Baked model dims (shared with the CUDA backend — model-specific, not arch-specific).
#include "fused_dims.h"

// Embedded per-arch HSACO images (Makefile: objcopy/.incbin -> these symbols).
extern "C" const unsigned char fluke_fp8_rotary_gfx1200[];
extern "C" const unsigned char fluke_fp8_rotary_gfx1201[];
extern "C" const unsigned char fluke_fp8_mlp_gfx1200[];
extern "C" const unsigned char fluke_fp8_mlp_gfx1201[];

struct arch_images {
    const char   *arch;    // gcnArchName prefix (before the ':feature' suffix)
    const void   *rotary;  // embedded qkv-rotary HSACO
    const void   *mlp;     // embedded dual-gemm+silu HSACO
};

// One row per shipped RDNA4 arch. Add a chip here (+ its embed symbols + Makefile rule) to
// support it; fluke_fp8_select matches by gcnArchName.
static const struct arch_images g_archs[] = {
    { "gfx1200", fluke_fp8_rotary_gfx1200, fluke_fp8_mlp_gfx1200 },
    { "gfx1201", fluke_fp8_rotary_gfx1201, fluke_fp8_mlp_gfx1201 },
};

struct fluke_fp8_backend {
    fluke_dims_t dims;
};

// Process-global kernel modules, loaded once by fluke_fp8_select.
static fp8_gemm_rotary_Module_t g_rotary_module;
static fp8_dual_silu_Module_t   g_mlp_module;
static int g_modules_loaded = 0;

fluke_fp8_backend_t *fluke_fp8_select(int device_index, fluke_dims_t dims) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device_index) != hipSuccess) {
        return NULL;
    }
    // gcnArchName is e.g. "gfx1201:sramecc+:xnack-"; match on the base gfx name.
    char arch[64];
    strncpy(arch, prop.gcnArchName, sizeof(arch) - 1);
    arch[sizeof(arch) - 1] = '\0';
    char *colon = strchr(arch, ':');
    if (colon) *colon = '\0';

    const struct arch_images *sel = NULL;
    for (size_t i = 0; i < sizeof(g_archs) / sizeof(g_archs[0]); ++i) {
        if (strcmp(g_archs[i].arch, arch) == 0) { sel = &g_archs[i]; break; }
    }
    if (!sel) {
        fprintf(stderr, "[fluke] no fp8 backend for GPU arch %s — using fp16\n", arch);
        return NULL;
    }
    if (dims.d_model != FLUKE_SUP_D_MODEL || dims.dim_feedforward != FLUKE_SUP_DIM_FF ||
        dims.nhead != FLUKE_SUP_NHEAD || dims.head_dim != FLUKE_SUP_HEAD_DIM) {
        fprintf(stderr, "[fluke] model dims do not match precompiled fp8 kernels "
                        "(need d_model=%d, dim_feedforward=%d, nhead=%d, head_dim=%d) — using fp16\n",
                FLUKE_SUP_D_MODEL, FLUKE_SUP_DIM_FF, FLUKE_SUP_NHEAD, FLUKE_SUP_HEAD_DIM);
        return NULL;
    }

    if (!g_modules_loaded) {
        if (fp8_gemm_rotary_Module_LoadData(&g_rotary_module, sel->rotary) != 0) return NULL;
        if (fp8_dual_silu_Module_LoadData(&g_mlp_module, sel->mlp) != 0) return NULL;
        g_modules_loaded = 1;
        fprintf(stderr, "[fluke] fp8 kernel backend active on device %d (%s)\n", device_index, arch);
    }

    static struct fluke_fp8_backend b;  // process-lifetime; all layers share it
    b.dims = dims;
    return &b;
}

int fluke_qkv_rotary_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *wqkv_fp8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
) {
    (void)b;
    return fp8_gemm_rotary_wrapper(
        &g_rotary_module,
        out, (void *)a_fp8, (void *)wqkv_fp8,
        (void *)scale_a, (void *)scale_b, (void *)sin, (void *)cos,
        (int32_t)M, (int32_t)seqlen, /*stream=*/0);
}

int fluke_gated_mlp_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *gate_fp8, const void *up_fp8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
) {
    (void)b;
    return fp8_dual_silu_wrapper(
        &g_mlp_module,
        out, (void *)a_fp8, (void *)gate_fp8, (void *)up_fp8,
        (void *)scale_a, (void *)scale_gate, (void *)scale_up,
        (int32_t)M, /*stream=*/0);
}

#else  // no ROCm — null backend; callers keep the fp16 path.

fluke_fp8_backend_t *fluke_fp8_select(int device_index, fluke_dims_t dims) {
    (void)device_index; (void)dims;
    return NULL;
}
int fluke_qkv_rotary_fp8_gpu(const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *wqkv_fp8, const void *scale_a, const void *scale_b,
    const void *sin, const void *cos, int M, int seqlen) {
    (void)b;(void)out;(void)a_fp8;(void)wqkv_fp8;(void)scale_a;(void)scale_b;
    (void)sin;(void)cos;(void)M;(void)seqlen;
    return -1;
}
int fluke_gated_mlp_fp8_gpu(const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *gate_fp8, const void *up_fp8,
    const void *scale_a, const void *scale_gate, const void *scale_up, int M) {
    (void)b;(void)out;(void)a_fp8;(void)gate_fp8;(void)up_fp8;
    (void)scale_a;(void)scale_gate;(void)scale_up;(void)M;
    return -1;
}

#endif
