// Fused FP8 DSL kernels (AMD/HIP) — arch dispatch only.
//
// The AMD counterpart of src/fused_cuda.cpp. Implements the ATen-free C ABI declared in
// <fluke/fluke.h> (fluke_fp8_select, fluke_qkv_rotary_fp8_gpu, fluke_gated_mlp_fp8_gpu).
//
// This TU holds NO arch-specific launch code. Each supported arch compiles to its own
// src/fp8_arch_<arch>.cpp, which includes that arch's generated header (baking its launch
// geometry) and exports a `fluke_fp8_ops_<arch>` vtable (see src/fp8_ops.h). Here we just
// match the device's gcnArchName to a row in g_archs[], load that arch's embedded HSACO
// images through its vtable, and dispatch launches through it. Because the per-arch launch
// code is isolated, one libfluke.a can embed many arches (a "fat" binary) even when their
// launch configs differ (RDNA4 WMMA/wave32 vs CDNA3 MFMA/wave64).
//
// The FlyDSL artifacts are standalone HSACO images; the Makefile embeds the per-arch bytes
// into libfluke.a (objcopy/.incbin -> fluke_fp8_<role>_<arch> symbols) and they are loaded
// at runtime with hipModuleLoadData, so the archive stays self-contained (no runtime file).
//
// On a device whose arch is not embedded (or a dims mismatch) fluke_fp8_select returns NULL
// and the caller keeps its fp16 path.

#include <fluke/fluke.h>

#include <stdio.h>
#include <string.h>

#if defined(HAVE_ROCM) && !defined(FLUKE_NO_FUSED)

#include <hip/hip_runtime.h>

#include "fp8_ops.h"
// Baked model dims (shared with the CUDA backend — model-specific, not arch-specific).
#include "fused_dims.h"

// Per-arch vtables (defined in src/fp8_arch_<arch>.cpp).
extern const struct fluke_fp8_ops fluke_fp8_ops_gfx1200;
extern const struct fluke_fp8_ops fluke_fp8_ops_gfx1201;

// Embedded per-arch HSACO images (Makefile: objcopy/.incbin -> these symbols).
extern "C" const unsigned char fluke_fp8_rotary_gfx1200[];
extern "C" const unsigned char fluke_fp8_rotary_gfx1201[];
extern "C" const unsigned char fluke_fp8_mlp_gfx1200[];
extern "C" const unsigned char fluke_fp8_mlp_gfx1201[];

struct arch_row {
    const char               *arch;    // gcnArchName base (before the ':feature' suffix)
    const struct fluke_fp8_ops *ops;   // that arch's launch vtable
    const void               *rotary;  // embedded qkv-rotary HSACO
    const void               *mlp;     // embedded dual-gemm+silu HSACO
};

// One row per embedded arch. Add a chip = new fp8_arch_<arch>.cpp (+ its embed symbols +
// Makefile rule) and a row here; fluke_fp8_select matches by gcnArchName.
static const struct arch_row g_archs[] = {
    { "gfx1200", &fluke_fp8_ops_gfx1200, fluke_fp8_rotary_gfx1200, fluke_fp8_mlp_gfx1200 },
    { "gfx1201", &fluke_fp8_ops_gfx1201, fluke_fp8_rotary_gfx1201, fluke_fp8_mlp_gfx1201 },
};

struct fluke_fp8_backend {
    fluke_dims_t dims;
    const struct fluke_fp8_ops *ops;  // selected arch's launch vtable
};

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

    const struct arch_row *sel = NULL;
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
        if (sel->ops->load(sel->rotary, sel->mlp) != 0) return NULL;
        g_modules_loaded = 1;
        fprintf(stderr, "[fluke] fp8 kernel backend active on device %d (%s)\n", device_index, arch);
    }

    static struct fluke_fp8_backend b;  // process-lifetime; all layers share it
    b.dims = dims;
    b.ops  = sel->ops;
    return &b;
}

int fluke_qkv_rotary_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *wqkv_fp8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
) {
    return b->ops->qkv_rotary(out, a_fp8, wqkv_fp8, scale_a, scale_b, sin, cos, M, seqlen);
}

int fluke_gated_mlp_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *gate_fp8, const void *up_fp8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
) {
    return b->ops->gated_mlp(out, a_fp8, gate_fp8, up_fp8, scale_a, scale_gate, scale_up, M);
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
