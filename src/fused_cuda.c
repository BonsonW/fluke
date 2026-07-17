// Fused INT8 DSL kernels (CUDA) — arch dispatch for the transformer fused kernels.
//
// Implements the ATen-free C ABI declared in <fluke/fluke.h> (fluke_int8_select,
// fluke_qkv_rotary_i8_gpu, fluke_gated_mlp_i8_gpu) over the AOT-exported CuTe kernels. This TU
// holds NO kernel-launch code: each SM arch compiles to its own src/i8_arch.c, which includes
// that arch's generated headers and exports a `fluke_i8_ops_<sm>` vtable (see src/i8_ops.h). Here
// we select the vtable by compute capability and dispatch launches through it. Isolating per-arch
// launch code lets one libfluke.a carry several SM arches (a "fat" binary).
//
// The generated headers use CUDA 12's library-management API. On older toolkits or non-CUDA
// builds (or with fused disabled) this compiles to null-backend stubs (fluke_int8_select returns
// NULL) and callers keep fp16.

#include <fluke/fluke.h>

#include <stdio.h>
#include <stddef.h>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>  /* defines CUDART_VERSION */
#endif

#if defined(HAVE_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(FLUKE_NO_FUSED)

#include "i8_ops.h"
/* Baked model dims (shared with the HIP backend — model-specific, not arch-specific).
   fluke_int8_select verifies the requested dims against these and bows out (NULL) on mismatch. */
#include "fused_dims.h"

/* Per-arch vtables (defined in src/i8_arch.c, one object per SM arch). */
extern const struct fluke_i8_ops fluke_i8_ops_sm80;

/* cc -> arch vtable. sm_80 cubins run on Ampere (80/86) and Ada (89); add {90, &..._sm90} etc.
   as arches ship. Exact-match on compute capability (mirrors the HIP gcnArchName table). */
struct fluke_cc_row { int cc; const struct fluke_i8_ops *ops; };
static const struct fluke_cc_row g_archs[] = {
    { 80, &fluke_i8_ops_sm80 },
    { 86, &fluke_i8_ops_sm80 },
    { 89, &fluke_i8_ops_sm80 },
};

static const struct fluke_i8_ops *ops_for_cc(int cc) {
    size_t i;
    for (i = 0; i < sizeof(g_archs) / sizeof(g_archs[0]); ++i)
        if (g_archs[i].cc == cc) return g_archs[i].ops;
    return NULL;
}

struct fluke_int8_backend {
    fluke_dims_t dims;
    int cc;
    const struct fluke_i8_ops *ops;
};

static int g_modules_loaded = 0;

fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims) {
    int major = 0, minor = 0, cc;
    const struct fluke_i8_ops *ops;
    static struct fluke_int8_backend b;  /* process-lifetime; all layers share it */

    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess) {
        return NULL;
    }
    cc = major * 10 + minor;
    ops = ops_for_cc(cc);
    if (!ops) {
        fprintf(stderr, "[fluke] no int8 backend for compute capability %d.%d — using fp16\n", major, minor);
        return NULL;
    }
    if (dims.d_model != FLUKE_SUP_D_MODEL || dims.dim_feedforward != FLUKE_SUP_DIM_FF ||
        dims.nhead != FLUKE_SUP_NHEAD || dims.head_dim != FLUKE_SUP_HEAD_DIM) {
        fprintf(stderr, "[fluke] model dims do not match precompiled kernels "
                        "(need d_model=%d, dim_feedforward=%d, nhead=%d, head_dim=%d) — using fp16\n",
                FLUKE_SUP_D_MODEL, FLUKE_SUP_DIM_FF, FLUKE_SUP_NHEAD, FLUKE_SUP_HEAD_DIM);
        return NULL;
    }

    if (!g_modules_loaded) {
        if (ops->load() != 0) return NULL;
        g_modules_loaded = 1;
        fprintf(stderr, "[fluke] int8 kernel backend active on device %d (sm_%d)\n", device_index, cc);
    }

    b.dims = dims;
    b.cc   = cc;
    b.ops  = ops;
    return &b;
}

fluke_dims_t fluke_int8_dims(const fluke_int8_backend_t *b) { return b->dims; }

int fluke_qkv_rotary_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *wqkv_i8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
) {
    return b->ops->qkv_rotary(out, a_i8, wqkv_i8, scale_a, scale_b, sin, cos, M, seqlen);
}

int fluke_gated_mlp_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *gate_i8, const void *up_i8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
) {
    return b->ops->gated_mlp(out, a_i8, gate_i8, up_i8, scale_a, scale_gate, scale_up, M);
}

#else  /* no CUDA-12 fused-kernel support — null backend; callers keep the fp16 path. */

fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims) {
    (void)device_index; (void)dims;
    return NULL;
}
fluke_dims_t fluke_int8_dims(const fluke_int8_backend_t *b) {
    fluke_dims_t d = {0,0,0,0,0}; (void)b; return d;
}
int fluke_qkv_rotary_i8_gpu(const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *wqkv_i8, const void *scale_a, const void *scale_b,
    const void *sin, const void *cos, int M, int seqlen) {
    (void)b;(void)out;(void)a_i8;(void)wqkv_i8;(void)scale_a;(void)scale_b;(void)sin;(void)cos;(void)M;(void)seqlen;
    return -1;
}
int fluke_gated_mlp_i8_gpu(const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *gate_i8, const void *up_i8,
    const void *scale_a, const void *scale_gate, const void *scale_up, int M) {
    (void)b;(void)out;(void)a_i8;(void)gate_i8;(void)up_i8;(void)scale_a;(void)scale_gate;(void)scale_up;(void)M;
    return -1;
}

#endif
