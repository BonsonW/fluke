// Per-arch fused-INT8 launch wrappers (CUDA) — compiled ONCE PER SM ARCH from this single
// source. Mirror of src/fp8_arch.c. One TU per arch is required (each arch's generated headers
// define the same CuTe descriptor typedefs / Module types), so the Makefile compiles this source
// once per arch with:
//   -DFLUKE_I8_ARCH_NAME=<smNN>                 the arch tag (names the exported vtable)
//   -DFLUKE_I8_ROTARY_HDR="artifacts/.../....h" this arch's two generated headers
//   -DFLUKE_I8_MLP_HDR="artifacts/.../....h"
// Each object exports one external symbol, the vtable `fluke_i8_ops_<arch>`, bound by compute
// capability in src/fused_cuda.c. Host code (drives the CUDA library/launch API — the device
// kernels live in the linked-in CuTe .o), so it builds with the host C compiler. See src/i8_ops.h.
#include <fluke/fluke.h>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>  /* defines CUDART_VERSION */
#endif

#if defined(HAVE_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000 && !defined(FLUKE_NO_FUSED)

#include "i8_ops.h"
#include FLUKE_I8_ROTARY_HDR    /* gemm_i8_rotary_*    : Module_t / Module_Load / Tensor_* / _wrapper */
#include FLUKE_I8_MLP_HDR       /* gemm_i8_dual_silu_* */
#include "fused_dims.h"         /* baked model dims (FLUKE_SUP_*) */

#ifndef FLUKE_I8_ARCH_NAME
#error "i8_arch.c: define FLUKE_I8_ARCH_NAME (e.g. sm80) — see the Makefile rule"
#endif

#define FLUKE_CAT_(a, b) a##b
#define FLUKE_CAT(a, b) FLUKE_CAT_(a, b)
#define FLUKE_OPS_SYM FLUKE_CAT(fluke_i8_ops_, FLUKE_I8_ARCH_NAME)

/* Generated kernel-name prefixes (they bake the model dims). Kept as macros so this single source
   tracks the dims from one place; if a config changes, update fused_dims.h and these together. */
#define ROTARY_K   gemm_i8_rotary_N1536_K512_H8D64R64S2048
#define MLP_K      gemm_i8_dual_silu_N2048_K512
#define P(pfx, s)  FLUKE_CAT(pfx, s)   /* e.g. P(ROTARY_K, _Tensor_mA_t) */

/* Process-global kernel modules for this arch, bound once by i8_load(). */
static P(ROTARY_K, _Kernel_Module_t) g_rotary_module;
static P(MLP_K,    _Kernel_Module_t) g_mlp_module;

/* Fill a 2D [rows, cols] row-major descriptor (dynamic_shapes[3] / dynamic_strides[2]). The
   kernels expect [M,K] memrefs with innermost stride 1; the caller guarantees contiguity. */
static void fill_desc2d(int32_t shapes[3], int64_t strides[2], int rows, int cols) {
    shapes[0] = rows; shapes[1] = cols; shapes[2] = 1;
    strides[0] = cols; strides[1] = 1;
}

static int i8_load(void) {
    P(ROTARY_K, _Kernel_Module_Load)(&g_rotary_module);
    P(MLP_K,    _Kernel_Module_Load)(&g_mlp_module);
    return 0;
}

static int i8_qkv_rotary(void *out, const void *a_i8, const void *wqkv_i8,
                         const void *scale_a, const void *scale_b,
                         const void *sin, const void *cos, int M, int seqlen) {
    const int d_model = FLUKE_SUP_D_MODEL;
    const int N = 3 * d_model;

    P(ROTARY_K, _Tensor_mA_t) mA; mA.data = (void *)a_i8;
    fill_desc2d(mA.dynamic_shapes, mA.dynamic_strides, M, d_model);
    P(ROTARY_K, _Tensor_mB_t) mB; mB.data = (void *)wqkv_i8;
    fill_desc2d(mB.dynamic_shapes, mB.dynamic_strides, N, d_model);
    P(ROTARY_K, _Tensor_mC_t) mC; mC.data = out;
    fill_desc2d(mC.dynamic_shapes, mC.dynamic_strides, M, N);
    P(ROTARY_K, _Tensor_mScaleA_t) mScaleA; mScaleA.data = (void *)scale_a;
    P(ROTARY_K, _Tensor_mScaleB_t) mScaleB; mScaleB.data = (void *)scale_b;
    P(ROTARY_K, _Tensor_mSin_t)    mSin;    mSin.data    = (void *)sin;
    P(ROTARY_K, _Tensor_mCos_t)    mCos;    mCos.data    = (void *)cos;

    /* Runtime seqlen: kernel indexes rotary as seq = row % seqlen (any seqlen <= baked max_seq). */
    return FLUKE_CAT(cute_dsl_, P(ROTARY_K, _wrapper))(
        &g_rotary_module, &mA, &mB, &mC, &mScaleA, &mScaleB, &mSin, &mCos, (int32_t)seqlen);
}

static int i8_gated_mlp(void *out, const void *a_i8, const void *gate_i8, const void *up_i8,
                        const void *scale_a, const void *scale_gate, const void *scale_up, int M) {
    const int d_model = FLUKE_SUP_D_MODEL;
    const int dim_ff  = FLUKE_SUP_DIM_FF;

    P(MLP_K, _Tensor_mA_t) mA; mA.data = (void *)a_i8;
    fill_desc2d(mA.dynamic_shapes, mA.dynamic_strides, M, d_model);
    P(MLP_K, _Tensor_mB_gate_t) mB_gate; mB_gate.data = (void *)gate_i8;
    fill_desc2d(mB_gate.dynamic_shapes, mB_gate.dynamic_strides, dim_ff, d_model);
    P(MLP_K, _Tensor_mB_up_t) mB_up; mB_up.data = (void *)up_i8;
    fill_desc2d(mB_up.dynamic_shapes, mB_up.dynamic_strides, dim_ff, d_model);
    P(MLP_K, _Tensor_mC_t) mC; mC.data = out;
    fill_desc2d(mC.dynamic_shapes, mC.dynamic_strides, M, dim_ff);
    P(MLP_K, _Tensor_mScaleA_t)      mScaleA;      mScaleA.data      = (void *)scale_a;
    P(MLP_K, _Tensor_mScaleB_gate_t) mScaleB_gate; mScaleB_gate.data = (void *)scale_gate;
    P(MLP_K, _Tensor_mScaleB_up_t)   mScaleB_up;   mScaleB_up.data   = (void *)scale_up;

    return FLUKE_CAT(cute_dsl_, P(MLP_K, _wrapper))(
        &g_mlp_module, &mA, &mB_gate, &mB_up, &mC, &mScaleA, &mScaleB_gate, &mScaleB_up);
}

/* A plain const file-scope object has external linkage in C; fused_cuda.c references this
   symbol by its arch-qualified name. */
const struct fluke_i8_ops FLUKE_OPS_SYM = {
    i8_load, i8_qkv_rotary, i8_gated_mlp
};

#endif  /* HAVE_CUDA && CUDART>=12 && !FLUKE_NO_FUSED */
