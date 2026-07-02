// Fused INT8 DSL kernels — arch dispatch + module load + descriptor plumbing.
//
// Implements the ATen-free C ABI declared in <fluke/fluke.h> (fluke_int8_select,
// fluke_qkv_rotary_i8_gpu, fluke_gated_mlp_i8_gpu) over the AOT-exported CuTe kernels
// (artifacts/<arch>/*.h + *.o, bundled into libfluke.a). This is the code that used to
// live in slorado's thirdparty/fluke/fluke.cpp — moved here so a single libfluke.a serves
// slorado and adding an arch/backend is fluke-internal.
//
// The generated headers use CUDA 12's library-management API (cudaLibrary_t,
// cudaLibraryLoadData, cudaLaunchKernelEx). On older toolkits or non-CUDA builds this
// compiles to null-backend stubs (fluke_int8_select returns NULL) and callers keep fp16.

#include <fluke/fluke.h>

#include <stdio.h>
#include <stdlib.h>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>  // defines CUDART_VERSION
#endif

#if defined(HAVE_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000

#include "artifacts/sm80/gemm_i8_rotary_N1536_K512_H8D64R64S2048.h"
#include "artifacts/sm80/gemm_i8_dual_silu_N2048_K512.h"

// Baked kernel dims (specialized into the cubin symbols). fluke_int8_select verifies the
// requested dims against these and bows out (NULL) on mismatch.
#define FLUKE_SM80_D_MODEL      512
#define FLUKE_SM80_DIM_FF       2048
#define FLUKE_SM80_NHEAD        8
#define FLUKE_SM80_HEAD_DIM     64

struct fluke_int8_backend {
    fluke_dims_t dims;
    int cc;  // compute capability major*10+minor
};

// Process-global kernel modules, loaded once by fluke_int8_select.
static gemm_i8_rotary_N1536_K512_H8D64R64S2048_Kernel_Module_t g_rotary_module;
static gemm_i8_dual_silu_N2048_K512_Kernel_Module_t            g_mlp_module;
static int g_modules_loaded = 0;

// Fill a 2D [rows, cols] row-major descriptor (dynamic_shapes[3] / dynamic_strides[2]).
// The kernels expect [M,K] memrefs with innermost stride 1; the caller (fluke_wrapper on
// the slorado side) guarantees contiguity by reshaping before the call.
static void fill_desc2d(int32_t shapes[3], int64_t strides[2], int rows, int cols) {
    shapes[0] = rows; shapes[1] = cols; shapes[2] = 1;
    strides[0] = cols; strides[1] = 1;
}

fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims) {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess) {
        return NULL;
    }
    const int cc = major * 10 + minor;
    // sm_80 cubins run on Ampere (80/86) and Ada (89). Add sm90/gfx branches here as arches ship.
    if (cc != 80 && cc != 86 && cc != 89) {
        fprintf(stderr, "[fluke] no int8 backend for compute capability %d.%d — using fp16\n", major, minor);
        return NULL;
    }
    if (dims.d_model != FLUKE_SM80_D_MODEL || dims.dim_feedforward != FLUKE_SM80_DIM_FF ||
        dims.nhead != FLUKE_SM80_NHEAD || dims.head_dim != FLUKE_SM80_HEAD_DIM) {
        fprintf(stderr, "[fluke] model dims do not match precompiled kernels "
                        "(need d_model=%d, dim_feedforward=%d, nhead=%d, head_dim=%d) — using fp16\n",
                FLUKE_SM80_D_MODEL, FLUKE_SM80_DIM_FF, FLUKE_SM80_NHEAD, FLUKE_SM80_HEAD_DIM);
        return NULL;
    }

    if (!g_modules_loaded) {
        gemm_i8_rotary_N1536_K512_H8D64R64S2048_Kernel_Module_Load(&g_rotary_module);
        gemm_i8_dual_silu_N2048_K512_Kernel_Module_Load(&g_mlp_module);
        g_modules_loaded = 1;
        fprintf(stderr, "[fluke] int8 kernel backend active on device %d (sm_%d)\n", device_index, cc);
    }

    static fluke_int8_backend_t b;  // process-lifetime; all layers share it
    b.dims = dims;
    b.cc = cc;
    return &b;
}

int fluke_qkv_rotary_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *wqkv_i8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
) {
    const int d_model = b->dims.d_model;
    const int N = 3 * d_model;

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mA_t mA;
    mA.data = (void *)a_i8;   fill_desc2d(mA.dynamic_shapes, mA.dynamic_strides, M, d_model);
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mB_t mB;
    mB.data = (void *)wqkv_i8; fill_desc2d(mB.dynamic_shapes, mB.dynamic_strides, N, d_model);
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mC_t mC;
    mC.data = out;            fill_desc2d(mC.dynamic_shapes, mC.dynamic_strides, M, N);

    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mScaleA_t mScaleA; mScaleA.data = (void *)scale_a;
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mScaleB_t mScaleB; mScaleB.data = (void *)scale_b;
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mSin_t    mSin;    mSin.data    = (void *)sin;
    gemm_i8_rotary_N1536_K512_H8D64R64S2048_Tensor_mCos_t    mCos;    mCos.data    = (void *)cos;

    // Runtime seqlen: kernel indexes rotary as seq = row % seqlen (any seqlen <= baked max_seq).
    return cute_dsl_gemm_i8_rotary_N1536_K512_H8D64R64S2048_wrapper(
        &g_rotary_module, &mA, &mB, &mC, &mScaleA, &mScaleB, &mSin, &mCos, (int32_t)seqlen);
}

int fluke_gated_mlp_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *gate_i8, const void *up_i8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
) {
    const int d_model = b->dims.d_model;
    const int dim_ff = b->dims.dim_feedforward;

    gemm_i8_dual_silu_N2048_K512_Tensor_mA_t mA;
    mA.data = (void *)a_i8;    fill_desc2d(mA.dynamic_shapes, mA.dynamic_strides, M, d_model);
    gemm_i8_dual_silu_N2048_K512_Tensor_mB_gate_t mB_gate;
    mB_gate.data = (void *)gate_i8; fill_desc2d(mB_gate.dynamic_shapes, mB_gate.dynamic_strides, dim_ff, d_model);
    gemm_i8_dual_silu_N2048_K512_Tensor_mB_up_t mB_up;
    mB_up.data = (void *)up_i8;     fill_desc2d(mB_up.dynamic_shapes, mB_up.dynamic_strides, dim_ff, d_model);
    gemm_i8_dual_silu_N2048_K512_Tensor_mC_t mC;
    mC.data = out;             fill_desc2d(mC.dynamic_shapes, mC.dynamic_strides, M, dim_ff);

    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleA_t      mScaleA;    mScaleA.data      = (void *)scale_a;
    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleB_gate_t mScaleB_gate; mScaleB_gate.data = (void *)scale_gate;
    gemm_i8_dual_silu_N2048_K512_Tensor_mScaleB_up_t   mScaleB_up; mScaleB_up.data   = (void *)scale_up;

    return cute_dsl_gemm_i8_dual_silu_N2048_K512_wrapper(
        &g_mlp_module, &mA, &mB_gate, &mB_up, &mC, &mScaleA, &mScaleB_gate, &mScaleB_up);
}

#else  // no CUDA-12 fused-kernel support — null backend; callers keep the fp16 path.

fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims) {
    (void)device_index; (void)dims;
    return NULL;
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
