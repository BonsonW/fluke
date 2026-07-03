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
#include "artifacts/sm80/factored_lstm_i8_H1024_Khh128_R128.h"
#include "artifacts/sm80/down_proj_i8_R128_K1024.h"

// Baked kernel dims (shared with the HIP backend — model-specific, not arch-specific).
// fluke_int8_select verifies the requested dims against these and bows out (NULL) on mismatch.
#include "fused_dims.h"

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
    if (dims.d_model != FLUKE_SUP_D_MODEL || dims.dim_feedforward != FLUKE_SUP_DIM_FF ||
        dims.nhead != FLUKE_SUP_NHEAD || dims.head_dim != FLUKE_SUP_HEAD_DIM) {
        fprintf(stderr, "[fluke] model dims do not match precompiled kernels "
                        "(need d_model=%d, dim_feedforward=%d, nhead=%d, head_dim=%d) — using fp16\n",
                FLUKE_SUP_D_MODEL, FLUKE_SUP_DIM_FF, FLUKE_SUP_NHEAD, FLUKE_SUP_HEAD_DIM);
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

// ── Factored-LSTM (CRF/LSTM model): int8 down-projection + fused step ──────────
struct fluke_flstm_backend {
    int H, K_hh, R;
    int cc;
};

static factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t g_lstm_step_module;
static down_proj_i8_R128_K1024_Kernel_Module_t            g_down_proj_module;
static int g_lstm_loaded = 0;

fluke_flstm_backend_t *fluke_flstm_select(int device_index, int H, int K_hh, int R) {
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_index) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_index) != cudaSuccess) {
        return NULL;
    }
    const int cc = major * 10 + minor;
    if (cc != 80 && cc != 86 && cc != 89) {
        fprintf(stderr, "[fluke] no int8 factored-LSTM backend for compute capability %d.%d — using fp16\n", major, minor);
        return NULL;
    }
    if (H != FLUKE_LSTM_H || K_hh != FLUKE_LSTM_K_HH || R != FLUKE_LSTM_R) {
        fprintf(stderr, "[fluke] LSTM dims do not match precompiled kernels "
                        "(need H=%d, K_hh=%d, R=%d) — using fp16\n",
                FLUKE_LSTM_H, FLUKE_LSTM_K_HH, FLUKE_LSTM_R);
        return NULL;
    }
    if (!g_lstm_loaded) {
        factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_Load(&g_lstm_step_module);
        down_proj_i8_R128_K1024_Kernel_Module_Load(&g_down_proj_module);
        g_lstm_loaded = 1;
        fprintf(stderr, "[fluke] int8 factored-LSTM backend active on device %d (sm_%d)\n", device_index, cc);
    }
    static fluke_flstm_backend_t b;  // process-lifetime; all layers share it
    b.H = H; b.K_hh = K_hh; b.R = R; b.cc = cc;
    return &b;
}

int fluke_down_proj_i8_gpu(
    const fluke_flstm_backend_t *b, void *out,
    const void *a_i8, const void *w_i8,
    const void *scale_a, const void *scale_b,
    int M
) {
    const int H = b->H;   // contraction K
    const int R = b->R;   // output N

    down_proj_i8_R128_K1024_Tensor_mA_t mA;
    mA.data = (void *)a_i8; fill_desc2d(mA.dynamic_shapes, mA.dynamic_strides, M, H);
    down_proj_i8_R128_K1024_Tensor_mB_t mB;
    mB.data = (void *)w_i8; fill_desc2d(mB.dynamic_shapes, mB.dynamic_strides, R, H);
    down_proj_i8_R128_K1024_Tensor_mC_t mC;
    mC.data = out;          fill_desc2d(mC.dynamic_shapes, mC.dynamic_strides, M, R);
    down_proj_i8_R128_K1024_Tensor_mScaleA_t mScaleA; mScaleA.data = (void *)scale_a;
    down_proj_i8_R128_K1024_Tensor_mScaleB_t mScaleB; mScaleB.data = (void *)scale_b;

    return cute_dsl_down_proj_i8_R128_K1024_wrapper(
        &g_down_proj_module, &mA, &mB, &mC, &mScaleA, &mScaleB);
}

int fluke_flstm_step_i8_gpu(
    const fluke_flstm_backend_t *b, void *h_i8,
    const void *a_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, int B
) {
    const int H = b->H;
    const int Kc = b->K_hh + b->R;   // merged f16 up-proj contraction
    #define FL_(s) factored_lstm_i8_H1024_Khh128_R128_Tensor_##s

    FL_(mA_t)   mA;   mA.data   = (void *)a_f16; fill_desc2d(mA.dynamic_shapes,   mA.dynamic_strides,   B, Kc);
    FL_(mB_i_t) mB_i; mB_i.data = (void *)Bi;    fill_desc2d(mB_i.dynamic_shapes, mB_i.dynamic_strides, H, Kc);
    FL_(mB_f_t) mB_f; mB_f.data = (void *)Bf;    fill_desc2d(mB_f.dynamic_shapes, mB_f.dynamic_strides, H, Kc);
    FL_(mB_g_t) mB_g; mB_g.data = (void *)Bg;    fill_desc2d(mB_g.dynamic_shapes, mB_g.dynamic_strides, H, Kc);
    FL_(mB_o_t) mB_o; mB_o.data = (void *)Bo;    fill_desc2d(mB_o.dynamic_shapes, mB_o.dynamic_strides, H, Kc);
    FL_(mBias_i_t) mBias_i; mBias_i.data = (void *)bias_i;
    FL_(mBias_f_t) mBias_f; mBias_f.data = (void *)bias_f;
    FL_(mBias_g_t) mBias_g; mBias_g.data = (void *)bias_g;
    FL_(mBias_o_t) mBias_o; mBias_o.data = (void *)bias_o;
    FL_(mC_c_t)   mC_c;   mC_c.data   = c_f32; fill_desc2d(mC_c.dynamic_shapes,   mC_c.dynamic_strides,   B, H);
    FL_(mH_out_t) mH_out; mH_out.data = h_i8;  fill_desc2d(mH_out.dynamic_shapes, mH_out.dynamic_strides, B, H);
    #undef FL_

    return cute_dsl_factored_lstm_i8_H1024_Khh128_R128_wrapper(
        &g_lstm_step_module, &mA, &mB_i, &mB_f, &mB_g, &mB_o,
        &mBias_i, &mBias_f, &mBias_g, &mBias_o, &mC_c, &mH_out);
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
fluke_flstm_backend_t *fluke_flstm_select(int device_index, int H, int K_hh, int R) {
    (void)device_index; (void)H; (void)K_hh; (void)R;
    return NULL;
}
int fluke_down_proj_i8_gpu(const fluke_flstm_backend_t *b, void *out,
    const void *a_i8, const void *w_i8, const void *scale_a, const void *scale_b, int M) {
    (void)b;(void)out;(void)a_i8;(void)w_i8;(void)scale_a;(void)scale_b;(void)M;
    return -1;
}
int fluke_flstm_step_i8_gpu(const fluke_flstm_backend_t *b, void *h_i8,
    const void *a_f16, const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, int B) {
    (void)b;(void)h_i8;(void)a_f16;(void)Bi;(void)Bf;(void)Bg;(void)Bo;
    (void)bias_i;(void)bias_f;(void)bias_g;(void)bias_o;(void)c_f32;(void)B;
    return -1;
}

#endif
