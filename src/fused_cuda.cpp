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
#include <vector>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>  // defines CUDART_VERSION
#endif

#if defined(HAVE_CUDA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000

#include "artifacts/sm80/gemm_i8_rotary_N1536_K512_H8D64R64S2048.h"
#include "artifacts/sm80/gemm_i8_dual_silu_N2048_K512.h"
#include "artifacts/sm80/factored_lstm_i8_H1024_Khh128_R128.h"
#include "artifacts/sm80/factored_lstm_fused_i8_H1024_Khh128_R128.h"
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

fluke_dims_t fluke_int8_dims(const fluke_int8_backend_t *b) { return b->dims; }

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
    int max_fused_N;   // largest N whose fused-step grid is fully co-resident on THIS device (else 2-kernel)
};

static factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t       g_lstm_step_module;
static factored_lstm_fused_i8_H1024_Khh128_R128_Kernel_Module_t g_lstm_fused_module;
static down_proj_i8_R128_K1024_Kernel_Module_t                  g_down_proj_module;
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
        factored_lstm_fused_i8_H1024_Khh128_R128_Kernel_Module_Load(&g_lstm_fused_module);
        down_proj_i8_R128_K1024_Kernel_Module_Load(&g_down_proj_module);
        g_lstm_loaded = 1;
        fprintf(stderr, "[fluke] int8 factored-LSTM backend active on device %d (sm_%d)\n", device_index, cc);
    }
    // Largest N the fused single-launch step can run without deadlock: its whole grid must be
    // CO-RESIDENT (consumers spin on same-grid producers). Grid = (N/64)*(H/32) CTAs. This kernel is
    // register-limited to 3 CTAs/SM on all supported archs, but its ~50KB smem/CTA caps sm86/89
    // (100KB smem/SM) at 2 CTAs/SM; sm80 (164KB/SM) allows 3. Scale by the queried SM count and cap
    // at the ~512 perf crossover. Decided ONCE here (per device) so the recurrence stays kernel- and
    // device-agnostic; a different arch's recurrence kernel sets its own bound in its own select.
    int sm_count = 1;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_index);
    const int ctas_per_sm = (cc == 80) ? 3 : 2;
    const int cols = H / 32;                                   // consumer CTAs per fused row-group (bN=32)
    int max_n = (sm_count * ctas_per_sm / cols) * 64;          // rows whose grid is fully co-resident
    if (max_n > 512) max_n = 512;                              // fused stops winning above ~512 rows
    if (max_n < 0)  max_n = 0;

    static fluke_flstm_backend_t b;  // process-lifetime; all layers share it
    b.H = H; b.K_hh = K_hh; b.R = R; b.cc = cc; b.max_fused_N = max_n;
    fprintf(stderr, "[fluke] fused-step co-residency bound on this device: N<=%d (%d SMs, %d CTAs/SM)\n",
            max_n, sm_count, ctas_per_sm);
    return &b;
}

int fluke_down_proj_i8_gpu(
    const fluke_flstm_backend_t *b, void *out,
    const void *a_i8, const void *w_i8,
    const void *scale_a, const void *scale_b,
    int M, void *stream
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
        &g_down_proj_module, &mA, &mB, &mC, &mScaleA, &mScaleB, (cudaStream_t)stream);
}

int fluke_flstm_step_i8_gpu(
    const fluke_flstm_backend_t *b, void *h_i8,
    const void *a_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, int B, void *stream
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
        &mBias_i, &mBias_f, &mBias_g, &mBias_o, &mC_c, &mH_out, (cudaStream_t)stream);
}

// Max rows per fused launch. The consumers spin on same-grid producers, so the whole
// grid must be co-resident: grid = ceil(rows/bM)*(H/bN) CTAs. At the baked config
// (bM=64, bN=32, H=1024 -> 32 CTAs/row-group, 3 CTAs/SM on A100 = 324 resident), 512
// rows -> 8*32=256 CTAs, safely under the limit. Larger B is tiled into 512-row chunks.
// Must be a multiple of bM (64) so every chunk stays tile-aligned (the kernel has no
// M-boundary predication); the caller guarantees B % 64 == 0.
#define FLUKE_FUSED_CHUNK 512
#define FLUKE_FUSED_BM 64

int fluke_flstm_fused_step_i8_gpu(
    const fluke_flstm_backend_t *b, void *h_i8,
    const void *h_prev_i8, const void *w_dn_i8, const void *comb_scale, const void *x_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, void *hh_stage, void *flags, int B, void *stream
) {
    const int H = b->H;
    const int K_hh = b->K_hh;
    const int R = b->R;
    const int Kc = K_hh + R;
    #define FF_(s) factored_lstm_fused_i8_H1024_Khh128_R128_Tensor_##s

    // Row-tile B into <= FLUKE_FUSED_CHUNK-row launches (grid co-residency). Each chunk is
    // independent (batch dim), row-offset into the per-token tensors; the weights, biases,
    // combined scale, and flags are shared (flags self-clean per launch, so the same buffer
    // is reused across chunks — launches are serial on the stream).
    for (int r0 = 0; r0 < B; r0 += FLUKE_FUSED_CHUNK) {
        const int rows = (B - r0 < FLUKE_FUSED_CHUNK) ? (B - r0) : FLUKE_FUSED_CHUNK;
        const char *hp  = (const char *)h_prev_i8 + (size_t)r0 * H * 1;      // int8 [B,H]
        const char *xp  = (const char *)x_f16     + (size_t)r0 * R * 2;      // f16  [B,R]
        char       *cp  = (char *)c_f32           + (size_t)r0 * H * 4;      // f32  [B,H]
        char       *hop = (char *)h_i8            + (size_t)r0 * H * 1;      // int8 [B,H]
        char       *hhp = (char *)hh_stage        + (size_t)r0 * K_hh * 2;   // f16  [B,K_hh]

        FF_(mHprev_t)  mHprev;  mHprev.data = (void *)hp; fill_desc2d(mHprev.dynamic_shapes, mHprev.dynamic_strides, rows, H);
        FF_(mWdn_t)    mWdn;    mWdn.data   = (void *)w_dn_i8;   fill_desc2d(mWdn.dynamic_shapes,   mWdn.dynamic_strides,   K_hh, H);
        FF_(mWdnScale_t) mWdnScale; mWdnScale.data = (void *)comb_scale;
        FF_(mX_t)      mX;      mX.data     = (void *)xp;        fill_desc2d(mX.dynamic_shapes,     mX.dynamic_strides,     rows, R);
        FF_(mB_i_t) mB_i; mB_i.data = (void *)Bi; fill_desc2d(mB_i.dynamic_shapes, mB_i.dynamic_strides, H, Kc);
        FF_(mB_f_t) mB_f; mB_f.data = (void *)Bf; fill_desc2d(mB_f.dynamic_shapes, mB_f.dynamic_strides, H, Kc);
        FF_(mB_g_t) mB_g; mB_g.data = (void *)Bg; fill_desc2d(mB_g.dynamic_shapes, mB_g.dynamic_strides, H, Kc);
        FF_(mB_o_t) mB_o; mB_o.data = (void *)Bo; fill_desc2d(mB_o.dynamic_shapes, mB_o.dynamic_strides, H, Kc);
        FF_(mBias_i_t) mBias_i; mBias_i.data = (void *)bias_i;
        FF_(mBias_f_t) mBias_f; mBias_f.data = (void *)bias_f;
        FF_(mBias_g_t) mBias_g; mBias_g.data = (void *)bias_g;
        FF_(mBias_o_t) mBias_o; mBias_o.data = (void *)bias_o;
        FF_(mC_c_t)   mC_c;   mC_c.data   = cp;   fill_desc2d(mC_c.dynamic_shapes,   mC_c.dynamic_strides,   rows, H);
        FF_(mH_out_t) mH_out; mH_out.data = hop;  fill_desc2d(mH_out.dynamic_shapes, mH_out.dynamic_strides, rows, H);
        FF_(mHH_t)    mHH;    mHH.data    = hhp;  fill_desc2d(mHH.dynamic_shapes,    mHH.dynamic_strides,    rows, K_hh);
        FF_(mFlags_t) mFlags; mFlags.data = flags;

        int rc = cute_dsl_factored_lstm_fused_i8_H1024_Khh128_R128_wrapper(
            &g_lstm_fused_module, &mHprev, &mWdn, &mWdnScale, &mX,
            &mB_i, &mB_f, &mB_g, &mB_o, &mBias_i, &mBias_f, &mBias_g, &mBias_o,
            &mC_c, &mH_out, &mHH, &mFlags, (cudaStream_t)stream);
        if (rc != 0) return rc;
    }
    #undef FF_
    return 0;
}

// ── Unified recurrence: fluke owns the T-step loop + CUDA graph + fused/two-kernel choice ─────────
struct fluke_flstm_rec {
    const fluke_flstm_backend_t *b;
    int N, T, num_layers, C, K;
    bool use_fused;                        // per-N/device kernel choice (baked at create)
    cudaStream_t cap;                      // private capture/replay stream
    std::vector<cudaGraphExec_t> graph;    // one per layer, NULL until captured
    void *hh_stage; void *flags;           // fused scratch
    void *hh_down;  void *a_scratch; float *ones;   // two-kernel scratch (ones = scale_a == 1)
};

fluke_flstm_rec_t *fluke_flstm_rec_create(const fluke_flstm_backend_t *b, int N, int T, int num_layers) {
    if (!b) return NULL;
    fluke_flstm_rec_t *r = new fluke_flstm_rec();
    r->b = b; r->N = N; r->T = T; r->num_layers = num_layers; r->C = b->H; r->K = b->K_hh;
    // Path choice = pure capability lookup: the backend already computed (per device, at select) the
    // largest N whose fused grid is co-resident. No kernel-resource math or magic 512 here — a
    // different arch/kernel just reports a different bound. N must also be a multiple of the 64-row
    // tile (the fused kernel has no M-boundary predication).
    r->use_fused = (N <= b->max_fused_N) && (N % 64 == 0);
    cudaStreamCreateWithFlags(&r->cap, cudaStreamNonBlocking);
    r->graph.assign(num_layers > 0 ? num_layers : 1, (cudaGraphExec_t)NULL);
    r->hh_stage = r->flags = r->hh_down = r->a_scratch = NULL; r->ones = NULL;
    if (r->use_fused) {
        const int grid_m = (N + 63) / 64;
        cudaMalloc(&r->hh_stage, (size_t)N * r->K * 2);
        cudaMalloc(&r->flags,    (size_t)grid_m * 4 * sizeof(int));
        cudaMemset(r->flags, 0,  (size_t)grid_m * 4 * sizeof(int));   // zeroed once; self-cleaning
    } else {
        cudaMalloc(&r->hh_down,   (size_t)N * r->K * 2);
        cudaMalloc(&r->a_scratch, (size_t)N * 2 * r->K * 2);
        cudaMalloc(&r->ones,      (size_t)N * sizeof(float));
        std::vector<float> h(N, 1.0f);
        cudaMemcpy(r->ones, h.data(), (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    }
    return r;
}

void fluke_flstm_rec_free(fluke_flstm_rec_t *r) {
    if (!r) return;
    for (auto g : r->graph) if (g) cudaGraphExecDestroy(g);
    if (r->cap) cudaStreamDestroy(r->cap);
    cudaFree(r->hh_stage); cudaFree(r->flags);
    cudaFree(r->hh_down);  cudaFree(r->a_scratch); cudaFree(r->ones);
    delete r;
}

int fluke_flstm_recurrence(
    fluke_flstm_rec_t *r, int layer_idx,
    void *hh_all, void *cell, const void *x_down,
    const void *w_dn, const void *comb_scale,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    int reverse, void *stream
) {
    if (!r || layer_idx < 0 || layer_idx >= (int)r->graph.size()) return -1;
    const int N = r->N, T = r->T, C = r->C, K = r->K;
    cudaStream_t cs  = r->cap;              // capture stream (can't capture the default stream)
    cudaStream_t run = (cudaStream_t)stream; // caller's stream (may be the default stream, handle 0):
                                             // memset + graph LAUNCH here. Must NOT fall back to cs —
                                             // if the caller runs on the default stream, launching the
                                             // graph on cs instead would race the precompute (the bug
                                             // the per-layer syncs were masking).

    // Boundary hidden + cell = 0 on the caller's stream, right before the graph launch (the graph
    // re-reads them each replay). Everything runs on `run`, so it is naturally ordered after the
    // caller's x_down precompute and before the next layer — no event handshake / full-device sync.
    const int boundary = reverse ? T : 0;
    cudaMemsetAsync((char *)hh_all + (size_t)boundary * N * C, 0, (size_t)N * C, run);
    cudaMemsetAsync(cell, 0, (size_t)N * C * sizeof(float), run);

    // One recurrence step at ring index i; only fluke launches + D2D copies (capture-safe). The
    // launches target `cs` because they run only during capture; the instantiated graph then replays
    // on whatever stream cudaGraphLaunch is given (`run`).
    auto step = [&](int i) {
        const int t    = reverse ? (T - 1 - i) : i;
        const int prev = reverse ? (t + 1) : t;
        const int out  = reverse ? t : (t + 1);
        void *h_prev = (char *)hh_all + (size_t)prev * N * C;
        void *h_out  = (char *)hh_all + (size_t)out  * N * C;
        const void *x_t = (const char *)x_down + (size_t)t * N * K * 2;   // f16
        if (r->use_fused) {
            fluke_flstm_fused_step_i8_gpu(r->b, h_out, h_prev, w_dn, comb_scale, x_t,
                Bi, Bf, Bg, Bo, bias_i, bias_f, bias_g, bias_o, cell, r->hh_stage, r->flags, N, cs);
        } else {
            // down_proj with scale_a = ones, scale_b = comb_scale  ->  hh_down = comb_scale * (h@w)
            // (identical to fused's per-channel scale), then concat [hh_down | x] via D2D, then step.
            fluke_down_proj_i8_gpu(r->b, r->hh_down, h_prev, w_dn, r->ones, comb_scale, N, cs);
            const size_t Kb = (size_t)K * 2;
            cudaMemcpy2DAsync(r->a_scratch,              2 * Kb, r->hh_down, Kb, Kb, N, cudaMemcpyDeviceToDevice, cs);
            cudaMemcpy2DAsync((char *)r->a_scratch + Kb, 2 * Kb, x_t,        Kb, Kb, N, cudaMemcpyDeviceToDevice, cs);
            fluke_flstm_step_i8_gpu(r->b, h_out, r->a_scratch,
                Bi, Bf, Bg, Bo, bias_i, bias_f, bias_g, bias_o, cell, N, cs);
        }
    };

    if (r->graph[layer_idx] == NULL) {
        cudaGraph_t g = NULL;
        if (cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return -2;
        for (int i = 0; i < T; ++i) step(i);
        if (cudaStreamEndCapture(cs, &g) != cudaSuccess) return -3;
        if (cudaGraphInstantiate(&r->graph[layer_idx], g, 0) != cudaSuccess) return -4;
        cudaGraphDestroy(g);
    }
    cudaGraphLaunch(r->graph[layer_idx], run);   // replay on the caller's stream (natural ordering)
    return 0;
}

#else  // no CUDA-12 fused-kernel support — null backend; callers keep the fp16 path.

fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims) {
    (void)device_index; (void)dims;
    return NULL;
}
fluke_dims_t fluke_int8_dims(const fluke_int8_backend_t *b) { (void)b; fluke_dims_t d = {0,0,0,0,0}; return d; }
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
    const void *a_i8, const void *w_i8, const void *scale_a, const void *scale_b, int M, void *stream) {
    (void)b;(void)out;(void)a_i8;(void)w_i8;(void)scale_a;(void)scale_b;(void)M;(void)stream;
    return -1;
}
int fluke_flstm_step_i8_gpu(const fluke_flstm_backend_t *b, void *h_i8,
    const void *a_f16, const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, int B, void *stream) {
    (void)b;(void)h_i8;(void)a_f16;(void)Bi;(void)Bf;(void)Bg;(void)Bo;
    (void)bias_i;(void)bias_f;(void)bias_g;(void)bias_o;(void)c_f32;(void)B;(void)stream;
    return -1;
}
int fluke_flstm_fused_step_i8_gpu(const fluke_flstm_backend_t *b, void *h_i8,
    const void *h_prev_i8, const void *w_dn_i8, const void *comb_scale, const void *x_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, void *hh_stage, void *flags, int B, void *stream) {
    (void)b;(void)h_i8;(void)h_prev_i8;(void)w_dn_i8;(void)comb_scale;(void)x_f16;
    (void)Bi;(void)Bf;(void)Bg;(void)Bo;(void)bias_i;(void)bias_f;(void)bias_g;(void)bias_o;
    (void)c_f32;(void)hh_stage;(void)flags;(void)B;(void)stream;
    return -1;
}
fluke_flstm_rec_t *fluke_flstm_rec_create(const fluke_flstm_backend_t *b, int N, int T, int num_layers) {
    (void)b;(void)N;(void)T;(void)num_layers; return NULL;
}
void fluke_flstm_rec_free(fluke_flstm_rec_t *rec) { (void)rec; }
int fluke_flstm_recurrence(fluke_flstm_rec_t *rec, int layer_idx,
    void *hh_all, void *cell, const void *x_down, const void *w_dn, const void *comb_scale,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    int reverse, void *stream) {
    (void)rec;(void)layer_idx;(void)hh_all;(void)cell;(void)x_down;(void)w_dn;(void)comb_scale;
    (void)Bi;(void)Bf;(void)Bg;(void)Bo;(void)bias_i;(void)bias_f;(void)bias_g;(void)bias_o;
    (void)reverse;(void)stream;
    return -1;
}

#endif
