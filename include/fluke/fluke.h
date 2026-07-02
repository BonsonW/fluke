#ifndef FLUKE_H
#define FLUKE_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "fluke_error.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rotary position embedding (rotate-half convention).
//
// `sincos_width` is the width of the sin/cos tables: sin_buf and cos_buf MUST be
// laid out as [seq_len, sincos_width]. The kernel rotates 2*sincos_width elements
// of each head, pairing element i with element i+sincos_width:
//     out_i               = x_i*cos_j - x_{i+sincos_width}*sin_j
//     out_{i+sincos_width} = x_i*sin_j + x_{i+sincos_width}*cos_j   (j = i, table col)
// so it rotates 2*sincos_width dims (require 2*sincos_width <= head_dim).
//
// NOTE: fused GEMM+rotary kernels (see cute/ampere) may instead expect FULL-width
// tables [seq_len, 2*sincos_width] (each column duplicated). Do not confuse the two
// — the table's second dimension must always equal the width the consumer documents.
void fluke_rotary_emb_cpu(
    void *x,
    const void *sin_buf,
    const void *cos_buf,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,   // width of sin/cos tables: [seq_len, sincos_width]; rotates 2*sincos_width dims
    int stride_batch,
    int stride_seq,
    int stride_head,
    int n_threads
);

// RMSNorm over a residual add: out = rmsnorm(in + alpha*residual) * weight, where
// rmsnorm(v) = v / sqrt(mean(v^2) + eps), computed per row (token). in/residual are
// [n_tokens, hidden_dim], weight is [hidden_dim]. fp32 math. (CPU reference for the
// non-quant kernel; the quantizing variant is GPU-only.)
void fluke_rmsnorm_cpu(
    void *in,
    const void *residual,
    const void *weight,
    void *out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps,
    int n_threads
);

// fLSTM step (single timestep, elementwise gate epilogue). scratch/ih_t are
// [batch_size, 4*hidden_dim] gate pre-activations; cell [batch_size, hidden_dim] is updated
// in place; hh_next [batch_size, hidden_dim] receives the new hidden state. fp32 CPU reference.
void fluke_flstm_step_cpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim,
    int n_threads
);

// Model dimensions a backend must match (kernels are dimension-specialized). Build-agnostic: the
// ATen wrapper's public API takes it on every build (int8 selection is a no-op on CPU-only builds).
typedef struct { int d_model, dim_feedforward, nhead, head_dim, max_seq; } fluke_dims_t;

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

// out = rmsnorm(in + alpha*residual) * weight. in/residual/weight/out are fp16
// ([n_tokens, hidden_dim]; weight [hidden_dim]); fp32 math. hidden_dim even, <= 1024.
void fluke_rmsnorm_gpu(
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

// Fused RMSNorm + per-token INT8 quantize, in place. `residual` (int8) and
// `residual_scale` (f32, per-token) hold the previous quantized residual on entry;
// on return they hold the newly quantized rmsnorm(in + alpha*dequant(residual))*weight.
void fluke_rmsnorm_quant_int8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

// See fluke_rotary_emb_cpu: sin_gpu/cos_gpu are [seq_len, sincos_width] (rotate-half),
// rotating 2*sincos_width dims per head (require 2*sincos_width <= head_dim).
void fluke_rotary_emb_gpu(
    void *x,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,   // width of sin/cos tables: [seq_len, sincos_width]; rotates 2*sincos_width dims
    int stride_batch,
    int stride_seq,
    int stride_head
);

// SiLU-gated multiply. `in` is [n_tokens, 2*hidden_dim] laid out as [y | gate]
// (y = first hidden_dim cols, gate = next hidden_dim cols); `out` is
// [n_tokens, hidden_dim] with out[t,k] = silu(gate[t,k]) * y[t,k].
void fluke_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
);

// fLSTM step (single timestep) on fp16. scratch/ih_t [batch_size, 4*hidden_dim];
// cell [batch_size, hidden_dim] updated in place; hh_next [batch_size, hidden_dim] out.
// No internal sync (called per-timestep in a loop; the caller syncs).
void fluke_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim
);

// ── Fused INT8 DSL kernels (AOT-exported CuTe/Fly artifacts) ──────────────────
// A stable, ATen-free C ABI over the arch-specialized fused kernels. The arch
// detection, module load, and descriptor plumbing live inside fluke (src/fused_*.c);
// consumers (e.g. slorado) just pass device pointers + dims. Only available where a
// matching precompiled backend exists (CUDA >= 12 on a supported arch); otherwise
// fluke_int8_select returns NULL and the caller keeps its fp16 path.

// Opaque, process-lifetime backend handle. Loads the arch's kernel module(s) once.
// Returns NULL if no precompiled backend matches this device's arch and `dims`.
// The returned pointer is shared across callers and must NOT be freed.
typedef struct fluke_int8_backend fluke_int8_backend_t;
fluke_int8_backend_t *fluke_int8_select(int device_index, fluke_dims_t dims);

// Fused int8 wqkv GEMM + rotary. a_i8: [M, d_model] int8 (+per-token scale_a). wqkv_i8:
// [3*d_model, d_model] int8 (+per-out-channel scale_b). sin/cos: fp32 [seq, head_dim/2]
// (rotate-half). out: fp16 [M, 3*d_model] (row-major, contiguous). seqlen = tokens/seq
// (rotary indexes seq = row % seqlen). Returns 0 on success.
int fluke_qkv_rotary_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *wqkv_i8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
);

// Fused int8 dual GEMM (gate, up) + SiLU. a_i8: [M, d_model] int8 (+per-token scale_a).
// gate_i8/up_i8: [dim_feedforward, d_model] int8 (+per-out-channel scales). out: fp16
// [M, dim_feedforward] = silu(gate) * up (row-major, contiguous). Returns 0 on success.
int fluke_gated_mlp_i8_gpu(
    const fluke_int8_backend_t *b, void *out,
    const void *a_i8, const void *gate_i8, const void *up_i8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
);

// ── Fused FP8 DSL kernels (AOT-exported FlyDSL/RDNA artifacts) ────────────────
// The AMD/RDNA counterpart of the int8 ABI above. Same fused ops, but inputs are
// fp8 (e4m3) with the fp8 amax/448 scale convention instead of int8 amax/127, run on
// RDNA4 WMMA (f32 accumulate). Weights are PRESHUFFLED to the WMMA B layout
// [N/16, K/16, 2, 16, 8] (see fly.rdna4 preshuffle_b_fp8). Implemented in
// src/fused_hip.cpp over the per-arch HSACO artifacts embedded into libfluke.a; the
// arch detection + module load live inside fluke. Only available on a supported RDNA4
// device (gfx1200/gfx1201) with matching precompiled kernels — otherwise
// fluke_fp8_select returns NULL and the caller keeps its fp16 path.

// Opaque, process-lifetime backend handle. Loads the device arch's kernel module(s)
// once (selected from the embedded per-arch HSACOs by gcnArchName). Returns NULL if no
// precompiled backend matches this device's arch and `dims`. Do NOT free.
typedef struct fluke_fp8_backend fluke_fp8_backend_t;
fluke_fp8_backend_t *fluke_fp8_select(int device_index, fluke_dims_t dims);

// Fused fp8 wqkv GEMM + rotary. a_fp8: [M, d_model] fp8 (+per-token scale_a). wqkv_fp8:
// PRESHUFFLED [3*d_model/16, d_model/16, 2, 16, 8] fp8 (+per-out-channel scale_b [3*d_model]).
// sin/cos: fp32 [seqlen, rotary_dim/2] (rotate-half). out: fp16 [M, 3*d_model] (row-major,
// contiguous). seqlen = tokens/seq (rotary indexes seq = row % seqlen). Returns 0 on success.
int fluke_qkv_rotary_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *wqkv_fp8,
    const void *scale_a, const void *scale_b,
    const void *sin, const void *cos,
    int M, int seqlen
);

// Fused fp8 dual GEMM (gate, up) + SiLU. a_fp8: [M, d_model] fp8 (+per-token scale_a).
// gate_fp8/up_fp8: PRESHUFFLED [dim_feedforward/16, d_model/16, 2, 16, 8] fp8 (+per-out-channel
// scales [dim_feedforward]). out: fp16 [M, dim_feedforward] = silu(gate) * up (row-major,
// contiguous). Returns 0 on success.
int fluke_gated_mlp_fp8_gpu(
    const fluke_fp8_backend_t *b, void *out,
    const void *a_fp8, const void *gate_fp8, const void *up_fp8,
    const void *scale_a, const void *scale_gate, const void *scale_up,
    int M
);

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef __cplusplus
}
#endif

#endif // FLUKE_H
