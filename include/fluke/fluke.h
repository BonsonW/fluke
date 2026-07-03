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

// Standalone per-token INT8 quantize (no RMSNorm). GPU analogue of quantize_tensor(x, dim=-1):
// scale (f32 [n_tokens]) is the dequant multiplier amax/128 (reciprocal pre-applied, fp ≈ out*scale);
// out (int8) = clamp(round(in / scale), -127, 127). hidden_dim must be even and <= 2048.
void fluke_quant_int8_gpu(
    const void* in,     /* f16  [n_tokens, hidden_dim] input  */
    void*       out,    /* int8 [n_tokens, hidden_dim] output */
    void*       scale,  /* f32  [n_tokens]             per-token dequant scale output */
    int         n_tokens,
    int         hidden_dim
);

// Standalone per-token fp8 (E4M3FN) quantize (no RMSNorm). scale (f32 [n_tokens]) = amax/448
// (dequant multiplier); out (fp8 bytes) = float_to_e4m3fn(clamp(in / scale, -448, 448)). Software
// fp8 (portable). hidden_dim must be even and <= 2048.
void fluke_quant_fp8_gpu(
    const void* in,     /* f16  [n_tokens, hidden_dim] input  */
    void*       out,    /* uint8[n_tokens, hidden_dim] E4M3FN output */
    void*       scale,  /* f32  [n_tokens]             per-token dequant scale output */
    int         n_tokens,
    int         hidden_dim
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

// Fused RMSNorm + per-token fp8 (E4M3FN) quantize, in place. `residual` (fp8 bytes) and
// `residual_scale` (f32, per-token) hold the previous quantized residual on entry; on return
// the newly quantized rmsnorm(in + alpha*dequant(residual))*weight. Software fp8 (portable).
void fluke_rmsnorm_quant_fp8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
);

// Dequantize + transpose in one pass: in fp8 [n_timesteps, batch_size, n_channels] ->
// out f16 [batch_size, n_timesteps, n_channels], out[n,t,c] = fp8(in[t,n,c]) * scale.
void fluke_dequant_fp8_transpose_gpu(
    const void* in,
    void*       out,
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
);

// INT8 analogue of fluke_dequant_fp8_transpose_gpu: in int8 [n_timesteps, batch_size, n_channels]
// -> out f16 [batch_size, n_timesteps, n_channels], out[n,t,c] = (float)in[t,n,c] * scale.
void fluke_dequant_int8_transpose_gpu(
    const void* in,
    void*       out,
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
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

// ── Factored-LSTM (CRF/LSTM model) int8 kernels ───────────────────────────────
// The CRF/LSTM basecaller path (slorado lstm_model.cpp), a separate model shape from the
// transformer dims above: the int8 down-projection GEMM + the fused factored-LSTM step.
// Selected on the LSTM shape (H/K_hh/R); NULL if the arch/shape don't match the precompiled
// kernels (caller keeps its fp16 path). The handle is process-lifetime; do NOT free.
typedef struct fluke_flstm_backend fluke_flstm_backend_t;
fluke_flstm_backend_t *fluke_flstm_select(int device_index, int H, int K_hh, int R);

// int8 down-projection: out[M, R] f16 = (a_i8[M, H] * scale_a[M]) @ (w_i8[R, H] * scale_b[R])^T.
// Projects hidden H -> rank R (the recurrent hh_down per step and the input x_down precompute).
// a_i8 [M,H] int8 (+per-token scale_a[M]); w_i8 [R,H] int8 (+per-channel scale_b[R]). Returns 0 on success.
// stream: the CUDA stream (cudaStream_t) to launch on; NULL uses the default stream. Passing the
// capture stream lets this recurrent per-step kernel be captured into a CUDA graph.
int fluke_down_proj_i8_gpu(
    const fluke_flstm_backend_t *b, void *out,
    const void *a_i8, const void *w_i8,
    const void *scale_a, const void *scale_b,
    int M, void *stream
);

// Fused factored-LSTM step. a_f16 [B, Kc] (Kc = K_hh + R; concat hh_down | x_down). Gate weights
// Bi/Bf/Bg/Bo f16 [H, Kc] (concat up_hh_g | up_ih_g). biases bias_{i,f,g,o} f32 [H]. cell c_f32
// [B, H] read + written in place. Output h_i8 [B, H] int8 (fixed scale 1/127). Does two f16
// up-projections into 4 gate accumulators + gates + cell update. Returns 0 on success.
// stream: the CUDA stream (cudaStream_t) to launch on; NULL uses the default stream. Passing the
// capture stream lets this recurrent per-step kernel be captured into a CUDA graph.
int fluke_flstm_step_i8_gpu(
    const fluke_flstm_backend_t *b, void *h_i8,
    const void *a_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, int B, void *stream
);

// Single-launch fused factored-LSTM step: does the recurrent int8 hh down-projection AND
// the gate step in ONE kernel (removes the inter-kernel launch/DRAM round-trip; ~1.3-1.6x
// over the down_proj + step pair on A100). Replaces fluke_down_proj_i8_gpu (hh) +
// fluke_flstm_step_i8_gpu for the per-step recurrence only (the ih precompute still uses
// fluke_down_proj_i8_gpu).
//   h_prev_i8 [B, H] int8 (fixed 1/127); w_dn_i8 [K_hh, H] int8 recurrent down-weight;
//   comb_scale [K_hh] f32 = per-channel w_dn scale * (1/127), host-folded;
//   x_f16 [B, R] f16 = this step's x_down slice;
//   Bi/Bf/Bg/Bo f16 [H, Kc], bias_{i,f,g,o} f32 [H]; c_f32 [B, H] updated in place;
//   h_i8 [B, H] int8 out (1/127); hh_stage f16 [B, K_hh] scratch (producer-written, no init);
//   flags int32 [ceil(B/64)*4] zeroed at allocation (self-cleaning across steps).
// RESIDENCY: the whole grid must be co-resident (consumers spin on same-grid producers).
// At the baked tile config that means B <= 512; the caller MUST fall back to the two-kernel
// path for larger B. Returns 0 on success. stream as above (CUDA-graph capturable).
int fluke_flstm_fused_step_i8_gpu(
    const fluke_flstm_backend_t *b, void *h_i8,
    const void *h_prev_i8, const void *w_dn_i8, const void *comb_scale, const void *x_f16,
    const void *Bi, const void *Bf, const void *Bg, const void *Bo,
    const void *bias_i, const void *bias_f, const void *bias_g, const void *bias_o,
    void *c_f32, void *hh_stage, void *flags, int B, void *stream
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
