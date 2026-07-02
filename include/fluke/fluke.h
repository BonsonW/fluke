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

#if defined(HAVE_CUDA) || defined(HAVE_ROCM)

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

#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

#ifdef __cplusplus
}
#endif

#endif // FLUKE_H
