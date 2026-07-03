#include <fluke/fluke.h>
#include "error.h"
#include "cuda_utils.h"
#include "nn_kernel_cuda.h"

#include <fluke/fluke_error.h>

#include <cuda_fp16.h>
#include <stdlib.h>

void fluke_rotary_emb_gpu(
    void *x,
    const void *sin_gpu,
    const void *cos_gpu,
    int batch_size,
    int seq_len,
    int n_heads,
    int head_dim,
    int sincos_width,
    int stride_batch,
    int stride_seq,
    int stride_head
) {
    int thread_h = 32;
    dim3 block_size(sincos_width, thread_h, 1);
	dim3 grid_size(batch_size, n_heads, 1);

    rotary_emb<<<grid_size, block_size>>>(
        (half *)x,
        (const float *)cos_gpu,
        (const float *)sin_gpu,
        seq_len,
        stride_batch,
        stride_seq,
        stride_head,
        sincos_width
    );
    checkKernel();
}

void fluke_silu_mul_gpu(
    const void *in,
    void *out,
    int n_tokens,
    int hidden_dim
) {
    int threads = 1024;
    int blocks = n_tokens;

    silu_mul<<<blocks, threads>>>(
        (const half *)in,
        (half *)out,
        hidden_dim,
        n_tokens
    );
    checkKernel();
}

void fluke_rmsnorm_gpu(
    const void* in,
    const void* residual,
    const void* weight,
    void* out,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);
    ASSERT(hidden_dim % 2 == 0);  // kernel is half2-vectorized: one thread per adjacent pair

    int threads = hidden_dim / 2;
    int blocks = n_tokens;

    rmsnorm<<<blocks, threads>>>(
        (half *)in, (half *)residual, (half *)weight, (half *)out, n_tokens, hidden_dim, alpha, eps
    );
    checkKernel();
}

void fluke_rmsnorm_quant_int8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);
    ASSERT(hidden_dim % 2 == 0);  // kernel is half2-vectorized: one thread per adjacent pair

    int threads = hidden_dim / 2;
    int blocks = n_tokens;

    rmsnorm_quant_int8<<<blocks, threads>>>(
        (half *)in, (half *)weight, (int8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkKernel();
}

void fluke_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim
) {
    int threads = (hidden_dim < 1024) ? hidden_dim : 1024;
    flstm_step<<<batch_size, threads>>>(
        (const half*)scratch, (const half*)ih_t,
        (half*)cell, (half*)hh_next,
        4 * hidden_dim, hidden_dim
    );
    checkKernel();
}

void fluke_rmsnorm_quant_fp8_gpu(
    const void* in,
    const void* weight,
    void* residual,
    void* residual_scale,
    int n_tokens,
    int hidden_dim,
    float alpha,
    float eps
) {
    ASSERT(hidden_dim <= 1024);
    ASSERT(hidden_dim % 2 == 0);  // kernel is half2-vectorized: one thread per adjacent pair

    int threads = hidden_dim / 2;
    int blocks = n_tokens;

    rmsnorm_quant_fp8<<<blocks, threads>>>(
        (half *)in, (half *)weight, (uint8_t *)residual, (float *)residual_scale, n_tokens, hidden_dim, alpha, eps
    );
    checkKernel();
}

void fluke_dequant_fp8_transpose_gpu(
    const void* in,
    void*       out,
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
) {
    ASSERT(n_channels <= 1024);

    dequant_fp8_transpose<<<n_timesteps * batch_size, n_channels>>>(
        (const uint8_t *)in, (half *)out, n_timesteps, batch_size, n_channels, scale
    );
    checkKernel();
}

void fluke_dequant_int8_transpose_gpu(
    const void* in,
    void*       out,
    int         n_timesteps,
    int         batch_size,
    int         n_channels,
    float       scale
) {
    ASSERT(n_channels <= 1024);

    dequant_int8_transpose<<<n_timesteps * batch_size, n_channels>>>(
        (const int8_t *)in, (half *)out, n_timesteps, batch_size, n_channels, scale
    );
    checkKernel();
}

void fluke_quant_int8_gpu(
    const void* in,
    void*       out,
    void*       scale,
    int         n_tokens,
    int         hidden_dim
) {
    ASSERT(hidden_dim <= 2048);
    ASSERT(hidden_dim % 2 == 0);  // half2/char2-vectorized: one thread per adjacent pair

    quant_int8<<<n_tokens, hidden_dim / 2>>>(
        (const half *)in, (int8_t *)out, (float *)scale, n_tokens, hidden_dim
    );
    checkKernel();
}

void fluke_quant_fp8_gpu(
    const void* in,
    void*       out,
    void*       scale,
    int         n_tokens,
    int         hidden_dim
) {
    ASSERT(hidden_dim <= 2048);
    ASSERT(hidden_dim % 2 == 0);  // half2/uchar2-vectorized: one thread per adjacent pair

    quant_fp8<<<n_tokens, hidden_dim / 2>>>(
        (const half *)in, (uint8_t *)out, (float *)scale, n_tokens, hidden_dim
    );
    checkKernel();
}
