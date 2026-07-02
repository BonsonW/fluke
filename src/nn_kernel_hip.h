// The MIT License (MIT)

// Copyright (c) 2025 Bonson Wong

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef NN_KERNEL_HIP_H
#define NN_KERNEL_HIP_H

#include <math.h>
#include <float.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

static __global__ void rotary_emb(
	half *x,
    const float *_cos,
    const float *_sin,
    const uint64_t seq_len,
    const uint64_t stride_batch,
    const uint64_t stride_seq,
    const uint64_t stride_head,
    const uint64_t sincos_width
) {
    const uint64_t batch = blockIdx.x;
    const uint64_t head = blockIdx.y;
    const uint64_t rot = threadIdx.x;
    const uint64_t tid = threadIdx.y;
    const uint64_t n_threads = blockDim.y;

    if (tid >= seq_len) return;

    half *_o0 = x + (batch * stride_batch) + (head * stride_head) + rot;
    half *_o1 = x + (batch * stride_batch) + (head * stride_head) + sincos_width + rot;

    for (int seq = tid; seq < seq_len; seq += n_threads) {
        float cos = *(_cos + (seq * sincos_width) + rot);
        float sin = *(_sin + (seq * sincos_width) + rot);

        half *o0 = _o0 + (seq * stride_seq);
        half *o1 = _o1 + (seq * stride_seq);

        float x0 = __half2float(*o0);
        float x1 = __half2float(*o1);

        *o0 = __float2half(x0 * cos - x1 * sin);
        *o1 = __float2half(x0 * sin + x1 * cos);
    }
}

static __global__ void silu_mul(
	const half *in,
	half *out,
    const uint64_t hidden_dim,
    const uint64_t n_tokens
) {
    uint64_t j = blockIdx.x;

    for (uint64_t k = threadIdx.x; k < hidden_dim; k += blockDim.x) {
        uint64_t i = k + j * (hidden_dim * 2);

        half y = in[i];
        half gate = in[i + hidden_dim];

        float g = __half2float(gate);
        float silu = g / (1.0f + __expf(-g));

        out[k + j * hidden_dim] = __float2half(silu * __half2float(y));
    }
}

#ifdef __cplusplus
}
#endif

#endif // NN_KERNEL_HIP_H
