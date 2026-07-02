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
    checkCudaError();
    cudaDeviceSynchronize();
    checkCudaError();
}
