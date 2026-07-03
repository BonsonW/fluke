// Model dimensions the precompiled fused kernels are specialized for. These are
// MODEL-specific, not arch-specific — the CUDA (fused_cuda.cpp) and HIP (fused_hip.cpp)
// backends bake the same shape, so the constants are shared here. fluke_{int8,fp8}_select
// verifies the caller's fluke_dims_t against these and bows out (NULL -> fp16) on mismatch.
#ifndef FLUKE_FUSED_DIMS_H
#define FLUKE_FUSED_DIMS_H

#define FLUKE_SUP_D_MODEL   512
#define FLUKE_SUP_DIM_FF    2048
#define FLUKE_SUP_NHEAD     8
#define FLUKE_SUP_HEAD_DIM  64

// Factored-LSTM (CRF/LSTM basecaller) shape. H = hidden, K_hh = hidden rank, R = input rank;
// Kc = K_hh + R is the merged f16 up-projection contraction.
#define FLUKE_LSTM_H        1024
#define FLUKE_LSTM_K_HH     128
#define FLUKE_LSTM_R        128

#endif // FLUKE_FUSED_DIMS_H
