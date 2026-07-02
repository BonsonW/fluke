
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_factored_lstm_i8_H1024_Khh128_R128_cuda_init(void **);
void _mlir_factored_lstm_i8_H1024_Khh128_R128_cuda_load_to_device(void **);
static inline void factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_Load(factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_factored_lstm_i8_H1024_Khh128_R128_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_factored_lstm_i8_H1024_Khh128_R128_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_Unload(factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mA_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_i_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_f_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_g_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_o_t;


typedef struct {
    void *data;
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_i_t;


typedef struct {
    void *data;
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_f_t;


typedef struct {
    void *data;
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_g_t;


typedef struct {
    void *data;
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_o_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mC_c_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} factored_lstm_i8_H1024_Khh128_R128_Tensor_mH_out_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_factored_lstm_i8_H1024_Khh128_R128__mlir_ciface_cutlass___call___factored_lstm_i8TensorOpFactoredLstmI8_object_at__Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64di(void **args, int32_t num_args);

static inline int32_t cute_dsl_factored_lstm_i8_H1024_Khh128_R128_wrapper(factored_lstm_i8_H1024_Khh128_R128_Kernel_Module_t *module, factored_lstm_i8_H1024_Khh128_R128_Tensor_mA_t *mA, factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_i_t *mB_i, factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_f_t *mB_f, factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_g_t *mB_g, factored_lstm_i8_H1024_Khh128_R128_Tensor_mB_o_t *mB_o, factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_i_t *mBias_i, factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_f_t *mBias_f, factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_g_t *mBias_g, factored_lstm_i8_H1024_Khh128_R128_Tensor_mBias_o_t *mBias_o, factored_lstm_i8_H1024_Khh128_R128_Tensor_mC_c_t *mC_c, factored_lstm_i8_H1024_Khh128_R128_Tensor_mH_out_t *mH_out) {
    int32_t ret;
    void *args[12] = {
        mA, mB_i, mB_f, mB_g, mB_o, mBias_i, mBias_f, mBias_g, mBias_o, mC_c, mH_out,
        &ret
    };
    _mlir_factored_lstm_i8_H1024_Khh128_R128__mlir_ciface_cutlass___call___factored_lstm_i8TensorOpFactoredLstmI8_object_at__Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64div81i64div8_Tensorgmemodiv8i64di(args, 12);
    return ret;
}
