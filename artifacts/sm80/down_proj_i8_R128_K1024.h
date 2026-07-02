
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
} down_proj_i8_R128_K1024_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_down_proj_i8_R128_K1024_cuda_init(void **);
void _mlir_down_proj_i8_R128_K1024_cuda_load_to_device(void **);
static inline void down_proj_i8_R128_K1024_Kernel_Module_Load(down_proj_i8_R128_K1024_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_down_proj_i8_R128_K1024_cuda_init((void **)(&initArgs));
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
        _mlir_down_proj_i8_R128_K1024_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void down_proj_i8_R128_K1024_Kernel_Module_Unload(down_proj_i8_R128_K1024_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} down_proj_i8_R128_K1024_Tensor_mA_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} down_proj_i8_R128_K1024_Tensor_mB_t;


typedef struct {
    void *data;
    int32_t dynamic_shapes[3];
    int64_t dynamic_strides[2];
} down_proj_i8_R128_K1024_Tensor_mC_t;


typedef struct {
    void *data;
} down_proj_i8_R128_K1024_Tensor_mScaleA_t;


typedef struct {
    void *data;
} down_proj_i8_R128_K1024_Tensor_mScaleB_t;

#ifdef __cplusplus
extern "C"
#endif
void _mlir_down_proj_i8_R128_K1024__mlir_ciface_cutlass___call___gemm_i8_quantTensorOpGemmI8_object_at__Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv8i64div81i64div8_FakeTensorFloat3212811128(void **args, int32_t num_args);

static inline int32_t cute_dsl_down_proj_i8_R128_K1024_wrapper(down_proj_i8_R128_K1024_Kernel_Module_t *module, down_proj_i8_R128_K1024_Tensor_mA_t *mA, down_proj_i8_R128_K1024_Tensor_mB_t *mB, down_proj_i8_R128_K1024_Tensor_mC_t *mC, down_proj_i8_R128_K1024_Tensor_mScaleA_t *mScaleA, down_proj_i8_R128_K1024_Tensor_mScaleB_t *mScaleB) {
    int32_t ret;
    void *args[6] = {
        mA, mB, mC, mScaleA, mScaleB,
        &ret
    };
    _mlir_down_proj_i8_R128_K1024__mlir_ciface_cutlass___call___gemm_i8_quantTensorOpGemmI8_object_at__Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv16i64div161i64div16_Tensorgmemodiv8i64div81i64div8_FakeTensorFloat3212811128(args, 6);
    return ret;
}
