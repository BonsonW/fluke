// Measure tuned int8 GEMM efficiency (cuBLAS GemmEx, IMMA/tensor-op) on the two
// per-step factored-LSTM shapes, to decide persistent-kernel vs tuned-two-kernel.
//   gate:     M=2048, K=256,  N=4096   int8->int32
//   down-proj M=2048, K=1024, N=128    int8->int32
// TN layout (opA=T, opB=N): A[K,M] col-major, B[K,N] col-major, C[M,N] col-major.
// Warm clocks first, loop many iters, report median µs/call.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define BK(x) do{cublasStatus_t s=(x); if(s){printf("cublas err %s:%d %d\n",__FILE__,__LINE__,(int)s);exit(1);}}while(0)

static double bench_shape(cublasHandle_t h, int M,int N,int K,const char* name,int iters){
  int8_t *dA,*dB; int32_t *dC;
  CK(cudaMalloc(&dA,(size_t)M*K)); CK(cudaMalloc(&dB,(size_t)K*N)); CK(cudaMalloc(&dC,(size_t)M*N*4));
  CK(cudaMemset(dA,1,(size_t)M*K)); CK(cudaMemset(dB,1,(size_t)K*N));
  int32_t alpha=1, beta=0;
  // C[M,N] = op(A)[M,K]*op(B)[K,N]; A stored [K,M] col-major lda=K, B [K,N] col-major ldb=K, C [M,N] ldc=M
  auto call=[&](){ return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
      &alpha, dA, CUDA_R_8I, K, dB, CUDA_R_8I, K, &beta, dC, CUDA_R_32I, M,
      CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP); };
  cublasStatus_t s = call();
  if(s){ printf("  %-9s cublasGemmEx FAILED status=%d (shape unsupported by this combo)\n",name,(int)s);
         cudaFree(dA);cudaFree(dB);cudaFree(dC); return -1; }
  CK(cudaDeviceSynchronize());
  // warm
  for(int i=0;i<200;++i) call();
  CK(cudaDeviceSynchronize());
  // time in chunks, take median of 7
  std::vector<double> us;
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  for(int r=0;r<7;++r){
    cudaEventRecord(e0);
    for(int i=0;i<iters;++i) call();
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    us.push_back((double)ms/iters*1000.0);
  }
  std::sort(us.begin(),us.end());
  double med=us[us.size()/2];
  double macs=(double)M*(double)N*(double)K;
  double roof_us = macs/312.0e12*1e6;              // 312 TMAC/s = 624 TOPS int8
  double pct_roof = roof_us/med*100.0;
  double tops = 2.0*macs/(med*1e-6)/1e12;
  printf("  %-9s M=%d K=%d N=%d : %7.2f us/call  | roofline %.2f us -> %.1f%% roofline | %.0f TOPS (%.1f%% of 624)\n",
         name,M,K,N, med, roof_us, pct_roof, tops, tops/624.0*100.0);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return med;
}

int main(int argc,char**argv){
  int dev = argc>1?atoi(argv[1]):0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp p; cudaGetDeviceProperties(&p,dev);
  printf("Device %d: %s  (%d SMs)\n",dev,p.name,p.multiProcessorCount);
  cublasHandle_t h; BK(cublasCreate(&h));
  BK(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
  // warm clocks: hammer a big GEMM
  { int8_t*a,*b; int32_t*c; CK(cudaMalloc(&a,4096*4096));CK(cudaMalloc(&b,4096*4096));CK(cudaMalloc(&c,(size_t)4096*4096*4));
    int32_t al=1,be=0;
    for(int i=0;i<300;++i) cublasGemmEx(h,CUBLAS_OP_T,CUBLAS_OP_N,4096,4096,4096,&al,a,CUDA_R_8I,4096,b,CUDA_R_8I,4096,&be,c,CUDA_R_32I,4096,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CK(cudaDeviceSynchronize()); cudaFree(a);cudaFree(b);cudaFree(c); }
  printf("cuBLAS int8 GEMM (IMMA, TN), warm, median of 7 x %d iters:\n", 2000);
  double g  = bench_shape(h,2048,4096,256 ,"gate"    ,2000);
  double d  = bench_shape(h,2048,128 ,1024,"downproj",2000);
  // also a big square for reference efficiency
  bench_shape(h,4096,4096,4096,"ref4096",1000);
  if(g>0&&d>0) printf("\nSUM per-step (gate+downproj) = %.2f us  | vs persistent 89us, two-kernel ~64us, dorado ~10us\n", g+d);
  cublasDestroy(h);
  return 0;
}
