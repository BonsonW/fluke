// Plain CUTLASS int8 device GEMM parity check on the gate shape (M=2048,K=256,N=4096).
// Sanity gate: CUTLASS's multistage cp.async mainloop should match/beat cuBLAS 32us.
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/arch/arch.h"

#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

template<typename TB, typename WS, int Stages>
double run_cfg(const char* name,int M,int N,int K,int iters){
  using Gemm = cutlass::gemm::device::Gemm<
    int8_t, cutlass::layout::RowMajor,
    int8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor,
    int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TB, WS, cutlass::gemm::GemmShape<16,8,32>,
    cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
  int8_t *dA,*dB; int32_t *dC;
  CK(cudaMalloc(&dA,(size_t)M*K)); CK(cudaMalloc(&dB,(size_t)N*K)); CK(cudaMalloc(&dC,(size_t)M*N*4));
  CK(cudaMemset(dA,1,(size_t)M*K)); CK(cudaMemset(dB,1,(size_t)N*K));
  Gemm op;
  typename Gemm::Arguments args({M,N,K},{dA,K},{dB,K},{dC,N},{dC,N},{1,0});
  if(op.can_implement(args)!=cutlass::Status::kSuccess){ printf("  %-28s UNSUPPORTED\n",name);
    cudaFree(dA);cudaFree(dB);cudaFree(dC); return -1; }
  op.initialize(args);
  for(int i=0;i<200;++i) op();
  CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<7;++r){ cudaEventRecord(e0); for(int i=0;i<iters;++i) op();
    cudaEventRecord(e1); cudaEventSynchronize(e1); float ms; cudaEventElapsedTime(&ms,e0,e1);
    us.push_back((double)ms/iters*1000.0); }
  std::sort(us.begin(),us.end());
  printf("  %-28s %7.2f us  (%.0f TOPS, %.1f%% peak)\n",name,us[3],
    2.0*M*N*K/(us[3]*1e-6)/1e12, 2.0*M*N*K/(us[3]*1e-6)/1e12/624*100);
  cudaFree(dA);cudaFree(dB);cudaFree(dC);
  return us[3];
}

int main(int argc,char**argv){
  int dev=argc>1?atoi(argv[1]):0; CK(cudaSetDevice(dev));
  // warm clocks
  { int8_t*a;CK(cudaMalloc(&a,64<<20)); for(int i=0;i<400;++i) cudaMemset(a,i,64<<20); CK(cudaDeviceSynchronize()); cudaFree(a); }
  printf("CUTLASS int8 gate GEMM M=2048 K=256 N=4096 (cuBLAS ref: 32.1us):\n");
  using G = cutlass::gemm::GemmShape<128,128,64>;
  run_cfg<cutlass::gemm::GemmShape<128,128,64>,cutlass::gemm::GemmShape<64,64,64>,3>("128x128x64/64x64x64/s3",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<128,128,64>,cutlass::gemm::GemmShape<64,64,64>,4>("128x128x64/64x64x64/s4",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<128,256,64>,cutlass::gemm::GemmShape<64,64,64>,3>("128x256x64/64x64x64/s3",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<256,128,64>,cutlass::gemm::GemmShape<64,64,64>,3>("256x128x64/64x64x64/s3",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<128,128,128>,cutlass::gemm::GemmShape<64,64,64>,2>("128x128x128/64x64x64/s2",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<64,128,64>,cutlass::gemm::GemmShape<32,64,64>,4>("64x128x64/32x64x64/s4",2048,4096,256,2000);
  run_cfg<cutlass::gemm::GemmShape<64,64,64>,cutlass::gemm::GemmShape<32,32,64>,4>("64x64x64/32x32x64/s4",2048,4096,256,2000);
  printf("warp-k-split probes (gate shape):\n");

  run_cfg<cutlass::gemm::GemmShape<128,128,128>,cutlass::gemm::GemmShape<64,64,64>,2>("128x128x128/64x64x64/s2 kP2",2048,4096,256,2000);
  printf("down-proj M=2048 K=1024 N=128 (cuBLAS ref: 8.1us):\n");
  run_cfg<cutlass::gemm::GemmShape<128,64,64>,cutlass::gemm::GemmShape<64,32,64>,4>("128x64x64/64x32x64/s4",2048,128,1024,2000);
  run_cfg<cutlass::gemm::GemmShape<64,64,64>,cutlass::gemm::GemmShape<32,32,64>,4>("64x64x64/32x32x64/s4",2048,128,1024,2000);
  return 0;
}
// warp-k-split probe (appended): does sm80 multistage accept WarpShape::kK < TB::kK?
