// koi8_gate.cu -- Phase 1: replicate koi's 8:1 IMMA:LDSM fragment reuse on the isolated
// int8 gate GEMM.  Decoded schedule (from koicubin/flstm_main.txt gate loop):
//   warp tile = 64 rows (4 m16-tiles) x WN cols; A[64,256] resident (LDS'd from smem);
//   per n8-tile: ldmatrix.x4 loads B for 2 k-tiles (b for kt,kt+1), each B-frag reused
//   across the 4 M-tiles -> 8 IMMA per LDSM.x4.  8 k-tiles => 4 ldmatrix.x4 + 32 IMMA / n-tile.
// Plain GEMM first (no epilogue). Pass/fail = OUR cuobjdump IMMA:LDSM ~= 8:1 with 0 spill.
//
// PHASE-1 VERDICT (the last lever, tested):
//   * 8:1 IMMA:LDSM ACHIEVED, 0 spill, 168 regs, EXACT. => the "8:1 needs huge
//     accumulators that spill" fear is FALSE: accumulator-tiling per n-tile + A-frag
//     resident fits in 168 regs. Register wall is a myth.
//   * BUT ZERO speedup: 87us gate vs fused_cutlass ~28us (3x SLOWER).
//   * WHY (ncu): L1TEX 72.5% / imma 4.5% / occ 12.5% / short_scoreboard 4.9 = LATENCY +
//     A-LDS bound, NOT smem-ratio bound. The 8:1 only counts B(weight) ldmatrix; hitting
//     it SHIFTS the smem-pipe cost onto the A(activation) LDS (A[64,256] resident = 128
//     scalar LDS/warp saturates L1TEX at 72.5%).
//   * Config-invariance proves it: WM=4/ratio8:1/1-CTA-SM = 87us == WM=2/ratio4:1/2-CTA-SM
//     = 88us. Neither ratio nor occupancy changes the time -> bottleneck is the MISSING
//     SOFTWARE PIPELINE (no cp.async<->ldmatrix<->IMMA overlap); every hand kernel this
//     session lacked it (imma 1-4.5%).
//   CONCLUSION: koi's 2x is its multistage software-pipelined mainloop, NOT the 8:1 ratio
//     (the 8:1 is a side-effect of koi's reuse; alone it does nothing). A single-GEMM
//     (safe) must reload all of A each step (128 LDS -> smem-saturated); koi amortizes A
//     via persistence, which has the proven-dead sync wall. Matching koi needs a from-
//     scratch CUTLASS-multistage-pipeline + resident weights + persistence -- and the
//     persistence half is already proven unsafe/slow. fused_cutlass 37us/step is the ceiling.
// Build: nvcc -arch=sm_80 -O3 -diag-suppress 177 koi8_gate.cu -o koi8_gate
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#define KC 256
#ifndef WM_TILES
#define WM_TILES 4       // m16 tiles per warp (64 rows) -> the reuse dimension
#endif
#ifndef WN_TILES
#define WN_TILES 4       // n8 tiles per warp (32 cols)
#endif
#define WM (WM_TILES*16)
#define WN (WN_TILES*8)

__device__ __forceinline__ void mma_m16n8k32(int&c0,int&c1,int&c2,int&c3,
    int a0,int a1,int a2,int a3,int b0,int b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
// ldmatrix.x4 .b16 (loads 4 regs/thread = for int8: 2 k-tiles of one B n8-tile, i.e. b[kt],b[kt+1])
__device__ __forceinline__ void ldm_x4(int&r0,int&r1,int&r2,int&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ uint32_t smem_addr(const void* p){
  return (uint32_t)__cvta_generic_to_shared(p);
}

// CTA tile: BM = 2*WM rows, BN = 4*WN cols; 8 warps (2 warp-rows x 4 warp-cols).
#define BM_C (2*WM)      // 128
#define BN_C (4*WN)      // 128
// gate GEMM: C[M,N] = A[M,Kc] @ B[N,Kc]^T (B row-major [N,Kc]; op over Kc).
// Grid (M/BM_C, N/BN_C); 8 warps/CTA. Per-warp 8:1 schedule (A[64,256] resident, B ldmatrix reused).
__global__ void __launch_bounds__(256,1) gate_probe(
    const int8_t* __restrict__ A, const int8_t* __restrict__ B, int32_t* __restrict__ C,
    int M, int N)
{
  const int tid=threadIdx.x, warp=tid>>5, lane=tid&31, gid=lane>>2, tg=lane&3;
  const int wm=warp>>2, wn=warp&3;                 // warp-row(0,1), warp-col(0..3)
  const int cbm=blockIdx.x*BM_C, cbn=blockIdx.y*BN_C;
  const int m0 = cbm + wm*WM;                       // this warp's 64 rows
  const int n0 = cbn + wn*WN;                       // this warp's 32 cols
  extern __shared__ int8_t sm[];
  int8_t* sA = sm;                 // [BM_C, Kc]
  int8_t* sB = sA + BM_C*KC;       // [BN_C, Kc]
  // stage A, B to smem via cp.async (all 256 threads)
  for(int i=tid;i<BM_C*KC/16;i+=256){ int r=i/(KC/16),c=i%(KC/16);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(smem_addr(&sA[r*KC+c*16])),"l"(&A[(long)(cbm+r)*KC+c*16])); }
  for(int i=tid;i<BN_C*KC/16;i+=256){ int r=i/(KC/16),c=i%(KC/16);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(smem_addr(&sB[r*KC+c*16])),"l"(&B[(long)(cbn+r)*KC+c*16])); }
  asm volatile("cp.async.commit_group;\n"); asm volatile("cp.async.wait_all;\n");
  __syncthreads();
  int8_t* sAw = sA + wm*WM*KC;      // this warp-row's A block
  int8_t* sBw = sB + wn*WN*KC;      // this warp-col's B block
  #define A_SMEM sAw
  #define B_SMEM sBw

  // A resident: WM_TILES m-tiles x 8 k-tiles, each a m16k32 frag = 4 regs. LDS'd once, reused.
  int Ar[WM_TILES][8][4];
  #pragma unroll
  for(int mt=0;mt<WM_TILES;++mt)
    #pragma unroll
    for(int kt=0;kt<8;++kt){
      int rowlo=mt*16+gid, rowhi=rowlo+8, co=kt*32+tg*4;
      Ar[mt][kt][0]=*(const int*)&sAw[rowlo*KC+co];
      Ar[mt][kt][1]=*(const int*)&sAw[rowhi*KC+co];
      Ar[mt][kt][2]=*(const int*)&sAw[rowlo*KC+co+16];
      Ar[mt][kt][3]=*(const int*)&sAw[rowhi*KC+co+16];
    }

  // n8-tile loop: per n-tile, 4 ldmatrix.x4 (8 k-tiles, 2/load) B reused across WM_TILES M -> 8 IMMA/LDSM
  #pragma unroll 1
  for(int nt=0; nt<WN_TILES; ++nt){
    int acc[WM_TILES][4];
    #pragma unroll
    for(int mt=0;mt<WM_TILES;++mt) for(int e=0;e<4;++e) acc[mt][e]=0;
    #pragma unroll
    for(int kk=0; kk<4; ++kk){            // 4 ldmatrix.x4, each = 2 k-tiles
      // B n8-tile (n0+nt*8 .. ), k-tiles 2kk,2kk+1. smem B row-major [WN,Kc].
      // ldmatrix.x4 addr: each lane l points to B[nt*8 + (l%8)][ (2kk)*32 + (l/8)*16 ] region.
      int nrow = nt*8 + (lane&7);
      int koff = (2*kk)*32 + (lane>>3)*16;
      uint32_t addr = smem_addr(&sBw[nrow*KC + koff]);
      int b0,b1,b2,b3; ldm_x4(b0,b1,b2,b3,addr);   // b0,b1 = kt(2kk); b2,b3 = kt(2kk+1)
      #pragma unroll
      for(int mt=0;mt<WM_TILES;++mt){
        mma_m16n8k32(acc[mt][0],acc[mt][1],acc[mt][2],acc[mt][3],
                     Ar[mt][2*kk][0],Ar[mt][2*kk][1],Ar[mt][2*kk][2],Ar[mt][2*kk][3], b0,b1);
        mma_m16n8k32(acc[mt][0],acc[mt][1],acc[mt][2],acc[mt][3],
                     Ar[mt][2*kk+1][0],Ar[mt][2*kk+1][1],Ar[mt][2*kk+1][2],Ar[mt][2*kk+1][3], b2,b3);
      }
    }
    // write C[m, n] : m = mt*16 + {gid,gid+8}, n = nt*8 + 2tg + {0,1}
    #pragma unroll
    for(int mt=0;mt<WM_TILES;++mt)
      #pragma unroll
      for(int e=0;e<4;++e){
        int r=m0+mt*16+((e<2)?gid:gid+8), c=n0+nt*8+2*tg+(e&1);
        C[(long)r*N+c]=acc[mt][e];
      }
  }
}

int main(int argc,char**argv){
  int M=1536,N=4096,bench=0,dev=0;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--M"))M=atoi(argv[++i]); else if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(M%BM_C)M=((M+BM_C-1)/BM_C)*BM_C; if(N%BN_C)N=((N+BN_C-1)/BN_C)*BN_C;
  printf("koi8_gate M=%d Kc=%d N=%d CTA %dx%d 8warps grid=(%d,%d)\n",M,KC,N,BM_C,BN_C,M/BM_C,N/BN_C);
  std::mt19937 rng(1); std::uniform_int_distribution<int> d(-127,127);
  std::vector<int8_t> hA((size_t)M*KC),hB((size_t)N*KC); std::vector<int32_t> hC((size_t)M*N),ref((size_t)M*N);
  for(auto&v:hA)v=d(rng); for(auto&v:hB)v=d(rng);
  int8_t*dA,*dB; int32_t*dC; CK(cudaMalloc(&dA,hA.size()));CK(cudaMalloc(&dB,hB.size()));CK(cudaMalloc(&dC,hC.size()*4));
  CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dB,hB.data(),hB.size(),cudaMemcpyHostToDevice));
  size_t smem=(size_t)BM_C*KC+(size_t)BN_C*KC;
  CK(cudaFuncSetAttribute(gate_probe,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  dim3 grid(M/BM_C,N/BN_C);
  gate_probe<<<grid,256,smem>>>(dA,dB,dC,M,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  if(!bench){
    CK(cudaMemcpy(hC.data(),dC,hC.size()*4,cudaMemcpyDeviceToHost));
    // ref: check a subset of rows
    int nr=std::min(M,64); long mism=0,maxd=0;
    for(int r=0;r<nr;++r)for(int c=0;c<N;++c){ long a=0; for(int k=0;k<KC;++k) a+=(int)hA[(long)r*KC+k]*(int)hB[(long)c*KC+k];
      long d2=llabs((long)hC[(long)r*N+c]-a); if(d2){mism++; if(d2>maxd)maxd=d2;} }
    printf("correctness (first %d rows): mism=%ld maxd=%ld -> %s\n",nr,mism,maxd,mism==0?"EXACT PASS":"FAIL");
  } else {
    for(int w=0;w<20;++w) gate_probe<<<grid,256,smem>>>(dA,dB,dC,M,N); CK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);
    std::vector<double> us;
    for(int r=0;r<7;++r){ cudaEventRecord(e0); for(int i=0;i<200;++i) gate_probe<<<grid,256,smem>>>(dA,dB,dC,M,N);
      cudaEventRecord(e1);cudaEventSynchronize(e1); float ms;cudaEventElapsedTime(&ms,e0,e1); us.push_back(ms/200*1000); }
    std::sort(us.begin(),us.end());
    printf("gate GEMM: %.2f us  (fused_cutlass gate ~28us @N2048)\n",us[3]);
  }
  return 0;
}
