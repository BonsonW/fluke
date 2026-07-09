// PERSISTENT variant of the winning fused_cutlass recipe (37.08us/step per-launch).
// One launch per layer; loops T in-kernel.  Per step: phase 1 down-proj tiles ->
// global-atomic barrier -> phase 2 gate tiles (DefaultMma multistage cp.async mainloop,
// interleaved single-acc 128x128, fused in-register LSTM epilogue, f16 cell,
// smem-staged writeout) -> barrier.  NO all-reduce (the down-proj is a plain GEMM over
// tiles, each tile owns its full K) - persistence is viable BECAUSE the interleave
// killed the register pressure and the output-split killed the combine.
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013 \
//        -I../thirdparty/cutlass/include -DFP16CELL persistent_cutlass.cu -o persistent_cutlass
//
// RESULT (A100, N=2048, T=2048, warm, median of 7): *** 37.86 us/step @ grid=216 ***
//   -> TIES the per-launch graph-captured fused_cutlass.cu (37.08) but does NOT beat it.
//   Bit-exact fwd+reverse.  254 regs, 0 spill, 2 CTA/SM (12.5% occ), imma 10.1%,
//   DRAM write ~1MB/step (no C).  Decomposition: barriers +3.0us/step (34.82 with
//   -DNO_BARRIER, timing-only); grid sweep 108=47.9 / 160=46.0 / 171=39.6 / 192=39.9 /
//   216(max resident)=37.9.  WHY persistence is a wash here: the CUDA-graph replay had
//   already amortized launch overhead to ~0, and the barrier quantizes the gate phase
//   to ceil(512 tiles / 216 resident CTAs) = 3 serial tile-waves -- the same critical
//   path as the per-launch version, plus ~3us of barrier.  The remaining gap to ~10us
//   is NOT launch/persistence overhead; it is per-tile efficiency (each 128x128 gate
//   tile runs ~3x its traffic floor at 11% occupancy, wait-stall on the 128-int32
//   accumulator dependency chains) -- shrinking THAT (deeper k-parallelism per warp or
//   a fundamentally wider-K problem shape) is the only remaining lever on this shape.
//   SHIPPABLE WINNER remains fused_cutlass.cu per-launch + graph: 37.08us/step.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

#define H 1024
#define K_HH 128
#define R 128
#define KC 256
#define NG (4*H)
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#include <cuda_fp16.h>
#ifdef FP16CELL
typedef __half cellT;
#else
typedef float cellT;
#endif

__device__ __forceinline__ int8_t clamp_i8(float q){ q=fminf(fmaxf(q,-127.f),127.f); return (int8_t)(int)q; }
__device__ __forceinline__ int8_t epi_elem(int gi,int gf,int gg,int go,
    float si,float sf,float sg,float so,float bi,float bf,float bg,float bo,float as,float&cell){
  float vi=__fmaf_rn((float)gi,as*si,bi), vf=__fmaf_rn((float)gf,as*sf,bf);
  float vg=__fmaf_rn((float)gg,as*sg,bg), vo=__fmaf_rn((float)go,as*so,bo);
  float I=fminf(fmaxf(__fmaf_rn(vi,0.2f,0.5f),0.f),1.f);
  float F=fminf(fmaxf(__fmaf_rn(vf,0.2f,0.5f),0.f),1.f);
  float O=fminf(fmaxf(__fmaf_rn(vo,0.2f,0.5f),0.f),1.f);
  float G=fminf(fmaxf(vg,-1.f),1.f);
  cell=__fmaf_rn(F,cell,I*G);
  return clamp_i8(rintf(O*tanhf(cell)*127.0f));
}

// ---- Mma types: identical to the per-launch winner ----
#ifndef STG
#define STG 4
#endif
using TbShape  = cutlass::gemm::GemmShape<128,128,64>;
using WpShape  = cutlass::gemm::GemmShape<64,64,64>;
using InShape  = cutlass::gemm::GemmShape<16,8,32>;
using MmaDef = cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TbShape, WpShape, InShape, STG, cutlass::arch::OpMultiplyAddSaturate>;
using Mma = MmaDef::ThreadblockMma;
constexpr int IterM = WpShape::kM/16, IterN = WpShape::kN/8, NB32 = IterN/4;
constexpr int CH_TB = TbShape::kN/4;

using TbShapeD = cutlass::gemm::GemmShape<64,64,64>;
using WpShapeD = cutlass::gemm::GemmShape<32,32,64>;
using MmaDefD = cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16, int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TbShapeD, WpShapeD, InShape, 4, cutlass::arch::OpMultiplyAddSaturate>;
using MmaD = MmaDefD::ThreadblockMma;
constexpr int IterMD = WpShapeD::kM/16, IterND = WpShapeD::kN/8;

// smem plan: [0, MAINLOOP_MAX) shared by both phases' mainloops (sH writeout staging
// aliases it post-mainloop); then cell tile + scales (live ACROSS the gate mainloop).
constexpr size_t MAINLOOP_MAX =
    sizeof(typename Mma::SharedStorage) > sizeof(typename MmaD::SharedStorage) ?
    sizeof(typename Mma::SharedStorage) : sizeof(typename MmaD::SharedStorage);
constexpr size_t SMEM_TOTAL = MAINLOOP_MAX + (size_t)TbShape::kM*CH_TB*sizeof(cellT)
                              + (size_t)8*CH_TB*4;

// ---- phase 1: one down-proj tile (tm,tn) ----
__device__ void downproj_tile(char* smem, const int8_t* hprev, const int8_t* Wd,
    const float* comb, int8_t* Aout, int Nrows, int tm, int tn){
  typename MmaD::SharedStorage* shared = reinterpret_cast<typename MmaD::SharedStorage*>(smem);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int tbM0 = tm*TbShapeD::kM, tbN0 = tn*TbShapeD::kN;
  typename MmaD::IteratorA::Params pA(cutlass::layout::RowMajor(H));
  typename MmaD::IteratorB::Params pB(cutlass::layout::ColumnMajor(H));
  typename MmaD::IteratorA itA(pA, const_cast<int8_t*>(hprev), {Nrows, H}, tid, {tbM0, 0});
  typename MmaD::IteratorB itB(pB, const_cast<int8_t*>(Wd), {H, K_HH}, tid, {0, tbN0});
  MmaD mma(*shared, tid, warp, lane);
  typename MmaD::FragmentC acc; acc.clear();
  mma(H/TbShapeD::kK, acc, itA, itB, acc);
  constexpr int WCM = TbShapeD::kM/WpShapeD::kM;
  const int wm = warp % WCM, wn = warp / WCM;
  const int32_t* af = reinterpret_cast<const int32_t*>(&acc);
  #pragma unroll
  for(int m_it=0;m_it<IterMD;++m_it)
    #pragma unroll
    for(int n_it=0;n_it<IterND;++n_it)
      #pragma unroll
      for(int e=0;e<4;++e){
        int row = tbM0 + wm*WpShapeD::kM + m_it*16 + gid + 8*(e>>1);
        int col = tbN0 + wn*WpShapeD::kN + n_it*8 + 2*tg + (e&1);
        int v = af[(m_it + n_it*IterMD)*4 + e];
        Aout[(long)row*KC + col] = clamp_i8(rintf((float)v*comb[col]));
      }
  __syncthreads();   // smem reuse safety before the next tile's prologue
}

// ---- phase 2: one gate tile (tm,tn) with fused LSTM epilogue ----
__device__ void gate_tile(char* smem, const int8_t* A, const int8_t* Bw,
    const float* wscale, const float* bias, cellT* cell, int8_t* hout,
    int Nrows, int tm, int tn){
  typename Mma::SharedStorage* shared = reinterpret_cast<typename Mma::SharedStorage*>(smem);
  cellT* sCell = reinterpret_cast<cellT*>(smem + MAINLOOP_MAX);
  float* sSc   = reinterpret_cast<float*>(sCell + TbShape::kM*CH_TB);
  int8_t* sH   = reinterpret_cast<int8_t*>(smem);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int tbM0 = tm*TbShape::kM, tbN0 = tn*TbShape::kN;
  const int ch0 = tbN0/4;
  const float AS=1.0f/127.0f;

  {   // cell + scale prefetch, first commit group (overlaps the mainloop)
    const int rowsz = CH_TB*sizeof(cellT);
    for(int i=tid; i<TbShape::kM*(rowsz/16); i+=blockDim.x){
      int r=i/(rowsz/16), c=i%(rowsz/16);
      void* dst=(char*)&sCell[r*CH_TB]+c*16;
      const void* src=(const char*)&cell[(long)(tbM0+r)*H+ch0]+c*16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    }
    for(int i=tid; i<8*CH_TB/4; i+=blockDim.x){
      int g=i/(CH_TB/4), c=i%(CH_TB/4);
      const float* srcv = (g<4)? &wscale[g*H+ch0] : &bias[(g-4)*H+ch0];
      void* dst=(char*)&sSc[g*CH_TB]+c*16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"((unsigned)__cvta_generic_to_shared(dst)), "l"((const void*)((const char*)srcv+c*16)));
    }
    asm volatile("cp.async.commit_group;\n");
  }

  typename Mma::IteratorA::Params pA(cutlass::layout::RowMajor(KC));
  typename Mma::IteratorB::Params pB(cutlass::layout::ColumnMajor(KC));
  typename Mma::IteratorA itA(pA, const_cast<int8_t*>(A), {Nrows, KC}, tid, {tbM0, 0});
  typename Mma::IteratorB itB(pB, const_cast<int8_t*>(Bw), {KC, NG}, tid, {0, tbN0});
  Mma mma(*shared, tid, warp, lane);
  typename Mma::FragmentC acc; acc.clear();
  mma(KC/TbShape::kK, acc, itA, itB, acc);

  constexpr int WCM = TbShape::kM/WpShape::kM;
  const int wm = warp % WCM, wn = warp / WCM;
  const int32_t* af = reinterpret_cast<const int32_t*>(&acc);
  asm volatile("cp.async.wait_all;\n");
  __syncthreads();

  #pragma unroll
  for(int m_it=0;m_it<IterM;++m_it)
    #pragma unroll
    for(int b=0;b<NB32;++b)
      #pragma unroll
      for(int e=0;e<4;++e){
        int rloc = wm*WpShape::kM + m_it*16 + gid + 8*(e>>1);
        int cloc = (wn*WpShape::kN + b*32)/4 + 2*tg + (e&1);
        int gi = af[(m_it + (b*4+0)*IterM)*4 + e];
        int gf = af[(m_it + (b*4+1)*IterM)*4 + e];
        int gg = af[(m_it + (b*4+2)*IterM)*4 + e];
        int go = af[(m_it + (b*4+3)*IterM)*4 + e];
        float cv = (float)sCell[rloc*CH_TB + cloc];
        int8_t hn = epi_elem(gi,gf,gg,go,
            sSc[0*CH_TB+cloc],sSc[1*CH_TB+cloc],sSc[2*CH_TB+cloc],sSc[3*CH_TB+cloc],
            sSc[4*CH_TB+cloc],sSc[5*CH_TB+cloc],sSc[6*CH_TB+cloc],sSc[7*CH_TB+cloc], AS, cv);
        sCell[rloc*CH_TB + cloc] = (cellT)cv;
        sH[rloc*CH_TB + cloc] = hn;
      }
  __syncthreads();
  for(int i=tid; i<TbShape::kM*(CH_TB/4); i+=blockDim.x){
    int r=i/(CH_TB/4), c=i%(CH_TB/4);
    *(int*)&hout[(long)(tbM0+r)*H + ch0 + c*4] = *(const int*)&sH[r*CH_TB + c*4];
  }
  constexpr int CB16 = CH_TB*sizeof(cellT)/16;
  for(int i=tid; i<TbShape::kM*CB16; i+=blockDim.x){
    int r=i/CB16, c=i%CB16;
    *(int4*)((char*)&cell[(long)(tbM0+r)*H + ch0] + c*16) = *(const int4*)((const char*)&sCell[r*CH_TB] + c*16);
  }
  __syncthreads();   // smem reuse safety before the next tile
}

// ---- the persistent kernel ----
__global__ void __launch_bounds__(128,2) persistent_kernel(
    const int8_t* __restrict__ w_dn, const float* __restrict__ comb,
    const int8_t* __restrict__ Bw, const float* __restrict__ wscale,
    const float* __restrict__ bias, cellT* __restrict__ cell,
    int8_t* __restrict__ hh_all, int8_t* __restrict__ Aring,
    int* __restrict__ flags, int N, int T, int reverse)
{
  extern __shared__ char smem[];
  const int bid = blockIdx.x, G = gridDim.x, tid = threadIdx.x;
  const int ntile_d = (N/TbShapeD::kM)*(K_HH/TbShapeD::kN);
  const int ntile_g = (N/TbShape::kM)*(NG/TbShape::kN);

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int prev = reverse ? (tt+1) : tt;
    int out  = reverse ? tt : (tt+1);
    const int8_t* hprev = hh_all + (size_t)prev*N*H;
    int8_t* hout = hh_all + (size_t)out*N*H;
    int8_t* At = Aring + (size_t)tt*N*KC;

    for(int tile=bid; tile<ntile_d; tile+=G){
      downproj_tile(smem, hprev, w_dn, comb, At, N,
                    tile % (N/TbShapeD::kM), tile / (N/TbShapeD::kM));
    }
#ifndef NO_BARRIER
    __threadfence(); __syncthreads();
    if(tid==0){ atomicAdd(flags,1); int need=G*(2*t+1);
      while(atomicAdd(flags,0)<need){} }
    __syncthreads();
#endif
    for(int tile=bid; tile<ntile_g; tile+=G){
      gate_tile(smem, At, Bw, wscale, bias, cell, hout, N,
                tile % (N/TbShape::kM), tile / (N/TbShape::kM));
    }
#ifndef NO_BARRIER
    __threadfence(); __syncthreads();
    if(tid==0){ atomicAdd(flags,1); int need=G*(2*t+2);
      while(atomicAdd(flags,0)<need){} }
    __syncthreads();
#endif
  }
}

// ---- naive GPU reference (identical epi math) ----
__global__ void ref_kernel(const int8_t* x,const int8_t* w_dn,const float* comb,
    const int8_t* Bi,const int8_t* Bf,const int8_t* Bg,const int8_t* Bo,
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/127.0f; const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  int8_t* h=new int8_t[H]; cellT* c=new cellT[H]; int8_t hh[K_HH];
  for(int i=0;i<H;++i){h[i]=0;c[i]=(cellT)0.f;}
  for(int t=0;t<T;++t){ int tt=reverse?(T-1-t):t; int ws=reverse?tt:(tt+1);
    for(int k=0;k<K_HH;++k){ int a=0; for(int cc=0;cc<H;++cc) a+=(int)h[cc]*(int)w_dn[k*H+cc];
      hh[k]=clamp_i8(rintf((float)a*comb[k])); }
    const int8_t* xr=x+((long)tt*N+r)*R;
    for(int oc=0;oc<H;++oc){ int g[4];
      for(int gg=0;gg<4;++gg){ const int8_t* w=Bp[gg]+(long)oc*KC; int a=0;
        for(int kc=0;kc<K_HH;++kc) a+=(int)hh[kc]*(int)w[kc];
        for(int kc=0;kc<R;++kc) a+=(int)xr[kc]*(int)w[K_HH+kc]; g[gg]=a; }
      float cv=(float)c[oc];
      int8_t hn=epi_elem(g[0],g[1],g[2],g[3],wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv); c[oc]=(cellT)cv;
      hh_all[((long)ws*N+r)*H+oc]=hn; }
    for(int oc=0;oc<H;++oc) h[oc]=hh_all[((long)ws*N+r)*H+oc];
  }
  delete[] h; delete[] c;
}

__global__ void fill_x_kernel(const int8_t* x, int8_t* Aring, int N, int T){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  long total=(long)T*N*(R/4);
  if(i>=total) return;
  long w=i; int c4=w%(R/4); w/=(R/4); int r=w%N; long t=w/N;
  ((int*)&Aring[(t*N+r)*KC + K_HH])[c4] = ((const int*)&x[(t*N+r)*R])[c4];
}

int main(int argc,char**argv){
  int N=256,T=64,reverse=0,bench=0,n_cmp=32,dev=0,G=216;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T"))T=atoi(argv[++i]); else if(!strcmp(argv[i],"--reverse"))reverse=1;
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--ncmp"))n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--grid"))G=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(N%TbShape::kM) N=((N+TbShape::kM-1)/TbShape::kM)*TbShape::kM;
  printf("PERSISTENT fused-cutlass: grid=%d TB %dx%dx%d w%dx%d s%d cell=%s N=%d T=%d rev=%d smem=%zu\n",
    G,TbShape::kM,TbShape::kN,TbShape::kK,WpShape::kM,WpShape::kN,STG,
    sizeof(cellT)==2?"f16":"f32",N,T,reverse,SMEM_TOTAL);

  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ux(-1,1);
  std::vector<int8_t> hx((size_t)T*N*R); for(size_t i=0;i<hx.size();++i) hx[i]=(int8_t)lrintf(fminf(fmaxf(ux(rng),-1.f),1.f)*127.f);
  std::vector<int8_t> hwdn((size_t)K_HH*H); std::vector<float> hcm(K_HH);
  for(int k=0;k<K_HH;++k){ float mx=1e-8f; std::vector<float> row(H);
    for(int c=0;c<H;++c){row[c]=nd(rng)*0.02f;mx=fmaxf(mx,fabsf(row[c]));} float sc=mx/127.f; hcm[k]=sc;
    for(int c=0;c<H;++c) hwdn[(size_t)k*H+c]=(int8_t)lrintf(row[c]/sc); }
  std::vector<int8_t> hB[4]; std::vector<float> hws((size_t)4*H),hbs((size_t)4*H);
  for(int g=0;g<4;++g){ hB[g].resize((size_t)H*KC);
    for(int oc=0;oc<H;++oc){ float mx=1e-8f; std::vector<float> row(KC);
      for(int kc=0;kc<KC;++kc){row[kc]=nd(rng)*0.1f;mx=fmaxf(mx,fabsf(row[kc]));}
      float sc=mx/127.f; hws[(size_t)g*H+oc]=sc; hbs[(size_t)g*H+oc]=nd(rng)*0.05f;
      for(int kc=0;kc<KC;++kc) hB[g][(size_t)oc*KC+kc]=(int8_t)lrintf(row[kc]/sc); } }
  std::vector<int8_t> hBint((size_t)NG*KC);
  for(int n=0;n<NG;++n){ int B32=n/32, g=(n%32)/8, p=n%8; int ch=B32*8+p;
    memcpy(&hBint[(size_t)n*KC], &hB[g][(size_t)ch*KC], KC); }

  int8_t *dx,*dwdn,*dB[4],*dBint,*dring,*dAring; float *dcm,*dws,*dbs; cellT *dcell; int* dflags;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size())); CK(cudaMalloc(&dBint,hBint.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dAring,(size_t)T*N*KC)); CK(cudaMalloc(&dcell,(size_t)N*H*sizeof(cellT)));
  CK(cudaMalloc(&dflags,64));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBint,hBint.data(),hBint.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));

  CK(cudaFuncSetAttribute(persistent_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)SMEM_TOTAL));
  { long tot=(long)T*N*(R/4); fill_x_kernel<<<(tot+255)/256,256>>>(dx,dAring,N,T); CK(cudaDeviceSynchronize()); }

  auto run=[&](int rev){
    CK(cudaMemset(dcell,0,(size_t)N*H*sizeof(cellT)));
    CK(cudaMemset(dflags,0,64));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H));
    persistent_kernel<<<G,128,SMEM_TOTAL>>>(dwdn,dcm,dBint,dws,dbs,dcell,dring,dAring,dflags,N,T,rev);
  };

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    int8_t* dref; CK(cudaMalloc(&dref,ring_bytes));
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    run(reverse); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ref_kernel<<<(n_cmp+31)/32,32>>>(dx,dwdn,dcm,dB[0],dB[1],dB[2],dB[3],dws,dbs,dref,N,T,reverse,n_cmp);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> a(ring_bytes),b(ring_bytes);
    CK(cudaMemcpy(a.data(),dring,ring_bytes,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(),dref,ring_bytes,cudaMemcpyDeviceToHost));
    long mism=0,maxd=0,tot=0;
    for(int s=0;s<=T;++s){ if((s==0&&!reverse)||(s==T&&reverse))continue;
      for(int r=0;r<n_cmp;++r)for(int oc=0;oc<H;++oc){ long idx=((long)s*N+r)*H+oc;
        int d=abs((int)a[idx]-(int)b[idx]); tot++; if(d){mism++; if(d>maxd)maxd=d;} } }
    printf("[%s] mism=%ld/%ld maxd=%ld -> %s\n",reverse?"reverse":"forward",mism,tot,maxd,mism==0?"BIT-EXACT PASS":"FAIL");
    return mism==0?0:1;
  }
  for(int w=0;w<3;++w){ run(reverse); } CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<7;++r){ cudaEventRecord(e0); run(reverse); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); us.push_back((double)ms/T*1000.0); }
  std::sort(us.begin(),us.end());
  printf("PERSISTENT step: %.2f us/step (median of 7, T=%d, one launch)\n",us[3],T);
  return 0;
}
