// CUTLASS-pipelined fused factored-LSTM step: multistage cp.async mainloop
// (cutlass::gemm::threadblock::DefaultMma, the pipeline every hand kernel lacked)
// + hand-written fused LSTM epilogue (the 32MB int32 C is NEVER written).
//
// Key trick (koi-style): gate weights are interleaved host-side in 8-column blocks --
// GEMM column n = B32*32 + g*8 + p  ->  gate g of channel (B32*8 + p).  With the
// m16n8k32 accumulator layout (thread holds cols 2tg,2tg+1 of each n8 tile), four
// consecutive n8-tiles = one 32-col block = the SAME thread holds gates i,f,g,o of its
// channels in-fragment -> the cell update is thread-local.  ONE accumulator set, normal
// large tiles (no 4-accumulator bN=32 pressure).
//
// Per step (graph-captured): [down-proj GEMM+quant epilogue -> A ring] then
// [gate GEMM + LSTM epilogue -> int8 h + f32 cell].  x half of A pre-filled once.
//
// Build (headers vendored in fluke/thirdparty/cutlass):
//   nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013 \
//        -I../thirdparty/cutlass/include -DSTG=4 -DFP16CELL fused_cutlass.cu -o fused_cutlass
//
// RESULTS (A100, N=2048, T=2048, warm, graph-captured, median of 7):
//   *** 37.08 us/step *** = down-proj 8.7us + fused gate 30.2us  (f16 cell; f32: 39.8)
//   vs two-kernel baseline 64us, persistent hand-CUDA 89us, cuBLAS C-materializing
//   gate GEMM alone 32us, dorado ~10us.  Bit-exact vs the naive GPU ref, fwd+reverse,
//   any T (explicit __fmaf_rn epilogue -> no fp-contract drift).
//   WINNING CONFIG: gate TB 128x128x64 / warp 64x64 / stages 4 (whole K=256 prefetched)
//   + f16 cell; down-proj TB 64x64x64 / warp 32x32 / stages 4.  Swept: TB {128x256,
//   256x128} worse (tail/traffic); warp {32x64,64x32,32x32} worse (mma efficiency);
//   down-proj TB 32x64 worse.
//   ncu (gate): DRAM write 384 B/step (the 32MB int32 C is GONE - everything
//   L2-resident), long_scoreboard 33.7 -> 0.18 (the multistage cp.async pipeline +
//   cell/scale prefetch killed the latency wall), imma-util 3.2 -> 10.9%, 230 regs.
//   Residual wall: wait-stall at 11% occupancy (128 int32 accum fragment/thread at
//   warp 64x64) - the short-K (4 k-iters) GEMM is prologue/latency-bound, ~3x its
//   ~10us traffic floor.
//
//   FINAL OPTIMIZATION ROUND (all measured NEGATIVE - 37.08 is the converged number):
//   * persistence (persistent_cutlass.cu): 37.86 - launch overhead was already graph-
//     amortized; barrier quantizes the gate to 3 tile-waves (+3us barriers).
//   * warp-k-split (TB kK=128 / warp kK=64, kPartitionsK=2, 8 warps): plain GEMM
//     42.3us vs 36.5 - the k-partition smem reduce costs more than the warps recover.
//     (warp kK=32 statically impossible: needs >=2 warp-gemm iterations.)
//   * more warps / smaller warp tiles (32x64/64x32/32x32 @256thr): 34.9-36.2 gate
//     (vs 30.3) - mma efficiency loss beats the latency hiding.
//   * split-K down-proj (-DUSE_SPLITK, S=2/4/8): partial kernel 9.6/11.7/17.1us vs
//     8.7 baseline + reduce 3.6-4.7us - NOT fill-bound; prologue-bound like the gate;
//     at cuBLAS parity (8.1us) the down-proj is at its floor.
//   * fused down-proj in gate prologue: dead on arithmetic (redundant recompute =
//     +8.6 GMAC/step = 4x total MMA work ~ 27us at 100% util) - not built.
//   ncu final: issue_active 32% (latency-bound, not issue-bound), L2 26% (headroom
//   irrelevant - the wall is dependency latency at 8 warps/SM).  This shape
//   (M=2048,K=256,N=4096 int8 + K=1024,N=128) is scheduling-converged on A100.
//
//   A/B EXPERIMENT ROUND (post-koi-SASS-decode, N fixed=2048):
//   A0 (compass): gate is L1TEX/smem-pipe bound (44.5% SOL); DRAM only 9.2% (5.8MB/step,
//     weights already L2-hot: 1MB<<40MB L2); imma-util 11%; occupancy 11% (smem->2CTA/SM,
//     4 warps each); ALL stalls low (wait 1.79 top). So 37us is throughput-bound on the
//     L1/smem pipe at low occupancy - NOT weight-DRAM-reload, NOT compute, NOT a stall.
//   A1 (L2-pin gate_w via cudaAccessPolicyWindow, -DL2PIN): ZERO effect (124.58us both at
//     N=8192) - exactly as A0 predicts (DRAM already low). No sync-free L2 win exists.
//   A2 (pipeline stages 2/3/4): 38.7/37.5/37.0us - marginal; STG4 best. Short-K prologue
//     is NOT the dominant cost.
//   B1 (persistent koi_flstm.cu + nanosleep-backoff release/acquire SEQ handoff -- the
//     CORRECT non-hot-spin sync): 193us/step (N=1536 one-wave) / 386us (N=2048 multiwave).
//     Bit-exact but ~5x WORSE. Fixing the hot-spin did NOT rescue it: the factorised-LSTM
//     recurrence is SEQUENTIAL (down(t)->gate(t)->h(t)->down(t+1)), so the down & gate
//     CTAs ALTERNATE (no overlap) and every step pays cross-CTA handoff latency x2.
//   B2 (cooperative grid.sync): STRUCTURALLY IMPOSSIBLE at N=2048 -- one-wave persistent
//     needs <=108 co-resident CTAs, but grid(9,N/bM) at bM>=171 (for <=108) needs bM=256
//     whose A[256,256]=64KB + 128KB resident weights = 192KB > 163KB smem. Won't fit.
//   VERDICT: at fixed N=2048, NOTHING beats 37.05us. koi's ~9 ns/row is its N=1536-one-
//   wave resident-weight config + its UB no-sync race; neither is available at forced
//   N=2048 (>108 CTAs, one-wave doesn't fit smem). 37.05us is the SAFE, bit-exact ceiling.
//
//   SMEM-PIPE hypothesis test (does resident-weights relieve the A0 44.5% L1TEX wall?):
//   ncu gate shared-mem wavefronts: LOADS(ldmatrix) 1.09M (74%) vs STORES(cp.async) 0.30M
//   (20%). Resident weights only remove WEIGHT cp.async-stores (~half of stores = ~10% of
//   the pipe); the ldmatrix-READS that DOMINATE (74%) are IDENTICAL whether weights are
//   streamed or resident (you ldmatrix weights from smem every k-iter either way).
//   => resident weights save at most ~10% of the smem pipe, NOT koi's 2x. HYPOTHESIS
//   FALSE by measurement (no persistent build needed to refute it).
//   koi's real 2x = fragment REUSE: koi SASS does 8 IMMA/LDSM; CUTLASS does 393216/110592
//   = 3.56 IMMA/LDSM (2.25x more ldmatrix per unit compute -> 2.25x the smem-pipe on the
//   dominant term). That 3.56 is fixed by CUTLASS's fragment scheme, NOT tile-config-
//   tunable (TB256x64 gave identical 393216/110592, and was SLOWER at N=2048: 40.7us).
//   Matching koi's 8:1 needs its exact hand fragment-reuse mainloop, which the from-
//   scratch persistent kernels never achieved with pipeline-quality+one-wave+sync.
//   FINAL: 37.05us is the safe ceiling at N=2048.
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
#define NG (4*H)      // gemm N for the gate
#include <cuda_fp16.h>
#ifdef FP16CELL
typedef __half cellT;   // halves the cell DRAM round-trip (gate's dominant traffic).
#else
typedef float cellT;    // reference/default
#endif
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ int8_t clamp_i8(float q){ q=fminf(fmaxf(q,-127.f),127.f); return (int8_t)(int)q; }
// deterministic epilogue: every mul-add is an explicit __fmaf_rn so codegen is identical
// in the fused kernel and the reference (no fp-contract ambiguity -> bit-exact at any T).
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

// ---------------- gate kernel: pipelined mainloop + fused LSTM epilogue ----------------
#ifndef TBM
#define TBM 128
#endif
#ifndef TBN
#define TBN 128
#endif
#ifndef STG
#define STG 3
#endif
#ifndef WPM
#define WPM 64
#endif
#ifndef WPN
#define WPN 64
#endif
using TbShape  = cutlass::gemm::GemmShape<TBM,TBN,64>;
using WpShape  = cutlass::gemm::GemmShape<WPM,WPN,64>;
using InShape  = cutlass::gemm::GemmShape<16,8,32>;
constexpr int kStages = STG;

using MmaDef = cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TbShape, WpShape, InShape, kStages,
    cutlass::arch::OpMultiplyAddSaturate>;
using Mma = MmaDef::ThreadblockMma;
constexpr int IterM = WpShape::kM/16;   // 4
constexpr int IterN = WpShape::kN/8;    // 8
constexpr int NB32  = IterN/4;          // 32-col blocks per warp (2)

constexpr int CH_TB = TbShape::kN/4;    // channels per CTA (32): CTA covers a CONTIGUOUS
                                        // channel window [tbN0/4, +CH_TB) under the interleave.
__global__ void gate_kernel(
    const int8_t* __restrict__ A,     // [N, Kc]  (hh_down | x)
    const int8_t* __restrict__ Bw,    // [NG, Kc] interleaved gate weights (col-major B)
    const float* __restrict__ wscale, // [4,H]
    const float* __restrict__ bias,   // [4,H]
    cellT* __restrict__ cell,         // [N,H]
    int8_t* __restrict__ hout,        // [N,H]  (ring slot)
    int Nrows)
{
  extern __shared__ char smem[];
  typename Mma::SharedStorage* shared = reinterpret_cast<typename Mma::SharedStorage*>(smem);
  cellT* sCell = reinterpret_cast<cellT*>(smem + sizeof(typename Mma::SharedStorage)); // [TBM][CH_TB]
  float* sSc   = reinterpret_cast<float*>(sCell + TbShape::kM*CH_TB);  // [8][CH_TB] (scales, biases)
  int8_t* sH   = reinterpret_cast<int8_t*>(smem);  // reuse mainloop smem AFTER mma
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int tbM0 = blockIdx.x*TbShape::kM, tbN0 = blockIdx.y*TbShape::kN;
  const int ch0 = tbN0/4;                    // first channel of this CTA's window
  const float AS=1.0f/127.0f;

  // prefetch cell tile [128, CH_TB] f32 + scales/biases via cp.async, OVERLAPPED with the
  // mainloop (independent of the GEMM; first-committed group completes first).
  {
    const int rowsz = CH_TB*sizeof(cellT);   // bytes per staged cell row
    for(int i=tid; i<TbShape::kM*(rowsz/16); i+=blockDim.x){
      int r=i/(rowsz/16), c=i%(rowsz/16);
      void* dst=(char*)&sCell[r*CH_TB]+c*16;
      const void* src=(const char*)&cell[(long)(tbM0+r)*H+ch0]+c*16;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src));
    }
    for(int i=tid; i<8*CH_TB/4; i+=blockDim.x){   // 8 rows of [CH_TB] f32, 4B granule
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

  // warp raster: m fastest (cutlass convention)
  constexpr int WCM = TbShape::kM/WpShape::kM;
  const int wm = warp % WCM, wn = warp / WCM;
  const int32_t* af = reinterpret_cast<const int32_t*>(&acc);

  asm volatile("cp.async.wait_all;\n");
  __syncthreads();     // mainloop smem consumed; sCell/sSc ready; sH may now alias mainloop

  // fused LSTM epilogue -> smem staging: frag idx = (m_it + n_it*IterM)*4 + e ; n_it = b*4+gate.
  #pragma unroll
  for(int m_it=0;m_it<IterM;++m_it)
    #pragma unroll
    for(int b=0;b<NB32;++b)
      #pragma unroll
      for(int e=0;e<4;++e){
        int rloc = wm*WpShape::kM + m_it*16 + gid + 8*(e>>1);
        int cloc = (wn*WpShape::kN + b*32)/4 + 2*tg + (e&1);   // channel within window
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
  // coalesced writeout: h tile [TBM, CH_TB] int8 (int words), cell tile (int4 chunks)
  for(int i=tid; i<TbShape::kM*(CH_TB/4); i+=blockDim.x){
    int r=i/(CH_TB/4), c=i%(CH_TB/4);
    *(int*)&hout[(long)(tbM0+r)*H + ch0 + c*4] = *(const int*)&sH[r*CH_TB + c*4];
  }
  constexpr int CB16 = CH_TB*sizeof(cellT)/16;   // 16B chunks per cell row
  for(int i=tid; i<TbShape::kM*CB16; i+=blockDim.x){
    int r=i/CB16, c=i%CB16;
    *(int4*)((char*)&cell[(long)(tbM0+r)*H + ch0] + c*16) = *(const int4*)((const char*)&sCell[r*CH_TB] + c*16);
  }
}

// ---------------- down-proj kernel: pipelined mainloop + quant epilogue ----------------
#ifndef DTBM
#define DTBM 64
#endif
#ifndef DWPM
#define DWPM 32
#endif
using TbShapeD = cutlass::gemm::GemmShape<DTBM,64,64>;
using WpShapeD = cutlass::gemm::GemmShape<DWPM,32,64>;
using MmaDefD = cutlass::gemm::threadblock::DefaultMma<
    int8_t, cutlass::layout::RowMajor, 16,
    int8_t, cutlass::layout::ColumnMajor, 16,
    int32_t, cutlass::layout::RowMajor,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    TbShapeD, WpShapeD, InShape, 4,
    cutlass::arch::OpMultiplyAddSaturate>;
using MmaD = MmaDefD::ThreadblockMma;
constexpr int IterMD = WpShapeD::kM/16;   // 4
constexpr int IterND = WpShapeD::kN/8;    // 4

__global__ void downproj_kernel(
    const int8_t* __restrict__ hprev,   // [N,H]
    const int8_t* __restrict__ Wd,      // [K_hh, H] row-major = B col-major [H x K_hh]
    const float* __restrict__ comb,     // [K_hh]
    int8_t* __restrict__ Aout,          // [N, Kc]  (writes cols 0..K_hh-1, stride Kc)
    int Nrows)
{
  extern __shared__ char smem[];
  typename MmaD::SharedStorage* shared = reinterpret_cast<typename MmaD::SharedStorage*>(smem);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int tbM0 = blockIdx.x*TbShapeD::kM, tbN0 = blockIdx.y*TbShapeD::kN;
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
}

// ---- split-K down-proj (Angle 2): grid.z = SPLITK slices of K=H; int32 partials to a
// workspace, then reduce+quant (int32 adds associative -> bit-exact preserved). ----
#ifndef SPLITK
#define SPLITK 4
#endif
__global__ void downproj_splitk_kernel(
    const int8_t* __restrict__ hprev, const int8_t* __restrict__ Wd,
    int32_t* __restrict__ ws,           // [SPLITK, N, K_hh] partials
    int Nrows)
{
  extern __shared__ char smem[];
  typename MmaD::SharedStorage* shared = reinterpret_cast<typename MmaD::SharedStorage*>(smem);
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int tbM0 = blockIdx.x*TbShapeD::kM, tbN0 = blockIdx.y*TbShapeD::kN;
  const int z = blockIdx.z;
  constexpr int KSL = H/SPLITK;                 // K slice per z
  typename MmaD::IteratorA::Params pA(cutlass::layout::RowMajor(H));
  typename MmaD::IteratorB::Params pB(cutlass::layout::ColumnMajor(H));
  // k-window [z*KSL, (z+1)*KSL): offset the start; extent caps the end so the
  // iterator's predicates stop this slice at its boundary.
  typename MmaD::IteratorA itA(pA, const_cast<int8_t*>(hprev), {Nrows, (z+1)*KSL}, tid, {tbM0, z*KSL});
  typename MmaD::IteratorB itB(pB, const_cast<int8_t*>(Wd), {(z+1)*KSL, K_HH}, tid, {z*KSL, tbN0});
  MmaD mma(*shared, tid, warp, lane);
  typename MmaD::FragmentC acc; acc.clear();
  mma(KSL/TbShapeD::kK, acc, itA, itB, acc);
  constexpr int WCM = TbShapeD::kM/WpShapeD::kM;
  const int wm = warp % WCM, wn = warp / WCM;
  const int32_t* af = reinterpret_cast<const int32_t*>(&acc);
  int32_t* wsz = ws + (size_t)z*Nrows*K_HH;
  #pragma unroll
  for(int m_it=0;m_it<IterMD;++m_it)
    #pragma unroll
    for(int n_it=0;n_it<IterND;++n_it)
      #pragma unroll
      for(int e=0;e<4;++e){
        int row = tbM0 + wm*WpShapeD::kM + m_it*16 + gid + 8*(e>>1);
        int col = tbN0 + wn*WpShapeD::kN + n_it*8 + 2*tg + (e&1);
        wsz[(long)row*K_HH + col] = af[(m_it + n_it*IterMD)*4 + e];
      }
}
__global__ void downproj_reduce_kernel(
    const int32_t* __restrict__ ws, const float* __restrict__ comb,
    int8_t* __restrict__ Aout, int Nrows)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i >= Nrows*K_HH) return;
  int row = i / K_HH, col = i % K_HH;
  int s = 0;
  #pragma unroll
  for(int z=0; z<SPLITK; ++z) s += ws[(size_t)z*Nrows*K_HH + i];
  Aout[(long)row*KC + col] = clamp_i8(rintf((float)s*comb[col]));
}

// ---------------- naive GPU reference (identical epi math -> bit-exact) ----------------
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

// x pre-fill: A ring x-half for all T (once, outside the graph)
__global__ void fill_x_kernel(const int8_t* x, int8_t* Aring, int N, int T){
  long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
  long total=(long)T*N*(R/4);
  if(i>=total) return;
  long w=i; int c4=w%(R/4); w/=(R/4); int r=w%N; long t=w/N;
  ((int*)&Aring[(t*N+r)*KC + K_HH])[c4] = ((const int*)&x[(t*N+r)*R])[c4];
}

int main(int argc,char**argv){
  int N=256,T=64,reverse=0,bench=0,n_cmp=32,dev=0;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T"))T=atoi(argv[++i]); else if(!strcmp(argv[i],"--reverse"))reverse=1;
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--ncmp"))n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(N%TbShape::kM) N=((N+TbShape::kM-1)/TbShape::kM)*TbShape::kM;
  printf("CUTLASS-fused: TB %dx%dx%d warp %dx%d stages %d  N=%d T=%d rev=%d\n",
    TbShape::kM,TbShape::kN,TbShape::kK,WpShape::kM,WpShape::kN,kStages,N,T,reverse);

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
  // interleaved gate weights: gemm col n = B32*32 + g*8 + p -> gate g, channel B32*8+p
  std::vector<int8_t> hBint((size_t)NG*KC);
  for(int n=0;n<NG;++n){ int B32=n/32, g=(n%32)/8, p=n%8; int ch=B32*8+p;
    memcpy(&hBint[(size_t)n*KC], &hB[g][(size_t)ch*KC], KC); }
  // down-proj B: W_dn [K_hh, H] row-major == B col-major [H x K_hh] (k=hidden, n=khh)? NO:
  // gemm: hh_down[m, n=khh] = sum_k h[m,k]*W_dn[n,k]; B col-major (k,n) at n*ldb+k = W_dn[n*H+k]. OK.

  int8_t *dx,*dwdn,*dB[4],*dBint,*dring,*dAring; float *dcm,*dws,*dbs; cellT *dcell;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size())); CK(cudaMalloc(&dBint,hBint.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dAring,(size_t)T*N*KC));          // per-t A = [hh_down | x]
  CK(cudaMalloc(&dcell,(size_t)N*H*sizeof(cellT)));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBint,hBint.data(),hBint.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));

  size_t smem_g=sizeof(typename Mma::SharedStorage) + (size_t)TbShape::kM*CH_TB*sizeof(cellT) + (size_t)8*CH_TB*4;
  size_t smem_d=sizeof(typename MmaD::SharedStorage);
  CK(cudaFuncSetAttribute(gate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_g));
  CK(cudaFuncSetAttribute(downproj_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_d));
#ifdef L2PIN
  { size_t l2=0; cudaDeviceGetAttribute((int*)&l2,cudaDevAttrMaxPersistingL2CacheSize,dev);
    size_t pin=hBint.size(); if(pin>l2)pin=l2; cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,pin);
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dBint; av.accessPolicyWindow.num_bytes=pin;
    av.accessPolicyWindow.hitRatio=1.0f; av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&av);
    printf("[A1] gate_w L2-pinned %zu bytes\n",pin); }
#endif
  CK(cudaFuncSetAttribute(downproj_splitk_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_d));
  int32_t* dwsK; CK(cudaMalloc(&dwsK,(size_t)SPLITK*N*K_HH*4));   // split-K workspace
  printf("smem: gate %zu  down %zu  splitk=%d\n",smem_g,smem_d,SPLITK);

  // pre-fill x halves of all A_t (outside graph)
  { long tot=(long)T*N*(R/4); fill_x_kernel<<<(tot+255)/256,256>>>(dx,dAring,N,T); CK(cudaDeviceSynchronize()); }

  constexpr int NT_G = 32*(TbShape::kM/WpShape::kM)*(TbShape::kN/WpShape::kN);
  constexpr int NT_D = 32*(TbShapeD::kM/WpShapeD::kM)*(TbShapeD::kN/WpShapeD::kN);
  auto step=[&](int i,cudaStream_t s){ int t=reverse?(T-1-i):i; int prev=reverse?(t+1):t; int out=reverse?t:(t+1);
    int8_t* hprev=dring+(size_t)prev*N*H; int8_t* hout=dring+(size_t)out*N*H;
    int8_t* At=dAring+(size_t)t*N*KC;
#ifdef USE_SPLITK
    downproj_splitk_kernel<<<dim3(N/TbShapeD::kM, K_HH/TbShapeD::kN, SPLITK),NT_D,smem_d,s>>>(hprev,dwdn,dwsK,N);
    downproj_reduce_kernel<<<(N*K_HH+255)/256,256,0,s>>>(dwsK,dcm,At,N);
#else
    downproj_kernel<<<dim3(N/TbShapeD::kM, K_HH/TbShapeD::kN),NT_D,smem_d,s>>>(hprev,dwdn,dcm,At,N);
#endif
    gate_kernel<<<dim3(N/TbShape::kM, NG/TbShape::kN),NT_G,smem_g,s>>>(At,dBint,dws,dbs,dcell,hout,N);
  };
  auto init=[&](int rev){ CK(cudaMemset(dcell,0,(size_t)N*H*sizeof(cellT)));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H)); };

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    int8_t* dref; CK(cudaMalloc(&dref,ring_bytes));
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    init(reverse); for(int i=0;i<T;++i) step(i,0); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
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
  cudaStream_t cs; CK(cudaStreamCreate(&cs));
  init(reverse);
  cudaGraph_t g; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(cs,cudaStreamCaptureModeThreadLocal));
  for(int i=0;i<T;++i) step(i,cs);
  CK(cudaStreamEndCapture(cs,&g)); CK(cudaGraphInstantiate(&ge,g,0));
  for(int w=0;w<3;++w) CK(cudaGraphLaunch(ge,0)); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<7;++r){ cudaEventRecord(e0); CK(cudaGraphLaunch(ge,0)); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); us.push_back((double)ms/T*1000.0); }
  std::sort(us.begin(),us.end());
  printf("CUTLASS-FUSED step: %.2f us/step (median of 7, T=%d, graph)\n",us[3],T);
  // isolated
  int8_t* hp=dring; int8_t* ho=dring+(size_t)N*H;
  auto timeit=[&](const char*nm, auto fn){ for(int i=0;i<20;++i) fn(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<500;++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); printf("  %-10s %.2f us/launch\n",nm,ms/500*1000.0); };
  timeit("downproj",[&](){ downproj_kernel<<<dim3(N/TbShapeD::kM,K_HH/TbShapeD::kN),NT_D,smem_d>>>(hp,dwdn,dcm,dAring,N); });
#ifdef USE_SPLITK
  timeit("dp_splitk",[&](){ downproj_splitk_kernel<<<dim3(N/TbShapeD::kM,K_HH/TbShapeD::kN,SPLITK),NT_D,smem_d>>>(hp,dwdn,dwsK,N); });
  timeit("dp_reduce",[&](){ downproj_reduce_kernel<<<(N*K_HH+255)/256,256>>>(dwsK,dcm,dAring,N); });
#endif
  timeit("gate",[&](){ gate_kernel<<<dim3(N/TbShape::kM,NG/TbShape::kN),NT_G,smem_g>>>(dAring,dBint,dws,dbs,dcell,ho,N); });
  return 0;
}
