// Hand-CUDA persistent M-split factored-LSTM recurrence for A100 (sm80).
//
// Algorithm (validated in the CuTe DSL prototype factored_lstm_msplit_i8.py):
//   Persistent over T (one launch/layer, loops t in-kernel).  Grid (GX,GY):
//   GX CTAs split hidden H into HX=H/GX slices; GY = batch tiles (bMg rows).
//   256 threads = 8 warps, M-split: each warp owns 16 batch rows (one m16 MMA).
//   Per step, per warp:
//     1. partial down-proj  h[16,HX] @ W_dn[K_hh,HX]^T -> C-frag[16,K_hh]  (int8 mma.sync m16n8k32)
//     2. cross-CTA all-reduce the partial C-frags across the GX CTAs of the gy-group
//        (global-atomic arrive/wait barrier; single-buffered gmem scratch; L2-resident at small bMg)
//     3. scale reduced acc -> hh int8, intra-warp shuffle C-frag -> gate A-operand (proven recipe)
//     4. x_down half of gate A: direct int8 gmem -> A registers (no smem)
//     5. gate int8 mma.sync m16n8k32 vs smem-resident int8 gate weights, per n-tile
//     6. epilogue: f32 cell update, h=int8(o*tanh(c)); write h to sHid slice + hh_all ring
//
// Correctness: bit-exact vs a naive GPU reference kernel (same epilogue/tanh/round
// device functions -> identical bits), fwd + reverse.
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math msplit_flstm.cu -o msplit_flstm
//
// RESULTS (A100 40GB, warm, N=2048, T=2048, int8, const 1/127 scale):
//   WINNING CONFIG: RESB (default), GX=8 (HX=128), bMg=128, double-buffered scratch = 89us/step.
//   Progression: 151us (first correct) -> vectorized int4 gate-B/reduce loads ->
//     double-buffered scratch (1 barrier) -> smem-staged wscale/bias -> nt-unroll = 108us
//     (M-split, gate weights in smem) -> REGISTER-RESIDENT gate weights (RESB) = 89us.
//   Bit-exact vs naive GPU reference, FORWARD + REVERSE, all tested sizes (0 mismatches).
//
//   TWO gate layouts in this file (compile flag):
//   * M-split (-DNO_RESB): 8 warps split ROWS; hh_down(C-frag)->gate-A via intra-warp
//     __shfl (no smem-A).  Gate weights REPLICATED across 8 warps -> must live in smem ->
//     108us; wall = short_scoreboard 4.2 (gate-MMA stalls on B operand from smem).
//   * RESB (default): 8 warps split HIDDEN COLS (each owns HX/8, all 4 gates) -> gate
//     weight slice is NOT replicated -> fits in REGISTERS (rB, ~128 regs @GX8, loaded
//     once, held across all T).  hh_down all-reduced -> smem A-buffer, N-split gate reads
//     A + resident B.  Kills the B-operand stall: short_scoreboard 4.2->1.8, 108->89us.
//   Cost decomposition (RESB GX8): core 77us | barrier+reduce 13us | ring 1us.
//   ncu RESB GX8: 0 spill(*), occupancy 12.5% (1 CTA/SM), imma-util 3->4.8%; new
//     dominant stalls long_scoreboard 4.6 (all-reduce scratch reads) + mio_throttle 3.0
//     (the A-buffer is now the 8x-replicated smem read - replication moved B->A, and A
//     is half B's size, hence the win).
//   Still above the <64us milestone: the residual wall is the 8x-replicated smem operand
//   (A here, B in M-split) + all-reduce global latency at 12.5% occupancy.  Next lever
//   would be a 2D warp split (halve A replication) or cross-step pipelining of the
//   all-reduce behind the gate MMA (register-blocked at bMg=128).  dorado's ~10us needs
//   the GX=9 frugal layout (does not divide H=1024).
//   (*) minor 200-300B spill under RESB; M-split path is exactly 0.
//   Compile-flag isolators: NO_RESB/NO_BARRIER/NO_REDUCE/NO_RING/NO_GATE/NO_EPI/GATE_GMEM/MINB.
//
//   OCCUPANCY-PIVOT EXPERIMENT (--dorado : dorado-style OUTPUT-channel split, NO input-hidden
//   split -> NO all-reduce; each CTA reads the full hidden h from the ring and recomputes
//   hh_down LOCALLY (redundant across the GX channel-CTAs); weights STREAMED from L2 (pinned
//   via cudaAccessPolicyWindow), tiny footprint -> >=2 CTA/SM).  Bit-exact fwd+reverse.
//   FINDING: occupancy DID rise (12.5% -> 23%) but tensor-util STAYED PINNED (~2.5%) and
//   per-step got WORSE (276us).  ncu: L2 traffic 1.07GB/step (vs 509MB RESB), mio_throttle
//   22.7 + long_scoreboard 15 dominant.  The redundant full-K=H down-proj re-reads full h +
//   full W_dn per CTA -> memory-traffic bound; occupancy can't hide bandwidth.  DRAM stayed
//   low (4.4MB) so the L2 pin works - the wall is L2 bandwidth, not DRAM eviction.
//   Synthesis test (all-reduce + streamed+pinned weights + 2 CTA/SM, -DGATE_GMEM -DMINB=2):
//   occ 15.6%, util 3.8%, 98us - still slower than RESB 89us.
//   CONCLUSION: for this shape (short-K gate + full-H down-proj at N=2048) the A100 floor is
//   MEMORY/LATENCY-bound, NOT occupancy-bound.  Raising occupancy via streamed weights adds
//   traffic that offsets the gain; redundant down-proj adds catastrophic traffic.  Resident
//   weights (RESB, minimal traffic, 1 CTA/SM) win at 89us.  Streaming-for-occupancy is the
//   right instinct on compute-bound shapes but this recurrence is traffic-bound.
//   Extra flags: --dorado (runtime), DBM (batch rows/CTA), DMINB (dorado CTAs/SM).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#define H     1024
#define K_HH  128
#define R     128
#define KC    256        // K_hh + R
#define WARPS 8
#define ROWS_PER_WARP 16
#define BMG   128        // 8 warps * 16 rows
#define NTHREADS 256

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

// ---------------- shared device math (identical in kernel + reference) ----------------
__device__ __forceinline__ int8_t clamp_i8(float q){
  q = fminf(fmaxf(q, -127.f), 127.f);
  return (int8_t)(int)q;
}
// one LSTM output-channel epilogue.  gi..go = int32 gate accumulators (sum A_i8*W_i8);
// s* = per-channel gate-weight scales; b* = biases; ascale = activation scale (1/127).
__device__ __forceinline__ int8_t epi_elem(
    int gi,int gf,int gg,int go,
    float si,float sf,float sg,float so,
    float bi,float bf,float bg,float bo,
    float ascale, float &cell){
  float vi = (float)gi*ascale*si + bi;
  float vf = (float)gf*ascale*sf + bf;
  float vg = (float)gg*ascale*sg + bg;
  float vo = (float)go*ascale*so + bo;
  float I = fminf(fmaxf(vi*0.2f+0.5f, 0.f), 1.f);
  float F = fminf(fmaxf(vf*0.2f+0.5f, 0.f), 1.f);
  float O = fminf(fmaxf(vo*0.2f+0.5f, 0.f), 1.f);
  float G = fminf(fmaxf(vg, -1.f), 1.f);
  cell = F*cell + I*G;
  float hh = O*tanhf(cell);
  return clamp_i8(rintf(hh*127.0f));
}

// mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32  (D = A*B + C, accumulate in place)
__device__ __forceinline__ void mma_m16n8k32(
    int &c0,int &c1,int &c2,int &c3,
    int a0,int a1,int a2,int a3, int b0,int b1){
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

__device__ __forceinline__ int extract_byte(int w,int p){ return (w>>(p*8))&0xFF; }

// ============================ persistent M-split kernel ============================
// scratch:  [GY, GX, WARPS, (K_hh/8 * 4) , 32]  int32  (partial C-frags, lane-fastest)
// flags:    [GY] int32  (monotonic arrive counter per gy-group)
#ifndef MINB
#define MINB 1
#endif
#ifndef UNROLL_NT
#define UNROLL_NT 1
#endif
// Default = RESB: register-resident gate weights + N-split-by-hidden-col gate (best,
// 89us/step @ GX=8).  Build with -DNO_RESB for the M-split shuffle path (108us).
#if !defined(RESB) && !defined(NO_RESB)
#define RESB
#endif
template<int GX>
__global__ void __launch_bounds__(NTHREADS,MINB) msplit_kernel(
    const int8_t* __restrict__ x,        // [T, N, R] int8
    const int8_t* __restrict__ w_dn,     // [K_hh, H] int8
    const float*  __restrict__ comb_mult,// [K_hh]  (= w_dn per-channel scale)
    const int8_t* __restrict__ wg,       // [4, H, Kc] int8  (gate weights, row=out chan)
    const float*  __restrict__ wscale,   // [4, H] gate per-channel scales
    const float*  __restrict__ bias,     // [4, H]
    int8_t* __restrict__ hh_all,         // [T+1, N, H] int8 ring
    float*  __restrict__ cell_out,       // [N, H]  final cell
    int*    __restrict__ scratch,
    int*    __restrict__ flags,
    int N, int T, int reverse)
{
  constexpr int HX = H / GX;
  constexpr int KTX = HX / 32;          // down-proj k-tiles (over HX)
  constexpr int NT  = HX / 8;           // gate output n-tiles per gate
  const float ASCALE = 1.0f/127.0f;

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int gid  = lane >> 2;           // 0..7  (row within 8-group)
  const int tg   = lane & 3;            // 0..3  (thread within 4-group)
  const int gx   = blockIdx.x;
  const int gy   = blockIdx.y;
  const int row0 = gy*BMG + warp*ROWS_PER_WARP;   // this warp's first global row
  const int hbase= gx*HX;                          // this CTA's hidden-col base

  constexpr int CPW = HX/WARPS;            // hidden cols owned per warp (RESB N-split)
  constexpr int NTW = CPW/8;               // gate n8-tiles per warp (RESB)
  extern __shared__ int8_t smem[];
  int8_t* sHid = smem;                     // [BMG, HX]  h slice (int8)
#ifdef RESB
  int8_t* sA   = sHid + BMG*HX;            // [BMG, Kc]  gate A-buffer (hh|x) ; weights in regs
  float*  sWs  = (float*)(sA + BMG*KC);    // [4, HX] gate per-channel scales
#elif defined(GATE_GMEM)
  int8_t* sWg  = sHid + BMG*HX;            // unused (gate weights read from L2)
  float*  sWs  = (float*)(sHid + BMG*HX);  // [4, HX] gate per-channel scales
#else
  int8_t* sWg  = sHid + BMG*HX;            // [4, HX, Kc] gate weights slice (W_dn read from L2)
  float*  sWs  = (float*)(sWg + 4*HX*KC);  // [4, HX] gate per-channel scales
#endif
  float*  sBs  = sWs + 4*HX;               // [4, HX] biases

  // ---- load persistent weights (once). W_dn stays in L2; gate weights are pre-repacked. ----
#if !defined(GATE_GMEM) && !defined(RESB)
  for(int g=0; g<4; ++g){
    const int8_t* src = wg + ((long)g*H + hbase)*KC;   // pre-repacked layout
    int8_t* dst = sWg + g*HX*KC;
    for(int i=tid; i<HX*KC; i+=NTHREADS) dst[i] = src[i];
  }
#endif
#ifdef RESB
  // REGISTER-RESIDENT gate weights (N-split by hidden col): warp owns CPW hidden cols,
  // 4 gates each -> rB = 4*NTW*8*2 int32 held across all T (kills the per-step B smem
  // load->mma stall).  Loaded once from pre-repacked wg.
  int rB[4][NTW?NTW:1][8][2];   // RESB requires HX>=64 (GX<=16); GX=32 unsupported
  #pragma unroll
  for(int g=0; g<4; ++g)
    #pragma unroll
    for(int nn=0; nn<NTW; ++nn){
      int oc = g*H + hbase + warp*CPW + nn*8 + gid;
      const int8_t* wp = wg + (long)oc*KC + tg*64;
      #pragma unroll
      for(int kt=0; kt<8; ++kt){ rB[g][nn][kt][0]=*(const int*)(wp+kt*8);
                                 rB[g][nn][kt][1]=*(const int*)(wp+kt*8+4); }
    }
#endif
  // stage this CTA's per-channel gate scales + biases into smem (const across T)
  for(int i=tid; i<4*HX; i+=NTHREADS){
    int g=i/HX, col=i%HX; sWs[i]=wscale[g*H+hbase+col]; sBs[i]=bias[g*H+hbase+col];
  }
  // init hidden slice = 0
  for(int i=tid; i<BMG*HX; i+=NTHREADS) sHid[i]=0;
  __syncthreads();

  // cell registers: per lane, cell[nt][e]  (e=0..3 gate C-frag elements)
  float cell[NT][4];
  #pragma unroll
  for(int nt=0; nt<NT; ++nt)
    #pragma unroll
    for(int e=0;e<4;++e) cell[nt][e]=0.f;

  // scratch layout per (gy,gx,warp): [jt(16)][lane(32)][e(4)] int32  (e-contiguous ->
  // int4 vectorized reduce loads / writes).
  const long BLK = (long)(K_HH/8)*32*4;            // 2048 ints
  const long HALF = (long)(N/BMG)*GX*WARPS*BLK;    // double-buffer half size
  long sc_base0 = (((long)(gy*GX + gx))*WARPS + warp) * BLK;   // this warp's write base
  long sc_rd00  = (((long)gy*GX)*WARPS + warp) * BLK;          // gx=0 read base (this warp)
  const long sc_gxstride = (long)WARPS*BLK;
  int* flag = flags + gy;

  for(int t=0; t<T; ++t){
    int tt = reverse ? (T-1-t) : t;
    int wslot = reverse ? tt : (tt+1);
    long sc_base = sc_base0 + (t&1)*HALF;   // double-buffered -> only 1 barrier/step
    long sc_rd0  = sc_rd00  + (t&1)*HALF;

    // ---- 1. partial down-proj (streamed by 4 n8-tiles) -> scratch ----
    // Keeping only a 4-tile C-frag live (16 int32) instead of the full 64 avoids
    // the register spill the DSL couldn't escape.  A-frag reloaded per chunk (cheap).
    #pragma unroll
    for(int jtg=0; jtg<K_HH/8; jtg+=4){
      int acc[4][4];
      #pragma unroll
      for(int j=0;j<4;++j)
        #pragma unroll
        for(int e=0;e<4;++e) acc[j][e]=0;
      #pragma unroll
      for(int kt=0; kt<KTX; ++kt){
        int rbase = warp*ROWS_PER_WARP;
        int coff  = kt*32 + tg*4;
        int a0 = *(const int*)&sHid[(rbase+gid  )*HX + coff];
        int a1 = *(const int*)&sHid[(rbase+gid+8)*HX + coff];
        int a2 = *(const int*)&sHid[(rbase+gid  )*HX + coff+16];
        int a3 = *(const int*)&sHid[(rbase+gid+8)*HX + coff+16];
        #pragma unroll
        for(int j=0;j<4;++j){
          int nrow = (jtg+j)*8 + gid;   // W_dn[khh, hbase + hidcol] direct from L2
          int b0 = *(const int*)&w_dn[nrow*H + hbase + kt*32 + tg*4];
          int b1 = *(const int*)&w_dn[nrow*H + hbase + kt*32 + 16 + tg*4];
          mma_m16n8k32(acc[j][0],acc[j][1],acc[j][2],acc[j][3], a0,a1,a2,a3,b0,b1);
        }
      }
      #pragma unroll
      for(int j=0;j<4;++j)
        *(int4*)&scratch[sc_base + (long)(jtg+j)*128 + lane*4] =
            make_int4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
    }

    // ---- barrier 1 (arrive/wait among GX CTAs of this gy) ----
#ifndef NO_BARRIER
    __threadfence();
    __syncthreads();
    if(tid==0){
      atomicAdd(flag, 1);
      int need = GX*(t+1);
      while(atomicAdd(flag,0) < need) { }
    }
    __syncthreads();
#endif

#ifdef RESB
    // ---- reduce GX partials + scale->hh int8 -> smem A-buffer (hh half) ----
    #pragma unroll
    for(int jt=0; jt<K_HH/8; ++jt){
      int r0,r1,r2,r3; r0=r1=r2=r3=0;
      long boff = sc_rd0 + (long)jt*128 + lane*4;
#ifdef NO_REDUCE
      #pragma unroll
      for(int g=0; g<1; ++g){
#else
      #pragma unroll
      for(int g=0; g<GX; ++g){
#endif
        int4 v = *(const int4*)&scratch[boff + (long)g*sc_gxstride];
        r0 += v.x; r1 += v.y; r2 += v.z; r3 += v.w;
      }
      int col0 = jt*8 + 2*tg, col1 = col0+1;
      int rlo = warp*ROWS_PER_WARP + gid, rhi = rlo + 8;
      sA[rlo*KC + col0] = clamp_i8(rintf((float)r0*comb_mult[col0]));
      sA[rlo*KC + col1] = clamp_i8(rintf((float)r1*comb_mult[col1]));
      sA[rhi*KC + col0] = clamp_i8(rintf((float)r2*comb_mult[col0]));
      sA[rhi*KC + col1] = clamp_i8(rintf((float)r3*comb_mult[col1]));
    }
    // x_down half -> smem A-buffer (each lane 4 bytes/row over 16 rows)
    #pragma unroll
    for(int r=0; r<ROWS_PER_WARP; ++r){
      int gr = row0 + r;
      *(int*)&sA[(warp*ROWS_PER_WARP+r)*KC + K_HH + lane*4] =
          *(const int*)&x[((long)tt*N + gr)*R + lane*4];
    }
    __syncthreads();   // sA ready across warps for N-split gate

    // ---- N-split gate: warp owns CPW hidden cols (all 4 gates), REGISTER-RESIDENT B,
    //      loop the BMG rows in m16 row-tiles, A ldmatrix-free from sA. ----
#ifndef NO_GATE
    #pragma unroll
    for(int rt=0; rt<BMG/16; ++rt){
      int A[8][4];   // [ktile][a0..a3]
      #pragma unroll
      for(int kt=0; kt<8; ++kt){
        int rb=rt*16, co=kt*32+tg*4;
        A[kt][0]=*(const int*)&sA[(rb+gid  )*KC+co];
        A[kt][1]=*(const int*)&sA[(rb+gid+8)*KC+co];
        A[kt][2]=*(const int*)&sA[(rb+gid  )*KC+co+16];
        A[kt][3]=*(const int*)&sA[(rb+gid+8)*KC+co+16];
      }
      #pragma unroll
      for(int nn=0; nn<NTW; ++nn){
        int cg[4][4];
        #pragma unroll
        for(int g=0;g<4;++g)
          #pragma unroll
          for(int e=0;e<4;++e) cg[g][e]=0;
        #pragma unroll
        for(int kt=0; kt<8; ++kt)
          #pragma unroll
          for(int g=0; g<4; ++g)
            mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],
                         A[kt][0],A[kt][1],A[kt][2],A[kt][3], rB[g][nn][kt][0], rB[g][nn][kt][1]);
        int cidx = rt*NTW+nn;
        #pragma unroll
        for(int e=0; e<4; ++e){
          int lcol = warp*CPW + nn*8 + 2*tg + (e&1);   // within HX
          int rin16 = (e<2)? gid : gid+8;
          int oc = hbase + lcol;
          int gr = gy*BMG + rt*16 + rin16;
#ifdef NO_EPI
          cell[cidx][e] += (float)cg[0][e];
          int8_t hnew = (int8_t)(cg[0][e]&0x7F);
#else
          float si=sWs[0*HX+lcol], sf=sWs[1*HX+lcol], sg=sWs[2*HX+lcol], so=sWs[3*HX+lcol];
          float bi=sBs[0*HX+lcol], bf=sBs[1*HX+lcol], bg=sBs[2*HX+lcol], bo=sBs[3*HX+lcol];
          int8_t hnew = epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
                                 si,sf,sg,so, bi,bf,bg,bo, ASCALE, cell[cidx][e]);
#endif
          sHid[(rt*16+rin16)*HX + lcol] = hnew;
#ifndef NO_RING
          hh_all[((long)wslot*N + gr)*H + oc] = hnew;
#endif
        }
      }
    }
#else
    for(int i=tid;i<BMG*HX;i+=NTHREADS) sHid[i]=sA[i];   // keep recurrence cycling
#endif
#else   /* !RESB : M-split shuffle path */
    // ---- 2+3. reduce GX partials + scale->hh int8 + shuffle C->gate A, streamed ----
    // per ktile group (4 n8-tiles): peak live = red4(16)+word4(4)+hhA -> no 64-wide spill.
    int hhA[4][4];   // [ktile][a0..a3]
    int group_base = (lane>>2)*4;
    int src_lo = group_base + (tg%2)*2;
    int src_hi = src_lo + 1;
    int useB = (tg>=2)?1:0;
    #pragma unroll
    for(int kt=0; kt<4; ++kt){
      int word4[4];
      #pragma unroll
      for(int j=0; j<4; ++j){
        int jt = kt*4 + j;
        int r0,r1,r2,r3; r0=r1=r2=r3=0;
        long boff = sc_rd0 + (long)jt*128 + lane*4;
#ifdef NO_REDUCE
        #pragma unroll
        for(int g=0; g<1; ++g){
#else
        #pragma unroll
        for(int g=0; g<GX; ++g){
#endif
          int4 v = *(const int4*)&scratch[boff + (long)g*sc_gxstride];
          r0 += v.x; r1 += v.y; r2 += v.z; r3 += v.w;
        }
        int col0 = jt*8 + 2*tg, col1 = col0+1;
        float m0 = comb_mult[col0], m1 = comb_mult[col1];
        int b0 = clamp_i8(rintf((float)r0*m0));
        int b1 = clamp_i8(rintf((float)r1*m1));
        int b2 = clamp_i8(rintf((float)r2*m0));
        int b3 = clamp_i8(rintf((float)r3*m1));
        word4[j] = (b0&0xFF)|((b1&0xFF)<<8)|((b2&0xFF)<<16)|((b3&0xFF)<<24);
      }
      #pragma unroll
      for(int band=0; band<2; ++band){
        int wA_lo = __shfl_sync(0xffffffff, word4[band*2],   src_lo);
        int wA_hi = __shfl_sync(0xffffffff, word4[band*2],   src_hi);
        int wB_lo = __shfl_sync(0xffffffff, word4[band*2+1], src_lo);
        int wB_hi = __shfl_sync(0xffffffff, word4[band*2+1], src_hi);
        int w_lo = useB ? wB_lo : wA_lo;
        int w_hi = useB ? wB_hi : wA_hi;
        #pragma unroll
        for(int rh=0; rh<2; ++rh){
          int p = rh*2;
          hhA[kt][band*2+rh] =
              (extract_byte(w_lo,p)      ) |
              (extract_byte(w_lo,p+1)<<8 ) |
              (extract_byte(w_hi,p)  <<16) |
              (extract_byte(w_hi,p+1)<<24);
        }
      }
    }

    // (barrier 2 eliminated: scratch is double-buffered by t parity)

    // ---- 4. x_down half of gate A (k-tiles 4..7), direct gmem -> registers ----
    int xA[4][4];    // [xtile][a0..a3]
    {
      int r_lo = row0 + gid;
      int r_hi = row0 + gid + 8;
      const int8_t* xrow_lo = x + ((long)tt*N + r_lo)*R;
      const int8_t* xrow_hi = x + ((long)tt*N + r_hi)*R;
      #pragma unroll
      for(int xt=0; xt<4; ++xt){
        int coff = xt*32 + tg*4;
        xA[xt][0] = *(const int*)&xrow_lo[coff];
        xA[xt][1] = *(const int*)&xrow_hi[coff];
        xA[xt][2] = *(const int*)&xrow_lo[coff+16];
        xA[xt][3] = *(const int*)&xrow_hi[coff+16];
      }
    }

    // ---- 5+6. gate MMA per n-tile + epilogue ----
#ifndef NO_GATE
    #pragma unroll
    for(int nt=0; nt<NT; ++nt){
      int cg[4][4];   // [gate][c0..c3]
      #pragma unroll
      for(int g=0;g<4;++g)
        #pragma unroll
        for(int e=0;e<4;++e) cg[g][e]=0;

      // preload all 4 gates' B (16 int4 loads issued together -> latency overlap)
      int B0[4][8], B1[4][8];   // [gate][ktile]
      #pragma unroll
      for(int g=0; g<4; ++g){
        // repacked gate weights: 8 k-tiles' (b0,b1) = 64 contiguous bytes -> 4x int4.
#ifdef GATE_GMEM
        const int8_t* wgs = wg + ((long)g*H + hbase + nt*8+gid)*KC + tg*64;   // from L2
#else
        const int8_t* wgs = sWg + g*HX*KC + (nt*8+gid)*KC + tg*64;
#endif
        int4 w0=*(const int4*)(wgs), w1=*(const int4*)(wgs+16),
             w2=*(const int4*)(wgs+32), w3=*(const int4*)(wgs+48);
        B0[g][0]=w0.x;B1[g][0]=w0.y;B0[g][1]=w0.z;B1[g][1]=w0.w;
        B0[g][2]=w1.x;B1[g][2]=w1.y;B0[g][3]=w1.z;B1[g][3]=w1.w;
        B0[g][4]=w2.x;B1[g][4]=w2.y;B0[g][5]=w2.z;B1[g][5]=w2.w;
        B0[g][6]=w3.x;B1[g][6]=w3.y;B0[g][7]=w3.z;B1[g][7]=w3.w;
      }
      // kt-outer: 4 independent gate mma-chains interleaved -> tensor pipe stays fed.
      #pragma unroll
      for(int kt=0; kt<8; ++kt){
        int a0,a1,a2,a3;
        if(kt<4){ a0=hhA[kt][0]; a1=hhA[kt][1]; a2=hhA[kt][2]; a3=hhA[kt][3]; }
        else    { int xt=kt-4; a0=xA[xt][0]; a1=xA[xt][1]; a2=xA[xt][2]; a3=xA[xt][3]; }
        #pragma unroll
        for(int g=0; g<4; ++g)
          mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3], a0,a1,a2,a3, B0[g][kt], B1[g][kt]);
      }

      // epilogue for the 4 C-frag elements of this n-tile
      #pragma unroll
      for(int e=0; e<4; ++e){
        int col = nt*8 + 2*tg + (e&1);           // local col within HX
        int rloc = (e<2)? gid : gid+8;           // local row within warp
        int oc = hbase + col;                    // global hidden channel
        int rg = row0 + rloc;                    // global batch row
#ifdef NO_EPI
        cell[nt][e] += (float)cg[0][e];
        int8_t hnew = (int8_t)(cg[0][e]&0x7F);
#else
        float si=sWs[0*HX+col], sf=sWs[1*HX+col], sg=sWs[2*HX+col], so=sWs[3*HX+col];
        float bi=sBs[0*HX+col], bf=sBs[1*HX+col], bg=sBs[2*HX+col], bo=sBs[3*HX+col];
        int8_t hnew = epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
                               si,sf,sg,so, bi,bf,bg,bo, ASCALE, cell[nt][e]);
#endif
        // write to hidden slice (for next down-proj) and ring
        sHid[(warp*ROWS_PER_WARP + rloc)*HX + col] = hnew;
#ifndef NO_RING
        hh_all[((long)wslot*N + rg)*H + oc] = hnew;
#endif
      }
    }
#else
    // NO_GATE: write hh (from shuffle) back to sHid so the recurrence still cycles
    for(int i=tid;i<BMG*HX;i+=NTHREADS) sHid[i]=(int8_t)(hhA[0][0]&0xFF);
#endif
#endif  /* RESB */
    __syncthreads();
  }

  // write final cell
#ifdef RESB
  #pragma unroll
  for(int rt=0; rt<BMG/16; ++rt)
    #pragma unroll
    for(int nn=0; nn<NTW; ++nn)
      #pragma unroll
      for(int e=0;e<4;++e){
        int lcol = warp*CPW + nn*8 + 2*tg + (e&1);
        int rin16 = (e<2)? gid : gid+8;
        cell_out[(long)(gy*BMG+rt*16+rin16)*H + hbase + lcol] = cell[rt*NTW+nn][e];
      }
#else
  #pragma unroll
  for(int nt=0; nt<NT; ++nt)
    #pragma unroll
    for(int e=0;e<4;++e){
      int col = nt*8 + 2*tg + (e&1);
      int rloc = (e<2)? gid : gid+8;
      int oc = hbase + col;
      int rg = row0 + rloc;
      cell_out[(long)rg*H + oc] = cell[nt][e];
    }
#endif
}

// ============================ naive GPU reference ============================
// one thread per (row) among first n_cmp rows; identical epilogue -> bit-exact.
__global__ void ref_kernel(
    const int8_t* __restrict__ x, const int8_t* __restrict__ w_dn,
    const float* __restrict__ comb_mult, const int8_t* __restrict__ wg,
    const float* __restrict__ wscale, const float* __restrict__ bias,
    int8_t* __restrict__ hh_all, float* __restrict__ cell_out,
    int N, int T, int reverse, int n_cmp)
{
  int r = blockIdx.x*blockDim.x + threadIdx.x;
  if(r>=n_cmp) return;
  const float ASCALE=1.0f/127.0f;
  int8_t* h = new int8_t[H];
  float*  c = new float[H];
  for(int i=0;i<H;++i){ h[i]=0; c[i]=0.f; }
  int8_t hh[K_HH];
  for(int t=0;t<T;++t){
    int tt = reverse?(T-1-t):t;
    int wslot = reverse?tt:(tt+1);
    // down-proj
    for(int k=0;k<K_HH;++k){
      int acc=0;
      for(int cc=0;cc<H;++cc) acc += (int)h[cc]*(int)w_dn[k*H+cc];
      hh[k]=clamp_i8(rintf((float)acc*comb_mult[k]));
    }
    // gate + epilogue
    const int8_t* xr = x + ((long)tt*N + r)*R;
    for(int oc=0; oc<H; ++oc){
      int g[4]={0,0,0,0};
      for(int gg=0; gg<4; ++gg){
        const int8_t* w = wg + ((long)gg*H + oc)*KC;
        int acc=0;
        for(int kc=0;kc<K_HH;++kc) acc += (int)hh[kc]*(int)w[kc];
        for(int kc=0;kc<R;++kc)    acc += (int)xr[kc]*(int)w[K_HH+kc];
        g[gg]=acc;
      }
      int8_t hn = epi_elem(g[0],g[1],g[2],g[3],
          wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc], ASCALE, c[oc]);
      hh_all[((long)wslot*N + r)*H + oc]=hn;
    }
    // commit new hidden (must read old h for whole down-proj/gate first)
    for(int oc=0; oc<H; ++oc) h[oc]=hh_all[((long)wslot*N + r)*H + oc];
  }
  for(int oc=0;oc<H;++oc) cell_out[(long)r*H+oc]=c[oc];
  delete[] h; delete[] c;
}

// ============================ dorado-structure kernel ============================
// OUTPUT-channel split (no input-hidden split -> NO all-reduce).  CTA(cb,bb) owns DBM
// batch rows + HX=H/GX hidden-output channels.  Each CTA reads the FULL hidden h[DBM,H]
// (all channels) from the ring's prev slot and computes hh_down LOCALLY (full-K=H reduce,
// high util), redundantly across the GX channel-CTAs of a batch block.  Weights are
// STREAMED from L2 (pinned), NOT resident -> tiny footprint -> HIGH OCCUPANCY (>=2 CTA/SM)
// to hide the short-K gate latency.  One producer/consumer barrier per step (h ready),
// no sum-reduce.  8 warps N-split the channels.
#ifndef DBM
#define DBM 32          // batch rows per CTA (sweep)
#endif
#ifndef DMINB
#define DMINB 2         // target CTAs/SM
#endif
template<int GX>
__global__ void __launch_bounds__(NTHREADS,DMINB) dorado_kernel(
    const int8_t* __restrict__ x, const int8_t* __restrict__ w_dn,
    const float* __restrict__ comb_mult, const int8_t* __restrict__ wg,
    const float* __restrict__ wscale, const float* __restrict__ bias,
    int8_t* __restrict__ hh_all, float* __restrict__ cell_out,
    int* __restrict__ flags, int N, int T, int reverse)
{
  constexpr int HX = H/GX;              // hidden-output channels per CTA
  constexpr int RT = DBM/16;            // batch row-tiles
  constexpr int CPW = HX/WARPS;         // gate channels per warp (N-split)
  constexpr int NTW = CPW/8;            // gate n8-tiles per warp
  const float ASCALE = 1.0f/127.0f;
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int cb=blockIdx.x, bb=blockIdx.y;
  const int row0=bb*DBM, hbase=cb*HX;

  extern __shared__ int8_t smem[];
  int8_t* sH = smem;                    // [DBM, H]  full hidden input (from ring)
  int8_t* sA = sH + DBM*H;              // [DBM, Kc] hh_down | x_down
  float*  sWs= (float*)(sA + DBM*KC);   // [4, HX]
  float*  sBs= sWs + 4*HX;
  for(int i=tid;i<4*HX;i+=NTHREADS){ int g=i/HX,c=i%HX; sWs[i]=wscale[g*H+hbase+c]; sBs[i]=bias[g*H+hbase+c]; }

  float cell[RT*NTW][4];
  #pragma unroll
  for(int i=0;i<RT*NTW;++i)
    #pragma unroll
    for(int e=0;e<4;++e) cell[i][e]=0.f;
  int* flag = flags + bb;

  for(int t=0; t<T; ++t){
    int tt = reverse ? (T-1-t) : t;
    int rd = reverse ? (tt+1) : tt;     // prev-h slot to read
    int wr = reverse ? tt : (tt+1);     // slot to write

    // 1. load full hidden h[DBM,H] from ring[rd] into smem (int4 coalesced)
    #pragma unroll
    for(int j=tid; j<DBM*(H/16); j+=NTHREADS){
      int r=j/(H/16), cc=j%(H/16);
      ((int4*)sH)[r*(H/16)+cc] = *(const int4*)&hh_all[((long)rd*N+row0+r)*H + cc*16];
    }
    __syncthreads();

    // 2. down-proj (redundant, full K=H): warp owns 2 of the 16 K_hh n8-tiles, all rows.
    #pragma unroll
    for(int rt=0; rt<RT; ++rt){
      #pragma unroll
      for(int loc=0; loc<(K_HH/8)/WARPS; ++loc){    // 16/8 = 2
        int ndp = warp*((K_HH/8)/WARPS) + loc;      // K_hh n8-tile 0..15
        int c0=0,c1=0,c2=0,c3=0;
        #pragma unroll
        for(int kt=0; kt<H/32; ++kt){               // 32 k-tiles over full H
          int rb=rt*16, co=kt*32+tg*4;
          int a0=*(const int*)&sH[(rb+gid  )*H+co];
          int a1=*(const int*)&sH[(rb+gid+8)*H+co];
          int a2=*(const int*)&sH[(rb+gid  )*H+co+16];
          int a3=*(const int*)&sH[(rb+gid+8)*H+co+16];
          int nrow=ndp*8+gid;                        // W_dn[nrow, kt*32+..]  (streamed L2)
          int b0=*(const int*)&w_dn[nrow*H + kt*32 + tg*4];
          int b1=*(const int*)&w_dn[nrow*H + kt*32 + 16 + tg*4];
          mma_m16n8k32(c0,c1,c2,c3, a0,a1,a2,a3, b0,b1);
        }
        int col0=ndp*8+2*tg, col1=col0+1;
        int rlo=rt*16+gid, rhi=rt*16+gid+8;
        sA[rlo*KC+col0]=clamp_i8(rintf((float)c0*comb_mult[col0]));
        sA[rlo*KC+col1]=clamp_i8(rintf((float)c1*comb_mult[col1]));
        sA[rhi*KC+col0]=clamp_i8(rintf((float)c2*comb_mult[col0]));
        sA[rhi*KC+col1]=clamp_i8(rintf((float)c3*comb_mult[col1]));
      }
    }
    // 3. x_down -> sA[.., K_hh:Kc]
    #pragma unroll
    for(int r=0; r<DBM; ++r)
      *(int*)&sA[r*KC + K_HH + lane*4] = *(const int*)&x[((long)tt*N + row0 + r)*R + lane*4];
    __syncthreads();

    // 4. gate (N-split): warp owns CPW channels (all 4 gates); stream gate_w from L2.
    #pragma unroll
    for(int rt=0; rt<RT; ++rt){
      int A[8][4];
      #pragma unroll
      for(int kt=0; kt<8; ++kt){
        int rb=rt*16, co=kt*32+tg*4;
        A[kt][0]=*(const int*)&sA[(rb+gid  )*KC+co];
        A[kt][1]=*(const int*)&sA[(rb+gid+8)*KC+co];
        A[kt][2]=*(const int*)&sA[(rb+gid  )*KC+co+16];
        A[kt][3]=*(const int*)&sA[(rb+gid+8)*KC+co+16];
      }
      #pragma unroll
      for(int nn=0; nn<NTW; ++nn){
        int cg[4][4];
        #pragma unroll
        for(int g=0;g<4;++g)
          #pragma unroll
          for(int e=0;e<4;++e) cg[g][e]=0;
        #pragma unroll
        for(int kt=0; kt<8; ++kt){
          #pragma unroll
          for(int g=0; g<4; ++g){
            int oc=g*H + hbase + warp*CPW + nn*8 + gid;   // streamed gate_w (original layout)
            int b0=*(const int*)&wg[(long)oc*KC + kt*32 + tg*4];
            int b1=*(const int*)&wg[(long)oc*KC + kt*32 + 16 + tg*4];
            mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],
                         A[kt][0],A[kt][1],A[kt][2],A[kt][3], b0,b1);
          }
        }
        int cidx=rt*NTW+nn;
        #pragma unroll
        for(int e=0; e<4; ++e){
          int lcol = warp*CPW + nn*8 + 2*tg + (e&1);
          int rin16 = (e<2)? gid : gid+8;
          int oc = hbase + lcol;
          int gr = row0 + rt*16 + rin16;
          float si=sWs[0*HX+lcol], sf=sWs[1*HX+lcol], sg=sWs[2*HX+lcol], so=sWs[3*HX+lcol];
          float bi=sBs[0*HX+lcol], bf=sBs[1*HX+lcol], bg=sBs[2*HX+lcol], bo=sBs[3*HX+lcol];
          int8_t hnew = epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
                                 si,sf,sg,so, bi,bf,bg,bo, ASCALE, cell[cidx][e]);
          hh_all[((long)wr*N + gr)*H + oc] = hnew;
        }
      }
    }

    // 5. producer/consumer barrier (all GX channel-CTAs wrote slot wr before next read)
    __threadfence();
    __syncthreads();
    if(tid==0){
      atomicAdd(flag, 1);
      int need = GX*(t+1);
      while(atomicAdd(flag,0) < need) { }
    }
    __syncthreads();
  }
  // final cell
  #pragma unroll
  for(int rt=0; rt<RT; ++rt)
    #pragma unroll
    for(int nn=0; nn<NTW; ++nn)
      #pragma unroll
      for(int e=0;e<4;++e){
        int lcol=warp*CPW+nn*8+2*tg+(e&1); int rin16=(e<2)?gid:gid+8;
        cell_out[(long)(row0+rt*16+rin16)*H + hbase+lcol] = cell[rt*NTW+nn][e];
      }
}

// ============================ host ============================
template<int GX>
static void launch(const int8_t*x,const int8_t*wdn,const float*cm,const int8_t*wg,
                   const float*ws,const float*bs,int8_t*ring,float*cell,int*scratch,int*flags,
                   int N,int T,int reverse,size_t smem){
  dim3 grid(GX, N/BMG);
  msplit_kernel<GX><<<grid, NTHREADS, smem>>>(x,wdn,cm,wg,ws,bs,ring,cell,scratch,flags,N,T,reverse);
}
template<int GX>
static void launch_dorado(const int8_t*x,const int8_t*wdn,const float*cm,const int8_t*wg,
                   const float*ws,const float*bs,int8_t*ring,float*cell,int*flags,
                   int N,int T,int reverse,size_t smem){
  dim3 grid(GX, N/DBM);
  dorado_kernel<GX><<<grid, NTHREADS, smem>>>(x,wdn,cm,wg,ws,bs,ring,cell,flags,N,T,reverse);
}

int main(int argc,char**argv){
  int GX=8, N=256, T=64, reverse=0, bench=0, n_cmp=64, dev=0, dorado=0;   // GX=8 = best (RESB)
  for(int i=1;i<argc;++i){
    if(!strcmp(argv[i],"--gx")) GX=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--N")) N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T")) T=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--reverse")) reverse=1;
    else if(!strcmp(argv[i],"--bench")) bench=1;
    else if(!strcmp(argv[i],"--ncmp")) n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev")) dev=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dorado")) dorado=1;
  }
  CUDA_CHECK(cudaSetDevice(dev));
#ifdef RESB
  if(GX==32 && !dorado){ fprintf(stderr,"RESB requires HX>=64 (GX<=16); use -DNO_RESB for GX=32\n"); return 1; }
#endif
  int PAD = dorado ? DBM : BMG;
  if(N%PAD){ N = ((N+PAD-1)/PAD)*PAD; }
  int HX = H/GX;
  printf("GX=%d HX=%d N=%d T=%d reverse=%d bench=%d\n",GX,HX,N,T,reverse,bench);

  // ---- host inputs ----
  std::mt19937 rng(1234);
  std::normal_distribution<float> nd(0.f,1.f);
  std::uniform_real_distribution<float> ux(-1.f,1.f);
  // x_down int8 in [-1,1] scaled by 127
  std::vector<int8_t> hx((size_t)T*N*R);
  for(size_t i=0;i<hx.size();++i) hx[i]=(int8_t)lrintf(fminf(fmaxf(ux(rng),-1.f),1.f)*127.f);
  // w_dn float -> per-channel int8 (dim over H)
  std::vector<int8_t> hwdn((size_t)K_HH*H); std::vector<float> hcm(K_HH);
  for(int k=0;k<K_HH;++k){
    float mx=1e-8f; std::vector<float> row(H);
    for(int cc=0;cc<H;++cc){ row[cc]=nd(rng)*0.02f; mx=fmaxf(mx,fabsf(row[cc])); }
    float sc=mx/127.f; hcm[k]=sc;
    for(int cc=0;cc<H;++cc) hwdn[(size_t)k*H+cc]=(int8_t)lrintf(row[cc]/sc);
  }
  // gate weights [4,H,Kc] per-channel (dim over Kc), scale + bias
  std::vector<int8_t> hwg((size_t)4*H*KC); std::vector<float> hws((size_t)4*H); std::vector<float> hbs((size_t)4*H);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc){
    float mx=1e-8f; std::vector<float> row(KC);
    for(int kc=0;kc<KC;++kc){ row[kc]=nd(rng)*0.1f; mx=fmaxf(mx,fabsf(row[kc])); }
    float sc=mx/127.f; hws[(size_t)g*H+oc]=sc; hbs[(size_t)g*H+oc]=nd(rng)*0.05f;
    for(int kc=0;kc<KC;++kc) hwg[((size_t)g*H+oc)*KC+kc]=(int8_t)lrintf(row[kc]/sc);
  }
  // repacked gate weights for the msplit kernel: per (g,oc) the KC bytes reordered to
  // [tg][kt][half][b] so a lane's 8 k-tiles (b0,b1) are 64 contiguous bytes -> int4 loads.
  std::vector<int8_t> hwg_rp((size_t)4*H*KC);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc)for(int kc=0;kc<KC;++kc){
    int kt=kc>>5, rem=kc&31, half=rem>>4, r2=rem&15, tgi=r2>>2, b=r2&3;
    hwg_rp[((size_t)g*H+oc)*KC + tgi*64 + kt*8 + half*4 + b] = hwg[((size_t)g*H+oc)*KC+kc];
  }

  // ---- device ----
  int8_t *dx,*dwdn,*dwg,*dwg_rp,*dring,*dring_ref; float *dcm,*dws,*dbs,*dcell,*dcell_ref;
  int *dscratch,*dflags;
  CUDA_CHECK(cudaMalloc(&dx,hx.size()));
  CUDA_CHECK(cudaMalloc(&dwdn,hwdn.size()));
  CUDA_CHECK(cudaMalloc(&dwg,hwg.size()));
  CUDA_CHECK(cudaMalloc(&dwg_rp,hwg_rp.size()));
  CUDA_CHECK(cudaMalloc(&dcm,K_HH*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dws,4*H*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dbs,4*H*sizeof(float)));
  size_t ring_bytes=(size_t)(T+1)*N*H;
  CUDA_CHECK(cudaMalloc(&dring,ring_bytes));
  CUDA_CHECK(cudaMalloc(&dcell,(size_t)N*H*sizeof(float)));
  int GY=N/BMG;
  size_t scratch_words=(size_t)2*GY*GX*WARPS*(K_HH/8*4)*32;   // x2 double-buffer
  CUDA_CHECK(cudaMalloc(&dscratch,scratch_words*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&dflags,(size_t)(N/16+1)*sizeof(int)));   // enough for any GY
  CUDA_CHECK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwg,hwg.data(),hwg.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dwg_rp,hwg_rp.data(),hwg_rp.size(),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dcm,hcm.data(),K_HH*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dws,hws.data(),4*H*sizeof(float),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dbs,hbs.data(),4*H*sizeof(float),cudaMemcpyHostToDevice));

  int HXd = H/GX;
#ifdef RESB
  size_t smem = (size_t)BMG*HX + (size_t)BMG*KC + (size_t)2*4*HX*sizeof(float);
#elif defined(GATE_GMEM)
  size_t smem = (size_t)BMG*HX + (size_t)2*4*HX*sizeof(float);
#else
  size_t smem = (size_t)BMG*HX + (size_t)4*HX*KC + (size_t)2*4*HX*sizeof(float);
#endif
  size_t smem_d = (size_t)DBM*H + (size_t)DBM*KC + (size_t)2*4*HXd*sizeof(float);
  auto set_attr=[&](const void*f,size_t s){ CUDA_CHECK(cudaFuncSetAttribute(f,
      cudaFuncAttributeMaxDynamicSharedMemorySize,(int)s)); };
  if(dorado){
    switch(GX){
      case 8:  set_attr((const void*)dorado_kernel<8>,smem_d); break;
      case 16: set_attr((const void*)dorado_kernel<16>,smem_d); break;
      default: fprintf(stderr,"dorado supports GX 8/16\n"); return 1;
    }
    printf("[dorado] DBM=%d smem=%zu bytes  grid=(%d,%d)\n",DBM,smem_d,GX,N/DBM);
    // pin gate weights + W_dn in L2 (persisting) so the ring stream doesn't evict them
    size_t pin = hwg.size(); size_t l2max=0;
    cudaDeviceGetAttribute((int*)&l2max, cudaDevAttrMaxPersistingL2CacheSize, dev);
    if(pin>l2max) pin=l2max;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin));
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dwg;
    av.accessPolicyWindow.num_bytes=pin; av.accessPolicyWindow.hitRatio=1.0f;
    av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &av);
  } else {
    switch(GX){
      case 8:  set_attr((const void*)msplit_kernel<8>,smem); break;
      case 16: set_attr((const void*)msplit_kernel<16>,smem); break;
      case 32: set_attr((const void*)msplit_kernel<32>,smem); break;
      default: fprintf(stderr,"unsupported GX %d\n",GX); return 1;
    }
    printf("smem=%zu bytes\n",smem);
    // pin gate weights in L2 (helps GATE_GMEM streamed path resist ring eviction)
    size_t pin=hwg_rp.size(); size_t l2max=0;
    cudaDeviceGetAttribute((int*)&l2max, cudaDevAttrMaxPersistingL2CacheSize, dev);
    if(pin>l2max) pin=l2max;
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, pin));
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dwg_rp;
    av.accessPolicyWindow.num_bytes=pin; av.accessPolicyWindow.hitRatio=1.0f;
    av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &av);
  }

  auto run=[&](int rev){
    CUDA_CHECK(cudaMemset(dflags,0,(size_t)(N/16+1)*sizeof(int)));
    if(dorado){
      // init the recurrence's initial h slot to 0 (fwd: slot0, rev: slotT)
      int init_slot = rev ? T : 0;
      CUDA_CHECK(cudaMemset(dring + (size_t)init_slot*N*H, 0, (size_t)N*H));
      switch(GX){
        case 8:  launch_dorado<8>(dx,dwdn,dcm,dwg,dws,dbs,dring,dcell,dflags,N,T,rev,smem_d); break;
        case 16: launch_dorado<16>(dx,dwdn,dcm,dwg,dws,dbs,dring,dcell,dflags,N,T,rev,smem_d); break;
      }
      return;
    }
    CUDA_CHECK(cudaMemset(dscratch,0,scratch_words*sizeof(int)));
    switch(GX){
      case 8:  launch<8>(dx,dwdn,dcm,dwg_rp,dws,dbs,dring,dcell,dscratch,dflags,N,T,rev,smem); break;
      case 16: launch<16>(dx,dwdn,dcm,dwg_rp,dws,dbs,dring,dcell,dscratch,dflags,N,T,rev,smem); break;
      case 32: launch<32>(dx,dwdn,dcm,dwg_rp,dws,dbs,dring,dcell,dscratch,dflags,N,T,rev,smem); break;
    }
  };

  if(!bench){
    // correctness vs reference
    n_cmp = n_cmp<N?n_cmp:N;
    CUDA_CHECK(cudaMalloc(&dring_ref,ring_bytes));
    CUDA_CHECK(cudaMalloc(&dcell_ref,(size_t)N*H*sizeof(float)));
    // enlarge malloc heap for reference new/delete
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    run(reverse);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    ref_kernel<<<(n_cmp+31)/32,32>>>(dx,dwdn,dcm,dwg,dws,dbs,dring_ref,dcell_ref,N,T,reverse,n_cmp);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // compare ring for first n_cmp rows, all slots
    std::vector<int8_t> a(ring_bytes), b(ring_bytes);
    CUDA_CHECK(cudaMemcpy(a.data(),dring,ring_bytes,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(),dring_ref,ring_bytes,cudaMemcpyDeviceToHost));
    long mism=0, maxd=0; long total=0;
    int first_slot = reverse? 0 : 1;
    for(int s=0;s<=T;++s){
      for(int r=0;r<n_cmp;++r)for(int oc=0;oc<H;++oc){
        long idx=((long)s*N+r)*H+oc;
        int d=abs((int)a[idx]-(int)b[idx]);
        if(s==0 && !reverse) continue; // slot0 unused fwd
        if(s==T && reverse)  continue;
        total++;
        if(d){ mism++; if(d>maxd)maxd=d; }
      }
    }
    printf("[%s] correctness: mismatches=%ld/%ld  maxdiff=%ld  -> %s\n",
      reverse?"reverse":"forward", mism,total,maxd, mism==0?"BIT-EXACT PASS":"FAIL");
    return mism==0?0:1;
  } else {
    // benchmark: warm GPU, then time
    // warmup
    for(int w=0;w<3;++w) run(reverse);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int iters=5;
    cudaEventRecord(e0);
    for(int it=0;it<iters;++it) run(reverse);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    double per_step_us = (double)ms/iters/T*1000.0;
    printf("BENCH: %.3f ms/launch  over T=%d  => %.3f us/step\n", ms/iters, T, per_step_us);
    return 0;
  }
}
