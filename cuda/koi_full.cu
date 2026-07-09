// koi_full.cu -- FULL FUSED STACK for the int8 factored-LSTM gate recurrence (A100/sm80).
// Assembling ALL of {software pipeline + persistence + resident weights + fused epilogue +
// one-wave} together -- the combination prior attempts only built in subsets.
//
// M1 (this file, gate-isolated): persistent-ize the gate mainloop with SMEM-RESIDENT gate
// weights (loaded once in the prologue) AND a cp.async multistage pipeline on the per-step
// ACTIVATION A=[hh_down|x] (double-buffered, prefetch A(t+1) while computing gate(t)).
// hh_down is fed from a PRECOMPUTED buffer (down-proj not fused yet) to isolate the gate.
// Fused in-register LSTM epilogue (f16 cell), bit-exact vs a naive GPU reference.
//
// THE M1 METRIC: does imma-util stay ~11%+ (fused_cutlass's pipelined level) after
// persistent-izing with resident weights, or collapse toward ~1-2% (resident_gate's
// no-pipeline failure)?  ncu the steady loop.
//
// Structure inherited from resident_gate_flstm.cu (proven bit-exact gate mainloop + fused
// epilogue + resident smem weights + persistence); the NEW ingredient is the cp.async
// double-buffered activation pipeline (PIPE=1) that resident_gate lacked (it had imma 2.6%).
//
// Config knobs (compile-time):
//   GX   : gate column split (hidden channels/CTA = H/GX).  weights smem = 4*(H/GX)*KC.
//   BMG  : gate rows/CTA (8 warps x 16).  A smem = STAGES_A*BMG*KC.
//   PIPE : 1 = double-buffered cp.async A prefetch (the pipeline); 0 = single-buffer, no overlap.
//   grid = (GX, N/BMG).  one-wave iff GX*(N/BMG) <= 108.
//
// Build:
//   nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013 \
//        -DGX=8 -DBMG=128 -DPIPE=1 -DFP16CELL koi_full.cu -o koi_full
//
// ============================ M1 RESULT (A100, N=1536, T=2048, warm, median of 9) =========
// DECISION METRIC: imma-util stays ~13-14% in EVERY viable config -- it does NOT collapse
// to ~1-2%. The "pipeline can't survive the persistent T-loop" fear is FALSE (refuted).
// BUT the per-step gate µs FAILS the <=28µs bar, and the pipeline is a NO-OP:
//   config                                    grid      smem   imma  occ   L1TEX  gate µs
//   smem-resident + NO pipe  (GX8  PIPE0) *   96 1-wave 160KB  14.5% 12.5% 91%    39.4   BIT-EXACT
//   smem-resident + pipe     (GX16 PIPE1)    192 2-wave 128KB  14.2% 12.5% 92%    40.1   BIT-EXACT
//   smem-resident + NO pipe  (GX16 PIPE0)    192 2-wave  96KB  13.9% 12.5% 90%    41.0   BIT-EXACT
//   LDG-weights   + pipe     (GX8  PIPE1) *   96 1-wave  64KB  12.8% 12.5% 82%    45.4   BIT-EXACT
//   LDG-weights   + pipe/2CTA(GX16 c2)       192 2-wave  64KB  13.2% 21.2% 88%    43.9   BIT-EXACT
//   ---- FALLBACK for reference: fused_cutlass gate @ N=1536 (CUTLASS ldmatrix) = 26.2µs ----
//   PIPE=1 + GX=8 (one-wave resident) DOES NOT BUILD: 192KB > 163KB smem -- one-wave +
//   resident-weights + double-buffered-A-pipeline CANNOT coexist in smem (structural wall).
//
// WHY M1 fails the bar (root cause, ncu): gpu__compute_memory_throughput = 100% SOL in ALL
// configs -- the kernel is hard-bound on the LSU/L1TEX (smem) pipe, NOT the tensor pipe
// (imma 13%) and NOT latency (so the pipeline can't help; it hides latency, not throughput).
// The hand mainloop issues 16.3M shared-load instructions/kernel (weights re-read via
// LDS.128 every IMMA-group + activation via scalar LDS.32). CUTLASS's ldmatrix path does the
// same loads in ~1.5x fewer smem-pipe transactions -> fused_cutlass gate 26µs vs this 39µs.
// Occupancy is NOT the lever: 12.5%->21% (2 CTA/SM) barely moved the time (39->44µs).
// Streaming weights L2->reg via LDG (koi's scheme, WLDG=1) moves weights OFF the smem pipe
// but is WORSE (45µs): at 12.5% one-wave occupancy the LDG latency isn't hidden and gpu mem
// SOL is still 100%.  "One-wave" (minimal CTAs) inherently forces 1 CTA/SM = 12.5% occ,
// which is fundamentally opposed to latency-hiding (wants many warps/SM).
//
// VERDICT: the full stack AS SPECIFIED (smem-resident weights + persistence + activation
// pipeline + one-wave) does NOT beat the fused_cutlass 37µs/step fallback -- it lands ~1.5x
// WORSE on the gate.  The pipeline survives persistence (good) but is irrelevant because
// resident-weights/one-wave are self-defeating (they saturate the smem/LSU pipe at forced
// 12.5% occupancy).  This confirms, WITH the pipeline now explicitly added and measured as a
// no-op, the prior resident_gate_flstm (121µs) and fused_cutlass-B2 (smem-won't-fit) walls.
// FAIL-FAST: stop at M1; do NOT proceed to M2 (adding cross-CTA producer/consumer sync would
// layer handoff latency onto an already-losing gate).  fused_cutlass 37µs stays the ceiling.
//
// THE ONE UNTRIED LEVER (koi's actual gate SASS, g07.txt = 320 IMMA : 40 LDSM : 18 LDGSTS :
// 128 LDG.CONSTANT): ldmatrix(LDSM) the ACTIVATION from a small smem pipe + stream WEIGHTS
// L2->reg via LDG.CONSTANT + reuse each weight fragment across 8 IMMA (8:1, weight-fragment
// M-reuse across 4 row-tiles x 2 k-tiles).  NO subset attempt combined ldmatrix-A WITH
// weight-M-reuse: fused_cutlass ldmatrix's BOTH A and weights (3.56:1); resident_gate/this
// LDS weights from smem; koi8_gate ldmatrix'd WEIGHTS (wrong operand) + scalar-loaded A.
// That is the genuine multi-day-class rewrite the honest framing flagged; payoff is bounded
// (CUTLASS ldmatrix already reaches 26µs; koi's 13.9 needs the exact 8:1 hand fragment loop).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define H 1024
#define K_HH 128
#define R 128
#define KC 256
#define WARPS 8
#define NTHREADS 256
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef GX
#define GX 8
#endif
#ifndef BMG
#define BMG 128
#endif
#ifndef PIPE
#define PIPE 1
#endif
#ifndef WLDG
#define WLDG 0        // 1 = stream gate weights L2->registers via LDG (koi scheme); smem holds only A.
#endif
#ifndef CTAPSM
#define CTAPSM 1      // CTAs/SM target for __launch_bounds__ (2 possible when smem small enough)
#endif
#define HX (H/GX)
#if PIPE
#define STAGES_A 2
#else
#define STAGES_A 1
#endif

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
__device__ __forceinline__ void mma_m16n8k32(int&c0,int&c1,int&c2,int&c3,
    int a0,int a1,int a2,int a3,int b0,int b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint32_t smem_addr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(void* dst,const void* src){
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(smem_addr(dst)),"l"(src));
}

// stage A tile [BMG,KC] for step t, this CTA's rows [row0,row0+BMG) -> sA buffer
__device__ __forceinline__ void load_A(int8_t* dstA, const int8_t* A_all, size_t t, int N, int row0){
  const int tid=threadIdx.x;
  const int nch = KC/16;               // 16 chunks of 16B per row
  const int8_t* base = A_all + (size_t)t*N*KC + (size_t)row0*KC;
  #pragma unroll
  for(int i=tid;i<BMG*nch;i+=NTHREADS){
    int r=i/nch, c=i%nch;
    cpa16(&dstA[r*KC+c*16], &base[(size_t)r*KC+c*16]);
  }
}

// ---- persistent gate kernel: resident smem weights + cp.async activation pipeline ----
__global__ void __launch_bounds__(NTHREADS,CTAPSM) gate_kernel(
    const int8_t* __restrict__ A_all,   // [T,N,KC] precomputed activations
    const int8_t* __restrict__ wg_rp,   // [4,H,KC] repacked gate weights
    const float* __restrict__ wscale, const float* __restrict__ bias,
    cellT* __restrict__ cell, int8_t* __restrict__ hout,
    int N, int T, int reverse, int store_all)
{
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int gx=blockIdx.x, gy=blockIdx.y;
  const int row0=gy*BMG, hbase=gx*HX;
  const float AS=1.0f/127.0f;

  extern __shared__ int8_t smem[];
#if WLDG
  int8_t* sA  = smem;                 // smem holds ONLY the activation pipeline
#else
  int8_t* sWg = smem;                 // [4*HX*KC] RESIDENT gate weights (repacked)
  int8_t* sA  = sWg + 4*HX*KC;        // [STAGES_A][BMG][KC]
  // prologue ONCE: resident gate weights slice -> smem
  for(int i=tid;i<4*HX*KC;i+=NTHREADS){
    int g=i/(HX*KC), rem=i%(HX*KC), col=rem/KC, kc=rem%KC;
    sWg[i]=wg_rp[((long)g*H + hbase+col)*KC + kc];
  }
#endif

  auto tstep=[&](int t)->size_t{ int tt = reverse ? (T-1-t):t; return (size_t)tt; };

  // prime: cp.async A(step 0)
  load_A(sA + 0, A_all, tstep(0), N, row0);
  asm volatile("cp.async.commit_group;\n");
  __syncthreads();  // weights resident & visible

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int out  = reverse ? tt : (tt+1);
    int8_t* hout_t = hout + (store_all ? (size_t)out*N*H : 0);
    int cur = t & (STAGES_A-1);
    int8_t* Acur = sA + (size_t)cur*BMG*KC;

#if PIPE
    if(t+1<T){ int nxt=(t+1)&(STAGES_A-1);
      load_A(sA + (size_t)nxt*BMG*KC, A_all, tstep(t+1), N, row0);
      asm volatile("cp.async.commit_group;\n");
    }
    asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES_A-1));  // A(t) landed; A(t+1) in flight
#else
    asm volatile("cp.async.wait_all;\n");
#endif
    __syncthreads();

    // ---- gate: M-split (warp = 16 rows), resident weights, all HX channels ----
    #pragma unroll 1
    for(int rt=0; rt<BMG/16; ++rt){
      if((rt & (WARPS-1)) != warp) continue;      // warp owns row-tile rt where rt%8==warp
      int A[8][4];
      #pragma unroll
      for(int kt=0;kt<8;++kt){ int rb=rt*16, co=kt*32+tg*4;
        A[kt][0]=*(const int*)&Acur[(rb+gid)*KC+co];    A[kt][1]=*(const int*)&Acur[(rb+gid+8)*KC+co];
        A[kt][2]=*(const int*)&Acur[(rb+gid)*KC+co+16]; A[kt][3]=*(const int*)&Acur[(rb+gid+8)*KC+co+16]; }
      for(int nn=0;nn<HX/8;++nn){
        int cg[4][4];
        #pragma unroll
        for(int g=0;g<4;++g) for(int e=0;e<4;++e) cg[g][e]=0;
        #pragma unroll
        for(int g=0;g<4;++g){
#if WLDG
          const int8_t* wgs = wg_rp + (long)g*H*KC + (long)(hbase+nn*8+gid)*KC + tg*64;  // L2->reg via LDG
          int4 w0=__ldg((const int4*)(wgs)), w1=__ldg((const int4*)(wgs+16)),
               w2=__ldg((const int4*)(wgs+32)), w3=__ldg((const int4*)(wgs+48));
#else
          const int8_t* wgs = sWg + g*HX*KC + (nn*8+gid)*KC + tg*64;   // resident smem, repacked
          int4 w0=*(const int4*)(wgs), w1=*(const int4*)(wgs+16), w2=*(const int4*)(wgs+32), w3=*(const int4*)(wgs+48);
#endif
          int b0[8]={w0.x,w0.z,w1.x,w1.z,w2.x,w2.z,w3.x,w3.z};
          int b1[8]={w0.y,w0.w,w1.y,w1.w,w2.y,w2.w,w3.y,w3.w};
          #pragma unroll
          for(int kt=0;kt<8;++kt)
            mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],A[kt][0],A[kt][1],A[kt][2],A[kt][3],b0[kt],b1[kt]);
        }
        #pragma unroll
        for(int e=0;e<4;++e){
          int lcol=nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
          int oc=hbase+lcol, gr=row0+rt*16+rin;
          float cv=(float)cell[(long)gr*H+oc];
          int8_t hn=epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
              wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
              bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv);
          cell[(long)gr*H+oc]=(cellT)cv;
          hout_t[(long)gr*H+oc]=hn;
        }
      }
    }
#if !PIPE
    // single-buffer: reload A(t+1) after compute (no overlap); wait at top of next iter
    __syncthreads();
    if(t+1<T){ load_A(sA + 0, A_all, tstep(t+1), N, row0); asm volatile("cp.async.commit_group;\n"); }
#else
    __syncthreads();  // safe to overwrite this buffer 2 iters later
#endif
  }
}

// ---- naive GPU reference: same A_all, plain gate weights, f16 cell recurrence ----
__global__ void ref_kernel(const int8_t* A_all,
    const int8_t* Bi,const int8_t* Bf,const int8_t* Bg,const int8_t* Bo,
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/127.0f; const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  cellT* c=new cellT[H];
  for(int i=0;i<H;++i) c[i]=(cellT)0.f;
  for(int t=0;t<T;++t){ int tt=reverse?(T-1-t):t; int ws=reverse?tt:(tt+1);
    const int8_t* Ar=A_all+((long)tt*N+r)*KC;
    for(int oc=0;oc<H;++oc){ int g[4];
      for(int gg=0;gg<4;++gg){ const int8_t* w=Bp[gg]+(long)oc*KC; int a=0;
        for(int kc=0;kc<KC;++kc) a+=(int)Ar[kc]*(int)w[kc]; g[gg]=a; }
      float cv=(float)c[oc];
      int8_t hn=epi_elem(g[0],g[1],g[2],g[3],wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv); c[oc]=(cellT)cv;
      hh_all[((long)ws*N+r)*H+oc]=hn; }
  }
  delete[] c;
}

int main(int argc,char**argv){
  int N=256,T=64,reverse=0,bench=0,n_cmp=32,dev=0;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T"))T=atoi(argv[++i]); else if(!strcmp(argv[i],"--reverse"))reverse=1;
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--ncmp"))n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(N%BMG) N=((N+BMG-1)/BMG)*BMG;
  size_t smem=(size_t)(WLDG?0:4*HX*KC) + (size_t)STAGES_A*BMG*KC;
  int ncta = GX*(N/BMG);
  printf("koi_full GATE: GX=%d HX=%d BMG=%d PIPE=%d WLDG=%d STAGES_A=%d ctapsm=%d cell=%s N=%d T=%d rev=%d grid=(%d,%d)=%d CTA %s smem=%.1fKB\n",
    GX,HX,BMG,PIPE,WLDG,STAGES_A,CTAPSM,sizeof(cellT)==2?"f16":"f32",N,T,reverse,GX,N/BMG,ncta,ncta<=108?"ONE-WAVE":"multi-wave",smem/1024.0);

  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ux(-1,1);
  // precomputed activations A_all[T,N,KC] int8 (== [hh_down|x], both quantized to int8)
  std::vector<int8_t> hA((size_t)T*N*KC);
  for(size_t i=0;i<hA.size();++i) hA[i]=(int8_t)lrintf(fminf(fmaxf(ux(rng),-1.f),1.f)*127.f);
  std::vector<int8_t> hB[4]; std::vector<float> hws((size_t)4*H),hbs((size_t)4*H);
  for(int g=0;g<4;++g){ hB[g].resize((size_t)H*KC);
    for(int oc=0;oc<H;++oc){ float mx=1e-8f; std::vector<float> row(KC);
      for(int kc=0;kc<KC;++kc){row[kc]=nd(rng)*0.1f;mx=fmaxf(mx,fabsf(row[kc]));}
      float sc=mx/127.f; hws[(size_t)g*H+oc]=sc; hbs[(size_t)g*H+oc]=nd(rng)*0.05f;
      for(int kc=0;kc<KC;++kc) hB[g][(size_t)oc*KC+kc]=(int8_t)lrintf(row[kc]/sc); } }
  // repacked weights for the resident-smem MMA (same repack as resident_gate_flstm.cu)
  std::vector<int8_t> hrp((size_t)4*H*KC);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc)for(int kc=0;kc<KC;++kc){
    int kt=kc>>5,rem=kc&31,half=rem>>4,r2=rem&15,tgi=r2>>2,b=r2&3;
    hrp[((size_t)g*H+oc)*KC + tgi*64+kt*8+half*4+b]=hB[g][(size_t)oc*KC+kc]; }

  int8_t *dA,*dB[4],*drp,*dring; float *dws,*dbs; cellT *dcell;
  CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&drp,hrp.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  CK(cudaMalloc(&dcell,(size_t)N*H*sizeof(cellT)));
  CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(gate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    size_t ring_bytes=(size_t)(T+1)*N*H;
    CK(cudaMalloc(&dring,ring_bytes));
    int8_t* dref; CK(cudaMalloc(&dref,ring_bytes));
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    CK(cudaMemset(dcell,0,(size_t)N*H*sizeof(cellT)));
    { int b=reverse?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H)); }
    gate_kernel<<<dim3(GX,N/BMG),NTHREADS,smem>>>(dA,drp,dws,dbs,dcell,dring,N,T,reverse,1);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ref_kernel<<<(n_cmp+31)/32,32>>>(dA,dB[0],dB[1],dB[2],dB[3],dws,dbs,dref,N,T,reverse,n_cmp);
    CK(cudaDeviceSynchronize());
    std::vector<int8_t> a(ring_bytes),bb(ring_bytes);
    CK(cudaMemcpy(a.data(),dring,ring_bytes,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(bb.data(),dref,ring_bytes,cudaMemcpyDeviceToHost));
    long mism=0,maxd=0,tot=0;
    for(int s=0;s<=T;++s){ if((s==0&&!reverse)||(s==T&&reverse))continue;
      for(int r=0;r<n_cmp;++r)for(int oc=0;oc<H;++oc){ long idx=((long)s*N+r)*H+oc;
        int d=abs((int)a[idx]-(int)bb[idx]); tot++; if(d){mism++; if(d>maxd)maxd=d;} } }
    printf("[%s] mism=%ld/%ld maxd=%ld -> %s\n",reverse?"reverse":"forward",mism,tot,maxd,mism==0?"BIT-EXACT PASS":"FAIL");
    return mism==0?0:1;
  }
  // bench: single overwrite buffer for h (store_all=0)
  CK(cudaMalloc(&dring,(size_t)N*H));
  CK(cudaMemset(dcell,0,(size_t)N*H*sizeof(cellT)));
  auto run=[&](){ gate_kernel<<<dim3(GX,N/BMG),NTHREADS,smem>>>(dA,drp,dws,dbs,dcell,dring,N,T,reverse,0); };
  for(int w=0;w<3;++w) run(); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<9;++r){ cudaEventRecord(e0); run(); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); us.push_back((double)ms/T*1000.0); }
  std::sort(us.begin(),us.end());
  printf("koi_full GATE step: %.2f us/step (median of 9, T=%d)  [fused_cutlass gate ~28us @N2048; koi ~13.9 full-step @N1536]\n",us[4],T);
  return 0;
}
