// koi_fused_pipe.cu -- CROSS-STEP PIPELINE TEST on the full fused kernel (2026-07-09).
// ==================================================================================
// Took koi_flstm's full-fused producer/consumer structure and grafted in koi_gate_pad's
// BEST gate mainloop (ldmatrix-A + conflict-free padded sA + __ldg constant-path weights
// + RESIDENT f32 cell) -- the 22%-imma-capable gate. Bit-exact fwd+reverse.
// RESULT: 188.96us/step @N1536 (377 @N2048) == koi_flstm's ORIGINAL 193us. The vastly
// better gate changed NOTHING. ncu: barrier-stall = 40.3/issue (DOMINANT), imma 3.1%,
// 12.5% occ. NO_BARRIER (unsafe) = 150us. VERDICT: the persistent fused kernel is
// SYNC/ALTERNATION-bound, NOT gate-compute-bound. The recurrence G[t]->h[t]->D[t+1]->G[t+1]
// forbids D/G cross-step overlap (proven: every phase depends on the immediately prior),
// so persistence forces a per-step cross-CTA sync that dominates -- and the FASTER the gate,
// the MORE it waits (40 vs the old 20). Improving within-step gate compute is irrelevant
// here. fused_cutlass.cu (per-step graph launches = free cross-step ordering, no in-kernel
// sync spin, CUTLASS within-step pipeline) is the correct structure = 37us. DEAD END confirmed
// with the best possible gate. See memory dorado-factorised-lstm-kernel round 40.
// ==================================================================================
//
// koi_flstm.cu -- ground-up rebuild modeled on the DECODED koi factorised_lstm SASS
// (dorado v2.0.0 sm80). Two disjoint CTA paths in ONE persistent kernel, grid (9, N/bM):
//   gx==8  -> DOWN-PROJ CTA: recurrent down-proj for gy's bM rows (256 IMMA, K=H=1024),
//            reads h[bM,H] from ring, writes hh_down[bM,128] to double-buffered scratch.
//   gx 0..7-> GATE CTA: owns gy's bM rows x gx's 128 hidden channels (4 gates = 512 out).
//            RESIDENT gate-weight slice in smem (128KB, loaded once, interleaved quads).
//            Pipelined steady loop: kt-outer 4-gate ILP; vectorized cell; STG h to ring.
// Decoupled producer/consumer via per-column flags (down runs ahead of gate -> overlap,
// unlike a phase-barrier which idles 8/9 of the wave). NO_BARRIER variant = koi's raw
// 1-step-skew scheme.
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math -diag-suppress 177 koi_flstm.cu -o koi_flstm
//
// RESULT (A100, N=2048, T=2048): 383 us/step (barrier); 192 us NO_BARRIER.  BIT-EXACT
// fwd+reverse (barrier variant). 126 regs, 0 spill, 1 CTA/SM.  This is koi's EXACT
// decoded structure built fresh -- and it reveals WHY koi's 13.9us is not reproducible
// as a correctness-guaranteed build at H=1024:
//
//   The factorised-LSTM recurrence is SEQUENTIAL per step: down-proj(t) -> gate(t) ->
//   h(t) -> down-proj(t+1).  So the down-proj CTA (gx==8) and the gate CTAs (gx0..7)
//   cannot overlap -- they ALTERNATE (down produces hh_down(t), gate consumes it, gate
//   produces h(t), down needs it for t+1).  A PERSISTENT kernel (required to keep gate
//   weights resident) must therefore do CROSS-CTA sync every step.
//   ncu: barrier-stall = 20.2/issue (dominant), imma-util 1%.  The per-step atomic-flag
//   producer/consumer spin (co-resident CTAs burning SMs while waiting) is catastrophic
//   over 2048 steps -- it costs FAR more than resident weights save.
//
//   koi hides this ONLY via its no-barrier timing-race (the flags param is UNUSED in koi
//   -- decoded from SASS): it relies on iteration_time (~14us) >> store->load latency so
//   the producer stays ahead without synchronization.  That is a hardware/schedule-
//   dependent race, not a correctness-guaranteed construct; our NO_BARRIER variant
//   (192us, and NOT bit-exact) confirms removing the sync isn't safe and still doesn't
//   win (the alternation + underutilized 16-CTA down phase remain).
//
//   fused_cutlass.cu (37us) SIDESTEPS all of it: per-step down+gate as graph-captured
//   kernel LAUNCHES.  The kernel boundary gives cross-step ordering FOR FREE (no atomic
//   spin, no race), the graph amortizes launch overhead to ~0, and CUTLASS pipelines the
//   gate.  It already has koi's sequential down->gate + pipelined gate; it only lacks
//   resident weights -- and the resident-weights gain is smaller than the in-kernel-sync
//   cost it would require.  DEFINITIVE: at H=1024, koi's persistent+resident structure
//   is NOT a net win over graph-captured per-step launches; parity would need koi's
//   unsafe no-barrier race.  SHIP fused_cutlass.cu = 37.08 us/step.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define H 1024
#define K_HH 128
#define R 128
#define KC 256
#define KCS (KC+16)          // padded smem A-row stride (bytes) -> conflict-free ldmatrix
#define WARPS 8
#define NTHREADS 256
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef BM
#define BM 128
#endif
#define GX 8                 // 8 gate CTAs; gx==8 is the down-proj CTA -> gridX=9
#define HXG (H/GX)           // hidden channels per gate CTA = 128

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
__device__ __forceinline__ void ldm_x4(int&r0,int&r1,int&r2,int&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ uint32_t smem_addr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void wait_ge(volatile int* p, int v){
#ifdef NANOSLEEP
  // B1: exponential-backoff poll (NOT hot-spin) -> SM not saturated re-issuing the load.
  if(threadIdx.x==0){ unsigned ns=32; while(*p < v){ __nanosleep(ns); ns=ns<1024?ns*2:1024; } }
#else
  if(threadIdx.x==0){ while(*p < v){} }
#endif
  __syncthreads();          // all threads wait for tid0's observation
  __threadfence();          // acquire: subsequent global reads see producer's data
}
__device__ __forceinline__ void signal(int* p){
  __syncthreads();          // all warps finished writing the produced data
  __threadfence();          // release: data visible in L2 before the flag bump
  if(threadIdx.x==0) atomicAdd(p,1);
  __syncthreads();
}

// ================= DOWN-PROJ CTA (gx==8) =================
// h[bM,H] staged in smem (128KB @ BM128); W_dn streamed L2 (cp.async k-chunks).
__device__ void downproj_cta(int8_t* smem, const int8_t* w_dn, const float* comb,
    const int8_t* hh_all, int8_t* scratch, int* fdown, int* fgate,
    int N, int T, int reverse, int gy){
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int row0=gy*BM;
  int8_t* sH = smem;                 // [BM, H]  (128KB)
  const int nkb=H/32;
  int* fd = fdown+gy; int* fg = fgate+gy;
  for(int t=0;t<T;++t){
    int tt=reverse?(T-1-t):t;
    int prev=reverse?(tt+1):tt;
#ifndef NO_BARRIER
    if(t>0) wait_ge(fg, GX*t);       // gate finished step t-1 -> h[prev] in ring
#endif
    const int8_t* hprev=hh_all+(size_t)prev*N*H;
    for(int j=tid;j<BM*(H/16);j+=NTHREADS){ int r=j/(H/16), c=j%(H/16);
      ((int4*)sH)[r*(H/16)+c]=*(const int4*)&hprev[(long)(row0+r)*H+c*16]; }
    __syncthreads();
    int8_t* sc = scratch + (size_t)(t&1)*N*K_HH;    // double-buffered
    // M-split: 8 warps x 16 rows; each warp all K_hh=128 output (16 n8-tiles), K=1024
#ifndef SKIP_DOWN
    for(int rt=0; rt<BM/16; ++rt){
      if((rt&7)!=warp) continue;
      for(int nt=0; nt<K_HH/8; ++nt){
        int c0=0,c1=0,c2=0,c3=0;
        for(int kt=0;kt<nkb;++kt){ int rb=rt*16, co=kt*32+tg*4;
          int a0=*(const int*)&sH[(rb+gid)*H+co], a1=*(const int*)&sH[(rb+gid+8)*H+co];
          int a2=*(const int*)&sH[(rb+gid)*H+co+16], a3=*(const int*)&sH[(rb+gid+8)*H+co+16];
          int nrow=nt*8+gid;
          int b0=*(const int*)&w_dn[nrow*H+kt*32+tg*4], b1=*(const int*)&w_dn[nrow*H+kt*32+16+tg*4];
          mma_m16n8k32(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        }
        int col0=nt*8+2*tg, col1=col0+1, rl=row0+rt*16+gid, rh=rl+8;
        sc[(long)rl*K_HH+col0]=clamp_i8(rintf((float)c0*comb[col0]));
        sc[(long)rl*K_HH+col1]=clamp_i8(rintf((float)c1*comb[col1]));
        sc[(long)rh*K_HH+col0]=clamp_i8(rintf((float)c2*comb[col0]));
        sc[(long)rh*K_HH+col1]=clamp_i8(rintf((float)c3*comb[col1]));
      }
    }
#endif
    signal(fd);                       // hh_down[t] ready
    __syncthreads();
  }
}

// ================= GATE CTA (gx 0..7) -- koi_gate_pad mainloop grafted in =================
// UPGRADE vs koi_flstm's original gate_cta (which ran ~1% imma): (1) ldmatrix-A from a
// BANK-CONFLICT-FREE padded sA (KCS stride); (2) __ldg gate weights L2->reg on the CONSTANT
// cache path (OFF the smem pipe) -- no 128KB resident smem slice; (3) recurrent cell RESIDENT
// in registers across T (creg, f32 -> bit-exact vs the f32 ref, no rounding). This is exactly
// koi_gate_pad's 22%-imma gate, reading A=[hh_down|x] from the fused scratch+x instead of a
// precomputed ring. MSET=1 at BM=128 (each warp owns 16 rows).
__device__ void gate_cta(int8_t* smem, const int8_t* wg_rp, const float* wscale,
    const float* bias, float* cell, int8_t* hh_all, const int8_t* x, const int8_t* scratch,
    int* fdown, int* fgate, int N, int T, int reverse, int gx, int gy){
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int row0=gy*BM, hbase=gx*HXG;
  const float AS=1.0f/127.0f;
  int8_t* sA = smem;                 // [BM][KCS] padded (no resident weights)
  const int wr0 = warp*16;           // MSET=1: each warp owns 16 rows
  float creg[HXG/8][4];              // resident recurrent cell (f32, no rounding)
  #pragma unroll
  for(int a=0;a<HXG/8;++a) for(int e=0;e<4;++e) creg[a][e]=0.f;
  int* fd=fdown+gy; int* fg=fgate+gy;
  for(int t=0;t<T;++t){
    int tt=reverse?(T-1-t):t;
    int out=reverse?tt:(tt+1);
    int8_t* hout=hh_all+(size_t)out*N*H;
#ifndef NO_BARRIER
    wait_ge(fd, t+1);                 // down produced hh_down[t]
#endif
    const int8_t* sc = scratch + (size_t)(t&1)*N*K_HH;
    const int8_t* xt = x + (size_t)tt*N*R;
    // assemble A=[hh_down(128) | x(128)] into padded sA
    for(int j=tid;j<BM*(K_HH/16);j+=NTHREADS){ int r=j/(K_HH/16), c=j%(K_HH/16);
      *(int4*)&sA[r*KCS + c*16]=*(const int4*)&sc[(long)(row0+r)*K_HH+c*16]; }
    for(int j=tid;j<BM*(R/16);j+=NTHREADS){ int r=j/(R/16), c=j%(R/16);
      *(int4*)&sA[r*KCS + K_HH + c*16]=*(const int4*)&xt[(long)(row0+r)*R+c*16]; }
    __syncthreads();
    // ldmatrix A frags (conflict-free padded stride), reused across all channels
    int Af[8][4];
    #pragma unroll
    for(int kt=0;kt<8;++kt){ int r=lane&15, koff=(lane>>4)*16;
      uint32_t a=smem_addr(&sA[(wr0+r)*KCS + kt*32 + koff]);
      ldm_x4(Af[kt][0],Af[kt][1],Af[kt][2],Af[kt][3],a); }
#ifndef SKIP_GATE
    #pragma unroll
    for(int nn=0;nn<HXG/8;++nn){
      int cg[4][4];
      #pragma unroll
      for(int g=0;g<4;++g) for(int e=0;e<4;++e) cg[g][e]=0;
      #pragma unroll
      for(int g=0;g<4;++g){
        const int8_t* wgs = wg_rp + (long)g*H*KC + (long)(hbase+nn*8+gid)*KC + tg*64;
        int4 w0=__ldg((const int4*)wgs), w1=__ldg((const int4*)(wgs+16)),
             w2=__ldg((const int4*)(wgs+32)), w3=__ldg((const int4*)(wgs+48));
        int b0[8]={w0.x,w0.z,w1.x,w1.z,w2.x,w2.z,w3.x,w3.z};
        int b1[8]={w0.y,w0.w,w1.y,w1.w,w2.y,w2.w,w3.y,w3.w};
        #pragma unroll
        for(int kt=0;kt<8;++kt)
          mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],
                       Af[kt][0],Af[kt][1],Af[kt][2],Af[kt][3],b0[kt],b1[kt]);
      }
      #pragma unroll
      for(int e=0;e<4;++e){
        int lcol=nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
        int oc=hbase+lcol, gr=row0+wr0+rin;
        float cv=creg[nn][e];
        int8_t hn=epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
            wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
            bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv);
        creg[nn][e]=cv;
        hout[(long)gr*H+oc]=hn;
      }
    }
#endif
    signal(fg);                       // this gate CTA finished step t
    __syncthreads();
  }
}

__global__ void __launch_bounds__(NTHREADS,1) koi_kernel(
    const int8_t* w_dn, const float* comb, const int8_t* wg_rp, const float* wscale,
    const float* bias, float* cell, int8_t* hh_all, const int8_t* x, int8_t* scratch,
    int* fdown, int* fgate, int N, int T, int reverse){
  extern __shared__ int8_t smem[];
  int gx=blockIdx.x, gy=blockIdx.y;
  if(gx==GX) downproj_cta(smem, w_dn, comb, hh_all, scratch, fdown, fgate, N,T,reverse,gy);
  else       gate_cta(smem, wg_rp, wscale, bias, cell, hh_all, x, scratch, fdown, fgate, N,T,reverse,gx,gy);
}

// ---- naive GPU reference ----
__global__ void ref_kernel(const int8_t* x,const int8_t* w_dn,const float* comb,
    const int8_t* Bi,const int8_t* Bf,const int8_t* Bg,const int8_t* Bo,
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/127.0f; const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  int8_t* h=new int8_t[H]; float* c=new float[H]; int8_t hh[K_HH];
  for(int i=0;i<H;++i){h[i]=0;c[i]=0.f;}
  for(int t=0;t<T;++t){ int tt=reverse?(T-1-t):t; int ws=reverse?tt:(tt+1);
    for(int k=0;k<K_HH;++k){ int a=0; for(int cc=0;cc<H;++cc) a+=(int)h[cc]*(int)w_dn[k*H+cc];
      hh[k]=clamp_i8(rintf((float)a*comb[k])); }
    const int8_t* xr=x+((long)tt*N+r)*R;
    for(int oc=0;oc<H;++oc){ int g[4];
      for(int gg=0;gg<4;++gg){ const int8_t* w=Bp[gg]+(long)oc*KC; int a=0;
        for(int kc=0;kc<K_HH;++kc) a+=(int)hh[kc]*(int)w[kc];
        for(int kc=0;kc<R;++kc) a+=(int)xr[kc]*(int)w[K_HH+kc]; g[gg]=a; }
      float cv=c[oc];
      int8_t hn=epi_elem(g[0],g[1],g[2],g[3],wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv); c[oc]=cv;
      hh_all[((long)ws*N+r)*H+oc]=hn; }
    for(int oc=0;oc<H;++oc) h[oc]=hh_all[((long)ws*N+r)*H+oc];
  }
  delete[] h; delete[] c;
}

int main(int argc,char**argv){
  int N=256,T=64,reverse=0,bench=0,n_cmp=32,dev=0;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T"))T=atoi(argv[++i]); else if(!strcmp(argv[i],"--reverse"))reverse=1;
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--ncmp"))n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(N%BM) N=((N+BM-1)/BM)*BM;
  size_t smem_g=(size_t)BM*KCS;                      // gate: padded A only (~34KB, weights via __ldg)
  size_t smem_d=(size_t)BM*H;                        // down: 128KB
  size_t smem=smem_g>smem_d?smem_g:smem_d;
  printf("koi_flstm: BM=%d grid=(9,%d)=%d CTAs 1CTA/SM smem=%.0fKB (gate %.0f down %.0f) N=%d T=%d rev=%d\n",
    BM,N/BM,9*(N/BM),smem/1024.0,smem_g/1024.0,smem_d/1024.0,N,T,reverse);

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
  std::vector<int8_t> hrp((size_t)4*H*KC);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc)for(int kc=0;kc<KC;++kc){
    int kt=kc>>5,rem=kc&31,half=rem>>4,r2=rem&15,tgi=r2>>2,b=r2&3;
    hrp[((size_t)g*H+oc)*KC + tgi*64+kt*8+half*4+b]=hB[g][(size_t)oc*KC+kc]; }

  int8_t *dx,*dwdn,*dB[4],*drp,*dring,*dsc; float *dcm,*dws,*dbs,*dcell; int *dfd,*dfg;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size())); CK(cudaMalloc(&drp,hrp.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dsc,(size_t)2*N*K_HH)); CK(cudaMalloc(&dcell,(size_t)N*H*4));
  CK(cudaMalloc(&dfd,(N/BM+1)*4)); CK(cudaMalloc(&dfg,(N/BM+1)*4));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(koi_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  { size_t l2=0; cudaDeviceGetAttribute((int*)&l2,cudaDevAttrMaxPersistingL2CacheSize,dev);
    size_t pin=hwdn.size(); if(pin>l2)pin=l2; cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,pin);
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dwdn; av.accessPolicyWindow.num_bytes=pin;
    av.accessPolicyWindow.hitRatio=1.0f; av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&av); }

  auto run=[&](int rev){
    CK(cudaMemset(dcell,0,(size_t)N*H*4)); CK(cudaMemset(dfd,0,(N/BM+1)*4)); CK(cudaMemset(dfg,0,(N/BM+1)*4));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H));
    koi_kernel<<<dim3(9,N/BM),NTHREADS,smem>>>(dwdn,dcm,drp,dws,dbs,dcell,dring,dx,dsc,dfd,dfg,N,T,rev);
  };

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    int8_t* dref; CK(cudaMalloc(&dref,ring_bytes));
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    run(reverse); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    ref_kernel<<<(n_cmp+31)/32,32>>>(dx,dwdn,dcm,dB[0],dB[1],dB[2],dB[3],dws,dbs,dref,N,T,reverse,n_cmp);
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
  for(int w=0;w<3;++w) run(reverse); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<7;++r){ cudaEventRecord(e0); run(reverse); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); us.push_back((double)ms/T*1000.0); }
  std::sort(us.begin(),us.end());
  printf("koi_flstm step: %.2f us/step  = %.2f ns/row @ N=%d  (koi ~9.0, fused 18.1)\n",us[3],us[3]*1000.0/N,N);
  return 0;
}
