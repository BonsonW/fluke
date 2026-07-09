// FINAL combination: fused_cutlass's SHARED down-proj + koi's SMEM-RESIDENT gate weights.
// Fixes round-25's flaw (redundant per-channel down-proj = 1GB/step) by computing the
// down-proj ONCE (M-split, each CTA its own rows, shared hh_down buffer), and adds the
// resident-weights lever ONLY to the gate.
//
// ONE persistent kernel, grid (GX, GY=N/BMG), 1 CTA/SM, block 256 (8 warps).
// CTA(gx,gy) owns gy's BMG rows and gx's HX=H/GX hidden channels.
//  prologue ONCE: load gx's gate_w slice (4*HX*Kc int8, repacked) -> smem.  Never re-read.
//  per step:
//   phase1 down-proj: this CTA does rows [gy*BMG + gx*(BMG/GX) ..] (BMG/GX rows), full
//     K=H, W_dn streamed L2 -> hh_down -> shared global buffer.  NON-redundant.
//   barrier.
//   phase2 gate: M-split over gy's BMG rows (8 warps x 16), read shared hh_down + x ->
//     A; ldmatrix RESIDENT gate_w from smem (0 LDG for weights); IMMA.16832; fused
//     in-register epilogue (sighard/tanh, f32 cell, int8 quant); STG h to ring.
//   barrier.
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math -diag-suppress 177 resident_gate_flstm.cu -o resident_gate_flstm
//
// RESULT (A100, N=2048, T=2048): 121 us/step (GX=8/BMG=128; GX=16/BMG=256: 125us).
// BIT-EXACT fwd+reverse. 148 regs, 0 spill, 1 CTA/SM.
// This is the requested final combination: SHARED (non-redundant) down-proj + koi's
// SMEM-RESIDENT gate weights.  It CORRECTLY fixes round-25's flaw -- ncu L2 dropped
// 1.03GB -> 395MB/step (the redundant down-proj is gone), and the gate weights are
// confirmed 0-LDG (resident, not in the traffic).  But it is still 3.3x slower than
// fused_cutlass (37us) and does NOT beat it.
//
// WHY (ncu GX=8): occupancy 12.5% (1 CTA/SM, forced by the 128KB resident gate slice),
// imma-util 2.6%, long_scoreboard 6.8 dominant.  The resident-weights lever works but is
// self-defeating: holding 64-128KB of weights resident => 1 CTA/SM => 12.5% occupancy,
// and at 8 warps/SM the per-step GLOBAL-STATE latency (recurrent h/hh_down/x/cell all
// round-trip the L2 ring every step: read h for down-proj, write+read hh_down, r/w cell,
// write h) is NOT hidden.  fused_cutlass wins the opposite way: MANY small CTAs (high
// machine occupancy) + CUTLASS's multistage pipeline hide latency, paying per-step gate-
// weight re-reads that L2 serves cheaply -- cheaper than losing 8x occupancy.
//
// The irreducible gap to koi: koi gets 0-LDG resident weights AND high utilization by
// keeping the recurrent STATE resident (not round-tripping the ring), which does not fit
// at H=1024 alongside a per-channel weight slice in 163KB smem.  Confirmed unresolvable
// at these dims across smem_resident_flstm.cu + this file.
//
// CONCLUSION of the resident-weights line of attack (rounds 25-27): the lever is real
// and now cleanly isolated, but on A100 at H=1024 it cannot beat the streamed-weights +
// high-occupancy + pipelined design.  SHIPPABLE WINNER remains fused_cutlass.cu 37.08us.
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
#define WARPS 8
#define NTHREADS 256
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef GX
#define GX 8
#endif
#ifndef BMG
#define BMG 128        // gate rows per CTA (M-split: 8 warps x 16)
#endif
#define HX (H/GX)      // hidden channels per CTA (128 @ GX=8)
#define RDP (BMG/GX)   // down-proj rows this CTA computes in phase1 (16 @ GX8/BMG128)

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

__global__ void __launch_bounds__(NTHREADS,1) rgate_kernel(
    const int8_t* __restrict__ w_dn, const float* __restrict__ comb,
    const int8_t* __restrict__ wg_rp, const float* __restrict__ wscale,
    const float* __restrict__ bias, float* __restrict__ cell,
    int8_t* __restrict__ hh_all, const int8_t* __restrict__ x,
    int8_t* __restrict__ hhbuf, int* __restrict__ flags, int N, int T, int reverse)
{
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int gx=blockIdx.x, gy=blockIdx.y;
  const int row0=gy*BMG, hbase=gx*HX;
  const float AS=1.0f/127.0f;

  extern __shared__ int8_t smem[];
  int8_t* sWg = smem;                 // [4*HX*Kc] RESIDENT gate weights (repacked)
  int8_t* sA  = sWg + 4*HX*KC;        // [BMG, Kc]  phase2 A (aliased by phase1 h tile)
  int8_t* sH  = sA;                   // phase1 h tile [RDP, H] aliases sA (RDP*H <= BMG*KC)

  // prologue ONCE: gate weights slice -> smem
  for(int i=tid;i<4*HX*KC;i+=NTHREADS){
    int g=i/(HX*KC), rem=i%(HX*KC), col=rem/KC, kc=rem%KC;
    sWg[i]=wg_rp[((long)g*H + hbase+col)*KC + kc];
  }
  __syncthreads();
  int* flag = flags + gy;
  const int nkb_dn = H/32;

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int prev = reverse ? (tt+1) : tt;
    int out  = reverse ? tt : (tt+1);
    const int8_t* hprev = hh_all + (size_t)prev*N*H;
    int8_t* hout = hh_all + (size_t)out*N*H;
    const int8_t* xt = x + (size_t)tt*N*R;

    // ---- phase1: down-proj this CTA's OWN RDP rows (non-redundant) -> hhbuf ----
    int dprow0 = row0 + gx*RDP;                    // rows this CTA down-projects
    for(int j=tid;j<RDP*(H/16);j+=NTHREADS){
      int r=j/(H/16), c=j%(H/16);
      ((int4*)sH)[r*(H/16)+c] = *(const int4*)&hprev[(long)(dprow0+r)*H + c*16];
    }
    __syncthreads();
    for(int rt=0; rt<RDP/16; ++rt)
      for(int nt=0; nt<K_HH/8; ++nt){        // all K_hh output n-tiles, this CTA's rows
        // 8 warps split the K_hh n-tiles: warp does nt where nt%8==warp
        if((nt & 7) != warp) continue;
        int c0=0,c1=0,c2=0,c3=0;
        for(int kt=0;kt<nkb_dn;++kt){
          int rb=rt*16, co=kt*32+tg*4;
          int a0=*(const int*)&sH[(rb+gid)*H+co], a1=*(const int*)&sH[(rb+gid+8)*H+co];
          int a2=*(const int*)&sH[(rb+gid)*H+co+16], a3=*(const int*)&sH[(rb+gid+8)*H+co+16];
          int nrow=nt*8+gid;
          int b0=*(const int*)&w_dn[nrow*H+kt*32+tg*4], b1=*(const int*)&w_dn[nrow*H+kt*32+16+tg*4];
          mma_m16n8k32(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        }
        int col0=nt*8+2*tg, col1=col0+1;
        int r_lo=dprow0+rt*16+gid, r_hi=r_lo+8;
        hhbuf[(long)r_lo*K_HH+col0]=clamp_i8(rintf((float)c0*comb[col0]));
        hhbuf[(long)r_lo*K_HH+col1]=clamp_i8(rintf((float)c1*comb[col1]));
        hhbuf[(long)r_hi*K_HH+col0]=clamp_i8(rintf((float)c2*comb[col0]));
        hhbuf[(long)r_hi*K_HH+col1]=clamp_i8(rintf((float)c3*comb[col1]));
      }
    // ---- barrier 1 (all rows' hh_down written) ----
    __threadfence(); __syncthreads();
    if(tid==0){ atomicAdd(flag,1); int need=GX*(2*t+1); while(atomicAdd(flag,0)<need){} }
    __syncthreads();

    // ---- phase2: build A[BMG,Kc] = [shared hh_down | x] for gy's BMG rows ----
    for(int j=tid;j<BMG*(K_HH/16);j+=NTHREADS){
      int r=j/(K_HH/16), c=j%(K_HH/16);
      ((int4*)sA)[r*(KC/16)+c] = *(const int4*)&hhbuf[(long)(row0+r)*K_HH + c*16];
    }
    for(int j=tid;j<BMG*(R/16);j+=NTHREADS){
      int r=j/(R/16), c=j%(R/16);
      ((int4*)sA)[r*(KC/16)+(K_HH/16)+c] = *(const int4*)&xt[(long)(row0+r)*R + c*16];
    }
    __syncthreads();

    // ---- gate: M-split (warp = 16 rows), resident weights, all HX channels ----
    for(int rt=0; rt<BMG/16; ++rt){
      if((rt & 7) != warp) continue;              // warp owns row-tile rt where rt%8==warp
      int A[8][4];
      for(int kt=0;kt<8;++kt){ int rb=rt*16, co=kt*32+tg*4;
        A[kt][0]=*(const int*)&sA[(rb+gid)*KC+co];   A[kt][1]=*(const int*)&sA[(rb+gid+8)*KC+co];
        A[kt][2]=*(const int*)&sA[(rb+gid)*KC+co+16];A[kt][3]=*(const int*)&sA[(rb+gid+8)*KC+co+16]; }
      for(int nn=0;nn<HX/8;++nn){
        int cg[4][4];
        for(int g=0;g<4;++g) for(int e=0;e<4;++e) cg[g][e]=0;
        for(int g=0;g<4;++g){
          const int8_t* wgs = sWg + g*HX*KC + (nn*8+gid)*KC + tg*64;   // resident, repacked
          int4 w0=*(const int4*)(wgs), w1=*(const int4*)(wgs+16), w2=*(const int4*)(wgs+32), w3=*(const int4*)(wgs+48);
          int b0[8]={w0.x,w0.z,w1.x,w1.z,w2.x,w2.z,w3.x,w3.z};
          int b1[8]={w0.y,w0.w,w1.y,w1.w,w2.y,w2.w,w3.y,w3.w};
          for(int kt=0;kt<8;++kt)
            mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],A[kt][0],A[kt][1],A[kt][2],A[kt][3],b0[kt],b1[kt]);
        }
        for(int e=0;e<4;++e){
          int lcol=nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
          int oc=hbase+lcol, gr=row0+rt*16+rin;
          float cv=cell[(long)gr*H+oc];
          int8_t hn=epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
              wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
              bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv);
          cell[(long)gr*H+oc]=cv;
          hout[(long)gr*H+oc]=hn;
        }
      }
    }
    // ---- barrier 2 ----
    __threadfence(); __syncthreads();
    if(tid==0){ atomicAdd(flag,1); int need=GX*(2*t+2); while(atomicAdd(flag,0)<need){} }
    __syncthreads();
  }
}

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
  if(N%BMG) N=((N+BMG-1)/BMG)*BMG;
  size_t smem=(size_t)4*HX*KC + (size_t)BMG*KC;   // scales read from gmem to save smem
  printf("RESIDENT-GATE: GX=%d HX=%d BMG=%d RDP=%d N=%d T=%d rev=%d grid=(%d,%d) smem=%.0fKB\n",
    GX,HX,BMG,RDP,N,T,reverse,GX,N/BMG,smem/1024.0);

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

  int8_t *dx,*dwdn,*dB[4],*drp,*dring,*dhh; float *dcm,*dws,*dbs,*dcell; int* dflags;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size())); CK(cudaMalloc(&drp,hrp.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dhh,(size_t)N*K_HH)); CK(cudaMalloc(&dcell,(size_t)N*H*4)); CK(cudaMalloc(&dflags,(N/BMG+1)*4));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(rgate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  { size_t l2=0; cudaDeviceGetAttribute((int*)&l2,cudaDevAttrMaxPersistingL2CacheSize,dev);
    size_t pin=hwdn.size(); if(pin>l2)pin=l2; cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,pin);
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dwdn; av.accessPolicyWindow.num_bytes=pin;
    av.accessPolicyWindow.hitRatio=1.0f; av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&av); }

  auto run=[&](int rev){
    CK(cudaMemset(dcell,0,(size_t)N*H*4)); CK(cudaMemset(dflags,0,(N/BMG+1)*4));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H));
    rgate_kernel<<<dim3(GX,N/BMG),NTHREADS,smem>>>(dwdn,dcm,drp,dws,dbs,dcell,dring,dx,dhh,dflags,N,T,rev);
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
  printf("RESIDENT-GATE step: %.2f us/step (median of 7)  = %.2f ns/row @ N=%d  (koi ~9.0)\n",
    us[3],us[3]*1000.0/N,N);
  return 0;
}
