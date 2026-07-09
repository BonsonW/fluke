// SMEM-RESIDENT-WEIGHTS persistent factored-LSTM = a transcription of koi's
// factorised_lstm SASS (dorado v2.0.0, sm80): int8 IMMA.16832, persistent T-loop,
// gate weights loaded to smem ONCE and ldmatrix'd every step (0 LDG/step for weights),
// per-step cp.async of the activation, STG.STRONG.GPU h to the ring, intra-block
// BAR.SYNC only + a per-gy global-atomic barrier for h ordering.
//
// Grid (GX, GY): GX = output-channel split (each CTA owns HX=H/GX hidden channels,
// i/f/g/o), GY = batch/bM.  Each CTA self-contained across T except the h it reads back
// from the ring (written by all GX siblings last step; the barrier orders it).
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177 \
//        smem_resident_flstm.cu -o smem_resident_flstm
//
// RESULT (A100, N=2048, T=2048): 264us/step @ GX=8/BM=16 (GX=16: 439us). BIT-EXACT
// fwd+reverse. 116 regs, 0 spill, 1 CTA/SM.  The resident-gate-weights lever WORKS
// (ncu: gate weights are NOT in the traffic) BUT the kernel is dominated by the
// REDUNDANT DOWN-PROJ: to gate its channel slice each CTA needs the full-H hh_down for
// its rows, so every one of the GX channel-CTAs re-reads full h[BM,H] + streams W_dn
// each step -> ncu L2 = 1.03 GB/step, imma-util 1.8%.  This is the same redundant-
// down-proj wall the earlier `--dorado` pivot hit (1.07GB).  BM must be tiny (16) so
// the 128KB resident gate slice + full-h stage fit 163KB smem -> 1024 CTAs, multi-wave,
// weights under-amortized.
//
// WHY IT LOSES TO fused_cutlass (37us): fused_cutlass does the down-proj ONCE (shared
// hh_down), not GX-redundantly.  koi avoids BOTH walls at once (0 LDG/step, 153KB) by a
// scheme this build could not fit at H=1024: its 117KB gate slice leaves no room to also
// keep W_dn(128KB) resident or stage full h, so koi must keep the recurrent h resident
// in smem across T (not round-trip the ring) with a row/channel ownership that needs no
// cross-CTA barrier -- an ownership that does not close at H=1024 within 163KB smem
// alongside a per-channel gate slice.  The resident-weights lever is real but is
// necessary-not-sufficient; it must be combined with a SHARED (non-redundant) down-proj,
// which is exactly what fused_cutlass.cu already does via its separate down-proj.
// SHIPPABLE WINNER remains fused_cutlass.cu: 37.08us/step (18.1 ns/row; koi ~9.0).
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
#ifndef BM
#define BM 16          // rows per CTA (N-split: 8 warps split the HX channels)
#endif
#define HX (H/GX)      // hidden channels per CTA (128 @ GX=8)
#define CPW (HX/WARPS) // channels per warp (16)
#define NTW (CPW/8)    // gate n8-tiles per warp (2)

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

// smem: sWg (RESIDENT gate weights, loaded once) | sH (h tile, per step) | sA (hh|x) | sWs/sBs
// sWg repacked [gate][col][tg*64+kt*8+half*4+b] for int4 loads (as fused_cutlass).
__global__ void __launch_bounds__(NTHREADS,1) srlstm_kernel(
    const int8_t* __restrict__ w_dn, const float* __restrict__ comb,
    const int8_t* __restrict__ wg_rp, const float* __restrict__ wscale,
    const float* __restrict__ bias, float* __restrict__ cell,
    int8_t* __restrict__ hh_all, const int8_t* __restrict__ x,
    int* __restrict__ flags, int N, int T, int reverse)
{
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int gx=blockIdx.x, gy=blockIdx.y;
  const int row0=gy*BM, hbase=gx*HX;

  extern __shared__ int8_t smem[];
  int8_t* sWg = smem;                    // [4*HX*Kc] resident gate weights (128KB @GX8)
  int8_t* sH  = sWg + 4*HX*KC;           // [BM, H] h tile for down-proj (16KB @BM16)
  int8_t* sA  = sH + BM*H;               // [BM, Kc] hh_down|x
  float*  sWs = (float*)(sA + BM*KC);    // [4, HX]
  float*  sBs = sWs + 4*HX;
  const float AS=1.0f/127.0f;

  // ---- prologue (ONCE): gate weights + scales/bias -> smem. NEVER re-read after this. ----
  for(int i=tid;i<4*HX*KC;i+=NTHREADS){
    int g=i/(HX*KC), rem=i%(HX*KC), col=rem/KC, kc=rem%KC;
    // source wg_rp is [4,H,Kc] repacked; take this CTA's HX cols
    sWg[i]=wg_rp[((long)g*H + hbase+col)*KC + kc];
  }
  for(int i=tid;i<4*HX;i+=NTHREADS){ int g=i/HX,c=i%HX; sWs[i]=wscale[g*H+hbase+c]; sBs[i]=bias[g*H+hbase+c]; }
  __syncthreads();

  int* flag = flags + gy;
  const int nkb_dn = H/32;               // down-proj k-tiles (32)

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int prev = reverse ? (tt+1) : tt;
    int out  = reverse ? tt : (tt+1);
    const int8_t* hprev = hh_all + (size_t)prev*N*H;
    int8_t* hout = hh_all + (size_t)out*N*H;
    const int8_t* xt = x + (size_t)tt*N*R;

    // ---- load full h[BM,H] (this CTA's rows, ALL channels) from ring via cp.async ----
    for(int j=tid;j<BM*(H/16);j+=NTHREADS){
      int r=j/(H/16), c=j%(H/16);
      ((int4*)sH)[r*(H/16)+c] = *(const int4*)&hprev[(long)(row0+r)*H + c*16];
    }
    __syncthreads();

    // ---- phase 1: down-proj hh_down[BM,K_hh] = sH @ W_dn (W_dn from L2-pinned).
    // 8 warps split K_hh=128 output -> 2 n8-tiles/warp; each does all BM rows (BM/16 tiles). ----
    for(int rt=0; rt<BM/16; ++rt)
      for(int loc=0; loc<(K_HH/8)/WARPS; ++loc){
        int ndp = warp*((K_HH/8)/WARPS) + loc;
        int c0=0,c1=0,c2=0,c3=0;
        for(int kt=0;kt<nkb_dn;++kt){
          int rb=rt*16, co=kt*32+tg*4;
          int a0=*(const int*)&sH[(rb+gid)*H+co], a1=*(const int*)&sH[(rb+gid+8)*H+co];
          int a2=*(const int*)&sH[(rb+gid)*H+co+16], a3=*(const int*)&sH[(rb+gid+8)*H+co+16];
          int nrow=ndp*8+gid;
          int b0=*(const int*)&w_dn[nrow*H+kt*32+tg*4], b1=*(const int*)&w_dn[nrow*H+kt*32+16+tg*4];
          mma_m16n8k32(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
        }
        int col0=ndp*8+2*tg, col1=col0+1, rlo=rt*16+gid, rhi=rlo+8;
        sA[rlo*KC+col0]=clamp_i8(rintf((float)c0*comb[col0]));
        sA[rlo*KC+col1]=clamp_i8(rintf((float)c1*comb[col1]));
        sA[rhi*KC+col0]=clamp_i8(rintf((float)c2*comb[col0]));
        sA[rhi*KC+col1]=clamp_i8(rintf((float)c3*comb[col1]));
      }
    // x half -> sA
    for(int r=0;r<BM;++r)
      *(int*)&sA[r*KC+K_HH+lane*4] = *(const int*)&xt[(long)(row0+r)*R+lane*4];
    __syncthreads();

    // ---- phase 2: gate (N-split: warp owns CPW channels, resident weights ldmatrix'd) ----
    for(int rt=0; rt<BM/16; ++rt){
      int A[8][4];
      for(int kt=0;kt<8;++kt){ int rb=rt*16, co=kt*32+tg*4;
        A[kt][0]=*(const int*)&sA[(rb+gid)*KC+co];   A[kt][1]=*(const int*)&sA[(rb+gid+8)*KC+co];
        A[kt][2]=*(const int*)&sA[(rb+gid)*KC+co+16];A[kt][3]=*(const int*)&sA[(rb+gid+8)*KC+co+16]; }
      for(int nn=0;nn<NTW;++nn){
        int cg[4][4];
        for(int g=0;g<4;++g) for(int e=0;e<4;++e) cg[g][e]=0;
        for(int g=0;g<4;++g){
          // resident gate weights from smem (repacked): 8 ktiles' (b0,b1) = 64 contiguous B
          const int8_t* wgs = sWg + g*HX*KC + (warp*CPW+nn*8+gid)*KC + tg*64;
          int4 w0=*(const int4*)(wgs), w1=*(const int4*)(wgs+16), w2=*(const int4*)(wgs+32), w3=*(const int4*)(wgs+48);
          int b0[8]={w0.x,w0.z,w1.x,w1.z,w2.x,w2.z,w3.x,w3.z};
          int b1[8]={w0.y,w0.w,w1.y,w1.w,w2.y,w2.w,w3.y,w3.w};
          for(int kt=0;kt<8;++kt)
            mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],A[kt][0],A[kt][1],A[kt][2],A[kt][3],b0[kt],b1[kt]);
        }
        for(int e=0;e<4;++e){
          int lcol=warp*CPW+nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
          int oc=hbase+lcol, gr=row0+rt*16+rin;
          float cv=cell[(long)gr*H+oc];
          int8_t hn=epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
              sWs[0*HX+lcol],sWs[1*HX+lcol],sWs[2*HX+lcol],sWs[3*HX+lcol],
              sBs[0*HX+lcol],sBs[1*HX+lcol],sBs[2*HX+lcol],sBs[3*HX+lcol],AS,cv);
          cell[(long)gr*H+oc]=cv;
          hout[(long)gr*H+oc]=hn;
        }
      }
    }
    // ---- per-gy barrier: all GX siblings wrote slot `out` before next step reads it ----
    __threadfence();
    __syncthreads();
    if(tid==0){ atomicAdd(flag,1); int need=GX*(t+1); while(atomicAdd(flag,0)<need){} }
    __syncthreads();
  }
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
  size_t smem=(size_t)4*HX*KC + (size_t)BM*H + (size_t)BM*KC + (size_t)2*4*HX*4;
  printf("SMEM-RESIDENT: GX=%d HX=%d BM=%d N=%d T=%d rev=%d grid=(%d,%d) smem=%zu (%.0fKB)\n",
    GX,HX,BM,N,T,reverse,GX,N/BM,smem,smem/1024.0);

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
  // repacked gate weights [4,H,Kc] -> [tg*64+kt*8+half*4+b] per (g,oc)
  std::vector<int8_t> hrp((size_t)4*H*KC);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc)for(int kc=0;kc<KC;++kc){
    int kt=kc>>5,rem=kc&31,half=rem>>4,r2=rem&15,tgi=r2>>2,b=r2&3;
    hrp[((size_t)g*H+oc)*KC + tgi*64+kt*8+half*4+b]=hB[g][(size_t)oc*KC+kc]; }

  int8_t *dx,*dwdn,*dB[4],*drp,*dring; float *dcm,*dws,*dbs,*dcell; int* dflags;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size())); CK(cudaMalloc(&drp,hrp.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dcell,(size_t)N*H*4)); CK(cudaMalloc(&dflags,(N/BM+1)*4));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(srlstm_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));
  // L2-pin W_dn (streamed each step for the down-proj)
  { size_t l2=0; cudaDeviceGetAttribute((int*)&l2,cudaDevAttrMaxPersistingL2CacheSize,dev);
    size_t pin=hwdn.size(); if(pin>l2)pin=l2; cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,pin);
    cudaStreamAttrValue av={}; av.accessPolicyWindow.base_ptr=dwdn; av.accessPolicyWindow.num_bytes=pin;
    av.accessPolicyWindow.hitRatio=1.0f; av.accessPolicyWindow.hitProp=cudaAccessPropertyPersisting;
    av.accessPolicyWindow.missProp=cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0,cudaStreamAttributeAccessPolicyWindow,&av); }

  auto run=[&](int rev){
    CK(cudaMemset(dcell,0,(size_t)N*H*4)); CK(cudaMemset(dflags,0,(N/BM+1)*4));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H));
    srlstm_kernel<<<dim3(GX,N/BM),NTHREADS,smem>>>(dwdn,dcm,drp,dws,dbs,dcell,dring,dx,dflags,N,T,rev);
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
  printf("SMEM-RESIDENT step: %.2f us/step (median of 7, T=%d)  = %.2f ns/row @ N=%d\n",
    us[3],T,us[3]*1000.0/N,N);
  return 0;
}
