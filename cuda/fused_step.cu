// NON-persistent, high-occupancy, per-step FUSED gate+epilogue factored-LSTM kernel.
//
// Audit finding: fluke's step kernel ALREADY fuses gate+epilogue (writes int8 h + f32
// cell, not the 32MB int32 C) but is latency-bound at ~50us (bN=32, 4-acc reg pressure,
// low occupancy).  cuBLAS's C-materializing gate GEMM is 32us (memory-bound on the 32MB
// int32 C write).  This kernel keeps the 4 int32 gate accumulators IN REGISTERS, does the
// LSTM cell update in-register, writes ONLY int8 h + cell -> C never materialized -> the
// redundant weight/A re-reads stay L2-resident -> should be ~compute/L2-bound (~10-20us).
//
// Structure: standard 2D tiling (batch M=N x hidden-channel H), MANY CTAs -> >=2 CTA/SM.
// CTA(cb,hb) owns FBM batch rows x FBN hidden channels; 8 warps N-split the FBN channels.
// Reduces over Kc=256 FULLY LOCAL (no split, no all-reduce).  Weights streamed from L2.
// down-proj is a cheap separate launch (hh_down[N,K_hh] int8) feeding the gate input.
//
// Build: nvcc -arch=sm_80 -O3 --use_fast_math fused_step.cu -o fused_step -lcublas? (no)
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
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef FBM
#define FBM 128         // batch rows per CTA (gate)
#endif
#ifndef FBN
#define FBN 64          // hidden channels per CTA (gate)
#endif
#ifndef DBM
#define DBM 32          // batch rows per CTA (down-proj)
#endif
#ifndef GMINB
#define GMINB 2
#endif

__device__ __forceinline__ int8_t clamp_i8(float q){ q=fminf(fmaxf(q,-127.f),127.f); return (int8_t)(int)q; }
__device__ __forceinline__ int8_t epi_elem(int gi,int gf,int gg,int go,
    float si,float sf,float sg,float so,float bi,float bf,float bg,float bo,float as,float&cell){
  float vi=(float)gi*as*si+bi, vf=(float)gf*as*sf+bf, vg=(float)gg*as*sg+bg, vo=(float)go*as*so+bo;
  float I=fminf(fmaxf(vi*0.2f+0.5f,0.f),1.f), F=fminf(fmaxf(vf*0.2f+0.5f,0.f),1.f);
  float O=fminf(fmaxf(vo*0.2f+0.5f,0.f),1.f), G=fminf(fmaxf(vg,-1.f),1.f);
  cell=F*cell+I*G; return clamp_i8(rintf(O*tanhf(cell)*127.0f));
}
__device__ __forceinline__ void mma_m16n8k32(int&c0,int&c1,int&c2,int&c3,
    int a0,int a1,int a2,int a3,int b0,int b1){
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    :"+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

// ---- down-proj: hh_down[N,K_hh] int8 = clamp(round((h@w_dn^T) * comb_mult)) ----
// CTA owns DBM rows, all K_hh cols; 8 warps split K_hh (2 n8-tiles each); K=H local.
__global__ void __launch_bounds__(256,2) downproj_kernel(
    const int8_t* __restrict__ h, const int8_t* __restrict__ w_dn,
    const float* __restrict__ comb_mult, int8_t* __restrict__ hh_down, int N){
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int row0=blockIdx.x*DBM;
  extern __shared__ int8_t sH[];  // [DBM,H]
  #pragma unroll
  for(int j=tid;j<DBM*(H/16);j+=256){ int r=j/(H/16),cc=j%(H/16);
    ((int4*)sH)[r*(H/16)+cc]=*(const int4*)&h[(long)(row0+r)*H+cc*16]; }
  __syncthreads();
  #pragma unroll
  for(int rt=0;rt<DBM/16;++rt)
    #pragma unroll
    for(int loc=0;loc<(K_HH/8)/8;++loc){
      int ndp=warp*((K_HH/8)/8)+loc, c0=0,c1=0,c2=0,c3=0;
      #pragma unroll
      for(int kt=0;kt<H/32;++kt){
        int rb=rt*16, co=kt*32+tg*4;
        int a0=*(const int*)&sH[(rb+gid)*H+co],a1=*(const int*)&sH[(rb+gid+8)*H+co];
        int a2=*(const int*)&sH[(rb+gid)*H+co+16],a3=*(const int*)&sH[(rb+gid+8)*H+co+16];
        int nrow=ndp*8+gid;
        int b0=*(const int*)&w_dn[nrow*H+kt*32+tg*4],b1=*(const int*)&w_dn[nrow*H+kt*32+16+tg*4];
        mma_m16n8k32(c0,c1,c2,c3,a0,a1,a2,a3,b0,b1);
      }
      int col0=ndp*8+2*tg,col1=col0+1,rlo=row0+rt*16+gid,rhi=rlo+8;
      hh_down[(long)rlo*K_HH+col0]=clamp_i8(rintf((float)c0*comb_mult[col0]));
      hh_down[(long)rlo*K_HH+col1]=clamp_i8(rintf((float)c1*comb_mult[col1]));
      hh_down[(long)rhi*K_HH+col0]=clamp_i8(rintf((float)c2*comb_mult[col0]));
      hh_down[(long)rhi*K_HH+col1]=clamp_i8(rintf((float)c3*comb_mult[col1]));
    }
}

// ---- fused gate + LSTM epilogue.  C stays in registers, only int8 h + f32 cell written ----
__global__ void __launch_bounds__(256,GMINB) fusedgate_kernel(
    const int8_t* __restrict__ hh_down, const int8_t* __restrict__ x_t,
    const int8_t* __restrict__ Bi, const int8_t* __restrict__ Bf,
    const int8_t* __restrict__ Bg, const int8_t* __restrict__ Bo,
    const float* __restrict__ wscale, const float* __restrict__ bias,
    float* __restrict__ cell, int8_t* __restrict__ hout, int N){
  constexpr int CPW=FBN/8, NTW=CPW/8, RT=FBM/16;
  const float AS=1.0f/127.0f;
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int row0=blockIdx.x*FBM, ch0=blockIdx.y*FBN;
  extern __shared__ int8_t smem[];
  int8_t* sA = smem;                 // [FBM, Kc]
  float* sWs = (float*)(sA + FBM*KC);// [4, FBN]
  float* sBs = sWs + 4*FBN;          // [4, FBN]
  for(int i=tid;i<4*FBN;i+=256){ int g=i/FBN,c=i%FBN; sWs[i]=wscale[g*H+ch0+c]; sBs[i]=bias[g*H+ch0+c]; }
  // build A = [hh_down | x]  for this row tile
  #pragma unroll
  for(int j=tid;j<FBM*(K_HH/16);j+=256){ int r=j/(K_HH/16),cc=j%(K_HH/16);
    ((int4*)sA)[r*(KC/16)+cc]=*(const int4*)&hh_down[(long)(row0+r)*K_HH+cc*16]; }
  #pragma unroll
  for(int j=tid;j<FBM*(R/16);j+=256){ int r=j/(R/16),cc=j%(R/16);
    ((int4*)sA)[r*(KC/16)+(K_HH/16)+cc]=*(const int4*)&x_t[(long)(row0+r)*R+cc*16]; }
  __syncthreads();
  const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  // hoist gate weights into registers ONCE per CTA (reused across all RT row-tiles ->
  // no per-row-tile weight re-read; loaded per launch, not resident across T).
  int rB[4][NTW][8][2];
  #pragma unroll
  for(int g=0;g<4;++g)
    #pragma unroll
    for(int nn=0;nn<NTW;++nn){
      int oc=ch0+warp*CPW+nn*8+gid;
      #pragma unroll
      for(int kt=0;kt<8;++kt){ rB[g][nn][kt][0]=*(const int*)&Bp[g][(long)oc*KC+kt*32+tg*4];
                               rB[g][nn][kt][1]=*(const int*)&Bp[g][(long)oc*KC+kt*32+16+tg*4]; }
    }
  #pragma unroll
  for(int rt=0;rt<RT;++rt){
    int A[8][4];
    #pragma unroll
    for(int kt=0;kt<8;++kt){ int rb=rt*16, co=kt*32+tg*4;
      A[kt][0]=*(const int*)&sA[(rb+gid)*KC+co];   A[kt][1]=*(const int*)&sA[(rb+gid+8)*KC+co];
      A[kt][2]=*(const int*)&sA[(rb+gid)*KC+co+16];A[kt][3]=*(const int*)&sA[(rb+gid+8)*KC+co+16]; }
    #pragma unroll
    for(int nn=0;nn<NTW;++nn){
      int cg[4][4];
      #pragma unroll
      for(int g=0;g<4;++g)
        #pragma unroll
        for(int e=0;e<4;++e) cg[g][e]=0;
      #pragma unroll
      for(int kt=0;kt<8;++kt)
        #pragma unroll
        for(int g=0;g<4;++g)
          mma_m16n8k32(cg[g][0],cg[g][1],cg[g][2],cg[g][3],A[kt][0],A[kt][1],A[kt][2],A[kt][3],
                       rB[g][nn][kt][0],rB[g][nn][kt][1]);
      #pragma unroll
      for(int e=0;e<4;++e){
        int lcol=warp*CPW+nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
        int oc=ch0+lcol, gr=row0+rt*16+rin;
        float cval=cell[(long)gr*H+oc];
        int8_t hn=epi_elem(cg[0][e],cg[1][e],cg[2][e],cg[3][e],
            sWs[0*FBN+lcol],sWs[1*FBN+lcol],sWs[2*FBN+lcol],sWs[3*FBN+lcol],
            sBs[0*FBN+lcol],sBs[1*FBN+lcol],sBs[2*FBN+lcol],sBs[3*FBN+lcol],AS,cval);
        cell[(long)gr*H+oc]=cval;
        hout[(long)gr*H+oc]=hn;
      }
    }
  }
}

// ---- naive GPU reference (bit-exact via same epi_elem/tanh/round) ----
__global__ void ref_kernel(const int8_t* x,const int8_t* w_dn,const float* comb_mult,
    const int8_t* Bi,const int8_t* Bf,const int8_t* Bg,const int8_t* Bo,
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/127.0f; const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  int8_t* h=new int8_t[H]; float* c=new float[H]; int8_t hh[K_HH];
  for(int i=0;i<H;++i){h[i]=0;c[i]=0.f;}
  for(int t=0;t<T;++t){ int tt=reverse?(T-1-t):t; int ws=reverse?tt:(tt+1);
    for(int k=0;k<K_HH;++k){ int a=0; for(int cc=0;cc<H;++cc) a+=(int)h[cc]*(int)w_dn[k*H+cc];
      hh[k]=clamp_i8(rintf((float)a*comb_mult[k])); }
    const int8_t* xr=x+((long)tt*N+r)*R;
    for(int oc=0;oc<H;++oc){ int g[4];
      for(int gg=0;gg<4;++gg){ const int8_t* w=Bp[gg]+(long)oc*KC; int a=0;
        for(int kc=0;kc<K_HH;++kc) a+=(int)hh[kc]*(int)w[kc];
        for(int kc=0;kc<R;++kc) a+=(int)xr[kc]*(int)w[K_HH+kc]; g[gg]=a; }
      int8_t hn=epi_elem(g[0],g[1],g[2],g[3],wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,c[oc]);
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
  int PAD=FBM>DBM?FBM:DBM; if(N%PAD) N=((N+PAD-1)/PAD)*PAD;
  printf("FBM=%d FBN=%d DBM=%d N=%d T=%d reverse=%d\n",FBM,FBN,DBM,N,T,reverse);

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

  int8_t *dx,*dwdn,*dB[4],*dring,*dhh; float *dcm,*dws,*dbs,*dcell;
  CK(cudaMalloc(&dx,hx.size())); CK(cudaMalloc(&dwdn,hwdn.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dcm,K_HH*4)); CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4));
  size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
  CK(cudaMalloc(&dhh,(size_t)N*K_HH)); CK(cudaMalloc(&dcell,(size_t)N*H*4));
  CK(cudaMemcpy(dx,hx.data(),hx.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dwdn,hwdn.data(),hwdn.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcm,hcm.data(),K_HH*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));

  size_t smem_dp=(size_t)DBM*H, smem_g=(size_t)FBM*KC + (size_t)2*4*FBN*sizeof(float);
  CK(cudaFuncSetAttribute(downproj_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_dp));
  CK(cudaFuncSetAttribute(fusedgate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem_g));

  auto step=[&](int i,cudaStream_t s){ int t=reverse?(T-1-i):i; int prev=reverse?(t+1):t; int out=reverse?t:(t+1);
    int8_t* hprev=dring+(size_t)prev*N*H; int8_t* hout=dring+(size_t)out*N*H;
    const int8_t* xt=dx+(size_t)t*N*R;
    downproj_kernel<<<dim3(N/DBM),256,smem_dp,s>>>(hprev,dwdn,dcm,dhh,N);
    fusedgate_kernel<<<dim3(N/FBM,H/FBN),256,smem_g,s>>>(dhh,xt,dB[0],dB[1],dB[2],dB[3],dws,dbs,dcell,hout,N);
  };
  auto init=[&](int rev){ CK(cudaMemset(dcell,0,(size_t)N*H*4));
    int b=rev?T:0; CK(cudaMemset(dring+(size_t)b*N*H,0,(size_t)N*H)); };

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    int8_t* dref; CK(cudaMalloc(&dref,ring_bytes));
    CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)256<<20));
    init(reverse); for(int i=0;i<T;++i) step(i,0); CK(cudaDeviceSynchronize());
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
  // bench: graph-capture the T-step recurrence, warm, time
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
  printf("FUSED step: %.2f us/step (median of 7, T=%d, graph)  [down-proj + fused-gate]\n",us[3],T);
  // isolated per-kernel timing (warm, 500 independent launches)
  int8_t* hp=dring; int8_t* ho=dring+(size_t)N*H; const int8_t* xt=dx;
  auto timeit=[&](const char*nm, auto fn){ for(int i=0;i<20;++i) fn(); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<500;++i) fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1); printf("  %-10s %.2f us/launch\n",nm,ms/500*1000.0); };
  timeit("downproj", [&](){ downproj_kernel<<<dim3(N/DBM),256,smem_dp>>>(hp,dwdn,dcm,dhh,N); });
  timeit("fusedgate",[&](){ fusedgate_kernel<<<dim3(N/FBM,H/FBN),256,smem_g>>>(dhh,xt,dB[0],dB[1],dB[2],dB[3],dws,dbs,dcell,ho,N); });
  return 0;
}
