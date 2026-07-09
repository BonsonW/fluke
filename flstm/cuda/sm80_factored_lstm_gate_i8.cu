// factored_lstm_gate_i8.cu -- int8 factored-LSTM gate recurrence, persistent single-launch.
// Hand-CUDA reference kernel. 18.0 us/step @ N=1536, T=2048 on A100 (sm80), bit-exact fwd+reverse.
//
// See FLSTM_GATE_GUIDE.md for the ground-up optimization rationale, how to reproduce this on
// other GPUs (including AMD/HIP), the profiling methodology, and the pitfalls to avoid.
//
// WHAT: one persistent kernel loops all T timesteps in-kernel. Per step it computes the 4 LSTM
// gates for a tile of (batch rows x hidden channels) via int8 tensor-core MMA (m16n8k32.s8),
// fuses the LSTM cell update in registers, and writes int8 h to the output ring. The recurrent
// cell stays RESIDENT in registers across all T (never round-trips global memory).
//
// SHAPE: H=1024 hidden, 4 gates. Per step the gate GEMM is [N rows, KC=256] x [4H=4096, KC=256]^T
// (KC = down-projected recurrent input + input contribution, concatenated). Grid (GX, N/BMG):
// GX CTAs split the 4H gate-output columns; N/BMG splits the batch rows.
//
// THE OPTIMIZATIONS (each independently measured; see guide for the why and the numbers):
//   1. int8 activation A -> ldmatrix from a small smem tile (efficient smem->register for MMA).
//   2. gate weights streamed L2->registers via __ldg (read-only / constant-cache path) -- OFF
//      the shared-memory pipe, NOT resident in smem. Each weight fragment reused across MSET
//      row-tiles (register blocking); each A fragment reused across all output channels.
//   3. recurrent cell RESIDENT in registers across T (no gmem round-trip); f16 storage.
//   4. smem A-tile row stride PADDED (KC+16) so the ldmatrix reads are bank-conflict-free.
//   5. per-channel dequant scale folded with the 1/127 activation scale ONCE (out of the loop).
//   6. output h staged to smem then written as coalesced 128-bit stores; the staging stores are
//      packed 16-bit (adjacent columns), never per-byte.
//
// CONFIG (compile-time): GX=16 BMG=256 -> MSET=2, ALDM=1, PIPE=0, CTAPSM=1, CELLREG=1, FP16CELL.
// This is the tuned A100 config; retune GX/BMG per GPU (guide has the procedure).
//
// Build:
//   nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013 \
//     -DGX=16 -DBMG=256 -DALDM=1 -DPIPE=0 -DCTAPSM=1 -DCELLREG=1 -DFP16CELL \
//     factored_lstm_gate_i8.cu -o flstm_gate
// Run:  ./flstm_gate --N 1536 --T 2048 [--reverse]    (correctness vs the built-in scalar ref)
//       ./flstm_gate --N 1536 --T 2048 --bench         (warm timing, median of 9)
// Profiling probes (see guide, "cost decomposition"):
//       add -DCORE     to time the MMA path only (epilogue+writeout+cell stripped)
//       add -DNOWRITE  to time MMA + epilogue + cell, but skip the output writeout
//
// Correctness controls: ALDM=0 uses a scalar-load A path with the proven fragment order (use it
// to validate the ldmatrix-A path); CELLREG=0 keeps the cell in global memory.
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
#define KC 256
#define WARPS 8
#define NTHREADS 256
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef GX
#define GX 16
#endif
#ifndef BMG
#define BMG 256          // rows/CTA; MSET = (BMG/16)/8 row-tiles per warp (weight M-reuse)
#endif
#ifndef ALDM
#define ALDM 1           // 1 = ldmatrix A, 0 = scalar-LDS A (correctness control, proven order)
#endif
#ifndef PIPE
#define PIPE 0           // 1 = double-buffered cp.async A prefetch across the T-loop
#endif
#ifndef CTAPSM
#define CTAPSM 1         // __launch_bounds__ CTAs/SM target
#endif
#ifndef CELLREG
#define CELLREG 1        // 1 = keep recurrent cell resident in registers (no gmem round-trip)
#endif
#define HX (H/GX)
#define MSET ((BMG/16)/WARPS)     // row-tiles per warp
#ifndef APAD
#define APAD 16                   // bytes of smem row padding on the A tile to kill ldmatrix
#endif                            // bank conflicts: stride (KC+APAD)/4 words must be !=0 mod 32
#define KCS (KC+APAD)             // padded smem row stride for A (bytes); gmem A stays KC
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
__device__ __forceinline__ void ldm_x4(int&r0,int&r1,int&r2,int&r3,uint32_t a){
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    :"=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3):"r"(a));
}
__device__ __forceinline__ uint32_t smem_addr(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(void* dst,const void* src){
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n"::"r"(smem_addr(dst)),"l"(src));
}
__device__ __forceinline__ void load_A(int8_t* dstA, const int8_t* A_all, size_t t, int N, int row0){
  const int tid=threadIdx.x, nch=KC/16;
  const int8_t* base = A_all + (size_t)t*N*KC + (size_t)row0*KC;
  for(int i=tid;i<BMG*nch;i+=NTHREADS){ int r=i/nch,c=i%nch; cpa16(&dstA[r*KCS+c*16],&base[(size_t)r*KC+c*16]); }
}

// ---- persistent gate: ldmatrix A (smem) + __ldg weights (L2->reg) + MSET weight-reuse ----
__global__ void __launch_bounds__(NTHREADS,CTAPSM) gate_kernel(
    const int8_t* __restrict__ A_all,   // [T,N,KC]
    const int8_t* __restrict__ wg_rp,   // [4,H,KC] repacked gate weights (read-only, LDG)
    const float* __restrict__ wscale, const float* __restrict__ bias,
    cellT* __restrict__ cell, int8_t* __restrict__ hout,
    int N, int T, int reverse, int store_all)
{
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int gx=blockIdx.x, gy=blockIdx.y;
  const int row0=gy*BMG, hbase=gx*HX;
  const float AS=1.0f/127.0f;
  const int wr0 = warp*MSET*16;         // this warp's first row within the CTA tile

  extern __shared__ int8_t smem[];
  int8_t* sA = smem;                    // [STAGES_A][BMG][KCS] (padded rows)
  float* sSc = (float*)(sA + (size_t)STAGES_A*BMG*KCS);  // [8][HX] resident scales/biases
  #define HXP (HX+16)
  int8_t* sHo = (int8_t*)(sSc + 8*HX);  // [BMG][HXP] padded staging (conflict-free STS)
  // resident scales/biases: constant across all T -> load ONCE (removes ~half the epilogue LDG)
  for(int i=tid;i<8*HX;i+=NTHREADS){ int g=i/HX, c=i%HX;
    sSc[i] = (g<4)? wscale[g*H+hbase+c]*AS : bias[(g-4)*H+hbase+c]; }  // fold as=1/127 into scales ONCE

#if CELLREG
  // recurrent CELL kept RESIDENT in registers across the whole T-loop (each thread owns a
  // fixed set of (row,channel) cells) -> NO cell gmem round-trip (the resident-state lever).
  float creg[HX/8][MSET][4];
  #pragma unroll
  for(int a=0;a<HX/8;++a) for(int b=0;b<MSET;++b) for(int c=0;c<4;++c) creg[a][b][c]=0.f;
#endif
#if defined(CORE) || defined(NOWRITE)
  int core_sink=0;   // profiling probe sink: keep the probed work live. CORE=MMA-only; NOWRITE=+epi/cell
#endif
  auto tstep=[&](int t)->size_t{ int tt=reverse?(T-1-t):t; return (size_t)tt; };
#if PIPE
  load_A(sA + 0, A_all, tstep(0), N, row0);        // prime step 0
  asm volatile("cp.async.commit_group;\n");
#endif

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int out  = reverse ? tt : (tt+1);
    int8_t* hout_t = hout + (store_all ? (size_t)out*N*H : 0);
    int cur = t & (STAGES_A-1);
    int8_t* Acur = sA + (size_t)cur*BMG*KCS;
#if PIPE
    if(t+1<T){ int nxt=(t+1)&(STAGES_A-1); load_A(sA+(size_t)nxt*BMG*KCS,A_all,tstep(t+1),N,row0);
      asm volatile("cp.async.commit_group;\n"); }
    asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES_A-1));
#else
    load_A(sA + 0, A_all, tstep(t), N, row0);        // load THIS step's activation
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
#endif
    __syncthreads();

    // load this warp's A fragments: MSET m-tiles x 8 k-tiles, reused across ALL channels
    int Af[MSET][8][4];
    #pragma unroll
    for(int m=0;m<MSET;++m){
      int mrow0 = wr0 + m*16;
#if ALDM
      #pragma unroll
      for(int kt=0;kt<8;++kt){
        int r=lane&15, koff=(lane>>4)*16;
        uint32_t a=smem_addr(&Acur[(mrow0+r)*KCS + kt*32 + koff]);
        ldm_x4(Af[m][kt][0],Af[m][kt][1],Af[m][kt][2],Af[m][kt][3],a);
      }
#else
      #pragma unroll
      for(int kt=0;kt<8;++kt){ int co=kt*32+tg*4;
        Af[m][kt][0]=*(const int*)&Acur[(mrow0+gid)*KCS+co];    Af[m][kt][1]=*(const int*)&Acur[(mrow0+gid+8)*KCS+co];
        Af[m][kt][2]=*(const int*)&Acur[(mrow0+gid)*KCS+co+16]; Af[m][kt][3]=*(const int*)&Acur[(mrow0+gid+8)*KCS+co+16]; }
#endif
    }

#if CELLREG
    #pragma unroll
#else
    #pragma unroll 1
#endif
    for(int nn=0;nn<HX/8;++nn){
      int cg[MSET][4][4];
      #pragma unroll
      for(int m=0;m<MSET;++m) for(int g=0;g<4;++g) for(int e=0;e<4;++e) cg[m][g][e]=0;
      #pragma unroll
      for(int g=0;g<4;++g){
        const int8_t* wgs = wg_rp + (long)g*H*KC + (long)(hbase+nn*8+gid)*KC + tg*64; // L2->reg
        int4 w0=__ldg((const int4*)(wgs)),  w1=__ldg((const int4*)(wgs+16)),
             w2=__ldg((const int4*)(wgs+32)),w3=__ldg((const int4*)(wgs+48));
        int b0[8]={w0.x,w0.z,w1.x,w1.z,w2.x,w2.z,w3.x,w3.z};
        int b1[8]={w0.y,w0.w,w1.y,w1.w,w2.y,w2.w,w3.y,w3.w};
        #pragma unroll
        for(int m=0;m<MSET;++m)             // weight fragment reused across MSET M-tiles
          #pragma unroll
          for(int kt=0;kt<8;++kt)
            mma_m16n8k32(cg[m][g][0],cg[m][g][1],cg[m][g][2],cg[m][g][3],
                         Af[m][kt][0],Af[m][kt][1],Af[m][kt][2],Af[m][kt][3],b0[kt],b1[kt]);
      }
#ifdef CORE
      // profiling probe: reduce accumulators into a sink to keep MMAs live; skip epilogue+writeout
      #pragma unroll
      for(int m=0;m<MSET;++m) for(int g=0;g<4;++g) for(int e=0;e<4;++e) core_sink += cg[m][g][e];
    }
#else
      #pragma unroll
      for(int m=0;m<MSET;++m){
        int mrow0 = wr0 + m*16;
        int8_t hh[4];
        #pragma unroll
        for(int e=0;e<4;++e){
          int lcol=nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
          int oc=hbase+lcol, gr=row0+mrow0+rin;
#if CELLREG
          float cv=creg[nn][m][e];
#else
          float cv=(float)cell[(long)gr*H+oc];
#endif
          hh[e]=epi_elem(cg[m][0][e],cg[m][1][e],cg[m][2][e],cg[m][3][e],
              sSc[0*HX+lcol],sSc[1*HX+lcol],sSc[2*HX+lcol],sSc[3*HX+lcol],
              sSc[4*HX+lcol],sSc[5*HX+lcol],sSc[6*HX+lcol],sSc[7*HX+lcol],1.0f,cv);  // as folded into sSc
#if CELLREG
          creg[nn][m][e]=(float)(cellT)cv;   // round to cell precision (f16) each step to match ref
#else
          cell[(long)gr*H+oc]=(cellT)cv;
#endif
        }
#ifdef NOWRITE
        core_sink += (int)hh[0]+hh[1]+hh[2]+hh[3];   // sink h: keep epi/cell live, skip writeout
#else
        // PACKED staging: e0,e1 -> row gid (adjacent cols); e2,e3 -> row gid+8. 32 STS.U16 vs 64 STS.U8.
        int col0=nn*8+2*tg;
        *(uint16_t*)&sHo[(size_t)(mrow0+gid  )*HXP + col0] = (uint16_t)((uint8_t)hh[0] | ((uint8_t)hh[1]<<8));
        *(uint16_t*)&sHo[(size_t)(mrow0+gid+8)*HXP + col0] = (uint16_t)((uint8_t)hh[2] | ((uint8_t)hh[3]<<8));
#endif
      }
    }
#ifndef NOWRITE
    __syncthreads();                          // all warps filled sHo
    // COALESCED WRITEOUT: sHo[BMG][HX] -> hout_t, 128-bit STG.128 (was 64x scattered STG.E.U8)
    #pragma unroll
    for(int i=tid;i<BMG*(HX/16);i+=NTHREADS){ int r=i/(HX/16), c=i%(HX/16);
      *(int4*)&hout_t[(long)(row0+r)*H + hbase + c*16] = *(const int4*)&sHo[r*HXP + c*16]; }
    __syncthreads();
#endif
#endif
  }
#if defined(CORE) || defined(NOWRITE)
  hout[tid]=(int8_t)core_sink;   // one write after the T-loop -> forces the probed work to run
#endif
}

__global__ void ref_kernel(const int8_t* A_all,
    const int8_t* Bi,const int8_t* Bf,const int8_t* Bg,const int8_t* Bo,
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/127.0f; const int8_t* Bp[4]={Bi,Bf,Bg,Bo};
  cellT* c=new cellT[H]; for(int i=0;i<H;++i) c[i]=(cellT)0.f;
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
  size_t smem=(size_t)STAGES_A*BMG*KCS + (size_t)8*HX*4 + (size_t)BMG*(HX+16);  // +sHo padded staging
  int ncta=GX*(N/BMG);
  printf("flstm_gate: GX=%d HX=%d BMG=%d MSET=%d ALDM=%d PIPE=%d ctapsm=%d cell=%s N=%d T=%d rev=%d grid=(%d,%d)=%d %s smem=%.1fKB\n",
    GX,HX,BMG,MSET,ALDM,PIPE,CTAPSM,sizeof(cellT)==2?"f16":"f32",N,T,reverse,GX,N/BMG,ncta,ncta<=108?"ONE-WAVE":"multi-wave",smem/1024.0);

  std::mt19937 rng(1234); std::normal_distribution<float> nd(0,1); std::uniform_real_distribution<float> ux(-1,1);
  std::vector<int8_t> hA((size_t)T*N*KC);
  for(size_t i=0;i<hA.size();++i) hA[i]=(int8_t)lrintf(fminf(fmaxf(ux(rng),-1.f),1.f)*127.f);
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

  int8_t *dA,*dB[4],*drp,*dring; float *dws,*dbs; cellT *dcell;
  CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&drp,hrp.size()));
  for(int g=0;g<4;++g) CK(cudaMalloc(&dB[g],hB[g].size()));
  CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4)); CK(cudaMalloc(&dcell,(size_t)N*H*sizeof(cellT)));
  CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  for(int g=0;g<4;++g) CK(cudaMemcpy(dB[g],hB[g].data(),hB[g].size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dws,hws.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbs,hbs.data(),4*H*4,cudaMemcpyHostToDevice));
  CK(cudaFuncSetAttribute(gate_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)smem));

  if(!bench){
    n_cmp=n_cmp<N?n_cmp:N;
    size_t ring_bytes=(size_t)(T+1)*N*H; CK(cudaMalloc(&dring,ring_bytes));
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
  CK(cudaMalloc(&dring,(size_t)N*H)); CK(cudaMemset(dcell,0,(size_t)N*H*sizeof(cellT)));
  auto run=[&](){ gate_kernel<<<dim3(GX,N/BMG),NTHREADS,smem>>>(dA,drp,dws,dbs,dcell,dring,N,T,reverse,0); };
  for(int w=0;w<3;++w) run(); CK(cudaDeviceSynchronize());
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  std::vector<double> us;
  for(int r=0;r<9;++r){ cudaEventRecord(e0); run(); cudaEventRecord(e1);
    cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); us.push_back((double)ms/T*1000.0); }
  std::sort(us.begin(),us.end());
  printf("flstm_gate step: %.2f us/step (median of 9, T=%d)\n",us[4],T);
  return 0;
}
