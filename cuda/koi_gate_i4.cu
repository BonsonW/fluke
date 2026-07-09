// koi_gate_i4.cu -- INT4xINT4 gate: the round-36 leapfrog lever, never built until now.
//
// Port of koi_gate.cu (int8 winner, 25.9us gate @N1536) to int4 x int4 via
// mma.sync.m16n8k64.s4.s4.  koi is 100% int8 (SASS has ZERO .S4) -> int4 is a lever
// koi structurally CANNOT use, and the sensitivity study (slorado sensitivity_results.tsv)
// clears int4 on the gate/up-proj on BOTH operands (~baseline 0.9837 identity).
//
// WHY int4 attacks EVERY term of koi_gate's measured 26us wall at once:
//   * weight LDG bytes HALVED  -> the 74% L1TEX weight-transit wall (koi_gate's residual)
//   * activation smem/ldmatrix HALVED
//   * m16n8k64 does 2x the K per MMA (4 k-tiles vs 8) -> half the IMMA + ldmatrix instrs
//   * 2x int4 TOPS (1248 vs 624)
// The s4 m16n8k64 FRAGMENT SHAPES == s8 m16n8k32 (A=4 regs, B=2 regs, C=4 int32); only the
// PTX op, k-step (32->64), and int4 packing differ -> clean port of the proven structure.
//
// This benchmark uses self-consistent RANDOM int4 data (values in [-7,7]) vs an int4 ref
// kernel: it measures SPEED (the deliverable) + validates the int4 GEMM+epilogue layout.
// True e2e accuracy is de-risked by the sensitivity study and deferred to a slorado run.
//
// Build:
//   nvcc -arch=sm_80 -O3 --use_fast_math -std=c++17 -diag-suppress 177,20013 \
//     -DGX=16 -DBMG=256 -DALDM=1 -DPIPE=0 -DCTAPSM=1 -DCELLREG=1 -DFP16CELL koi_gate_i4.cu -o koi_gate_i4
// Run:  ./koi_gate_i4 [--N 1536 --T 2048 [--reverse] [--bench]]   (no --bench = correctness)
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
#define KBYTES (KC/2)      // int4 packed: 128 bytes/row
#define KT (KC/64)         // 4 k-tiles of 64 (m16n8k64)
#define WARPS 8
#define NTHREADS 256
#define CK(x) do{cudaError_t e=(x); if(e){printf("cuda %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

#ifndef GX
#define GX 16
#endif
#ifndef BMG
#define BMG 256
#endif
#ifndef ALDM
#define ALDM 1
#endif
#ifndef PIPE
#define PIPE 0
#endif
#ifndef CTAPSM
#define CTAPSM 1
#endif
#ifndef CELLREG
#define CELLREG 1
#endif
#define HX (H/GX)
#define MSET ((BMG/16)/WARPS)
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
// m16n8k64 s4: A=4 regs, B=2 regs, C=4 int32 -- identical register shape to m16n8k32 s8.
__device__ __forceinline__ void mma_m16n8k64(int&c0,int&c1,int&c2,int&c3,
    int a0,int a1,int a2,int a3,int b0,int b1){
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
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
// A stored packed int4: [BMG][KBYTES] bytes.
__device__ __forceinline__ void load_A(int8_t* dstA, const int8_t* A_all, size_t t, int N, int row0){
  const int tid=threadIdx.x, nch=KBYTES/16;   // 128/16 = 8 chunks of 16 bytes
  const int8_t* base = A_all + (size_t)t*N*KBYTES + (size_t)row0*KBYTES;
  for(int i=tid;i<BMG*nch;i+=NTHREADS){ int r=i/nch,c=i%nch; cpa16(&dstA[r*KBYTES+c*16],&base[(size_t)r*KBYTES+c*16]); }
}

// ---- persistent int4 gate: ldmatrix A + __ldg int4 weights + MSET reuse + cell-resident ----
__global__ void __launch_bounds__(NTHREADS,CTAPSM) gate_kernel(
    const int8_t* __restrict__ A_all,   // [T,N,KBYTES] packed int4
    const int8_t* __restrict__ wg_rp,   // repacked int4 gate weights [4,H,KBYTES]
    const float* __restrict__ wscale, const float* __restrict__ bias,
    cellT* __restrict__ cell, int8_t* __restrict__ hout,
    int N, int T, int reverse, int store_all)
{
  const int tid=threadIdx.x, lane=tid&31, warp=tid>>5, gid=lane>>2, tg=lane&3;
  const int gx=blockIdx.x, gy=blockIdx.y;
  const int row0=gy*BMG, hbase=gx*HX;
  const float AS=1.0f/7.0f;                 // int4 activation dequant scale
  const int wr0 = warp*MSET*16;

  extern __shared__ int8_t smem[];
  int8_t* sA = smem;                         // [STAGES_A][BMG][KBYTES]
  float* sSc = (float*)(sA + (size_t)STAGES_A*BMG*KBYTES);
  for(int i=tid;i<8*HX;i+=NTHREADS){ int g=i/HX, c=i%HX;
    sSc[i] = (g<4)? wscale[g*H+hbase+c] : bias[(g-4)*H+hbase+c]; }

#if CELLREG
  float creg[HX/8][MSET][4];
  #pragma unroll
  for(int a=0;a<HX/8;++a) for(int b=0;b<MSET;++b) for(int c=0;c<4;++c) creg[a][b][c]=0.f;
#endif
  auto tstep=[&](int t)->size_t{ int tt=reverse?(T-1-t):t; return (size_t)tt; };
#if PIPE
  load_A(sA + 0, A_all, tstep(0), N, row0);
  asm volatile("cp.async.commit_group;\n");
#endif

  for(int t=0;t<T;++t){
    int tt = reverse ? (T-1-t) : t;
    int out  = reverse ? tt : (tt+1);
    int8_t* hout_t = hout + (store_all ? (size_t)out*N*H : 0);
    int cur = t & (STAGES_A-1);
    int8_t* Acur = sA + (size_t)cur*BMG*KBYTES;
#if PIPE
    if(t+1<T){ int nxt=(t+1)&(STAGES_A-1); load_A(sA+(size_t)nxt*BMG*KBYTES,A_all,tstep(t+1),N,row0);
      asm volatile("cp.async.commit_group;\n"); }
    asm volatile("cp.async.wait_group %0;\n"::"n"(STAGES_A-1));
#else
    load_A(sA + 0, A_all, tstep(t), N, row0);
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_all;\n");
#endif
    __syncthreads();

    // load this warp's A fragments: MSET m-tiles x KT k-tiles (each ldm/frag covers k=64)
    int Af[MSET][KT][4];
    #pragma unroll
    for(int m=0;m<MSET;++m){
      int mrow0 = wr0 + m*16;
#if ALDM
      #pragma unroll
      for(int kt=0;kt<KT;++kt){
        int r=lane&15, koff=(lane>>4)*16;
        uint32_t a=smem_addr(&Acur[(mrow0+r)*KBYTES + kt*32 + koff]);   // 64 int4 = 32 bytes/ktile
        ldm_x4(Af[m][kt][0],Af[m][kt][1],Af[m][kt][2],Af[m][kt][3],a);
      }
#else
      // scalar A frag (ground-truth m16n8k64.s4 layout): each reg = 8 int4 = 4 bytes.
      // a0=row gid,k[tg*8..+7]; a1=row gid+8 same; a2=row gid,k[32+tg*8..]; a3=row gid+8.
      #pragma unroll
      for(int kt=0;kt<KT;++kt){
        int lo=(kt*64 + tg*8)/2, hi=(kt*64 + 32 + tg*8)/2;   // byte offsets
        Af[m][kt][0]=*(const int*)&Acur[(mrow0+gid)*KBYTES+lo];
        Af[m][kt][1]=*(const int*)&Acur[(mrow0+gid+8)*KBYTES+lo];
        Af[m][kt][2]=*(const int*)&Acur[(mrow0+gid)*KBYTES+hi];
        Af[m][kt][3]=*(const int*)&Acur[(mrow0+gid+8)*KBYTES+hi];
      }
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
        #pragma unroll
        for(int kt=0;kt<KT;++kt){
          // weight (gate g, channel hbase+nn*8+gid), ktile kt: b0=k[tg*8..+7], b1=k[32+tg*8..+7]
          const int8_t* wbase = wg_rp + ((long)g*H + (hbase+nn*8+gid))*KBYTES + (long)kt*32 + tg*8;
          int b0=__ldg((const int*)wbase), b1=__ldg((const int*)(wbase+4));
          #pragma unroll
          for(int m=0;m<MSET;++m)                 // weight fragment reused across MSET M-tiles
            mma_m16n8k64(cg[m][g][0],cg[m][g][1],cg[m][g][2],cg[m][g][3],
                         Af[m][kt][0],Af[m][kt][1],Af[m][kt][2],Af[m][kt][3],b0,b1);
        }
      }
      #pragma unroll
      for(int m=0;m<MSET;++m){
        int mrow0 = wr0 + m*16;
        #pragma unroll
        for(int e=0;e<4;++e){
          int lcol=nn*8+2*tg+(e&1), rin=(e<2)?gid:gid+8;
          int oc=hbase+lcol, gr=row0+mrow0+rin;
#if CELLREG
          float cv=creg[nn][m][e];
#else
          float cv=(float)cell[(long)gr*H+oc];
#endif
          int8_t hn=epi_elem(cg[m][0][e],cg[m][1][e],cg[m][2][e],cg[m][3][e],
              sSc[0*HX+lcol],sSc[1*HX+lcol],sSc[2*HX+lcol],sSc[3*HX+lcol],
              sSc[4*HX+lcol],sSc[5*HX+lcol],sSc[6*HX+lcol],sSc[7*HX+lcol],AS,cv);
#if CELLREG
          creg[nn][m][e]=(float)(cellT)cv;
#else
          cell[(long)gr*H+oc]=(cellT)cv;
#endif
          hout_t[(long)gr*H+oc]=hn;
        }
      }
    }
    __syncthreads();
  }
}

// int4 reference: unpack A + weights, exact integer dot product over KC, same epilogue.
__device__ __forceinline__ int unpack4(const int8_t* p,int idx){
  int8_t byte=p[idx>>1]; int nib=(idx&1)?(byte>>4):(byte&0xF);
  nib &= 0xF; if(nib&0x8) nib-=16; return nib;   // sign-extend 4-bit
}
__global__ void ref_kernel(const int8_t* A_all,        // [T,N,KBYTES]
    const int8_t* Wg,                                  // [4,H,KBYTES] int4 (unpermuted)
    const float* wscale,const float* bias,int8_t* hh_all,int N,int T,int reverse,int n_cmp){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=n_cmp) return;
  const float AS=1.0f/7.0f;
  cellT* c=new cellT[H]; for(int i=0;i<H;++i) c[i]=(cellT)0.f;
  for(int t=0;t<T;++t){ int tt=reverse?(T-1-t):t; int ws=reverse?tt:(tt+1);
    const int8_t* Ar=A_all+((long)tt*N+r)*KBYTES;
    for(int oc=0;oc<H;++oc){ int g[4];
      for(int gg=0;gg<4;++gg){ const int8_t* w=Wg+((long)gg*H+oc)*KBYTES; int a=0;
        for(int kc=0;kc<KC;++kc) a+=unpack4(Ar,kc)*unpack4(w,kc); g[gg]=a; }
      float cv=(float)c[oc];
      int8_t hn=epi_elem(g[0],g[1],g[2],g[3],wscale[0*H+oc],wscale[1*H+oc],wscale[2*H+oc],wscale[3*H+oc],
          bias[0*H+oc],bias[1*H+oc],bias[2*H+oc],bias[3*H+oc],AS,cv); c[oc]=(cellT)cv;
      hh_all[((long)ws*N+r)*H+oc]=hn; }
  }
  delete[] c;
}

// pack a signed int4 value (in [-7,7]) into nibble idx of a byte array
static inline void pack4(std::vector<int8_t>& v,size_t idx,int val){
  int nib=val&0xF; int8_t& byte=v[idx>>1];
  if(idx&1) byte=(int8_t)((byte&0x0F)|(nib<<4)); else byte=(int8_t)((byte&0xF0)|nib);
}
static inline int unpack_host(const std::vector<int8_t>& v,size_t idx){
  int8_t byte=v[idx>>1]; int nib=(idx&1)?((byte>>4)&0xF):(byte&0xF);
  if(nib&0x8) nib-=16; return nib;
}

int main(int argc,char**argv){
  int N=256,T=64,reverse=0,bench=0,n_cmp=32,dev=0;
  for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"--N"))N=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--T"))T=atoi(argv[++i]); else if(!strcmp(argv[i],"--reverse"))reverse=1;
    else if(!strcmp(argv[i],"--bench"))bench=1; else if(!strcmp(argv[i],"--ncmp"))n_cmp=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev"))dev=atoi(argv[++i]); }
  CK(cudaSetDevice(dev));
  if(N%BMG) N=((N+BMG-1)/BMG)*BMG;
  size_t smem=(size_t)STAGES_A*BMG*KBYTES + (size_t)8*HX*4;
  int ncta=GX*(N/BMG);
  printf("koi_gate_i4: GX=%d HX=%d BMG=%d MSET=%d KT=%d ALDM=%d PIPE=%d ctapsm=%d cell=%s N=%d T=%d rev=%d grid=(%d,%d)=%d %s smem=%.1fKB\n",
    GX,HX,BMG,MSET,KT,ALDM,PIPE,CTAPSM,sizeof(cellT)==2?"f16":"f32",N,T,reverse,GX,N/BMG,ncta,ncta<=108?"ONE-WAVE":"multi-wave",smem/1024.0);

  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> a4(-7,7);
  std::normal_distribution<float> nd(0,1);
  // activation: random int4 [-7,7] packed [T,N,KBYTES]
  std::vector<int8_t> hA((size_t)T*N*KBYTES,0);
  for(size_t r=0;r<(size_t)T*N;++r) for(int kc=0;kc<KC;++kc) pack4(hA,r*KC+kc,a4(rng));
  // weights: random int4 [-7,7] packed [4,H,KBYTES] (unpermuted, for the ref) + per-channel scale
  std::vector<int8_t> hW((size_t)4*H*KBYTES,0);
  std::vector<float> hws((size_t)4*H),hbs((size_t)4*H);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc){ float mx=1e-8f; std::vector<int> row(KC);
    for(int kc=0;kc<KC;++kc){ row[kc]=a4(rng); mx=fmaxf(mx,fabsf((float)row[kc])); }
    hws[(size_t)g*H+oc]=(mx/7.f)*0.02f; hbs[(size_t)g*H+oc]=nd(rng)*0.05f;
    for(int kc=0;kc<KC;++kc) pack4(hW,((size_t)g*H+oc)*KC+kc,row[kc]); }
  // repacked weights for the kernel: per (g,oc) block = [KT][tg][b0(4B),b1(4B)] = KBYTES bytes
  std::vector<int8_t> hrp((size_t)4*H*KBYTES,0);
  for(int g=0;g<4;++g)for(int oc=0;oc<H;++oc)for(int kc=0;kc<KC;++kc){
    int val=unpack_host(hW,((size_t)g*H+oc)*KC+kc);
    int kt=kc>>6, rem=kc&63, half=rem>>5, sub=rem&31, tgi=sub>>3, e=sub&7;
    size_t blk=((size_t)g*H+oc)*KC;                 // int4 units within hrp
    // kernel wbase byte off = kt*32 + tg*8 -> int4 base = kt*64 + tg*16; b0=[+0..7], b1=[+8..15]
    size_t pos=blk + (size_t)kt*64 + tgi*16 + half*8 + e;
    pack4(hrp,pos,val); }

  int8_t *dA,*dW,*drp,*dring; float *dws,*dbs; cellT *dcell;
  CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&drp,hrp.size())); CK(cudaMalloc(&dW,hW.size()));
  CK(cudaMalloc(&dws,4*H*4)); CK(cudaMalloc(&dbs,4*H*4)); CK(cudaMalloc(&dcell,(size_t)N*H*sizeof(cellT)));
  CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(drp,hrp.data(),hrp.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW,hW.data(),hW.size(),cudaMemcpyHostToDevice));
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
    ref_kernel<<<(n_cmp+31)/32,32>>>(dA,dW,dws,dbs,dref,N,T,reverse,n_cmp);
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
  printf("koi_gate_i4 step: %.2f us/step (median of 9, T=%d)  [koi_gate int8 26.0us; fused_cutlass 26.2us; koi ~13.9 full-step]\n",us[4],T);
  return 0;
}
