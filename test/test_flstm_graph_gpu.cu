// CUDA-graph capture-safety regression for the int8 factored-LSTM kernels.
//
// The recurrent flstm path (slorado lstm_model.cpp) captures the whole T-step loop into a CUDA
// graph and replays it. That is only correct if the fluke int8 kernels (down_proj + fused step)
// bake their launch parameters into the graph node by value — i.e. a replay must not depend on
// the transient host-side descriptor structs that fluke_*_gpu fill per call. This test runs
// {down_proj; step} eager on a stream, then the same sequence captured into a graph and replayed,
// on identical fixed buffers, and requires bit-identical output. If they differ, the wrappers are
// not capture-safe and the graph path would silently produce wrong results.
//
// Build + run (CUDA only; needs the sm80/86/89 int8 backend):
//   make cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80" test-flstm-graph
//
// Exit: 0 = capture-safe (PASS), 1 = mismatch (FAIL), 2 = no backend (skipped).
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#include "fluke/fluke.h"

#define CK(x) do { cudaError_t e_=(x); if (e_!=cudaSuccess) { \
    printf("CUDA error %s @ %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

// Baked kernel shape (matches artifacts/sm80/{down_proj_i8_R128_K1024, factored_lstm_i8_H1024_Khh128_R128}).
static const int H = 1024, K_hh = 128, R = 128, Kc = K_hh + R, B = 384;

template <class T> static T *dalloc(size_t n) { void *p; CK(cudaMalloc(&p, n * sizeof(T))); return (T *)p; }

int main() {
    fluke_flstm_backend_t *be = fluke_flstm_select(0, H, K_hh, R);
    if (!be) { printf("SKIP: no int8 factored-LSTM backend on this device (arch/dims)\n"); return 2; }

    cudaStream_t s; CK(cudaStreamCreate(&s));

    // down_proj inputs: a_i8[B,H], w_i8[R,H], scale_a[B], scale_b[R]; output dp_out f16[B,R].
    std::vector<int8_t> h_a(B * H), h_w(R * H);
    for (size_t i = 0; i < h_a.size(); ++i) h_a[i] = (int8_t)((i % 13) - 6);
    for (size_t i = 0; i < h_w.size(); ++i) h_w[i] = (int8_t)((i % 7) - 3);
    std::vector<float> h_sa(B, 0.011f), h_sb(R, 0.019f);
    int8_t *a_i8 = dalloc<int8_t>(B * H), *w_i8 = dalloc<int8_t>(R * H);
    float *sa = dalloc<float>(B), *sb = dalloc<float>(R);
    __half *dp_out = dalloc<__half>(B * R);
    CK(cudaMemcpy(a_i8, h_a.data(), h_a.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(w_i8, h_w.data(), h_w.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sa, h_sa.data(), B * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(sb, h_sb.data(), R * 4, cudaMemcpyHostToDevice));

    // step inputs: a_f16[B,Kc], gate weights [H,Kc]x4, bias[H]x4, cell[B,H]; output h_i8[B,H].
    std::vector<__half> h_af(B * Kc), h_gw(H * Kc);
    for (size_t i = 0; i < h_af.size(); ++i) h_af[i] = __float2half(0.02f * ((int)(i % 17) - 8));
    for (size_t i = 0; i < h_gw.size(); ++i) h_gw[i] = __float2half(0.015f * ((int)(i % 11) - 5));
    std::vector<float> h_bias(H, 0.0f), h_c0(B * H);
    for (size_t i = 0; i < h_c0.size(); ++i) h_c0[i] = 0.1f * ((int)(i % 5) - 2);
    __half *af = dalloc<__half>(B * Kc);
    __half *bi = dalloc<__half>(H * Kc), *bf = dalloc<__half>(H * Kc), *bg = dalloc<__half>(H * Kc), *bo = dalloc<__half>(H * Kc);
    float *ci = dalloc<float>(H), *cf = dalloc<float>(H), *cg = dalloc<float>(H), *co = dalloc<float>(H);
    float *cell = dalloc<float>(B * H); int8_t *hout = dalloc<int8_t>(B * H);
    CK(cudaMemcpy(af, h_af.data(), h_af.size() * 2, cudaMemcpyHostToDevice));
    for (__half *g : {bi, bf, bg, bo}) CK(cudaMemcpy(g, h_gw.data(), h_gw.size() * 2, cudaMemcpyHostToDevice));
    for (float *g : {ci, cf, cg, co}) CK(cudaMemcpy(g, h_bias.data(), H * 4, cudaMemcpyHostToDevice));

    auto reset_cell = [&] { CK(cudaMemcpyAsync(cell, h_c0.data(), h_c0.size() * 4, cudaMemcpyHostToDevice, s)); };
    auto run = [&] {
        fluke_down_proj_i8_gpu(be, dp_out, a_i8, w_i8, sa, sb, B, s);
        fluke_flstm_step_i8_gpu(be, hout, af, bi, bf, bg, bo, ci, cf, cg, co, cell, B, s);
    };

    // Eager reference.
    reset_cell(); run(); CK(cudaStreamSynchronize(s));
    std::vector<int8_t> h_eager(B * H); std::vector<float> c_eager(B * H); std::vector<__half> dp_eager(B * R);
    CK(cudaMemcpy(h_eager.data(), hout, B * H, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c_eager.data(), cell, B * H * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dp_eager.data(), dp_out, B * R * 2, cudaMemcpyDeviceToHost));

    // Capture into a graph and replay (warmup first to prime any lazy first-launch init).
    reset_cell(); run(); CK(cudaStreamSynchronize(s));
    cudaGraph_t g; cudaGraphExec_t gexec;
    reset_cell();
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    run();
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&gexec, g, 0));
    reset_cell();                                   // graph records ops, not data: re-zero cell before replay
    CK(cudaGraphLaunch(gexec, s));
    CK(cudaStreamSynchronize(s));
    std::vector<int8_t> h_graph(B * H); std::vector<float> c_graph(B * H); std::vector<__half> dp_graph(B * R);
    CK(cudaMemcpy(h_graph.data(), hout, B * H, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(c_graph.data(), cell, B * H * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dp_graph.data(), dp_out, B * R * 2, cudaMemcpyDeviceToHost));

    // Compare: must be bit-identical.
    long h_diff = 0, h_max = 0;
    for (size_t i = 0; i < h_eager.size(); ++i) { int d = abs((int)h_eager[i] - (int)h_graph[i]); if (d) { h_diff++; if (d > h_max) h_max = d; } }
    double c_max = 0;
    for (size_t i = 0; i < c_eager.size(); ++i) { double d = fabs((double)c_eager[i] - (double)c_graph[i]); if (d > c_max) c_max = d; }
    long dp_diff = 0; double dp_max = 0;
    for (size_t i = 0; i < dp_eager.size(); ++i) { double d = fabs((double)__half2float(dp_eager[i]) - (double)__half2float(dp_graph[i])); if (d > 0) { dp_diff++; if (d > dp_max) dp_max = d; } }

    printf("[down_proj] eager-vs-graph  #diff=%ld/%d  max|d|=%.4g\n", dp_diff, B * R, dp_max);
    printf("[step h_i8] eager-vs-graph  #diff=%ld/%d  max|d|=%ld\n", h_diff, B * H, h_max);
    printf("[step cell] eager-vs-graph  max|d|=%.4g\n", c_max);

    bool ok = (dp_diff == 0) && (h_diff == 0) && (c_max < 1e-4);
    printf("%s\n", ok ? "PASS (CUDA-graph capture-safe: graph == eager)"
                      : "FAIL (graph != eager -> flstm kernels not capture-safe)");
    return ok ? 0 : 1;
}
