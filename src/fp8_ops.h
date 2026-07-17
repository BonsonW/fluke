// Arch-neutral vtable for the fused fp8 DSL kernels.
//
// Each supported GPU arch compiles to its OWN translation unit (one source, src/fp8_arch.c,
// compiled once per arch), which includes that arch's generated header (baking its launch
// geometry — wave width, tile sizes, grid math) and exposes exactly one external symbol: a
// `fluke_fp8_ops` vtable named `fluke_fp8_ops_<arch>`. src/fused_hip.c holds no arch-specific
// launch code — it just picks the matching vtable by the device's gcnArchName and dispatches
// through it. This is what lets one libfluke.a embed many arches (a "fat" binary): arches whose
// launch config differs (e.g. RDNA4 WMMA/wave32 vs CDNA3 MFMA/wave64) no longer share a single
// host wrapper, so they can coexist. Adding an arch = add it to the Makefile arch list + a
// g_archs[] row in fused_hip.c.
#ifndef FLUKE_FUSED_FP8_OPS_H
#define FLUKE_FUSED_FP8_OPS_H

// One arch's fused-fp8 entry points. Pointers reference file-local (static) functions in that
// arch's TU; only the vtable itself is externally visible. `load` binds
// the arch's kernel modules from the embedded HSACO images (called once by fluke_fp8_select);
// the launch fns mirror the fluke_*_fp8_gpu C ABI argument order.
struct fluke_fp8_ops {
    int (*load)(const void *rotary_image, const void *mlp_image);
    int (*qkv_rotary)(void *out, const void *a_fp8, const void *wqkv_fp8,
                      const void *scale_a, const void *scale_b,
                      const void *sin, const void *cos, int M, int seqlen);
    int (*gated_mlp)(void *out, const void *a_fp8, const void *gate_fp8, const void *up_fp8,
                     const void *scale_a, const void *scale_gate, const void *scale_up, int M);
};

#endif  // FLUKE_FUSED_FP8_OPS_H
