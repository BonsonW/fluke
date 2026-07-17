// Arch-neutral vtable for the fused INT8 DSL kernels (CUDA). Mirror of src/fp8_ops.h.
//
// Each supported SM arch compiles to its own translation unit (src/i8_arch.c, built once per
// arch with -D), which includes that arch's generated headers (artifacts/<sm>/*.h — the CuTe
// descriptor types + launch wrappers) and exports one external symbol: a `fluke_i8_ops` vtable
// named `fluke_i8_ops_<sm>`. src/fused_cuda.c holds no kernel-launch code — it selects the
// vtable by compute capability and dispatches through it. Isolating each arch's headers in its
// own TU lets one libfluke.a carry several SM arches (a "fat" binary) without the generated
// typedefs colliding.
//
// NOTE: the CuTe-generated launcher/Module_Load symbols live in the exported .o and are keyed
// by kernel dims, not arch — so shipping two SM arches with identical dims additionally needs
// the export to arch-qualify those symbols (function_prefix). This vtable is the host-side seam;
// that export change is the remaining piece for >1 SM arch.
#ifndef FLUKE_FUSED_I8_OPS_H
#define FLUKE_FUSED_I8_OPS_H

// One arch's fused-int8 entry points (transformer path: fused qkv-GEMM+rotary and dual-GEMM+SiLU).
// Pointers reference file-local (static) functions in that arch's TU; only the vtable
// itself is externally visible. load() binds the two kernel modules once. The launch fns mirror the
// fluke_*_i8_gpu C ABI argument order; baked shapes come from fused_dims.h inside the arch TU, so
// only the runtime M (and rotary seqlen) are passed.
struct fluke_i8_ops {
    int (*load)(void);
    int (*qkv_rotary)(void *out, const void *a_i8, const void *wqkv_i8,
                      const void *scale_a, const void *scale_b,
                      const void *sin, const void *cos, int M, int seqlen);
    int (*gated_mlp)(void *out, const void *a_i8, const void *gate_i8, const void *up_i8,
                     const void *scale_a, const void *scale_gate, const void *scale_up, int M);
};

#endif  // FLUKE_FUSED_I8_OPS_H
