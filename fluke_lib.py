"""Load libfluke.so and expose the C ABI via ctypes.

Tests call the real launch wrappers (fluke_rotary_emb_gpu, fluke_silu_mul_gpu,
fluke_rmsnorm_gpu, ...) directly on torch device pointers — the same C ABI slorado
uses — instead of JIT-recompiling the kernels. If lib/libfluke.so is missing it is
built via `make shared` for the current GPU's arch.

    import fluke_lib
    lib = fluke_lib.load()
    lib.fluke_rotary_emb_gpu(x.data_ptr(), sin.data_ptr(), cos.data_ptr(), ...)
"""
import ctypes
import os
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))   # fluke repo root
_SO = os.path.join(_ROOT, "lib", "libfluke.so")

_P, _I, _F = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
# (function name -> argtypes); restype is None (void) for all.
_SIGNATURES = {
    "fluke_rotary_emb_gpu":        [_P, _P, _P, _I, _I, _I, _I, _I, _I, _I, _I],
    "fluke_silu_mul_gpu":          [_P, _P, _I, _I],
    "fluke_rmsnorm_gpu":           [_P, _P, _P, _P, _I, _I, _F, _F],
    "fluke_rmsnorm_quant_int8_gpu":[_P, _P, _P, _P, _I, _I, _F, _F],
    "fluke_rmsnorm_quant_fp8_gpu": [_P, _P, _P, _P, _I, _I, _F, _F],
    "fluke_dequant_fp8_transpose_gpu":  [_P, _P, _I, _I, _I, _F],
    "fluke_dequant_int8_transpose_gpu": [_P, _P, _I, _I, _I, _F],
    "fluke_flstm_step_gpu":        [_P, _P, _P, _P, _I, _I],
}


def _build():
    """Build lib/libfluke.so for the current GPU backend (CUDA or HIP), matching the device arch.
    ROCm torch presents as torch.cuda.* but sets torch.version.hip."""
    import torch
    if getattr(torch.version, "hip", None):  # ROCm/HIP torch
        try:
            gfx = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]  # e.g. gfx1201
        except Exception:
            gfx = ""
        cmd = ["make", "shared", "rocm=1"] + ([f"ROCM_ARCH=--offload-arch={gfx}"] if gfx else [])
    else:  # CUDA torch
        cc = torch.cuda.get_device_capability()
        cmd = ["make", "shared", "cuda=1",
               f"CUDA_ARCH=-gencode arch=compute_{cc[0]}{cc[1]},code=sm_{cc[0]}{cc[1]}"]
    print(f">> building libfluke.so ({' '.join(cmd[1:])})")
    subprocess.run(cmd, cwd=_ROOT, check=True)


def load(rebuild=False):
    """Return the libfluke.so CDLL with argtypes set (building it first if needed)."""
    if rebuild or not os.path.isfile(_SO):
        _build()
    lib = ctypes.CDLL(_SO)
    for name, argtypes in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = argtypes
    return lib
