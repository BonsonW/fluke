"""Empirically dump the int8 m16n8k32 A-fragment and C-fragment per-lane layouts.

For a single warp (canonical atom, atom_layout (1,1,1)):
  - A operand tile (16 x 32): which (row, k) does each thread/element hold?
  - C operand tile (16 x 8):  which (row, n) does each thread/element hold?

We write coords into gmem [lane, elem, 2] and read back in torch to reason about
the C->A intra-warp shuffle for the M-split kernel.
"""
import os, sys
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ampere", "gemm"))
import gemm_i8_quant  # installs MmaI8Op


class Probe:
    @cute.jit
    def __call__(self, mA_out: cute.Tensor, mC_out: cute.Tensor):
        i8 = (16, 8, 32)
        op = cute.nvgpu.warp.MmaI8Op(cutlass.Int8, cutlass.Int8, cutlass.Int32, i8)
        tiled = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)),
                                    permutation_mnk=(16, 8, 32))
        self.kernel(mA_out, mC_out, tiled).launch(grid=(1, 1, 1), block=(32, 1, 1))

    @cute.kernel
    def kernel(self, mA_out: cute.Tensor, mC_out: cute.Tensor, tiled: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        thr = tiled.get_slice(tidx)
        cA = cute.make_identity_tensor((16, 32))
        cC = cute.make_identity_tensor((16, 8))
        tA = thr.partition_A(cA)   # (MMA, MMA_M, MMA_K)
        tC = thr.partition_C(cC)   # (MMA, MMA_M, MMA_N)
        aV = cute.size(tA, mode=[0]); aM = cute.size(tA, mode=[1]); aK = cute.size(tA, mode=[2])
        cV = cute.size(tC, mode=[0]); cM = cute.size(tC, mode=[1]); cN = cute.size(tC, mode=[2])
        for v in cutlass.range_constexpr(aV):
            for m in cutlass.range_constexpr(aM):
                for k in cutlass.range_constexpr(aK):
                    e = v + m * aV + k * aV * aM
                    crd = tA[v, m, k]
                    mA_out[tidx, e, 0] = cutlass.Int32(crd[0])
                    mA_out[tidx, e, 1] = cutlass.Int32(crd[1])
        for v in cutlass.range_constexpr(cV):
            for m in cutlass.range_constexpr(cM):
                for n in cutlass.range_constexpr(cN):
                    e = v + m * cV + n * cV * cM
                    crd = tC[v, m, n]
                    mC_out[tidx, e, 0] = cutlass.Int32(crd[0])
                    mC_out[tidx, e, 1] = cutlass.Int32(crd[1])


def main():
    dev = "cuda"
    A_out = torch.full((32, 16, 2), -1, dtype=torch.int32, device=dev)
    C_out = torch.full((32, 4, 2), -1, dtype=torch.int32, device=dev)
    mA = from_dlpack(A_out, assumed_align=16)
    mC = from_dlpack(C_out, assumed_align=16)
    p = Probe()
    compiled = cute.compile(p, mA, mC)
    compiled(mA, mC)
    torch.cuda.synchronize()
    A = A_out.cpu().numpy()
    C = C_out.cpu().numpy()
    print("=== A-fragment (16x32) per-lane, elem -> (row,k) ===")
    for lane in range(32):
        gid = lane // 4; tg = lane % 4
        elems = ",".join(f"{e}:({A[lane,e,0]:2d},{A[lane,e,1]:2d})" for e in range(16))
        print(f"lane{lane:2d} gid{gid} tg{tg}: {elems}")
    print("\n=== C-fragment (16x8) per-lane, elem -> (row,n) ===")
    for lane in range(32):
        gid = lane // 4; tg = lane % 4
        elems = ",".join(f"{e}:({C[lane,e,0]:2d},{C[lane,e,1]:2d})" for e in range(4))
        print(f"lane{lane:2d} gid{gid} tg{tg}: {elems}")


if __name__ == "__main__":
    main()
