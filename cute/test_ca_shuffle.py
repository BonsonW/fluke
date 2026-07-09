"""Crux unit test: intra-warp shuffle of the int8 m16n8k32 down-proj C-fragment
(hh_down [16,128]) into the int8 m16n8k32 gate A-operand layout [16,128], NO smem.

Loads a known int8 hh[16,128] into a C-frag (via partition_C of a down-proj tiled
MMA), runs the 2-round packed-word shuffle -> A-frag, writes the A-frag back to
[16,128] (via partition_A of a gate tiled MMA), and checks it equals hh exactly.

Single warp (rows 0..15). If this is bit-exact the C->A relayout is proven.
"""
import os, sys
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ampere", "gemm"))
import gemm_i8_quant  # installs MmaI8Op

KHH = 128


def shuffle_c_to_a(rC, rA, lane):
    """rC: down-proj C-frag (MMA=4, MMA_M=1, MMA_N=16) int8 in C-layout.
    rA: gate A-frag  (MMA=16, MMA_M=1, MMA_K=4) int8, filled here.
    lane: within-warp lane (Int32)."""
    # pack each n8-subtile's 4 C ints into one word (bytes: (gid,2tg),(gid,2tg+1),(gid+8,2tg),(gid+8,2tg+1))
    word = []
    for jt in range(16):
        c0 = rC[0, 0, jt].to(cutlass.Int32) & 0xFF
        c1 = rC[1, 0, jt].to(cutlass.Int32) & 0xFF
        c2 = rC[2, 0, jt].to(cutlass.Int32) & 0xFF
        c3 = rC[3, 0, jt].to(cutlass.Int32) & 0xFF
        word.append(c0 | (c1 << 8) | (c2 << 16) | (c3 << 24))

    t = lane % 4
    group_base = (lane // 4) * 4
    src_lo = group_base + (t % 2) * 2      # tg0,2->group+0 ; tg1,3->group+2
    src_hi = src_lo + 1
    useB = t >= 2                          # tg0,1 -> jtA ; tg2,3 -> jtB

    def byte(w, p):
        v = (w >> (p * 8)) & 0xFF
        return (v - ((v & 0x80) << 1)).to(cutlass.Int8)   # sign-extend low byte -> int8

    for kt in range(4):
        for band in range(2):
            jtA = (kt * 32 + band * 16) // 8
            jtB = jtA + 1
            wA_lo = cute.arch.shuffle_sync(word[jtA], src_lo)
            wA_hi = cute.arch.shuffle_sync(word[jtA], src_hi)
            wB_lo = cute.arch.shuffle_sync(word[jtB], src_lo)
            wB_hi = cute.arch.shuffle_sync(word[jtB], src_hi)
            w_lo = cutlass.Int32(useB) * (wB_lo - wA_lo) + wA_lo   # select without branch
            w_hi = cutlass.Int32(useB) * (wB_hi - wA_hi) + wA_hi
            for rowhalf in range(2):
                e_base = rowhalf * 4 + band * 8
                rA[e_base + 0, 0, kt] = byte(w_lo, rowhalf * 2 + 0)
                rA[e_base + 1, 0, kt] = byte(w_lo, rowhalf * 2 + 1)
                rA[e_base + 2, 0, kt] = byte(w_hi, rowhalf * 2 + 0)
                rA[e_base + 3, 0, kt] = byte(w_hi, rowhalf * 2 + 1)


class CAShuffleTest:
    @cute.jit
    def __call__(self, mHH: cute.Tensor, mOut: cute.Tensor):
        i8 = (16, 8, 32)
        op = cute.nvgpu.warp.MmaI8Op(cutlass.Int8, cutlass.Int8, cutlass.Int32, i8)
        dp = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, KHH, 32))
        gate = cute.make_tiled_mma(op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, KHH))
        self.kernel(mHH, mOut, dp, gate).launch(grid=(1, 1, 1), block=(32, 1, 1))

    @cute.kernel
    def kernel(self, mHH: cute.Tensor, mOut: cute.Tensor, dp: cute.TiledMma, gate: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        thr_dp = dp.get_slice(tidx)
        thr_g = gate.get_slice(tidx)
        gC = thr_dp.partition_C(mHH)          # (4,1,16) int8
        rC = cute.make_fragment_like(gC, cutlass.Int8)
        cute.autovec_copy(gC, rC)
        gA = thr_g.partition_A(mOut)          # (16,1,4) int8
        rA = cute.make_fragment_like(gA, cutlass.Int8)
        shuffle_c_to_a(rC, rA, tidx % 32)
        cute.autovec_copy(rA, gA)


def main():
    dev = "cuda"
    torch.manual_seed(0)
    hh = torch.randint(-128, 128, (16, KHH), dtype=torch.int8, device=dev)
    out = torch.zeros(16, KHH, dtype=torch.int8, device=dev)
    mHH = from_dlpack(hh, assumed_align=16)
    mOut = from_dlpack(out, assumed_align=16)
    t = CAShuffleTest()
    compiled = cute.compile(t, mHH, mOut)
    compiled(mHH, mOut)
    torch.cuda.synchronize()
    match = torch.equal(out, hh)
    ndiff = (out != hh).sum().item()
    print(f"C->A shuffle bit-exact: {match}  (mismatches={ndiff}/{16*KHH})")
    if not match:
        bad = (out != hh).nonzero()[:10]
        for r, c in bad.tolist():
            print(f"  [{r},{c}] got {out[r,c].item()} exp {hh[r,c].item()}")
    print("PASS" if match else "FAIL")
    sys.exit(0 if match else 1)


if __name__ == "__main__":
    main()
