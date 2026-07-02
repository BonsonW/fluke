"""Fused factored-LSTM step for NVIDIA Ampere (A100, sm80) — INT8 port.

CUDA port of fly/rdna_fp8_factored_lstm.py compile_fp8_factored_lstm. Computes one
LSTM timestep with BOTH projections kept low-rank (no K=H GEMM):

    gates[B, 4H] = hh_down[B,K_hh] @ up_hh[4H,K_hh]^T
                 + x_down[B,R]    @ up_ih[4H,R]^T   + bias_hh + bias_ih
    i,f,o = sighard(gate) = clamp(0.2*gate + 0.5, 0, 1)
    g     = clamp(gate, -1, 1)
    c_new = f*c + i*g                          (f32, written in place)
    h_new = o * tanh(c_new)  in [-1,1]  ->  int8 at fixed scale 1/127

Faithful to the RDNA reference, the two up-projections stay f16 (Ampere f16 tensor
cores via MmaF16BF16Op); only the recurrent down-projection (a separate kernel) and
the hidden-state output are int8. h_new is provably in [-1,1], so a fixed 1/127 scale
maps it onto int8 and the quantize is fused into the epilogue.

Structure mirrors ampere_dual_gemm_i8_silu.py (one shared A operand, several
accumulators, register epilogue with the gate/cell intermediates never touching DRAM),
generalized from 2 to 4 accumulators. The two projections are MERGED along K so a
single K=K_hh+R GEMM feeds all four gates:

    A      = concat([hh_down, x_down], dim=1)                 -> [B, K_hh+R]
    W_gate = concat([up_hh[gate], up_ih[gate]], dim=1)        -> [H,  K_hh+R]   (per gate)

The four gate weights W_i, W_f, W_g, W_o are the four H-row blocks of the (merged,
K-concatenated) up-projection weight; the caller passes them as four [H, K_hh+R]
tensors (contiguous row blocks of one [4H, K_hh+R] buffer). All four accumulators
cover the SAME (batch, hidden) output element, so the LSTM epilogue combines them
elementwise, exactly like silu(gate)*up in the dual kernel.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from gemm_f16 import TensorOpGemm

H_OUT_SCALE = 1.0 / 127.0   # fixed int8 hidden-output scale (h in [-1,1])


class TensorOpFactoredLstmI8(TensorOpGemm):
    """Fused factored-LSTM step: 4 gate accumulators, f16 up-projection, int8 h out.

    ab_dtype  : up-projection / down-projection activation dtype (Float16).
    out_dtype : hidden-state output dtype (Int8), fixed scale 1/127.
    Register-accumulator pressure is 4x a single GEMM, so keep bN small (default 32).
    """

    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        bm: int = 64,
        bn: int = 32,
        bk: int = 32,
        num_stages: int = 3,
    ):
        self.ab_dtype = ab_dtype
        self.out_dtype = out_dtype
        self.c_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.cta_tiler = (bm, bn, bk)
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        assert self.bM % (atom_lay_M * mmaM) == 0, "bM must be divisible by MMA instruction"
        assert self.bN % (atom_lay_N * mmaN) == 0, "bN must be divisible by MMA instruction"
        assert atom_lay_K == 1, "atom layout K > 1 unsupported"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction"
        assert self.num_stages >= 3, "num_stages must be >= 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,        # [B, Kc]  f16   (concat hh_down|x_down)
        mB_i: cute.Tensor,      # [H, Kc]  f16   gate-i weight (concat up_hh_i|up_ih_i)
        mB_f: cute.Tensor,      # [H, Kc]  f16
        mB_g: cute.Tensor,      # [H, Kc]  f16
        mB_o: cute.Tensor,      # [H, Kc]  f16
        mBias_i: cute.Tensor,   # [H]      f32   (bias_hh_i + bias_ih_i)
        mBias_f: cute.Tensor,   # [H]      f32
        mBias_g: cute.Tensor,   # [H]      f32
        mBias_o: cute.Tensor,   # [H]      f32
        mC_c: cute.Tensor,      # [B, H]   f32   cell state, read + written in place
        mH_out: cute.Tensor,    # [B, H]   int8  hidden output (fixed scale 1/127)
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB_i)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mH_out)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB_i.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        smem_size = (
            cute.size_in_bytes(mA.element_type, sA_layout)
            + 4 * cute.size_in_bytes(mB_i.element_type, sB_layout)
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type, num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB_i.element_type, self.b_major_mode, ab_copy_bits)

        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        grid_dim = cute.ceil_div(mH_out.shape, (self.bM, self.bN, 1))
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA, mB_i, mB_f, mB_g, mB_o,
            mBias_i, mBias_f, mBias_g, mBias_o,
            mC_c, mH_out,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma,
            raster_factor,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB_i: cute.Tensor, mB_f: cute.Tensor, mB_g: cute.Tensor, mB_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor, mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mC_c: cute.Tensor,
        mH_out: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mH_out.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            gA = cute.local_tile(mA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            gB_i = cute.local_tile(mB_i[None, None, bidz], tiler=self.cta_tiler,
                                   coord=tiler_coord, proj=(None, 1, 1))
            gB_f = cute.local_tile(mB_f[None, None, bidz], tiler=self.cta_tiler,
                                   coord=tiler_coord, proj=(None, 1, 1))
            gB_g = cute.local_tile(mB_g[None, None, bidz], tiler=self.cta_tiler,
                                   coord=tiler_coord, proj=(None, 1, 1))
            gB_o = cute.local_tile(mB_o[None, None, bidz], tiler=self.cta_tiler,
                                   coord=tiler_coord, proj=(None, 1, 1))
            gH = cute.local_tile(mH_out[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, 1, None))
            gCc = cute.local_tile(mC_c[None, None, bidz], tiler=self.cta_tiler,
                                  coord=tiler_coord, proj=(1, 1, None))

            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB_i = cute.domain_offset((0, residual_k, 0), gB_i)
            gB_f = cute.domain_offset((0, residual_k, 0), gB_f)
            gB_g = cute.domain_offset((0, residual_k, 0), gB_g)
            gB_o = cute.domain_offset((0, residual_k, 0), gB_o)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB_i = cute.make_tensor(gB_i.iterator.align(16), gB_i.layout)
            gB_f = cute.make_tensor(gB_f.iterator.align(16), gB_f.layout)
            gB_g = cute.make_tensor(gB_g.iterator.align(16), gB_g.layout)
            gB_o = cute.make_tensor(gB_o.iterator.align(16), gB_o.layout)

            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB_i.layout.shape)
            cA = cute.local_tile(mcA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            cB = cute.local_tile(mcB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB_i = smem.allocate_tensor(mB_i.element_type, sB_layout, 16)
            sB_f = smem.allocate_tensor(mB_f.element_type, sB_layout, 16)
            sB_g = smem.allocate_tensor(mB_g.element_type, sB_layout, 16)
            sB_o = smem.allocate_tensor(mB_o.element_type, sB_layout, 16)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB_i = thr_copy_B.partition_S(gB_i); tBsB_i = thr_copy_B.partition_D(sB_i)
            tBgB_f = thr_copy_B.partition_S(gB_f); tBsB_f = thr_copy_B.partition_D(sB_f)
            tBgB_g = thr_copy_B.partition_S(gB_g); tBsB_g = thr_copy_B.partition_D(sB_g)
            tBgB_o = thr_copy_B.partition_S(gB_o); tBsB_o = thr_copy_B.partition_D(sB_o)

            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ), cutlass.Boolean)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB_i.shape[0][1], cute.size(tBsB_i, mode=[1]), cute.size(tBsB_i, mode=[2])),
                    stride=(cute.size(tBsB_i, mode=[1]), 1, 0),
                ), cutlass.Boolean)
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB_i.shape[0])

            tAsA.fill(0)
            tBsB_i.fill(0); tBsB_f.fill(0); tBsB_g.fill(0); tBsB_o.fill(0)
            cute.arch.sync_threads()
            num_smem_stages = cute.size(tAsA, mode=[3])
            k_tile_count = cute.size(tAgA, mode=[3])
            k_tile_index = cutlass.Int32(0)

            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_A, tAgA[None, None, k, k_tile_index],
                              tAsA[None, None, k, 0], pred=tApA[None, None, k])
            for k in range(tBpB.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                    cute.copy(tiled_copy_B, tBgB_i[None, None, k, k_tile_index],
                              tBsB_i[None, None, k, 0], pred=tBpB[None, None, k])
                    cute.copy(tiled_copy_B, tBgB_f[None, None, k, k_tile_index],
                              tBsB_f[None, None, k, 0], pred=tBpB[None, None, k])
                    cute.copy(tiled_copy_B, tBgB_g[None, None, k, k_tile_index],
                              tBsB_g[None, None, k, 0], pred=tBpB[None, None, k])
                    cute.copy(tiled_copy_B, tBgB_o[None, None, k, k_tile_index],
                              tBsB_o[None, None, k, 0], pred=tBpB[None, None, k])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0); tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                          tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB_i[None, None, None, k_tile_index],
                          tBsB_i[None, None, None, k_tile], pred=tBpB)
                cute.copy(tiled_copy_B, tBgB_f[None, None, None, k_tile_index],
                          tBsB_f[None, None, None, k_tile], pred=tBpB)
                cute.copy(tiled_copy_B, tBgB_g[None, None, None, k_tile_index],
                          tBsB_g[None, None, None, k_tile], pred=tBpB)
                cute.copy(tiled_copy_B, tBgB_o[None, None, None, k_tile_index],
                          tBsB_o[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # MMA partitions + four gate accumulators
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB_i = thr_mma.partition_B(sB_i)
            tCsB_f = thr_mma.partition_B(sB_f)
            tCsB_g = thr_mma.partition_B(sB_g)
            tCsB_o = thr_mma.partition_B(sB_o)
            tCgH = thr_mma.partition_C(gH)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB_i = tiled_mma.make_fragment_B(tCsB_i[None, None, None, 0])
            tCrB_f = tiled_mma.make_fragment_B(tCsB_f[None, None, None, 0])
            tCrB_g = tiled_mma.make_fragment_B(tCsB_g[None, None, None, 0])
            tCrB_o = tiled_mma.make_fragment_B(tCsB_o[None, None, None, 0])
            tCrC_i = tiled_mma.make_fragment_C(tCgH)
            tCrC_f = tiled_mma.make_fragment_C(tCgH)
            tCrC_g = tiled_mma.make_fragment_C(tCgH)
            tCrC_o = tiled_mma.make_fragment_C(tCgH)
            tCrC_i.fill(0.0); tCrC_f.fill(0.0); tCrC_g.fill(0.0); tCrC_o.fill(0.0)

            num_vals = cute.size(tCrC_i, mode=[0])
            num_mma_m = cute.size(tCrC_i, mode=[1])
            num_mma_n = cute.size(tCrC_i, mode=[2])

            # Per-hidden-column bias (broadcast over M), one per gate.
            gBias_i = cute.local_tile(mBias_i[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))
            gBias_f = cute.local_tile(mBias_f[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))
            gBias_g = cute.local_tile(mBias_g[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))
            gBias_o = cute.local_tile(mBias_o[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))

            def bias_view(gBias):
                b2d = cute.make_tensor(gBias.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1)))
                return thr_mma.partition_C(b2d)

            tCgBias_i = bias_view(gBias_i)
            tCgBias_f = bias_view(gBias_f)
            tCgBias_g = bias_view(gBias_g)
            tCgBias_o = bias_view(gBias_o)

            rBias_i = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            rBias_f = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            rBias_g = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            rBias_o = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rBias_i[i, 0, n] = tCgBias_i[i, 0, n].to(cutlass.Float32)
                    rBias_f[i, 0, n] = tCgBias_f[i, 0, n].to(cutlass.Float32)
                    rBias_g[i, 0, n] = tCgBias_g[i, 0, n].to(cutlass.Float32)
                    rBias_o[i, 0, n] = tCgBias_o[i, 0, n].to(cutlass.Float32)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mA.element_type)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mB_i.element_type)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_i_cv = thr_copy_ldmatrix_B.partition_S(sB_i); tCrB_i_cv = thr_copy_ldmatrix_B.retile(tCrB_i)
            tCsB_f_cv = thr_copy_ldmatrix_B.partition_S(sB_f); tCrB_f_cv = thr_copy_ldmatrix_B.retile(tCrB_f)
            tCsB_g_cv = thr_copy_ldmatrix_B.partition_S(sB_g); tCrB_g_cv = thr_copy_ldmatrix_B.retile(tCrB_g)
            tCsB_o_cv = thr_copy_ldmatrix_B.partition_S(sB_o); tCrB_o_cv = thr_copy_ldmatrix_B.retile(tCrB_o)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1

            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_i_p = tCsB_i_cv[None, None, None, smem_pipe_read]
            tCsB_f_p = tCsB_f_cv[None, None, None, smem_pipe_read]
            tCsB_g_p = tCsB_g_cv[None, None, None, smem_pipe_read]
            tCsB_o_p = tCsB_o_cv[None, None, None, smem_pipe_read]

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_i_p[None, None, 0], tCrB_i_cv[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_f_p[None, None, 0], tCrB_f_cv[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_g_p[None, None, 0], tCrB_g_cv[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_o_p[None, None, 0], tCrB_o_cv[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_i_p = tCsB_i_cv[None, None, None, smem_pipe_read]
                        tCsB_f_p = tCsB_f_cv[None, None, None, smem_pipe_read]
                        tCsB_g_p = tCsB_g_cv[None, None, None, smem_pipe_read]
                        tCsB_o_p = tCsB_o_cv[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                              tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_i_p[None, None, k_block_next], tCrB_i_cv[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_f_p[None, None, k_block_next], tCrB_f_cv[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_g_p[None, None, k_block_next], tCrB_g_cv[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_o_p[None, None, k_block_next], tCrB_o_cv[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                                      tAsA[None, None, None, smem_pipe_write], pred=tApA)
                            cute.copy(tiled_copy_B, tBgB_i[None, None, None, k_tile_index],
                                      tBsB_i[None, None, None, smem_pipe_write], pred=tBpB)
                            cute.copy(tiled_copy_B, tBgB_f[None, None, None, k_tile_index],
                                      tBsB_f[None, None, None, smem_pipe_write], pred=tBpB)
                            cute.copy(tiled_copy_B, tBgB_g[None, None, None, k_tile_index],
                                      tBsB_g[None, None, None, smem_pipe_write], pred=tBpB)
                            cute.copy(tiled_copy_B, tBgB_o[None, None, None, k_tile_index],
                                      tBsB_o[None, None, None, smem_pipe_write], pred=tBpB)

                    cute.gemm(tiled_mma, tCrC_i, tCrA[None, None, k_block], tCrB_i[None, None, k_block], tCrC_i)
                    cute.gemm(tiled_mma, tCrC_f, tCrA[None, None, k_block], tCrB_f[None, None, k_block], tCrC_f)
                    cute.gemm(tiled_mma, tCrC_g, tCrA[None, None, k_block], tCrB_g[None, None, k_block], tCrC_g)
                    cute.gemm(tiled_mma, tCrC_o, tCrA[None, None, k_block], tCrB_o[None, None, k_block], tCrC_o)

                    if k_block == 0:
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Epilogue: LSTM gate combine, cell update (f32 in place), int8 hidden out.
            tCgCc = thr_mma.partition_C(gCc)
            rH = cute.make_fragment_like(tCrC_i, self.out_dtype)
            rC = cute.make_fragment_like(tCrC_i, cutlass.Float32)

            def clamp(x, lo, hi):
                # cute.arch only exposes fmax; min(y,hi) = -fmax(-y,-hi)
                return 0.0 - cute.arch.fmax(0.0 - cute.arch.fmax(x, lo), 0.0 - hi)

            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    for n in cutlass.range(num_mma_n, unroll_full=True):
                        raw_i = tCrC_i[i, m, n].to(cutlass.Float32) + rBias_i[i, 0, n]
                        raw_f = tCrC_f[i, m, n].to(cutlass.Float32) + rBias_f[i, 0, n]
                        raw_g = tCrC_g[i, m, n].to(cutlass.Float32) + rBias_g[i, 0, n]
                        raw_o = tCrC_o[i, m, n].to(cutlass.Float32) + rBias_o[i, 0, n]

                        i_a = clamp(raw_i * 0.2 + 0.5, 0.0, 1.0)
                        f_a = clamp(raw_f * 0.2 + 0.5, 0.0, 1.0)
                        o_a = clamp(raw_o * 0.2 + 0.5, 0.0, 1.0)
                        g_a = clamp(raw_g, -1.0, 1.0)

                        c_old = tCgCc[i, m, n].to(cutlass.Float32)
                        c_new = f_a * c_old + i_a * g_a
                        rC[i, m, n] = c_new

                        h_new = o_a * cute.math.tanh(c_new)
                        # quantize to int8 at fixed scale 1/127. h_new in [-1,1] so
                        # h_new*127 in [-127,127]; round half away from zero via +-0.5.
                        hs = h_new * 127.0
                        half = clamp(hs * 1e30, -0.5, 0.5)   # sign(hs)*0.5 (0 -> 0)
                        rH[i, m, n] = (hs + half).to(self.out_dtype)
            cute.autovec_copy(rC, tCgCc)
            cute.autovec_copy(rH, tCgH)
        return


# =============================================================================
# AOT export
# =============================================================================
def export_factored_lstm_i8(
    ab_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    out_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    file_path: str = "./artifacts",
    file_name: str = "factored_lstm_i8",
    function_prefix: str = "factored_lstm_i8",
    H: int = 1024,
    K_hh: int = 128,
    R: int = 128,
    bm: int = 64,
    bn: int = 32,
    bk: int = 32,
    num_stages: int = 3,
    b_size: int = 128,
) -> None:
    """AOT-compile the fused factored-LSTM step and emit a C header.

    Kc = K_hh + R (merged contraction). Emits <file_name>.{h,o} with an 11-argument
    wrapper in __call__ order: mA[B,Kc], mB_i/f/g/o[H,Kc], mBias_i/f/g/o[H], mC_c[B,H],
    mH_out[B,H]. B (M) is dynamic at runtime; H, Kc and the tile config are baked.
    """
    import torch
    import cutlass.torch as cutlass_torch

    Kc = K_hh + R

    def _cpt(mode0, mode1, dtype):
        """Row-major [mode0, mode1, 1] trace tensor with dynamic-shape marks.
        Local variant of create_and_permute_tensor that also accepts Float32."""
        torch_dtype = cutlass_torch.dtype(dtype)
        t = torch.zeros((1, mode0, mode1), dtype=torch_dtype).permute(1, 2, 0).cuda()
        ct = (
            from_dlpack(t, assumed_align=16)
            .mark_layout_dynamic(leading_dim=1)
            .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1),
                                        divisibility=(128 // dtype.width))
        )
        return ct

    fake_a = _cpt(b_size, Kc, ab_dtype)
    fake_bi = _cpt(H, Kc, ab_dtype)
    fake_bf = _cpt(H, Kc, ab_dtype)
    fake_bg = _cpt(H, Kc, ab_dtype)
    fake_bo = _cpt(H, Kc, ab_dtype)
    fake_bias_i = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_f = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_g = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_o = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_c = _cpt(b_size, H, cutlass.Float32)
    fake_h = _cpt(b_size, H, out_dtype)

    lstm = TensorOpFactoredLstmI8(ab_dtype, out_dtype, acc_dtype, atom_layout_mnk,
                                  bm=bm, bn=bn, bk=bk, num_stages=num_stages)
    print(f"Compiling TensorOpFactoredLstmI8  H={H} Kc={Kc}  tile={bm}x{bn}x{bk}  atom={atom_layout_mnk}")
    compiled = cute.compile(
        lstm, fake_a, fake_bi, fake_bf, fake_bg, fake_bo,
        fake_bias_i, fake_bias_f, fake_bias_g, fake_bias_o, fake_c, fake_h,
    )
    print(f"Exporting to {file_path}/{file_name}.h ...")
    compiled.export_to_c(file_path=file_path, file_name=file_name, function_prefix=function_prefix)
    print("Export complete!")


__all__ = ["TensorOpFactoredLstmI8", "export_factored_lstm_i8", "H_OUT_SCALE"]
