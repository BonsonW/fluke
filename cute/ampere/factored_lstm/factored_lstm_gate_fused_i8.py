"""Fused-interleaved INT8 gate factored-LSTM step for Ampere (A100, sm80).

The CUTLASS-C++ prototype (fluke/cuda/fused_cutlass.cu, 30us gate / 37us step at
N=2048) proven design, ported to the CuTe DSL:

  * ONE plain GEMM instead of 4: gate weights are INTERLEAVED host-side in 8-column
    blocks -- GEMM column n = B32*32 + g*8 + p -> gate g of hidden channel B32*8 + p.
    One accumulator set -> normal large tiles (bN=128), killing the 4-accumulator
    register pressure that forced bN=32 in factored_lstm_gate_i8.py (~50us).
  * The proven multistage cp.async pipelined mainloop (from gemm_i8_quant /
    factored_lstm_gate_i8) UNCHANGED, minus the 4x B duplication.
  * Fused LSTM epilogue, smem-staged (the [B,4H] int32 C is NEVER written to gmem):
      pass 1: each acc element's (row,col) comes from partition_C of an identity
              tensor (no fragment-layout math); raw int32 -> sRaw[gate][row][chan].
      pass 2: threads sweep [bM, bN/4] channels: read the 4 gate accs from smem,
              dequant (scale_a[m] * scale_w[g][ch] + bias), sighard/tanh cell update,
              write int8 h + f32 cell -- both coalesced (channel-contiguous).

  raw_g = acc_g * scale_a[m] * scale_w[g][n] + bias_g
  i,f,o = clamp(0.2*raw+0.5,0,1); g = clamp(raw,-1,1)
  c'    = f*c + i*g ;  h = o*tanh(c') -> int8 @ 1/127

RESULTS (A100, N=2048, warm, test_factored_lstm_gate_fused_i8.py):
  correct (h within one rounding quantum of the int8-recompute ref; cell ~exact;
  quantization-vs-fp16 max_abs ~0.012 = same as gate_i8).  Sweep winner
  bm=64 bn=64 bk=32 st=3 atom=(2,2,1): 48.3us.  vs gate_i8 4-acc bN=32: 43.8us;
  vs the CUDA/CUTLASS fused prototype (fused_cutlass.cu): 33us.  ncu: 80 regs (no
  spill -- the interleave DID kill the 4-acc pressure), DRAM-write ~0 (C never
  materialized), wall = long_scoreboard on the pass-2 scalar epilogue gmem traffic
  (partially cured by the cell cp.async prefetch: 53->48us) + tiny-tile mainloop.
  The DSL epilogue costs ~15us more than the CUDA hand epilogue (scalar smem/gmem
  ops vs vectorized staged writeout); large tiles (128x128, CUDA's winner) regress
  in the DSL (128-int32 fragment + pass-1 staging exceeds what ptxas handles well).
  NOTE mC_c must be passed with a STATIC layout (from_dlpack(assumed_align=16), no
  mark_layout_dynamic) or the cell-prefetch cp.async fails its alignment proof.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda_driver

from gemm_i8_quant import TensorOpGemmI8

H_OUT_SCALE = 1.0 / 127.0


def interleave_gate_weights(W_i8_list):
    """Host-side: [4 x (H,Kc) int8 torch tensors] -> (4H,Kc) interleaved.
    GEMM col n = B32*32 + g*8 + p  ->  gate g of channel B32*8 + p."""
    import torch
    Hn, Kc = W_i8_list[0].shape
    out = torch.empty(4 * Hn, Kc, dtype=W_i8_list[0].dtype, device=W_i8_list[0].device)
    # rows of gate g land at (ch//8)*32 + g*8 + (ch%8)
    ch = torch.arange(Hn, device=W_i8_list[0].device)
    for g in range(4):
        n = (ch // 8) * 32 + g * 8 + (ch % 8)
        out[n] = W_i8_list[g]
    return out


class TensorOpFactoredLstmGateFusedI8(TensorOpGemmI8):
    """One interleaved int8 gate GEMM (pipelined) + smem-staged fused LSTM epilogue."""

    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        bm: int = 128,
        bn: int = 128,
        bk: int = 64,
        num_stages: int = 3,
    ):
        self.a_dtype = ab_dtype
        self.b_dtype = ab_dtype
        self.ab_dtype = ab_dtype
        self.out_dtype = out_dtype
        self.c_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.use_k32 = True
        self.cta_tiler = (bm, bn, bk)
        self.num_stages = num_stages
        self.atom_layout_mnk = atom_layout_mnk
        atom_lay_M, atom_lay_N, atom_lay_K = atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32
        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 32)
        mmaM, mmaN, mmaK = self.mma_inst_shape
        assert self.bM % (atom_lay_M * mmaM) == 0
        assert self.bN % (atom_lay_N * mmaN) == 0
        assert self.bN % 32 == 0, "bN must cover whole 32-col (8-channel) gate blocks"
        assert atom_lay_K == 1
        assert self.bK % mmaK == 0
        assert self.num_stages >= 3

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,        # [B, Kc]   int8  (concat hh_down|x_down)
        mBint: cute.Tensor,     # [4H, Kc]  int8  INTERLEAVED gate weights
        mScaleA: cute.Tensor,   # [B, 1]    f32   per-row dequant scale
        mScaleW_i: cute.Tensor, mScaleW_f: cute.Tensor,   # [H,1] f32 per-channel
        mScaleW_g: cute.Tensor, mScaleW_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor,       # [H,1] f32
        mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mC_c: cute.Tensor,      # [B, H]   f32  cell, read+write in place
        mH_out: cute.Tensor,    # [B, H]   int8 hidden out (1/127)
        stream: cuda_driver.CUstream = None,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mBint)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mH_out)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages))
        sB_layout = self._make_smem_layout_AB(
            mBint.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages))
        # int32 accumulator staging in the NATURAL [bM, bN] fragment layout: stored via a
        # coalesced partition_C autovec_copy (vs the old scattered de-interleave write);
        # pass-2 de-interleaves the gate columns on READ.
        sRaw_layout = cute.make_layout((self.bM, self.bN), stride=(self.bN, 1))
        # cell tile prefetch staging [bM, bN/4] f32 (+ per-row scale_a [bM])
        sCell_layout = cute.make_layout((self.bM, self.bN // 4), stride=(self.bN // 4, 1))
        sSa_layout = cute.make_layout(self.bM)
        # int8 hidden-out staging [bM, bN/4] -> packed int32 coalesced writeout.
        sH_layout = cute.make_layout((self.bM, self.bN // 4), stride=(self.bN // 4, 1))
        # sRaw (epilogue acc staging, bM*bN int32) ALIASES the mainloop sA/sB smem when it
        # fits (dead after the mainloop) -> big occupancy win (drops ~16KB/block). If the
        # A/B tiles are too small (e.g. bK=32), fall back to a dedicated allocation.
        ab_bytes = (cute.size_in_bytes(mA.element_type, sA_layout)
                    + cute.size_in_bytes(mBint.element_type, sB_layout))
        raw_bytes = cute.size_in_bytes(cutlass.Int32, sRaw_layout)
        self._alias_raw = ab_bytes >= raw_bytes
        smem_size = (
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mBint.element_type, sB_layout)
            + cute.size_in_bytes(cutlass.Float32, sCell_layout)
            + cute.size_in_bytes(cutlass.Float32, sSa_layout)
            + cute.size_in_bytes(cutlass.Int8, sH_layout)
            + (0 if self._alias_raw else raw_bytes)
        )
        # cp.async tiled copy for the cell tile (f32, 128-bit)
        atom_cell_cp = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32, num_bits_per_copy=128)
        thr_k_c = (self.bN // 4) // 4
        thr_m_c = self.num_threads // thr_k_c
        tiled_copy_cell = cute.make_tiled_copy_tv(
            atom_cell_cp,
            cute.make_layout((thr_m_c, thr_k_c), stride=(thr_k_c, 1)),
            cute.make_layout((1, 4)))

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type, num_bits_per_copy=ab_copy_bits)
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mBint.element_type, self.b_major_mode, ab_copy_bits)

        op = cute.nvgpu.warp.MmaI8Op(self.a_dtype, self.b_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        NG = cute.size(mBint, mode=[0])                     # 4H
        M = cute.size(mA, mode=[0])
        grid_dim = (cute.ceil_div(M, self.bM), cute.ceil_div(NG, self.bN), 1)
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rr_grid = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            1,
        )

        _launcher = self.kernel(
            mA, mBint, mScaleA,
            mScaleW_i, mScaleW_f, mScaleW_g, mScaleW_o,
            mBias_i, mBias_f, mBias_g, mBias_o,
            mC_c, mH_out,
            sA_layout, sB_layout, sRaw_layout, sCell_layout, sSa_layout, sH_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_cell, tiled_mma,
            raster_factor,
        )
        launch_kwargs = dict(grid=rr_grid, block=[self.num_threads, 1, 1], smem=smem_size)
        if cutlass.const_expr(stream is not None):
            launch_kwargs["stream"] = stream
        _launcher.launch(**launch_kwargs)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor, mBint: cute.Tensor, mScaleA: cute.Tensor,
        mScaleW_i: cute.Tensor, mScaleW_f: cute.Tensor,
        mScaleW_g: cute.Tensor, mScaleW_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor,
        mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mC_c: cute.Tensor, mH_out: cute.Tensor,
        sA_layout: cute.ComposedLayout, sB_layout: cute.ComposedLayout,
        sRaw_layout: cute.Layout, sCell_layout: cute.Layout, sSa_layout: cute.Layout,
        sH_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy, tiled_copy_B: cute.TiledCopy,
        tiled_copy_cell: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        NG = cute.size(mBint, mode=[0])
        M = cute.size(mA, mode=[0])
        Hn = cute.size(mC_c, mode=[1])
        grid_dim = (cute.ceil_div(M, self.bM), cute.ceil_div(NG, self.bN), 1)
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            gA = cute.local_tile(mA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            gB = cute.local_tile(mBint[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))

            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mBint.layout.shape)
            cA = cute.local_tile(mcA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            cB = cute.local_tile(mcB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
            sB = smem.allocate_tensor(mBint.element_type, sB_layout, 16)
            sCell = smem.allocate_tensor(cutlass.Float32, sCell_layout, 16)
            sSa = smem.allocate_tensor(cutlass.Float32, sSa_layout, 16)
            sH = smem.allocate_tensor(cutlass.Int8, sH_layout, 16)
            # epilogue acc staging: alias the mainloop A/B smem (contiguous, dead after
            # the final cp_async_wait/syncthreads) when it fits; else a dedicated tile.
            if cutlass.const_expr(self._alias_raw):
                sRaw = cute.make_tensor(
                    cute.recast_ptr(sA.iterator, dtype=cutlass.Int32), sRaw_layout)
            else:
                sRaw = smem.allocate_tensor(cutlass.Int32, sRaw_layout, 16)

            # ---- prefetch the cell tile + per-row scale via cp.async as the FIRST
            # commit group (completes during the mainloop; wait_group(0) precedes use).
            CH_TB = cutlass.const_expr(self.bN // 4)
            row0 = offset_tile_x * cutlass.Int32(self.bM)
            ch0 = offset_tile_y * cutlass.Int32(CH_TB)
            gCell = cute.local_tile(mC_c[None, None, 0], tiler=(self.bM, CH_TB),
                                    coord=(offset_tile_x, offset_tile_y))
            gCell = cute.make_tensor(gCell.iterator.align(16), gCell.layout)
            thr_cell = tiled_copy_cell.get_slice(tidx)
            cute.copy(tiled_copy_cell, thr_cell.partition_S(gCell), thr_cell.partition_D(sCell))
            # per-row scale: bM f32 = small; scalar load by the first bM threads (once).
            if tidx < self.bM:
                sSa[tidx] = mScaleA[row0 + tidx, 0]
            cute.arch.cp_async_commit_group()

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0)), cutlass.Boolean)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0)), cutlass.Boolean)
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mBint.shape[0])

            tAsA.fill(0)
            tBsB.fill(0)
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
                    cute.copy(tiled_copy_B, tBgB[None, None, k, k_tile_index],
                              tBsB[None, None, k, 0], pred=tBpB[None, None, k])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0); tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                          tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                          tBsB[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            # identity coords for the epilogue (LOCAL (row,col) within the CTA tile);
            # fragment_C needs a REAL memref -> use a same-shape view over mC_c's iterator.
            cC = cute.make_identity_tensor((self.bM, self.bN))
            tCcC = thr_mma.partition_C(cC)
            c_ref = cute.make_tensor(mC_c.iterator,
                                     cute.make_layout((self.bM, self.bN), stride=(self.bN, 1)))
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(thr_mma.partition_C(c_ref))
            tCrC.fill(0)

            num_vals = cute.size(tCrC, mode=[0])
            num_mma_m = cute.size(tCrC, mode=[1])
            num_mma_n = cute.size(tCrC, mode=[2])

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mA.element_type)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mBint.element_type)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1
            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()
                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                              tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next],
                              tCrB_copy_view[None, None, k_block_next])
                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                                      tAsA[None, None, None, smem_pipe_write], pred=tApA)
                            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                                      tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                              tCrB[None, None, k_block], tCrC)
                    if k_block == 0:
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ---------------- fused LSTM epilogue (smem-staged) ----------------
            # pass 1: int32 acc -> sRaw in NATURAL [bM, bN] layout via a coalesced
            # partition_C autovec_copy (no scattered de-interleave write / bank storms).
            tCsRaw = thr_mma.partition_C(sRaw)
            cute.autovec_copy(tCrC, tCsRaw)
            cute.arch.sync_threads()

            # pass 2: per (row, channel): read 4 gate accs from smem, dequant, LSTM
            # update, write int8 h + f32 cell (channel-contiguous -> coalesced).
            def clamp(x, lo, hi):
                return 0.0 - cute.arch.fmax(0.0 - cute.arch.fmax(x, lo), 0.0 - hi)

            mC_flat = cute.make_tensor(mC_c.iterator, cute.make_layout(M * Hn))
            sw_i = cute.make_tensor(mScaleW_i.iterator, cute.make_layout(Hn))
            sw_f = cute.make_tensor(mScaleW_f.iterator, cute.make_layout(Hn))
            sw_g = cute.make_tensor(mScaleW_g.iterator, cute.make_layout(Hn))
            sw_o = cute.make_tensor(mScaleW_o.iterator, cute.make_layout(Hn))
            bs_i = cute.make_tensor(mBias_i.iterator, cute.make_layout(Hn))
            bs_f = cute.make_tensor(mBias_f.iterator, cute.make_layout(Hn))
            bs_g = cute.make_tensor(mBias_g.iterator, cute.make_layout(Hn))
            bs_o = cute.make_tensor(mBias_o.iterator, cute.make_layout(Hn))
            nt = cutlass.const_expr(self.num_threads)
            n_iter = cutlass.const_expr((self.bM * (self.bN // 4)) // self.num_threads)
            for j in cutlass.range_constexpr(n_iter):
                idx = tidx + j * nt
                r = idx // CH_TB
                c = idx % CH_TB
                gch = ch0 + c
                # de-interleave the 4 gate columns from the natural [bM, bN] acc tile:
                # channel c lives in 8-block (c//8); its i/f/g/o are 8 cols apart.
                base = (c // 8) * 32 + (c % 8)
                acc_i = sRaw[r, base + 0].to(cutlass.Float32)
                acc_f = sRaw[r, base + 8].to(cutlass.Float32)
                acc_g = sRaw[r, base + 16].to(cutlass.Float32)
                acc_o = sRaw[r, base + 24].to(cutlass.Float32)
                sa = sSa[r].to(cutlass.Float32)
                raw_i = acc_i * sa * sw_i[gch].to(cutlass.Float32) + bs_i[gch].to(cutlass.Float32)
                raw_f = acc_f * sa * sw_f[gch].to(cutlass.Float32) + bs_f[gch].to(cutlass.Float32)
                raw_g = acc_g * sa * sw_g[gch].to(cutlass.Float32) + bs_g[gch].to(cutlass.Float32)
                raw_o = acc_o * sa * sw_o[gch].to(cutlass.Float32) + bs_o[gch].to(cutlass.Float32)
                i_a = clamp(raw_i * 0.2 + 0.5, 0.0, 1.0)
                f_a = clamp(raw_f * 0.2 + 0.5, 0.0, 1.0)
                o_a = clamp(raw_o * 0.2 + 0.5, 0.0, 1.0)
                g_a = clamp(raw_g, -1.0, 1.0)
                c_new = f_a * sCell[r, c].to(cutlass.Float32) + i_a * g_a
                mC_flat[(row0 + r) * Hn + gch] = c_new          # cell: already coalesced
                h_new = o_a * cute.math.tanh(c_new)
                hs = h_new * 127.0
                half = clamp(hs * 1e30, -0.5, 0.5)
                sH[r, c] = (hs + half).to(self.out_dtype)       # stage h -> smem
            cute.arch.sync_threads()

            # pass 3: packed int32 coalesced writeout of the int8 hidden tile
            # (4 channel-contiguous int8 -> one 32-bit store; ch0 & Hn are 4-aligned).
            CH4 = cutlass.const_expr(CH_TB // 4)
            mH_i32 = cute.make_tensor(
                cute.recast_ptr(mH_out.iterator, dtype=cutlass.Int32),
                cute.make_layout(M * Hn // 4))
            sH_i32 = cute.make_tensor(
                cute.recast_ptr(sH.iterator, dtype=cutlass.Int32),
                cute.make_layout(self.bM * CH4))
            nword = cutlass.const_expr(self.bM * CH4)
            witer = cutlass.const_expr((nword + nt - 1) // nt)
            for j in cutlass.range_constexpr(witer):
                w = tidx + j * nt
                if w < nword:
                    r = w // CH4
                    c4 = w % CH4
                    mH_i32[((row0 + r) * Hn + ch0) // 4 + c4] = sH_i32[w]
        return


__all__ = ["TensorOpFactoredLstmGateFusedI8", "interleave_gate_weights", "H_OUT_SCALE"]
