"""Fused INT8 GEMM + rotary embedding for NVIDIA Ampere (A100, sm80).

CUDA INT8 port of fly/rdna_fp8_gemm_rotary.py.

  C[M, N] = rotary_embed(A[M,K] @ B[N,K]^T)

A (M,K) int8 K-major, B (N,K) int8 K-major (nn.Linear weight). Output fp16.
Scales: per-token scale_a[M], per-channel scale_b[N].

N = 3 * nhead * head_dim  (QKV concatenated):
  cols [0,                 nhead*head_dim): Q  — rotary applied
  cols [nhead*head_dim,  2*nhead*head_dim): K  — rotary applied
  cols [2*nhead*head_dim,                N): V  — passthrough

Rotary (matches openfish / the RDNA kernel):
  sincos_width = rotary_dim // 2
  for k in [0, sincos_width):
    x0 = head col k ; x1 = head col k+sincos_width
    out[k]            = x0*cos[k] - x1*sin[k]
    out[k+sincos_width] = x0*sin[k] + x1*cos[k]
sin/cos: [seqlen, sincos_width] fp32; row m uses seq = m % seqlen.

The rotary "companion" column (sincos_width away) is held by a different warp under
the (2,2,1) MMA layout, so the epilogue rounds the scaled tile through shared
memory. bN is pinned to head_dim so each N-tile is exactly one head and the
companion column is always within the tile.
"""

from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from gemm_i8_quant import (
    TensorOpGemmI8,
    _install_local_mma_i8_op,
    create_and_permute_tensor,
)

_install_local_mma_i8_op()


class TensorOpGemmI8Rotary(TensorOpGemmI8):
    """INT8 GEMM with fused rotary-embedding epilogue. bN must equal head_dim."""

    def __init__(self, *args, nhead, head_dim, rotary_dim, seqlen, **kwargs):
        super().__init__(*args, **kwargs)
        self.nhead = nhead
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.sincos_width = rotary_dim // 2
        self.seqlen = seqlen
        self.qk_cols = 2 * nhead * head_dim
        # bN must be a multiple of head_dim so each N-tile spans whole heads and the
        # rotary companion column (sincos_width away, within a head) stays in-tile.
        assert self.bN % head_dim == 0, f"bN({self.bN}) must be a multiple of head_dim({head_dim})"
        # tile must not straddle the Q/K|V boundary so Q/K-vs-V is uniform per tile
        assert self.qk_cols % self.bN == 0, "bN must divide 2*nhead*head_dim (Q/K region)"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mSin: cute.Tensor,
        mCos: cute.Tensor,
        seqlen: cutlass.Int32,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        # Epilogue-C staging layout (single stage, plain+padded). The rotary result
        # is written to smem in the scattered MMA-C fragment layout, then re-read
        # with a coalesced thread map for 128-bit global stores.
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )
        # A/B smem is reused for C in the epilogue, so the block only needs the
        # larger of the two — the C staging costs no extra footprint (occupancy).
        smem_size = max(
            cute.size_in_bytes(mA.element_type, sA_layout)
            + cute.size_in_bytes(mB.element_type, sB_layout),
            cute.size_in_bytes(mC.element_type, sC_layout),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mA.element_type, num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits)
        # Synchronous 128-bit universal copy for the coalesced smem->gmem store.
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mC.element_type, num_bits_per_copy=128,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, 128)

        op = cute.nvgpu.warp.MmaI8Op(
            self.a_dtype, self.b_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
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
            mA, mB, mC, mScaleA, mScaleB, mSin, mCos,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C, tiled_mma, raster_factor, seqlen,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB: cute.Tensor,
        mSin: cute.Tensor,
        mCos: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            gA = cute.local_tile(mA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            gB = cute.local_tile(mB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            gC = cute.local_tile(mC[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, 1, None))

            gScaleA = cute.local_tile(mScaleA[None, bidz], tiler=(self.bM,), coord=(offset_tile_x,))
            gScaleB = cute.local_tile(mScaleB[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))

            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB = cute.domain_offset((0, residual_k, 0), gB)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB.layout.shape)
            cA = cute.local_tile(mcA[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            cB = cute.local_tile(mcB[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(None, 1, 1))
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # A/B/C share one smem arena: the epilogue overwrites the A/B tiles
            # with the fp16 C staging tile (union), so no extra footprint is requested.
            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16]
                b: cute.struct.Align[
                    cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)], 16]

            @cute.struct
            class SharedStorageC:
                c: cute.struct.Align[
                    cute.struct.MemRange[mC.element_type, cute.cosize(sC_layout)], 16]

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(
                max(SharedStorageAB.size_in_bytes(), SharedStorageC.size_in_bytes()),
                byte_alignment=16,
            )
            sA = SharedStorageAB(storage).a.get_tensor(sA_layout)
            sB = SharedStorageAB(storage).b.get_tensor(sB_layout)
            sC = SharedStorageC(storage).c.get_tensor(sC_layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB = thr_copy_B.partition_S(gB)
            tBsB = thr_copy_B.partition_D(sB)
            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0)),
                cutlass.Boolean)
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                    stride=(cute.size(tBsB, mode=[1]), 1, 0)),
                cutlass.Boolean)
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

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
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                          tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                          tBsB[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCgC = thr_mma.partition_C(gC)
            tCsC = thr_mma.partition_C(sC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0)

            num_vals = int(cute.size(tCrC, mode=[0]))
            num_mma_m = int(cute.size(tCrC, mode=[1]))
            num_mma_n = int(cute.size(tCrC, mode=[2]))

            gScaleA_2d = cute.make_tensor(gScaleA.iterator, cute.make_layout((self.bM, self.bN), stride=(1, 0)))
            gScaleB_2d = cute.make_tensor(gScaleB.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1)))
            tCgScaleA = thr_mma.partition_C(gScaleA_2d)
            tCgScaleB = thr_mma.partition_C(gScaleB_2d)
            rScaleA = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_m, 1, 0)), cutlass.Float32)
            rScaleB = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    rScaleA[i, m, 0] = tCgScaleA[i, m, 0].to(cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rScaleB[i, 0, n] = tCgScaleB[i, 0, n].to(cutlass.Float32)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mA.element_type)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mB.element_type)
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
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                                      tAsA[None, None, None, smem_pipe_write], pred=tApA)

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index],
                                      tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////////
            # Rotary epilogue, entirely in registers (no smem round-trip).
            #
            # With the (waves_m, waves_n) MMA layout each MMA_N block covers
            # cols_per_mma_n contiguous columns, so the rotary companion column
            # (sincos_width away, in the same head) is held by the SAME thread at
            # a compile-time-known MMA_N index. companion_step = sincos_width //
            # cols_per_mma_n. A block is entirely in the first or second rotary
            # half (compile-time), so the rotation formula is selected at compile
            # time and the only runtime work is the sin/cos table lookup.
            # ///////////////////////////////////////////////////////////////////
            tCcC = thr_mma.partition_C(cute.make_identity_tensor((self.bM, self.bN)))
            cols_per_mma_n = self.bN // num_mma_n
            companion_step = self.sincos_width // cols_per_mma_n
            assert self.sincos_width % cols_per_mma_n == 0, \
                "sincos_width must be a multiple of cols_per_mma_n for register companion"

            tile_n0 = offset_tile_y * self.bN
            qk_f = cute.elem_less(
                cutlass.Int32(tile_n0), cutlass.Int32(self.qk_cols)
            ).to(cutlass.Float32)
            row0 = cutlass.Int32(offset_tile_x * self.bM)
            rh = self.sincos_width

            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            for i in cutlass.range_constexpr(num_vals):
                for m in cutlass.range_constexpr(num_mma_m):
                    for n in cutlass.range_constexpr(num_mma_n):
                        # compile-time: which rotary half this MMA_N block sits in
                        block_pos = (n * cols_per_mma_n) % self.head_dim
                        is_first = block_pos < rh

                        r = cutlass.Int32(tCcC[i, m, n][0])
                        c = cutlass.Int32(tCcC[i, m, n][1])
                        seq = (row0 + r) % seqlen
                        pos = c % cutlass.Int32(self.head_dim)
                        self_v = (tCrC[i, m, n].to(cutlass.Float32)
                                  * rScaleA[i, m, 0] * rScaleB[i, 0, n])

                        if cutlass.const_expr(is_first):   # self=x0, companion=x1
                            comp_n = n + companion_step
                            comp_v = (tCrC[i, m, comp_n].to(cutlass.Float32)
                                      * rScaleA[i, m, 0] * rScaleB[i, 0, comp_n])
                            rot_idx = pos
                            sin_v = mSin[seq, rot_idx].to(cutlass.Float32)
                            cos_v = mCos[seq, rot_idx].to(cutlass.Float32)
                            rotated = self_v * cos_v - comp_v * sin_v
                        else:                              # self=x1, companion=x0
                            comp_n = n - companion_step
                            comp_v = (tCrC[i, m, comp_n].to(cutlass.Float32)
                                      * rScaleA[i, m, 0] * rScaleB[i, 0, comp_n])
                            rot_idx = pos - cutlass.Int32(rh)
                            sin_v = mSin[seq, rot_idx].to(cutlass.Float32)
                            cos_v = mCos[seq, rot_idx].to(cutlass.Float32)
                            rotated = comp_v * sin_v + self_v * cos_v

                        # passthrough for V (and columns outside rotary span)
                        inrot_f = cute.elem_less(pos, cutlass.Int32(self.rotary_dim)).to(cutlass.Float32)
                        do_rot_f = qk_f * inrot_f
                        out_v = do_rot_f * rotated + (1.0 - do_rot_f) * self_v
                        tCrD[i, m, n] = out_v.to(self.c_dtype)

            # Stage the fp16 result through smem to convert the scattered MMA-C
            # fragment layout (the N-doubled permutation stores 8-column-strided
            # values per thread) into contiguous 128-bit global stores. The direct
            # register->gmem write (autovec_copy tCrD->tCgC) only fills 16 of every
            # 32 store bytes per sector; going via smem coalesces the store.
            cute.autovec_copy(tCrD, tCsC)
            cute.arch.sync_threads()
            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            # Predicate the store on the M/N extents (tiles may overhang the matrix).
            cCstore = cute.local_tile(
                cute.make_identity_tensor(mC.layout.shape)[None, None, bidz],
                tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None),
            )
            tCcCstore = thr_copy_C.partition_S(cCstore)
            tCpC = cute.make_rmem_tensor(
                cute.make_layout(
                    (tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]),
                     cute.size(tCgC_epilogue, mode=[2])),
                    stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tCpC.shape[0]):
                for m in range(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(
                        tCcCstore[(0, rest_v), m, 0][0], mC.shape[0]
                    )
            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcCstore[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n],
                                  tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])

        return

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        """Epilogue-C smem staging layout (single stage), plain + padded.

        The int8 MMA's doubled-N C partition makes autovec_copy's right_inverse
        reject a composed (swizzled) layout, so we stage through a plain tile. A
        plain fp16 row whose width is a multiple of 32 banks makes the MMA-fragment
        register->smem write hit a heavy bank conflict; padding the major stride by
        8 elems (16 B / 4 banks) decorrelates consecutive rows and removes it. The
        pad costs a couple KB of the C tile, still inside the A/B smem arena it
        unions over, so occupancy is unchanged. (Ported from dual_gemm_i8_silu.)
        """
        pad = 8
        if major_mode == utils.LayoutEnum.ROW_MAJOR:
            return cute.make_layout(smem_tiler, stride=(smem_tiler[1] + pad, 1))
        return cute.make_layout(smem_tiler, stride=(1, smem_tiler[0] + pad))

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        """Coalesced smem->gmem thread map for the C store. Mirrors gemm_f16."""
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bN) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


# =============================================================================
# AOT export
# =============================================================================
def export_gemm_i8_rotary(
    nhead: int,
    head_dim: int,
    rotary_dim: int,
    seqlen: int,
    a_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    b_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Int32,
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    file_path: str = "./artifacts",
    file_name: str = "gemm_i8_rotary",
    function_prefix: str = "gemm_i8_rotary",
    use_k32: bool = True,
    bm: int = 128,
    bn: int = 128,
    num_stages: int = 3,
    m_size: int = 128,
    k_size: int = 128,
    l_size: int = 1,
) -> None:
    """AOT-compile the fused INT8 GEMM + rotary kernel and emit a C header.

    Computes C[M,N] = rotary(A@B^T) for QKV, fp16 output. N = 3*nhead*head_dim.
    nhead/head_dim/rotary_dim and the tile/atom config are baked at export; M and
    the sequence length are dynamic at runtime. `seqlen` here sizes the baked
    sin/cos table (the MAX supported sequence length); the actual seqlen is passed
    as a runtime scalar at launch and may be any value in [1, seqlen].
    bn must be a multiple of head_dim.

    Emits <file_path>/<file_name>.h with structs for the 7 tensor arguments in
    __call__ order (mA, mB, mC, mScaleA, mScaleB, mSin, mCos) plus a trailing
    runtime int32 `seqlen` scalar argument.
    """
    n_size = 3 * nhead * head_dim
    sincos_width = rotary_dim // 2

    fake_a, _ = create_and_permute_tensor(l_size, m_size, k_size, False, a_dtype)
    fake_b, _ = create_and_permute_tensor(l_size, n_size, k_size, False, b_dtype)
    fake_c, _ = create_and_permute_tensor(l_size, m_size, n_size, False, c_dtype)
    fake_scale_a = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m_size, l_size), assumed_align=16)
    fake_scale_b = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_size, l_size), assumed_align=16)
    # sin/cos are a row-major [seqlen, sincos_width] table (contiguous, stride
    # (sincos_width, 1)). The table extent is baked to `seqlen` (the MAX supported
    # sequence length, e.g. 2048); the ACTUAL sequence length is a separate runtime
    # scalar (`fake_seqlen` below) so one export serves any seqlen in [1, seqlen].
    # make_fake_compact_tensor defaults to COLUMN-MAJOR (stride_order left-to-right
    # -> stride (1, seqlen)), which would bake the wrong strides into
    # mSin[seq, rot_idx] and read garbage/out-of-bounds over the AOT C ABI. Pin
    # row-major with stride_order=(1, 0). The baked extent never enters address
    # math (only stride0=sincos_width does), so a smaller runtime seqlen is fine.
    fake_sin = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (seqlen, sincos_width), stride_order=(1, 0), assumed_align=16)
    fake_cos = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (seqlen, sincos_width), stride_order=(1, 0), assumed_align=16)
    # Runtime seqlen scalar: pass a cutlass.Int32 (NOT a Python int) so it stays a
    # dynamic C-ABI parameter instead of being folded into the modulo. The value
    # here is only the trace placeholder; the caller sets the real seqlen at launch.
    fake_seqlen = cutlass.Int32(seqlen)

    gemm = TensorOpGemmI8Rotary(
        a_dtype, b_dtype, c_dtype, acc_dtype,
        atom_layout_mnk, use_k32, bm, bn=bn, num_stages=num_stages,
        nhead=nhead, head_dim=head_dim, rotary_dim=rotary_dim, seqlen=seqlen,
    )
    print(f"Compiling TensorOpGemmI8Rotary  N={n_size} K={k_size}  tile={bm}x{bn}x64  "
          f"atom={atom_layout_mnk}  nhead={nhead} head_dim={head_dim} "
          f"rotary_dim={rotary_dim} seqlen(max/table)={seqlen} (runtime-dynamic)")
    compiled = cute.compile(
        gemm, fake_a, fake_b, fake_c, fake_scale_a, fake_scale_b, fake_sin, fake_cos,
        fake_seqlen,
    )
    print(f"Exporting to {file_path}/{file_name}.h ...")
    compiled.export_to_c(
        file_path=file_path, file_name=file_name, function_prefix=function_prefix,
    )
    print("Export complete!")


# The host-side torch reference lives in the universal test (cute/test_rotary.py),
# not here — this module is just the kernel + its AOT export.
__all__ = [
    "TensorOpGemmI8Rotary", "export_gemm_i8_rotary", "create_and_permute_tensor",
]
