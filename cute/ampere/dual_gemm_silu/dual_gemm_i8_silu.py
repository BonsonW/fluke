"""Fused dual INT8 GEMM + silu_mul for NVIDIA Ampere (A100, sm80).

CUDA INT8 port of fly/rdna_fp8_dual_gemm_silu.py.

  out[M, N] = silu(A[M,K] @ B_gate[N,K]^T) * (A[M,K] @ B_up[N,K]^T)

A, B_gate, B_up are int8 (K-major, i.e. A is (M,K) row-major, B is (N,K) row-major
weight as in nn.Linear). Output is fp16.
Scales: per-token scale_a[M] for A, per-channel scale_b_gate[N] / scale_b_up[N].

Both GEMMs share operand A. The two accumulators (gate, up) are held in registers
simultaneously and the SiLU-gate*up activation is fused into the epilogue, so the
gate/up intermediates never touch DRAM.

Implementation mirrors ampere_gemm_i8_quant_rmem.py with a second B operand threaded
through the smem pipeline, the register pipeline, and the MMA mainloop.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

from gemm_i8_quant import (
    TensorOpGemmI8,
    _install_local_mma_i8_op,
    create_and_permute_tensor,
)

_install_local_mma_i8_op()


class TensorOpDualGemmI8Silu(TensorOpGemmI8):
    """Dual INT8 GEMM with fused SiLU(gate) * up epilogue.

    Inherits __init__ and the helper methods (_make_smem_layout_AB,
    _make_gmem_tiled_copy_AB, raster_tile) from TensorOpGemmI8. Note the
    register-accumulator pressure is doubled (gate + up), so bN should be
    smaller than the single-GEMM kernel. For the deployed K=512 shape bN=32
    (atom (2,2,1), 3 stages) is the more robust default: it ties bN=64 at large M
    (~215 TOPS at M=131072) and is clearly faster at small M (halving the N tile
    halves the doubled C accumulator, raising occupancy when there is little work
    to hide latency). Run autotune_dual.py to tune a specific shape.
    Constraint: bN >= atom_N * mmaN * 2 (=32 for atom_N=2, =64 for atom_N=4);
    smaller bN with a wider atom hangs compilation.
    """

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB_gate: cute.Tensor,
        mB_up: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB_gate: cute.Tensor,
        mScaleB_up: cute.Tensor,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB_gate)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB_gate.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        # Epilogue staging layout (no stages): the fp16 output tile is written to
        # smem, then re-read with a coalesced thread map for 128-bit global stores.
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[1]),
        )

        # A/B smem is reused for C in the epilogue, so the block only needs the
        # larger of the two — the C staging costs no extra footprint (occupancy).
        smem_size = max(
            cute.size_in_bytes(mA.element_type, sA_layout)
            + 2 * cute.size_in_bytes(mB_gate.element_type, sB_layout),
            cute.size_in_bytes(mC.element_type, sC_layout),
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB_gate.element_type, self.b_major_mode, ab_copy_bits
        )
        # Synchronous 128-bit universal copy for the coalesced smem->gmem store.
        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mC.element_type, num_bits_per_copy=c_copy_bits,
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits
        )

        op = cute.nvgpu.warp.MmaI8Op(
            self.a_dtype, self.b_dtype, self.acc_dtype, self.mma_inst_shape
        )
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
            mA, mB_gate, mB_up, mC,
            mScaleA, mScaleB_gate, mScaleB_up,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C, tiled_mma,
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
        mB_gate: cute.Tensor,
        mB_up: cute.Tensor,
        mC: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleB_gate: cute.Tensor,
        mScaleB_up: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(
            bidx, bidy, rasterization_factor
        )
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            gA = cute.local_tile(
                mA[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, None, 1),
            )
            gB_gate = cute.local_tile(
                mB_gate[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(None, 1, 1),
            )
            gB_up = cute.local_tile(
                mB_up[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(None, 1, 1),
            )
            gC = cute.local_tile(
                mC[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, 1, None),
            )

            gScaleA = cute.local_tile(
                mScaleA[None, bidz], tiler=(self.bM,), coord=(offset_tile_x,),
            )
            gScaleB_gate = cute.local_tile(
                mScaleB_gate[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,),
            )
            gScaleB_up = cute.local_tile(
                mScaleB_up[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,),
            )

            residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(
                gA, mode=[2]
            )
            gA = cute.domain_offset((0, residual_k, 0), gA)
            gB_gate = cute.domain_offset((0, residual_k, 0), gB_gate)
            gB_up = cute.domain_offset((0, residual_k, 0), gB_up)
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
            gB_gate = cute.make_tensor(gB_gate.iterator.align(16), gB_gate.layout)
            gB_up = cute.make_tensor(gB_up.iterator.align(16), gB_up.layout)

            mcA = cute.make_identity_tensor(mA.layout.shape)
            mcB = cute.make_identity_tensor(mB_gate.layout.shape)
            cA = cute.local_tile(
                mcA[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(1, None, 1),
            )
            cB = cute.local_tile(
                mcB[None, None, bidz], tiler=self.cta_tiler,
                coord=tiler_coord, proj=(None, 1, 1),
            )
            cA = cute.domain_offset((0, residual_k, 0), cA)
            cB = cute.domain_offset((0, residual_k, 0), cB)

            # A/B/C share one smem arena: the epilogue overwrites the A/B tiles
            # with the fp16 C tile (union), so no extra footprint is requested.
            @cute.struct
            class SharedStorageAB:
                a: cute.struct.Align[
                    cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)], 16]
                b_gate: cute.struct.Align[
                    cute.struct.MemRange[mB_gate.element_type, cute.cosize(sB_layout)], 16]
                b_up: cute.struct.Align[
                    cute.struct.MemRange[mB_up.element_type, cute.cosize(sB_layout)], 16]

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
            sB_gate = SharedStorageAB(storage).b_gate.get_tensor(sB_layout)
            sB_up = SharedStorageAB(storage).b_up.get_tensor(sB_layout)
            sC = SharedStorageC(storage).c.get_tensor(sC_layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            thr_copy_C = tiled_copy_C.get_slice(tidx)
            tCsC_epilogue = thr_copy_C.partition_S(sC)
            tCgC_epilogue = thr_copy_C.partition_D(gC)
            tAgA = thr_copy_A.partition_S(gA)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB_gate = thr_copy_B.partition_S(gB_gate)
            tBsB_gate = thr_copy_B.partition_D(sB_gate)
            tBgB_up = thr_copy_B.partition_S(gB_up)
            tBsB_up = thr_copy_B.partition_D(sB_up)

            tAcA = thr_copy_A.partition_S(cA)
            tBcB = thr_copy_B.partition_S(cB)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            tBpB = cute.make_rmem_tensor(
                cute.make_layout(
                    (tBsB_gate.shape[0][1], cute.size(tBsB_gate, mode=[1]), cute.size(tBsB_gate, mode=[2])),
                    stride=(cute.size(tBsB_gate, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(
                        tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                    )
            for rest_v in range(tBpB.shape[0]):
                for n in range(tBpB.shape[1]):
                    tBpB[rest_v, n, 0] = cute.elem_less(
                        tBcB[(0, rest_v), n, 0, 0][0], mB_gate.shape[0]
                    )

            tAsA.fill(0)
            tBsB_gate.fill(0)
            tBsB_up.fill(0)
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
                    cute.copy(tiled_copy_B, tBgB_gate[None, None, k, k_tile_index],
                              tBsB_gate[None, None, k, 0], pred=tBpB[None, None, k])
                    cute.copy(tiled_copy_B, tBgB_up[None, None, k, k_tile_index],
                              tBsB_up[None, None, k, 0], pred=tBpB[None, None, k])
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

            for k_tile in range(1, num_smem_stages - 1):
                if k_tile == k_tile_count:
                    tApA.fill(0)
                    tBpB.fill(0)
                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                          tAsA[None, None, None, k_tile], pred=tApA)
                cute.copy(tiled_copy_B, tBgB_gate[None, None, None, k_tile_index],
                          tBsB_gate[None, None, None, k_tile], pred=tBpB)
                cute.copy(tiled_copy_B, tBgB_up[None, None, None, k_tile_index],
                          tBsB_up[None, None, None, k_tile], pred=tBpB)
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

            # ///////////////////////////////////////////////////////////////////
            # MMA partitions and two accumulators (gate, up)
            # ///////////////////////////////////////////////////////////////////
            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB_gate = thr_mma.partition_B(sB_gate)
            tCsB_up = thr_mma.partition_B(sB_up)
            tCgC = thr_mma.partition_C(gC)
            tCsC = thr_mma.partition_C(sC)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB_gate = tiled_mma.make_fragment_B(tCsB_gate[None, None, None, 0])
            tCrB_up = tiled_mma.make_fragment_B(tCsB_up[None, None, None, 0])
            tCrC_gate = tiled_mma.make_fragment_C(tCgC)
            tCrC_up = tiled_mma.make_fragment_C(tCgC)
            tCrC_gate.fill(0)
            tCrC_up.fill(0)

            num_vals = cute.size(tCrC_gate, mode=[0])
            num_mma_m = cute.size(tCrC_gate, mode=[1])
            num_mma_n = cute.size(tCrC_gate, mode=[2])

            gScaleA_2d = cute.make_tensor(
                gScaleA.iterator, cute.make_layout((self.bM, self.bN), stride=(1, 0))
            )
            gScaleB_gate_2d = cute.make_tensor(
                gScaleB_gate.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1))
            )
            gScaleB_up_2d = cute.make_tensor(
                gScaleB_up.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1))
            )
            tCgScaleA = thr_mma.partition_C(gScaleA_2d)
            tCgScaleB_gate = thr_mma.partition_C(gScaleB_gate_2d)
            tCgScaleB_up = thr_mma.partition_C(gScaleB_up_2d)

            rScaleA = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_m, 1, 0)),
                cutlass.Float32,
            )
            rScaleB_gate = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)),
                cutlass.Float32,
            )
            rScaleB_up = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)),
                cutlass.Float32,
            )
            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    rScaleA[i, m, 0] = tCgScaleA[i, m, 0].to(cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rScaleB_gate[i, 0, n] = tCgScaleB_gate[i, 0, n].to(cutlass.Float32)
                    rScaleB_up[i, 0, n] = tCgScaleB_up[i, 0, n].to(cutlass.Float32)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mA.element_type,
            )
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                ),
                mB_gate.element_type,
            )
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
            thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
            thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCsB_gate_copy_view = thr_copy_ldmatrix_B.partition_S(sB_gate)
            tCrB_gate_copy_view = thr_copy_ldmatrix_B.retile(tCrB_gate)
            tCsB_up_copy_view = thr_copy_ldmatrix_B.partition_S(sB_up)
            tCrB_up_copy_view = thr_copy_ldmatrix_B.retile(tCrB_up)

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1

            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
            tCsB_gate_p = tCsB_gate_copy_view[None, None, None, smem_pipe_read]
            tCsB_up_p = tCsB_up_copy_view[None, None, None, smem_pipe_read]

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_gate_p[None, None, 0], tCrB_gate_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_up_p[None, None, 0], tCrB_up_copy_view[None, None, 0])

            # ///////////////////////////////////////////////////////////////////
            # Mainloop: two interleaved GEMMs sharing A
            # ///////////////////////////////////////////////////////////////////
            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_gate_p = tCsB_gate_copy_view[None, None, None, smem_pipe_read]
                        tCsB_up_p = tCsB_up_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next],
                              tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_gate_p[None, None, k_block_next],
                              tCrB_gate_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_up_p[None, None, k_block_next],
                              tCrB_up_copy_view[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index],
                                      tAsA[None, None, None, smem_pipe_write], pred=tApA)
                            cute.copy(tiled_copy_B, tBgB_gate[None, None, None, k_tile_index],
                                      tBsB_gate[None, None, None, smem_pipe_write], pred=tBpB)
                            cute.copy(tiled_copy_B, tBgB_up[None, None, None, k_tile_index],
                                      tBsB_up[None, None, None, smem_pipe_write], pred=tBpB)

                    cute.gemm(tiled_mma, tCrC_gate, tCrA[None, None, k_block],
                              tCrB_gate[None, None, k_block], tCrC_gate)
                    cute.gemm(tiled_mma, tCrC_up, tCrA[None, None, k_block],
                              tCrB_up[None, None, k_block], tCrC_up)

                    if k_block == 0:
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ///////////////////////////////////////////////////////////////////
            # Epilogue: out = silu(gate) * up = gate * sigmoid(gate) * up
            #
            # sigmoid(g) is computed via the exact identity 0.5*tanh(0.5*g)+0.5
            # rather than 1/(1+exp(-g)): tanh is a single MUFU op, whereas the
            # exp form needs MUFU.EX2 *and* MUFU.RCP. In this short-K (K=512, 8
            # k-tiles) kernel the epilogue is a large fraction of runtime and the
            # SFU/MUFU pipe is the throttled resource, so halving the MUFU count
            # here is ~12% end-to-end on A100 at the deployed shape. (tanh is the
            # sanctioned CuTe transcendental — see the factored-LSTM kernel.)
            # ///////////////////////////////////////////////////////////////////
            tCrD = cute.make_fragment_like(tCrC_gate, self.c_dtype)
            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    for n in cutlass.range(num_mma_n, unroll_full=True):
                        gate = (tCrC_gate[i, m, n].to(cutlass.Float32)
                                * rScaleA[i, m, 0] * rScaleB_gate[i, 0, n])
                        up = (tCrC_up[i, m, n].to(cutlass.Float32)
                              * rScaleA[i, m, 0] * rScaleB_up[i, 0, n])
                        sig = 0.5 * cute.math.tanh(0.5 * gate) + 0.5
                        tCrD[i, m, n] = (gate * sig * up).to(self.c_dtype)

            # Stage the fp16 result through smem to convert the scattered MMA-C
            # fragment layout into contiguous 128-bit global stores. The direct
            # register->gmem write (autovec_copy tCrD->tCgC) only fills 16 of every
            # 32 store bytes per sector; going via smem coalesces the store.
            cute.autovec_copy(tCrD, tCsC)
            cute.arch.sync_threads()
            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            # Predicate the store on the M/N extents (tiles may overhang the matrix).
            cC = cute.local_tile(
                cute.make_identity_tensor(mC.layout.shape)[None, None, bidz],
                tiler=self.cta_tiler, coord=tiler_coord, proj=(1, 1, None),
            )
            tCcC = thr_copy_C.partition_S(cC)
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
                        tCcC[(0, rest_v), m, 0][0], mC.shape[0]
                    )
            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n],
                                  tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])

        return

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        """Epilogue-C smem staging layout (single stage), plain + padded.

        Unlike gemm_f16's swizzled C layout, the int8 MMA's doubled-N C partition
        makes autovec_copy's right_inverse reject the composed (swizzled) layout,
        so we stage through a plain tile. A plain bN=64 fp16 row is exactly 32
        banks, so the MMA-fragment register->smem write hits a ~6.6-way bank
        conflict (65% of store wavefronts). Padding the major stride by 8 elems
        (16 B / 4 banks) decorrelates consecutive rows and removes it -> +7% at
        bn=64 (289 vs 270 TOPS, M=131072). The pad costs ~2 KB of the C tile, still
        far inside the A/B smem arena it unions over, so occupancy is unchanged.
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
def export_dual_gemm_i8_silu(
    a_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    b_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    c_dtype: Type[cutlass.Numeric] = cutlass.Float16,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Int32,
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    file_path: str = "./artifacts",
    file_name: str = "gemm_i8_dual_silu",
    function_prefix: str = "gemm_i8_dual_silu",
    use_k32: bool = True,
    bm: int = 128,
    bn: int = 32,
    num_stages: int = 3,
    m_size: int = 128,
    n_size: int = 128,
    k_size: int = 128,
    l_size: int = 1,
) -> None:
    """AOT-compile the dual INT8 GEMM + SiLU kernel and emit a C header.

    Computes out[M,N] = silu(A@B_gate^T) * (A@B_up^T), fp16 output.
    Dimensions are dynamic at runtime; the tile/atom config is baked at export.
    Mirrors export_tensor_op_gemm_i8 in ampere_gemm_i8_quant_rmem.py, but with two
    B operands and two per-channel scales.

    Emits <file_path>/<file_name>.h with a <function_prefix>_Kernel_Module_Load,
    a cute_dsl_<function_prefix>_wrapper, and tensor structs for all 7 arguments
    in __call__ order: mA, mB_gate, mB_up, mC, mScaleA, mScaleB_gate, mScaleB_up.
    """
    fake_a, _ = create_and_permute_tensor(l_size, m_size, k_size, False, a_dtype)
    fake_b_gate, _ = create_and_permute_tensor(l_size, n_size, k_size, False, b_dtype)
    fake_b_up, _ = create_and_permute_tensor(l_size, n_size, k_size, False, b_dtype)
    fake_c, _ = create_and_permute_tensor(l_size, m_size, n_size, False, c_dtype)
    fake_scale_a = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m_size, l_size), assumed_align=16)
    fake_scale_b_gate = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_size, l_size), assumed_align=16)
    fake_scale_b_up = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (n_size, l_size), assumed_align=16)

    gemm = TensorOpDualGemmI8Silu(
        a_dtype, b_dtype, c_dtype, acc_dtype,
        atom_layout_mnk, use_k32, bm, bn=bn, num_stages=num_stages,
    )
    print(f"Compiling TensorOpDualGemmI8Silu  tile={bm}x{bn}x64  atom={atom_layout_mnk}  stages={num_stages}")
    compiled = cute.compile(
        gemm, fake_a, fake_b_gate, fake_b_up, fake_c,
        fake_scale_a, fake_scale_b_gate, fake_scale_b_up,
    )
    print(f"Exporting to {file_path}/{file_name}.h ...")
    compiled.export_to_c(
        file_path=file_path, file_name=file_name, function_prefix=function_prefix,
    )
    print("Export complete!")


# =============================================================================
# Host-side reference / helpers
# =============================================================================
def dual_gemm_silu_ref(A_int8, B_gate_int8, B_up_int8, scale_a, scale_b_gate, scale_b_up):
    import torch
    A_dq = A_int8.float() * scale_a[:, None]
    Bg_dq = B_gate_int8.float() * scale_b_gate[:, None]
    Bu_dq = B_up_int8.float() * scale_b_up[:, None]
    gate = A_dq @ Bg_dq.T
    up = A_dq @ Bu_dq.T
    return (gate * torch.sigmoid(gate) * up)


__all__ = [
    "TensorOpDualGemmI8Silu", "export_dual_gemm_i8_silu",
    "dual_gemm_silu_ref", "create_and_permute_tensor",
]
