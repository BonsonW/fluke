"""Factored-LSTM step with an INT8 GATE GEMM for NVIDIA Ampere (A100, sm80).

Variant of factored_lstm_i8.TensorOpFactoredLstmI8 that runs the four gate
up-projections in INT8 tensor cores (mma.sync m16n8k32, int32 accumulate) instead
of f16. The recurrence gate GEMM is the step bottleneck (~50us/step at N=2048, only
~28% of the f16 tensor-core peak); int8 halves weight traffic + smem and should give
~1.5-1.8x. h_new is still provably in [-1,1] so the hidden output stays int8 @ 1/127.

    A      = concat([hh_down, x_down], dim=1)          -> [B, Kc]   INT8, per-row scale_a[B]
    W_gate = concat([up_hh[g], up_ih[g]], dim=1)       -> [H, Kc]   INT8, per-channel scale_w[g][H]
    acc_g  = A_int8 @ W_gate_int8^T                     -> [B, H]    INT32
    raw_g  = acc_g * scale_a[m] * scale_w[g][n] + bias_g            (f32)
    i,f,o  = sighard(raw) = clamp(0.2*raw + 0.5, 0, 1);  g = clamp(raw, -1, 1)
    c_new  = f*c + i*g   (f32, written in place)
    h_new  = o * tanh(c_new) in [-1,1]  ->  int8 @ 1/127

Composed from factored_lstm_i8.py (4-accumulator LSTM epilogue) and
gemm_i8_quant.py (int8 A/B, int32 accum, per-row/per-channel scale plumbing). The
int8 shared-memory swizzle (MBase=4, Swizzle<2,4,3>) and the int8 gmem tiled copy
come from TensorOpGemmI8, which this class subclasses.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda_driver  # cuda_driver.CUstream (AOT stream arg → CUDA-graph capture)

# Importing gemm_i8_quant installs the local MmaI8Op shim (DSL 4.4.2 dropped it) and
# gives us the int8-aware _make_smem_layout_AB / _make_gmem_tiled_copy_AB helpers.
from gemm_i8_quant import TensorOpGemmI8

H_OUT_SCALE = 1.0 / 127.0   # fixed int8 hidden-output scale (h in [-1,1])


class TensorOpFactoredLstmGateI8(TensorOpGemmI8):
    """Fused factored-LSTM step with an INT8 gate GEMM (4 int32 accumulators).

    ab_dtype  : gate GEMM activation/weight dtype (Int8).
    out_dtype : hidden-state output dtype (Int8), fixed scale 1/127.
    acc_dtype : accumulator dtype (Int32).
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
        self.mma_inst_shape = (16, 8, 32)   # int8 mma.sync m16n8k32
        mmaM, mmaN, mmaK = self.mma_inst_shape
        assert self.bM % (atom_lay_M * mmaM) == 0, "bM must be divisible by MMA instruction"
        assert self.bN % (atom_lay_N * mmaN) == 0, "bN must be divisible by MMA instruction"
        assert atom_lay_K == 1, "atom layout K > 1 unsupported"
        assert self.bK % mmaK == 0, "bK must be divisible by int8 MMA K=32"
        assert self.num_stages >= 3, "num_stages must be >= 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,        # [B, Kc]  int8   (concat hh_down|x_down)
        mB_i: cute.Tensor,      # [H, Kc]  int8   gate-i weight (concat up_hh_i|up_ih_i)
        mB_f: cute.Tensor,      # [H, Kc]  int8
        mB_g: cute.Tensor,      # [H, Kc]  int8
        mB_o: cute.Tensor,      # [H, Kc]  int8
        mScaleA: cute.Tensor,   # [B, 1]   f32   per-row (per-batch) dequant scale
        mScaleW_i: cute.Tensor, # [H, 1]   f32   per-output-channel dequant scale (gate i)
        mScaleW_f: cute.Tensor, # [H, 1]   f32
        mScaleW_g: cute.Tensor, # [H, 1]   f32
        mScaleW_o: cute.Tensor, # [H, 1]   f32
        mBias_i: cute.Tensor,   # [H, 1]   f32   (bias_hh_i + bias_ih_i)
        mBias_f: cute.Tensor,   # [H, 1]   f32
        mBias_g: cute.Tensor,   # [H, 1]   f32
        mBias_o: cute.Tensor,   # [H, 1]   f32
        mC_c: cute.Tensor,      # [B, H]   f32   cell state, read + written in place
        mH_out: cute.Tensor,    # [B, H]   int8  hidden output (fixed scale 1/127)
        stream: cuda_driver.CUstream = None,
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

        op = cute.nvgpu.warp.MmaI8Op(self.a_dtype, self.b_dtype, self.acc_dtype, self.mma_inst_shape)
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

        _launcher = self.kernel(
            mA, mB_i, mB_f, mB_g, mB_o,
            mScaleA, mScaleW_i, mScaleW_f, mScaleW_g, mScaleW_o,
            mBias_i, mBias_f, mBias_g, mBias_o,
            mC_c, mH_out,
            sA_layout, sB_layout,
            tiled_copy_A, tiled_copy_B, tiled_mma,
            raster_factor,
        )
        launch_kwargs = dict(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )
        if cutlass.const_expr(stream is not None):
            launch_kwargs["stream"] = stream
        _launcher.launch(**launch_kwargs)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB_i: cute.Tensor, mB_f: cute.Tensor, mB_g: cute.Tensor, mB_o: cute.Tensor,
        mScaleA: cute.Tensor,
        mScaleW_i: cute.Tensor, mScaleW_f: cute.Tensor, mScaleW_g: cute.Tensor, mScaleW_o: cute.Tensor,
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

            # MMA partitions + four gate accumulators (int32)
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
            tCrC_i.fill(0); tCrC_f.fill(0); tCrC_g.fill(0); tCrC_o.fill(0)

            num_vals = cute.size(tCrC_i, mode=[0])
            num_mma_m = cute.size(tCrC_i, mode=[1])
            num_mma_n = cute.size(tCrC_i, mode=[2])

            # Per-batch (M) dequant scale: broadcast over N (stride-0 in N).
            gScaleA = cute.local_tile(mScaleA[None, bidz], tiler=(self.bM,), coord=(offset_tile_x,))
            gScaleA_2d = cute.make_tensor(gScaleA.iterator, cute.make_layout((self.bM, self.bN), stride=(1, 0)))
            tCgScaleA = thr_mma.partition_C(gScaleA_2d)
            rScaleA = cute.make_rmem_tensor(
                cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_m, 1, 0)), cutlass.Float32)
            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    rScaleA[i, m, 0] = tCgScaleA[i, m, 0].to(cutlass.Float32)

            # Per-hidden-column (N) tensors: bias + per-channel weight scale, one per gate.
            # All broadcast over M (stride-0 in M), loaded through the same MMA thread map.
            def col_view(mT):
                gT = cute.local_tile(mT[None, bidz], tiler=(self.bN,), coord=(offset_tile_y,))
                t2d = cute.make_tensor(gT.iterator, cute.make_layout((self.bM, self.bN), stride=(0, 1)))
                return thr_mma.partition_C(t2d)

            tCgBias_i = col_view(mBias_i); tCgBias_f = col_view(mBias_f)
            tCgBias_g = col_view(mBias_g); tCgBias_o = col_view(mBias_o)
            tCgScaleW_i = col_view(mScaleW_i); tCgScaleW_f = col_view(mScaleW_f)
            tCgScaleW_g = col_view(mScaleW_g); tCgScaleW_o = col_view(mScaleW_o)

            def col_frag():
                return cute.make_rmem_tensor(
                    cute.make_layout((num_vals, num_mma_m, num_mma_n), stride=(num_mma_n, 0, 1)), cutlass.Float32)

            rBias_i = col_frag(); rBias_f = col_frag(); rBias_g = col_frag(); rBias_o = col_frag()
            rScaleW_i = col_frag(); rScaleW_f = col_frag(); rScaleW_g = col_frag(); rScaleW_o = col_frag()
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rBias_i[i, 0, n] = tCgBias_i[i, 0, n].to(cutlass.Float32)
                    rBias_f[i, 0, n] = tCgBias_f[i, 0, n].to(cutlass.Float32)
                    rBias_g[i, 0, n] = tCgBias_g[i, 0, n].to(cutlass.Float32)
                    rBias_o[i, 0, n] = tCgBias_o[i, 0, n].to(cutlass.Float32)
                    rScaleW_i[i, 0, n] = tCgScaleW_i[i, 0, n].to(cutlass.Float32)
                    rScaleW_f[i, 0, n] = tCgScaleW_f[i, 0, n].to(cutlass.Float32)
                    rScaleW_g[i, 0, n] = tCgScaleW_g[i, 0, n].to(cutlass.Float32)
                    rScaleW_o[i, 0, n] = tCgScaleW_o[i, 0, n].to(cutlass.Float32)

            # S2R ldmatrix atoms (int8: 16-bit ldmatrix units, 2 int8 per unit).
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

            # Epilogue: dequant (per-row scale_a * per-channel scale_w) + bias, then the
            # LSTM gate combine, cell update (f32 in place), int8 hidden out.
            tCgCc = thr_mma.partition_C(gCc)
            rH = cute.make_fragment_like(tCrC_i, self.out_dtype)
            rC = cute.make_fragment_like(tCrC_i, cutlass.Float32)

            def clamp(x, lo, hi):
                # cute.arch only exposes fmax; min(y,hi) = -fmax(-y,-hi)
                return 0.0 - cute.arch.fmax(0.0 - cute.arch.fmax(x, lo), 0.0 - hi)

            for i in cutlass.range(num_vals, unroll_full=True):
                for m in cutlass.range(num_mma_m, unroll_full=True):
                    for n in cutlass.range(num_mma_n, unroll_full=True):
                        sa = rScaleA[i, m, 0]
                        raw_i = tCrC_i[i, m, n].to(cutlass.Float32) * sa * rScaleW_i[i, 0, n] + rBias_i[i, 0, n]
                        raw_f = tCrC_f[i, m, n].to(cutlass.Float32) * sa * rScaleW_f[i, 0, n] + rBias_f[i, 0, n]
                        raw_g = tCrC_g[i, m, n].to(cutlass.Float32) * sa * rScaleW_g[i, 0, n] + rBias_g[i, 0, n]
                        raw_o = tCrC_o[i, m, n].to(cutlass.Float32) * sa * rScaleW_o[i, 0, n] + rBias_o[i, 0, n]

                        i_a = clamp(raw_i * 0.2 + 0.5, 0.0, 1.0)
                        f_a = clamp(raw_f * 0.2 + 0.5, 0.0, 1.0)
                        o_a = clamp(raw_o * 0.2 + 0.5, 0.0, 1.0)
                        g_a = clamp(raw_g, -1.0, 1.0)

                        c_old = tCgCc[i, m, n].to(cutlass.Float32)
                        c_new = f_a * c_old + i_a * g_a
                        rC[i, m, n] = c_new

                        h_new = o_a * cute.math.tanh(c_new)
                        hs = h_new * 127.0
                        half = clamp(hs * 1e30, -0.5, 0.5)   # sign(hs)*0.5 (0 -> 0)
                        rH[i, m, n] = (hs + half).to(self.out_dtype)
            cute.autovec_copy(rC, tCgCc)
            cute.autovec_copy(rH, tCgH)
        return


# =============================================================================
# AOT export
# =============================================================================
def export_factored_lstm_gate_i8(
    ab_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    out_dtype: Type[cutlass.Numeric] = cutlass.Int8,
    acc_dtype: Type[cutlass.Numeric] = cutlass.Int32,
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    file_path: str = "./artifacts",
    file_name: str = "factored_lstm_gate_i8",
    function_prefix: str = "factored_lstm_gate_i8",
    H: int = 1024,
    K_hh: int = 128,
    R: int = 128,
    bm: int = 64,
    bn: int = 32,
    bk: int = 64,
    num_stages: int = 3,
    b_size: int = 128,
) -> None:
    """AOT-compile the int8-gate factored-LSTM step and emit a C header.

    Kc = K_hh + R (merged contraction). 16-argument wrapper in __call__ order:
    mA[B,Kc], mB_i/f/g/o[H,Kc] (int8), mScaleA[B], mScaleW_i/f/g/o[H],
    mBias_i/f/g/o[H], mC_c[B,H], mH_out[B,H]. B (M) dynamic; H, Kc, tile baked.
    """
    import torch
    import cutlass.torch as cutlass_torch

    Kc = K_hh + R

    def _cpt(mode0, mode1, dtype):
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
    fake_scale_a = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (b_size, 1), assumed_align=16)
    fake_scale_wi = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_scale_wf = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_scale_wg = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_scale_wo = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_i = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_f = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_g = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_o = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_c = _cpt(b_size, H, cutlass.Float32)
    fake_h = _cpt(b_size, H, out_dtype)

    lstm = TensorOpFactoredLstmGateI8(ab_dtype, out_dtype, acc_dtype, atom_layout_mnk,
                                      bm=bm, bn=bn, bk=bk, num_stages=num_stages)
    print(f"Compiling TensorOpFactoredLstmGateI8  H={H} Kc={Kc}  tile={bm}x{bn}x{bk}  atom={atom_layout_mnk}")
    compiled = cute.compile(
        lstm, fake_a, fake_bi, fake_bf, fake_bg, fake_bo,
        fake_scale_a, fake_scale_wi, fake_scale_wf, fake_scale_wg, fake_scale_wo,
        fake_bias_i, fake_bias_f, fake_bias_g, fake_bias_o, fake_c, fake_h,
        stream=cute.runtime.make_fake_stream(),
    )
    print(f"Exporting to {file_path}/{file_name}.h ...")
    compiled.export_to_c(file_path=file_path, file_name=file_name, function_prefix=function_prefix)
    print("Export complete!")


__all__ = ["TensorOpFactoredLstmGateI8", "export_factored_lstm_gate_i8", "H_OUT_SCALE"]
