"""Fully-fused factored-LSTM step for NVIDIA Ampere (A100, sm80) — INT8, single launch.

Fuses the recurrent int8 down-projection INTO the fused gate/step kernel, removing
one kernel launch per timestep and the hh_down DRAM round-trip between them:

    hh_down[B,K_hh] = (h_i8[B,H] @ W_dn_i8[K_hh,H]^T) * comb_scale[K_hh]   (int8 GEMM)
    gates[B,4H]     = [hh_down | x_down] @ W_g[4H,Kc]^T + bias             (f16 GEMM x4)
    i,f,o = sighard; g = clamp; c' = f*c + i*g; h' = o*tanh(c') -> int8 (1/127)

Why fuse: both kernels are per-CTA latency-bound (see autotune sweep + ncu 2026-07:
schedulers idle ~80%, down-proj runs on 4 CTAs = 4 SMs). The fusion spreads the
down-projection across ALL CTAs (split-K) and overlaps it with the x-half of the
gate GEMM, which depends only on the precomputed x_down.

Single-kernel producer/consumer structure (per M row-group of bM rows):
  producer  (n_tile < S CTAs): producer j computes the FULL-K down-projection for
            its own K_hh/S output columns with the int8 MMA (direct gmem->rmem
            fragment loads, no smem), scales by comb_scale[col] (= w_dn_scale *
            1/127, folded on the host) and plain-stores f16 into mHH. Disjoint
            column slices: no atomics on data, no reduction step. (An earlier
            split-K variant deposited int32 partials via f32 atomic-add + a reducer
            CTA; the ~256K L2 atomics per launch made it slower than unfused.)
  consumer  (all CTAs): pass 1 of the gate GEMM over x_down (W k-tiles K_hh/bK..),
            acquire-spin until ready == S, pass 2 over mHH (W k-tiles 0..), both
            passes accumulating into the same 4 gate accumulators, then the usual
            epilogue.
  cleanup   flags are self-cleaning (last consumer resets ready/done), so
            back-to-back steps need no host-side memsets.

DEADLOCK CONSTRAINT: consumers spin-wait on producers of their own row-group inside
one grid, so the whole grid must be co-resident. Measured (ncu, A100, 2026-07):
134 regs/thread + 49.2KB smem -> 3 CTAs/SM -> 324 co-resident CTAs. At bn=32,
H=1024 the grid is (M/64)*32, so keep B <= 512 (grid 256). Larger grids MAY
complete (scheduling order usually favors producers) but are not guaranteed to —
this cannot be asserted here because M is dynamic in the AOT trace; the caller owns
the bound. Measured vs the two-kernel pipeline (warm A100, isolated):
B=128: 14.4us vs 22.7us (1.57x); B=256: 18.1 vs 24.1 (1.33x); B=512: 23.5 vs 37.0 (1.58x).

Layout requirements (asserted where possible, otherwise documented):
  - all row tensors padded to M_pad = ceil(B/bM)*bM rows (same as the unfused step)
  - mHH f16 [M_pad, K_hh] contiguous, no init needed (fully producer-written), and
    marked with the same divisibility as the other f16 operands (cp.async alignment)
  - mFlags int32 [grid_m * 4] zero-filled at allocation
  - mWdnScale f32 [K_hh, 1] = per-channel W_dn scale * 1/127 (host-folded)
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda_driver

from factored_lstm_i8 import TensorOpFactoredLstmI8, H_OUT_SCALE


class TensorOpFactoredLstmFusedI8(TensorOpFactoredLstmI8):
    """Single-launch fused step: split-K int8 down-proj producers + f16 gate consumers.

    H, K_hh, R are compile-time constants (they size the split-K chunks and the
    weight k-tile offsets). num_producers (S) CTAs per row-group each own an H/S
    K-chunk of the down-projection.
    """

    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        H: int = 1024,
        K_hh: int = 128,
        R: int = 128,
        num_producers: int = 4,
        bm: int = 64,
        bn: int = 32,
        bk: int = 32,
        num_stages: int = 3,
    ):
        super().__init__(ab_dtype, out_dtype, acc_dtype, atom_layout_mnk,
                         bm=bm, bn=bn, bk=bk, num_stages=num_stages)
        self.H = H
        self.K_hh = K_hh
        self.R = R
        self.S = num_producers
        atom_m, atom_n, _ = atom_layout_mnk
        assert K_hh % num_producers == 0, "K_hh must divide into num_producers column slices"
        self.dp_nn = K_hh // num_producers   # hh_down columns per producer
        self.dp_bk = 64                       # producer smem k-tile (128b cp.async, i8)
        assert self.dp_nn >= atom_n * 8 * 2, "producer N-slice must cover the MMA-N permutation"
        assert self.dp_nn % (atom_n * 8) == 0
        assert self.dp_nn % 32 == 0 and bm % 32 == 0, "producer gmem copy covers (32, dp_bk) patches"
        assert H % self.dp_bk == 0, "H must be a multiple of the producer k-tile"
        assert bm % (atom_m * 16) == 0
        assert K_hh % bk == 0 and R % bk == 0, "per-pass K must be a multiple of bK"
        assert H % bn == 0, "H must be a multiple of bN (padded grids unsupported)"
        self.k_tiles_hh = K_hh // bk   # pass-2 tile count; also pass-1 B k-tile offset
        self.k_tiles_x = R // bk       # pass-1 tile count

    def _make_smem_layout_i8(self, copy_bits, smem_tiler):
        """Row-major int8 smem layout, Swizzle<2,4,3>: 16-byte contiguity, 8-row atom
        (the inherited f16 helper hardcodes MBase=3, which breaks 128-bit i8 copies —
        same formula as gemm_i8_quant._make_smem_layout_AB)."""
        major_mode_size = min(smem_tiler[1], 64)
        swizzle_bits = min(int(math.log2(major_mode_size * 8 // copy_bits)), 3)
        mbase = int(math.log2(copy_bits // 8))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, mbase, 3),
            0,
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1)),
        )
        return cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))

    @cute.jit
    def __call__(
        self,
        mHprev: cute.Tensor,     # [M_pad, H]    int8  previous hidden (fixed 1/127)
        mWdn: cute.Tensor,       # [K_hh, H]     int8  down-proj weight
        mWdnScale: cute.Tensor,  # [K_hh, 1]     f32   per-channel scale * 1/127
        mX: cute.Tensor,         # [M_pad, R]    f16   this step's x_down slice
        mB_i: cute.Tensor,       # [H, Kc]       f16   gate weights (concat hh|x on K)
        mB_f: cute.Tensor,
        mB_g: cute.Tensor,
        mB_o: cute.Tensor,
        mBias_i: cute.Tensor,    # [H, 1]        f32
        mBias_f: cute.Tensor,
        mBias_g: cute.Tensor,
        mBias_o: cute.Tensor,
        mC_c: cute.Tensor,       # [M_pad, H]    f32   cell, updated in place
        mH_out: cute.Tensor,     # [M_pad, H]    int8  hidden out (fixed 1/127)
        mHH: cute.Tensor,        # [M_pad, K_hh] f16   hh_down staging (producer-written)
        mFlags: cute.Tensor,     # [grid_m * 4]  int32 (zeroed at alloc; self-cleaning)
        stream: cuda_driver.CUstream = None,
    ):
        self.a_major_mode = utils.LayoutEnum.from_tensor(mX)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB_i)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mH_out)

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mX.element_type, self.a_major_mode, ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB_i.element_type, self.b_major_mode, ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )
        # Producer staging: 2-stage int8 smem pipeline (disjoint from the consumer
        # f16 buffers; producers run before the consumer passes touch sA/sB).
        sAdp_layout = self._make_smem_layout_i8(ab_copy_bits, (self.bM, self.dp_bk, 2))
        sBdp_layout = self._make_smem_layout_i8(ab_copy_bits, (self.dp_nn, self.dp_bk, 2))

        smem_size = (
            cute.size_in_bytes(mX.element_type, sA_layout)
            + 4 * cute.size_in_bytes(mB_i.element_type, sB_layout)
            + cute.size_in_bytes(cutlass.Int8, sAdp_layout)
            + cute.size_in_bytes(cutlass.Int8, sBdp_layout)
        )

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            mX.element_type, num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mX.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB_i.element_type, self.b_major_mode, ab_copy_bits)

        # Producer gmem->smem copy: (32, 4) threads x (1, 16) i8 values covers a
        # (32, dp_bk) patch — fits both the A tile (bM rows) and the dp_nn-row B tile
        # (the shared _make_gmem_tiled_copy_AB helper assumes bK/128-thread coverage).
        atom_async_copy_dp = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            cutlass.Int8, num_bits_per_copy=ab_copy_bits,
        )
        dp_copy_elems = ab_copy_bits // 8
        dp_k_thr = self.dp_bk // dp_copy_elems
        tiled_copy_dp = cute.make_tiled_copy_tv(
            atom_async_copy_dp,
            cute.make_layout((self.num_threads // dp_k_thr, dp_k_thr), stride=(dp_k_thr, 1)),
            cute.make_layout((1, dp_copy_elems)),
        )

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

        _launcher = self.kernel(
            mHprev, mWdn, mWdnScale, mX,
            mB_i, mB_f, mB_g, mB_o,
            mBias_i, mBias_f, mBias_g, mBias_o,
            mC_c, mH_out,
            mHH, mFlags,
            sA_layout, sB_layout, sAdp_layout, sBdp_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_dp, tiled_mma,
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
        mHprev: cute.Tensor, mWdn: cute.Tensor, mWdnScale: cute.Tensor, mX: cute.Tensor,
        mB_i: cute.Tensor, mB_f: cute.Tensor, mB_g: cute.Tensor, mB_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor, mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mC_c: cute.Tensor, mH_out: cute.Tensor,
        mHH: cute.Tensor, mFlags: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sAdp_layout: cute.ComposedLayout,
        sBdp_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_dp: cute.TiledCopy,
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
            m_tile = offset_tile_x
            n_tile = offset_tile_y
            GN = self.H // self.bN            # consumers per row-group (compile-time)
            KOFF = self.k_tiles_hh            # pass-1 W k-tile offset (x half)

            # Smem: gate-GEMM pipeline buffers + one scalar for reducer election.
            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(mX.element_type, sA_layout, 16)
            sB_i = smem.allocate_tensor(mB_i.element_type, sB_layout, 16)
            sB_f = smem.allocate_tensor(mB_i.element_type, sB_layout, 16)
            sB_g = smem.allocate_tensor(mB_i.element_type, sB_layout, 16)
            sB_o = smem.allocate_tensor(mB_i.element_type, sB_layout, 16)
            sAdp = smem.allocate_tensor(cutlass.Int8, sAdp_layout, 16)
            sBdp = smem.allocate_tensor(cutlass.Int8, sBdp_layout, 16)

            ready_ptr = mFlags.iterator + (m_tile * 4 + 0)
            done_ptr = mFlags.iterator + (m_tile * 4 + 2)

            # ///////////////////////////////////////////////////////////////////////
            # Producer: column-split hh_down. Producer j (n_tile == j < S) computes
            # the FULL-K down-projection for its own dp_nn columns and plain-stores
            # f16 straight into mHH — disjoint outputs, no atomics on data, no
            # reduction step. The full-unrolled k loop gives ptxas independent loads
            # to pipeline (fragments are tiny: one k-step at a time).
            # ///////////////////////////////////////////////////////////////////////
            if n_tile < self.S:
                op_dp = cute.nvgpu.warp.MmaI8Op(cutlass.Int8, cutlass.Int8,
                                                cutlass.Int32, (16, 8, 32))
                perm_dp = (
                    self.atom_layout_mnk[0] * 16,
                    self.atom_layout_mnk[1] * 8 * 2,
                    self.atom_layout_mnk[2] * 32,
                )
                tiled_mma_dp = cute.make_tiled_mma(
                    op_dp, cute.make_layout(self.atom_layout_mnk), permutation_mnk=perm_dp)
                thr_dp = tiled_mma_dp.get_slice(tidx)

                # A: [bM, H] rows of h_prev; B: [dp_nn, H] rows of W_dn, k-tiled
                # into dp_bk columns for the 2-stage cp.async + ldmatrix pipeline.
                gAdp = cute.local_tile(mHprev[None, None, bidz],
                                       tiler=(self.bM, self.dp_bk),
                                       coord=(m_tile, None))     # (bM, dp_bk, NK)
                gBdp = cute.local_tile(mWdn[None, None, bidz],
                                       tiler=(self.dp_nn, self.dp_bk),
                                       coord=(n_tile, None))     # (dp_nn, dp_bk, NK)
                gHHp = cute.local_tile(mHH[None, None, bidz],
                                       tiler=(self.bM, self.dp_nn),
                                       coord=(m_tile, n_tile))
                gAdp = cute.make_tensor(gAdp.iterator.align(16), gAdp.layout)
                gBdp = cute.make_tensor(gBdp.iterator.align(16), gBdp.layout)

                thr_copy_dp = tiled_copy_dp.get_slice(tidx)
                tAdpg = thr_copy_dp.partition_S(gAdp)   # (CPY, CPY_M, CPY_K, NK)
                tAdps = thr_copy_dp.partition_D(sAdp)   # (CPY, CPY_M, CPY_K, 2)
                tBdpg = thr_copy_dp.partition_S(gBdp)
                tBdps = thr_copy_dp.partition_D(sBdp)

                tCsAdp = thr_dp.partition_A(sAdp)       # (MMA, MMA_M, MMA_K, 2)
                tCsBdp = thr_dp.partition_B(sBdp)
                tCgHHp = thr_dp.partition_C(gHHp)       # (MMA, MMA_M, MMA_N)
                tCrAdp = tiled_mma_dp.make_fragment_A(tCsAdp[None, None, None, 0])
                tCrBdp = tiled_mma_dp.make_fragment_B(tCsBdp[None, None, None, 0])
                tCrCdp = tiled_mma_dp.make_fragment_C(tCgHHp)
                tCrCdp.fill(0)

                atom_ldm_dp = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Int8)
                t_s2r_A_dp = cute.make_tiled_copy_A(atom_ldm_dp, tiled_mma_dp)
                t_s2r_B_dp = cute.make_tiled_copy_B(atom_ldm_dp, tiled_mma_dp)
                thr_ldm_A_dp = t_s2r_A_dp.get_slice(tidx)
                thr_ldm_B_dp = t_s2r_B_dp.get_slice(tidx)
                tCsAdp_cv = thr_ldm_A_dp.partition_S(sAdp)
                tCrAdp_cv = thr_ldm_A_dp.retile(tCrAdp)
                tCsBdp_cv = thr_ldm_B_dp.partition_S(sBdp)
                tCrBdp_cv = thr_ldm_B_dp.retile(tCrBdp)

                NK_DP = self.H // self.dp_bk
                nkb_dp = cute.size(tCrAdp, mode=[2])    # dp_bk / 32 mma-k blocks

                cute.copy(tiled_copy_dp, tAdpg[None, None, None, 0],
                          tAdps[None, None, None, 0])
                cute.copy(tiled_copy_dp, tBdpg[None, None, None, 0],
                          tBdps[None, None, None, 0])
                cute.arch.cp_async_commit_group()
                for k in cutlass.range_constexpr(NK_DP):
                    if cutlass.const_expr(k + 1 < NK_DP):
                        cute.copy(tiled_copy_dp, tAdpg[None, None, None, k + 1],
                                  tAdps[None, None, None, (k + 1) % 2])
                        cute.copy(tiled_copy_dp, tBdpg[None, None, None, k + 1],
                                  tBdps[None, None, None, (k + 1) % 2])
                        cute.arch.cp_async_commit_group()
                        cute.arch.cp_async_wait_group(1)
                    else:
                        cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    for kb in cutlass.range_constexpr(nkb_dp):
                        cute.copy(t_s2r_A_dp, tCsAdp_cv[None, None, kb, k % 2],
                                  tCrAdp_cv[None, None, kb])
                        cute.copy(t_s2r_B_dp, tCsBdp_cv[None, None, kb, k % 2],
                                  tCrBdp_cv[None, None, kb])
                        cute.gemm(tiled_mma_dp, tCrCdp, tCrAdp[None, None, kb],
                                  tCrBdp[None, None, kb], tCrCdp)
                    # all threads done reading this stage before iter k+1 overwrites it
                    cute.arch.sync_threads()

                # Scale per output column (comb_scale = w_dn_scale * 1/127) and store f16.
                idHH = cute.make_identity_tensor(mHH.layout.shape)
                cHHp = cute.local_tile(idHH[None, None, bidz],
                                       tiler=(self.bM, self.dp_nn),
                                       coord=(m_tile, n_tile))
                tCcHHp = thr_dp.partition_C(cHHp)
                rHHp = cute.make_fragment_like(tCrCdp, mHH.element_type)
                dp_vals = cute.size(tCrCdp, mode=[0])
                dp_m = cute.size(tCrCdp, mode=[1])
                dp_n = cute.size(tCrCdp, mode=[2])
                for i in cutlass.range(dp_vals, unroll_full=True):
                    for m in cutlass.range(dp_m, unroll_full=True):
                        for n in cutlass.range(dp_n, unroll_full=True):
                            col = tCcHHp[i, m, n][1]
                            s = mWdnScale[col, bidz].to(cutlass.Float32)
                            rHHp[i, m, n] = (tCrCdp[i, m, n].to(cutlass.Float32) * s).to(
                                mHH.element_type)
                cute.autovec_copy(rHHp, tCgHHp)

                # Publish: all stores visible before the ready count ticks.
                cute.arch.fence_acq_rel_gpu()
                cute.arch.sync_threads()
                if tidx == 0:
                    cute.arch.atomic_add(ptr=ready_ptr, val=cutlass.Int32(1),
                                         sem="release", scope="gpu")

            # ///////////////////////////////////////////////////////////////////////
            # Consumer: gate GEMM in two passes over the merged-K weights.
            #   pass 1: A = x_down,   W k-tiles [KOFF, KOFF + k_tiles_x)
            #   pass 2: A = hh_down,  W k-tiles [0, k_tiles_hh)      (after the wait)
            # ///////////////////////////////////////////////////////////////////////
            tiler_coord = (m_tile, n_tile, None)
            gX = cute.local_tile(mX[None, None, bidz], tiler=self.cta_tiler,
                                 coord=tiler_coord, proj=(1, None, 1))
            gHH = cute.local_tile(mHH[None, None, bidz], tiler=self.cta_tiler,
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

            gX = cute.make_tensor(gX.iterator.align(16), gX.layout)
            gHH = cute.make_tensor(gHH.iterator.align(16), gHH.layout)
            gB_i = cute.make_tensor(gB_i.iterator.align(16), gB_i.layout)
            gB_f = cute.make_tensor(gB_f.iterator.align(16), gB_f.layout)
            gB_g = cute.make_tensor(gB_g.iterator.align(16), gB_g.layout)
            gB_o = cute.make_tensor(gB_o.iterator.align(16), gB_o.layout)

            thr_copy_A = tiled_copy_A.get_slice(tidx)
            thr_copy_B = tiled_copy_B.get_slice(tidx)
            tAgX = thr_copy_A.partition_S(gX)
            tAgHH = thr_copy_A.partition_S(gHH)
            tAsA = thr_copy_A.partition_D(sA)
            tBgB_i = thr_copy_B.partition_S(gB_i); tBsB_i = thr_copy_B.partition_D(sB_i)
            tBgB_f = thr_copy_B.partition_S(gB_f); tBsB_f = thr_copy_B.partition_D(sB_f)
            tBgB_g = thr_copy_B.partition_S(gB_g); tBsB_g = thr_copy_B.partition_D(sB_g)
            tBgB_o = thr_copy_B.partition_S(gB_o); tBsB_o = thr_copy_B.partition_D(sB_o)

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

            gBias_i = cute.local_tile(mBias_i[None, bidz], tiler=(self.bN,), coord=(n_tile,))
            gBias_f = cute.local_tile(mBias_f[None, bidz], tiler=(self.bN,), coord=(n_tile,))
            gBias_g = cute.local_tile(mBias_g[None, bidz], tiler=(self.bN,), coord=(n_tile,))
            gBias_o = cute.local_tile(mBias_o[None, bidz], tiler=(self.bN,), coord=(n_tile,))

            def bias_view(gBias):
                b2d = cute.make_tensor(gBias.iterator,
                                       cute.make_layout((self.bM, self.bN), stride=(0, 1)))
                return thr_mma.partition_C(b2d)

            tCgBias_i = bias_view(gBias_i)
            tCgBias_f = bias_view(gBias_f)
            tCgBias_g = bias_view(gBias_g)
            tCgBias_o = bias_view(gBias_o)

            def bias_reg():
                return cute.make_rmem_tensor(
                    cute.make_layout((num_vals, num_mma_m, num_mma_n),
                                     stride=(num_mma_n, 0, 1)), cutlass.Float32)

            rBias_i = bias_reg(); rBias_f = bias_reg(); rBias_g = bias_reg(); rBias_o = bias_reg()
            for i in cutlass.range(num_vals, unroll_full=True):
                for n in cutlass.range(num_mma_n, unroll_full=True):
                    rBias_i[i, 0, n] = tCgBias_i[i, 0, n].to(cutlass.Float32)
                    rBias_f[i, 0, n] = tCgBias_f[i, 0, n].to(cutlass.Float32)
                    rBias_g[i, 0, n] = tCgBias_g[i, 0, n].to(cutlass.Float32)
                    rBias_o[i, 0, n] = tCgBias_o[i, 0, n].to(cutlass.Float32)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4),
                mX.element_type)
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

            num_smem_stages = cute.size(tAsA, mode=[3])
            num_k_block = cute.size(tCrA, mode=[2])

            def gemm_pass(tAgA_p, koff_b, k_tile_count):
                """One pipelined pass over k_tile_count A-tiles paired with W k-tiles
                [koff_b, koff_b + k_tile_count). Accumulates into the 4 gate accs.
                Enters and leaves with all cp.async groups drained. All tile indices
                are compile-time constants (fully unrolled passes)."""
                smem_pipe_read = 0
                smem_pipe_write = num_smem_stages - 1

                # Prologue: prefetch the first (stages-1) tiles.
                for stage in cutlass.range_constexpr(num_smem_stages - 1):
                    if cutlass.const_expr(stage < k_tile_count):
                        cute.copy(tiled_copy_A, tAgA_p[None, None, None, stage],
                                  tAsA[None, None, None, stage])
                        cute.copy(tiled_copy_B, tBgB_i[None, None, None, stage + koff_b],
                                  tBsB_i[None, None, None, stage])
                        cute.copy(tiled_copy_B, tBgB_f[None, None, None, stage + koff_b],
                                  tBsB_f[None, None, None, stage])
                        cute.copy(tiled_copy_B, tBgB_g[None, None, None, stage + koff_b],
                                  tBsB_g[None, None, None, stage])
                        cute.copy(tiled_copy_B, tBgB_o[None, None, None, stage + koff_b],
                                  tBsB_o[None, None, None, stage])
                    cute.arch.cp_async_commit_group()

                tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                tCsB_i_p = tCsB_i_cv[None, None, None, smem_pipe_read]
                tCsB_f_p = tCsB_f_cv[None, None, None, smem_pipe_read]
                tCsB_g_p = tCsB_g_cv[None, None, None, smem_pipe_read]
                tCsB_o_p = tCsB_o_cv[None, None, None, smem_pipe_read]

                if cutlass.const_expr(num_k_block > 1):
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                    cute.copy(tiled_copy_s2r_B, tCsB_i_p[None, None, 0], tCrB_i_cv[None, None, 0])
                    cute.copy(tiled_copy_s2r_B, tCsB_f_p[None, None, 0], tCrB_f_cv[None, None, 0])
                    cute.copy(tiled_copy_s2r_B, tCsB_g_p[None, None, 0], tCrB_g_cv[None, None, 0])
                    cute.copy(tiled_copy_s2r_B, tCsB_o_p[None, None, 0], tCrB_o_cv[None, None, 0])

                for k_tile in cutlass.range_constexpr(k_tile_count):
                    for k_block in cutlass.range_constexpr(num_k_block):
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
                        cute.copy(tiled_copy_s2r_B, tCsB_i_p[None, None, k_block_next],
                                  tCrB_i_cv[None, None, k_block_next])
                        cute.copy(tiled_copy_s2r_B, tCsB_f_p[None, None, k_block_next],
                                  tCrB_f_cv[None, None, k_block_next])
                        cute.copy(tiled_copy_s2r_B, tCsB_g_p[None, None, k_block_next],
                                  tCrB_g_cv[None, None, k_block_next])
                        cute.copy(tiled_copy_s2r_B, tCsB_o_p[None, None, k_block_next],
                                  tCrB_o_cv[None, None, k_block_next])

                        if k_block == 0:
                            if cutlass.const_expr(k_tile + num_smem_stages - 1 < k_tile_count):
                                kt_pref = k_tile + num_smem_stages - 1   # compile-time
                                cute.copy(tiled_copy_A, tAgA_p[None, None, None, kt_pref],
                                          tAsA[None, None, None, smem_pipe_write])
                                cute.copy(tiled_copy_B,
                                          tBgB_i[None, None, None, kt_pref + koff_b],
                                          tBsB_i[None, None, None, smem_pipe_write])
                                cute.copy(tiled_copy_B,
                                          tBgB_f[None, None, None, kt_pref + koff_b],
                                          tBsB_f[None, None, None, smem_pipe_write])
                                cute.copy(tiled_copy_B,
                                          tBgB_g[None, None, None, kt_pref + koff_b],
                                          tBsB_g[None, None, None, smem_pipe_write])
                                cute.copy(tiled_copy_B,
                                          tBgB_o[None, None, None, kt_pref + koff_b],
                                          tBsB_o[None, None, None, smem_pipe_write])

                        cute.gemm(tiled_mma, tCrC_i, tCrA[None, None, k_block],
                                  tCrB_i[None, None, k_block], tCrC_i)
                        cute.gemm(tiled_mma, tCrC_f, tCrA[None, None, k_block],
                                  tCrB_f[None, None, k_block], tCrC_f)
                        cute.gemm(tiled_mma, tCrC_g, tCrA[None, None, k_block],
                                  tCrB_g[None, None, k_block], tCrC_g)
                        cute.gemm(tiled_mma, tCrC_o, tCrA[None, None, k_block],
                                  tCrB_o[None, None, k_block], tCrC_o)

                        if k_block == 0:
                            cute.arch.cp_async_commit_group()
                            smem_pipe_write = smem_pipe_read
                            smem_pipe_read = smem_pipe_read + 1
                            if smem_pipe_read == num_smem_stages:
                                smem_pipe_read = 0

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

            # Pass 1: x half (no dependency on the producers).
            gemm_pass(tAgX, KOFF, self.k_tiles_x)

            # Wait for this row-group's hh_down (all S producer slices), then pass 2.
            if tidx == 0:
                ready = cutlass.Int32(0)
                while ready < self.S:
                    ready = cute.arch.atomic_add(ptr=ready_ptr, val=cutlass.Int32(0),
                                                 sem="acquire", scope="gpu")
            cute.arch.sync_threads()
            gemm_pass(tAgHH, 0, self.k_tiles_hh)

            # Epilogue: unchanged from the unfused step kernel.
            tCgCc = thr_mma.partition_C(gCc)
            rH = cute.make_fragment_like(tCrC_i, self.out_dtype)
            rC = cute.make_fragment_like(tCrC_i, cutlass.Float32)

            def clamp(x, lo, hi):
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
                        hs = h_new * 127.0
                        half = clamp(hs * 1e30, -0.5, 0.5)
                        rH[i, m, n] = (hs + half).to(self.out_dtype)
            cute.autovec_copy(rC, tCgCc)
            cute.autovec_copy(rH, tCgH)

            # Cleanup: the last consumer of the row-group resets the flags so the
            # next launch starts from zero (no host-side memsets between steps).
            if tidx == 0:
                old_done = cute.arch.atomic_add(ptr=done_ptr, val=cutlass.Int32(1),
                                                sem="acq_rel", scope="gpu")
                if old_done == GN - 1:
                    cute.arch.atomic_exch(ptr=ready_ptr, val=cutlass.Int32(0),
                                          sem="relaxed", scope="gpu")
                    cute.arch.atomic_exch(ptr=done_ptr, val=cutlass.Int32(0),
                                          sem="relaxed", scope="gpu")
        return


# =============================================================================
# AOT export
# =============================================================================
def export_factored_lstm_fused_i8(
    atom_layout_mnk: Tuple[int, int, int] = (2, 2, 1),
    file_path: str = "./artifacts",
    file_name: str = "factored_lstm_fused_i8",
    function_prefix: str = "factored_lstm_fused_i8",
    H: int = 1024,
    K_hh: int = 128,
    R: int = 128,
    num_producers: int = 4,
    bm: int = 64,
    bn: int = 32,
    bk: int = 32,
    num_stages: int = 3,
    b_size: int = 256,
) -> None:
    """AOT-compile the single-launch fused factored-LSTM step and emit a C header.

    Emits a 16-tensor + stream wrapper in __call__ order: mHprev[B,H] i8, mWdn[K_hh,H] i8,
    mWdnScale[K_hh,1] f32, mX[B,R] f16, mB_i/f/g/o[H,Kc] f16, mBias_i/f/g/o[H,1] f32,
    mC_c[B,H] f32, mH_out[B,H] i8, mHH[B,K_hh] f16, mFlags[grid_m*4] i32, CUstream.
    B (M) is dynamic at runtime; H, K_hh, R and the tile/producer config are baked.
    mFlags/mWdnScale are traced as data-only compact tensors (indexed by raw pointer),
    so the runtime buffer sizes are the caller's responsibility.
    """
    import torch
    import cutlass.torch as cutlass_torch

    Kc = K_hh + R

    def _cpt(mode0, mode1, dtype):
        torch_dtype = cutlass_torch.dtype(dtype)
        t = torch.zeros((1, mode0, mode1), dtype=torch_dtype).permute(1, 2, 0).cuda()
        return (from_dlpack(t, assumed_align=16)
                .mark_layout_dynamic(leading_dim=1)
                .mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1),
                                            divisibility=(128 // dtype.width)))

    fake_hprev = _cpt(b_size, H, cutlass.Int8)
    fake_wdn = _cpt(K_hh, H, cutlass.Int8)
    fake_wdn_scale = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (K_hh, 1), assumed_align=16)
    fake_x = _cpt(b_size, R, cutlass.Float16)
    fake_bi = _cpt(H, Kc, cutlass.Float16)
    fake_bf = _cpt(H, Kc, cutlass.Float16)
    fake_bg = _cpt(H, Kc, cutlass.Float16)
    fake_bo = _cpt(H, Kc, cutlass.Float16)
    fake_bias_i = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_f = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_g = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_bias_o = cute.runtime.make_fake_compact_tensor(cutlass.Float32, (H, 1), assumed_align=16)
    fake_c = _cpt(b_size, H, cutlass.Float32)
    fake_h = _cpt(b_size, H, cutlass.Int8)
    fake_hh = _cpt(b_size, K_hh, cutlass.Float16)
    fake_flags = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (4, 1), assumed_align=16)

    fused = TensorOpFactoredLstmFusedI8(
        cutlass.Float16, cutlass.Int8, cutlass.Float32, atom_layout_mnk,
        H=H, K_hh=K_hh, R=R, num_producers=num_producers,
        bm=bm, bn=bn, bk=bk, num_stages=num_stages)
    print(f"Compiling TensorOpFactoredLstmFusedI8  H={H} K_hh={K_hh} R={R}  "
          f"S={num_producers}  tile={bm}x{bn}x{bk}  atom={atom_layout_mnk}")
    compiled = cute.compile(
        fused, fake_hprev, fake_wdn, fake_wdn_scale, fake_x,
        fake_bi, fake_bf, fake_bg, fake_bo,
        fake_bias_i, fake_bias_f, fake_bias_g, fake_bias_o,
        fake_c, fake_h, fake_hh, fake_flags,
        stream=cute.runtime.make_fake_stream(),
    )
    print(f"Exporting to {file_path}/{file_name}.h ...")
    compiled.export_to_c(file_path=file_path, file_name=file_name, function_prefix=function_prefix)
    print("Export complete!")


__all__ = ["TensorOpFactoredLstmFusedI8", "export_factored_lstm_fused_i8"]
