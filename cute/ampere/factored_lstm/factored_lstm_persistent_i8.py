"""Persistent factorised-LSTM recurrence for NVIDIA Ampere (A100, sm80) — dorado design.

ONE launch runs the ENTIRE T-step recurrence.  Partitions BOTH hidden and batch so each
CTA reads only 1/GX of the weights, with a cheap row-split cross-CTA all-reduce of hh_down.

Grid = (GX, GY).  Batch-group = the GX CTAs sharing grid.y; together own bM_group rows.
CTA gx owns hidden columns [gx*HX, (gx+1)*HX), HX = H/GX.

Register-resident weights.  Gate weights (f16) and W_dn (int8) are loop-invariant across
all T, so each CTA ldmatrix'es its 1/GX slice into REGISTER fragments ONCE before the
T-loop and reuses them across all steps -> no per-step weight LDSM (this killed the MIO
wall: mio_throttle 7.8 -> ~1).  GX=16 (HX=64) so the resident f16 gate B fits (~128 regs).

Per timestep t (bM_group rows looped in register subtiles of bm_reg):
  1. PARTIAL down-proj: hh_partial[bm_reg,K_hh] = sHid @ resident-W_dn^T (int8) -> scratch.
  2. Row-split all-reduce (2 barriers, one monotonic flag): each CTA reduces bM_group/GX
     rows of the GX partials * comb_scale -> mScratchHH; all CTAs then read (not re-sum).
  3. GATE GEMM (f16): gates[bm_reg,HX] = A=[hh_down|x_down] @ resident gate_w^T + bias, 4 gates.
  4. EPILOGUE: cell update (f16 resident, in place), h_t int8 -> resident sHid + hh_all ring.

NOTE: an int8-gate variant (in-kernel per-row A quant + int8 resident gate_w) was tried;
it was SLOWER here (the per-row amax quant is a serial bottleneck and the register spill
persists), so the shipped path keeps f16 gate weights.  See the report for ncu evidence.
"""

import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cuda.bindings.driver as cuda_driver

import gemm_i8_quant  # noqa: F401  installs MmaI8Op shim on cute.nvgpu.warp

PERSIST_BM = 16
OUT_SCALE = 1.0 / 127.0


class TensorOpFactoredLstmPersistentI8:
    """Persistent hidden+batch-partitioned fLSTM, register-resident weights."""

    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        H: int, K_hh: int, R: int, T: int,
        bm: int = PERSIST_BM,
        GX: int = 16,
        bM_group: int = 384,
        bm_reg: int = 16,
        reverse: bool = False,
        int8_gate: bool = False,
        cell_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        variant: str = "full",     # full | core | core_ar | core_epi | core_i8
        **_ignored,
    ):
        self.ab_dtype = ab_dtype
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.cell_dtype = cell_dtype
        self.atom_layout_mnk = atom_layout_mnk
        self.H = H; self.K_hh = K_hh; self.R = R; self.Kc = K_hh + R; self.T = T
        self.GX = GX; self.HX = H // GX
        self.bM_group = bM_group; self.bm_reg = bm_reg
        self.reverse = reverse
        self.do_allreduce = variant in ("full", "core_ar")
        self.do_epilogue = variant in ("full", "core_epi")
        self.i8gate = variant == "core_i8"     # int8-gate microbench: int8 resident weights +
        self.variant = variant                 # int8 A (const scalar scale, no amax, no scale tensors)

        atom_m, atom_n, atom_k = atom_layout_mnk
        assert atom_m == 1 and atom_k == 1
        self.atom_n = atom_n
        self.num_threads = atom_n * 32
        self.f16_inst = (16, 8, 16)
        self.i8_inst = (16, 8, 32)
        self.perm_n = atom_n * 8

        assert H % GX == 0
        assert self.HX % self.perm_n == 0
        assert self.K_hh % self.perm_n == 0
        assert bM_group % bm_reg == 0
        assert bm_reg % 16 == 0
        assert bM_group % GX == 0
        self.rpc = bM_group // GX
        assert (self.rpc * K_hh) % self.num_threads == 0
        assert (bM_group * self.HX) % self.num_threads == 0
        assert (bm_reg * self.K_hh) % self.num_threads == 0
        assert (self.HX * self.Kc) % self.num_threads == 0
        assert (self.K_hh * self.HX) % self.num_threads == 0
        self.nsub = bM_group // bm_reg

    def _make_g2s(self, dtype, rows, cols):
        ce = 128 // dtype.width
        thr_k = cols // ce
        thr_m = self.num_threads // thr_k
        assert thr_m * thr_k == self.num_threads, f"g2s {rows}x{cols}"
        assert rows % thr_m == 0, f"g2s rows {rows}/{thr_m}"
        atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
            dtype, num_bits_per_copy=128)
        return cute.make_tiled_copy_tv(
            atom, cute.make_layout((thr_m, thr_k), stride=(thr_k, 1)),
            cute.make_layout((1, ce)))

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor, mW_dn: cute.Tensor, mCombScale: cute.Tensor,
        mW_i: cute.Tensor, mW_f: cute.Tensor, mW_g: cute.Tensor, mW_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor, mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mHH_all: cute.Tensor, mCell: cute.Tensor, mScratch: cute.Tensor, mFlags: cute.Tensor,
        mScratchHH: cute.Tensor, mScaleW: cute.Tensor, stream: cuda_driver.CUstream = None,
    ):
        HX = self.HX; cdt = self.cell_dtype
        adt = mX.element_type            # activation/gate dtype: Int8 (i8gate) or Float16
        sCell_layout = cute.make_layout((self.bM_group, HX), stride=(HX, 1))
        sHid_layout = cute.make_layout((self.bM_group, HX), stride=(HX, 1))
        stage_rows = max(HX, self.bm_reg)
        sStage_layout = cute.make_layout((stage_rows, self.Kc), stride=(self.Kc, 1))

        smem_bytes = (
            cute.size_in_bytes(cdt, sCell_layout)
            + cute.size_in_bytes(cutlass.Int8, sHid_layout)
            + cute.size_in_bytes(adt, sStage_layout))

        copy_x = self._make_g2s(adt, self.bm_reg, self.R)
        copy_wg = self._make_g2s(adt, HX, self.Kc)
        copy_wdn = self._make_g2s(cutlass.Int8, self.K_hh, HX)

        f16_op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.f16_inst)
        tiled_mma_f16 = cute.make_tiled_mma(
            f16_op, cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(self.f16_inst[0], self.perm_n, self.f16_inst[2]))
        i8_op = cute.nvgpu.warp.MmaI8Op(cutlass.Int8, cutlass.Int8, cutlass.Int32, self.i8_inst)
        tiled_mma_i8 = cute.make_tiled_mma(
            i8_op, cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(self.i8_inst[0], self.perm_n, self.i8_inst[2]))

        N = mCell.shape[0]
        GY = cute.ceil_div(N, self.bM_group)
        grid = (self.GX, GY, 1)

        launcher = self.kernel(
            mX, mW_dn, mCombScale, mW_i, mW_f, mW_g, mW_o,
            mBias_i, mBias_f, mBias_g, mBias_o, mHH_all, mCell, mScratch, mFlags, mScratchHH, mScaleW,
            sCell_layout, sHid_layout, sStage_layout,
            copy_x, copy_wg, copy_wdn, tiled_mma_f16, tiled_mma_i8, GY)
        launch_kwargs = dict(grid=grid, block=[self.num_threads, 1, 1], smem=smem_bytes,
                             min_blocks_per_mp=1)
        if cutlass.const_expr(stream is not None):
            launch_kwargs["stream"] = stream
        launcher.launch(**launch_kwargs)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor, mW_dn: cute.Tensor, mCombScale: cute.Tensor,
        mW_i: cute.Tensor, mW_f: cute.Tensor, mW_g: cute.Tensor, mW_o: cute.Tensor,
        mBias_i: cute.Tensor, mBias_f: cute.Tensor, mBias_g: cute.Tensor, mBias_o: cute.Tensor,
        mHH_all: cute.Tensor, mCell: cute.Tensor, mScratch: cute.Tensor, mFlags: cute.Tensor,
        mScratchHH: cute.Tensor, mScaleW: cute.Tensor,
        sCell_layout: cute.Layout, sHid_layout: cute.Layout, sStage_layout: cute.Layout,
        copy_x: cute.TiledCopy, copy_wg: cute.TiledCopy, copy_wdn: cute.TiledCopy,
        tiled_mma_f16: cute.TiledMma, tiled_mma_i8: cute.TiledMma, GY: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gx, gy, _ = cute.arch.block_idx()
        N = mCell.shape[0]
        HX = self.HX; K_hh = self.K_hh; Kc = self.Kc; H = self.H
        bMg = self.bM_group; bmr = self.bm_reg; cdt = self.cell_dtype
        nt = self.num_threads

        adt = mX.element_type            # Int8 (i8gate) or Float16
        smem = cutlass.utils.SmemAllocator()
        sCell = smem.allocate_tensor(cdt, sCell_layout, 16)
        sHid = smem.allocate_tensor(cutlass.Int8, sHid_layout, 16)
        sStage = smem.allocate_tensor(adt, sStage_layout, 16)
        sStage_i8 = cute.make_tensor(cute.recast_ptr(sStage.iterator, dtype=cutlass.Int8),
                                     cute.make_layout((K_hh, HX), stride=(HX, 1)))
        sA = cute.local_tile(sStage, (bmr, Kc), (0, 0))

        sCell_flat = cute.make_tensor(sCell.iterator, cute.make_layout(bMg * HX))
        ncz = (bMg * HX) // nt
        for j in cutlass.range_constexpr(ncz):
            sCell_flat[tidx + j * nt] = cdt(0.0)

        thr_mma_f16 = tiled_mma_f16.get_slice(tidx)
        thr_mma_i8 = tiled_mma_i8.get_slice(tidx)
        atom_ldm_f16 = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float16)
        atom_ldm_i8 = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Int8)
        tcA_f16 = cute.make_tiled_copy_A(atom_ldm_f16, tiled_mma_f16)
        tcB_f16 = cute.make_tiled_copy_B(atom_ldm_f16, tiled_mma_f16)
        tcA_i8 = cute.make_tiled_copy_A(atom_ldm_i8, tiled_mma_i8)
        tcB_i8 = cute.make_tiled_copy_B(atom_ldm_i8, tiled_mma_i8)
        thrA_f16 = tcA_f16.get_slice(tidx); thrB_f16 = tcB_f16.get_slice(tidx)
        thrA_i8 = tcA_i8.get_slice(tidx); thrB_i8 = tcB_i8.get_slice(tidx)
        thr_cx = copy_x.get_slice(tidx)
        thr_cwg = copy_wg.get_slice(tidx)
        thr_cwdn = copy_wdn.get_slice(tidx)

        # gate-path handles: int8 (i8gate microbench) or f16
        if cutlass.const_expr(self.i8gate):
            g_mma = tiled_mma_i8; g_thr = thr_mma_i8
            g_tcA = tcA_i8; g_thrA = thrA_i8; g_tcB = tcB_i8; g_thrB = thrB_i8
        else:
            g_mma = tiled_mma_f16; g_thr = thr_mma_f16
            g_tcA = tcA_f16; g_thrA = thrA_f16; g_tcB = tcB_f16; g_thrB = thrB_f16

        # ---------- resident weights, loaded ONCE ----------
        gWgates = [cute.local_tile(mW, (HX, Kc), (gx, 0)) for mW in (mW_i, mW_f, mW_g, mW_o)]
        rB_gate = []
        for g in cutlass.range_constexpr(4):
            cute.arch.sync_threads()
            cute.copy(copy_wg, thr_cwg.partition_S(gWgates[g]), thr_cwg.partition_D(sStage))
            cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            rB = g_mma.make_fragment_B(g_thr.partition_B(sStage))
            cute.copy(g_tcB, g_thrB.partition_S(sStage), g_thrB.retile(rB))
            rB_gate.append(rB)
        cute.arch.sync_threads()
        gWdn_sl = cute.local_tile(mW_dn, (K_hh, HX), (0, gx))
        cute.copy(copy_wdn, thr_cwdn.partition_S(gWdn_sl), thr_cwdn.partition_D(sStage_i8))
        cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        rW_dn = tiled_mma_i8.make_fragment_B(thr_mma_i8.partition_B(sStage_i8))
        cute.copy(tcB_i8, thrB_i8.partition_S(sStage_i8), thrB_i8.retile(rW_dn))
        cute.arch.sync_threads()
        nkb_g = cute.size(rB_gate[0], mode=[2])
        nkb_dp = cute.size(rW_dn, mode=[2])

        # representative gate C-fragment shape (same for every subtile) + resident bias
        sC0 = cute.local_tile(sCell, (bmr, HX), (0, 0))
        tCg0 = g_thr.partition_C(sC0)
        nv = cute.size(g_mma.make_fragment_C(tCg0), mode=[0])
        nm = cute.size(g_mma.make_fragment_C(tCg0), mode=[1])
        nn = cute.size(g_mma.make_fragment_C(tCg0), mode=[2])

        def bias_frag(mBias):
            gBb = cute.local_tile(mBias, (HX, 1), (gx, 0))
            b2d = cute.make_tensor(gBb.iterator, cute.make_layout((bmr, HX), stride=(0, 1)))
            tc = g_thr.partition_C(b2d)
            r = cute.make_rmem_tensor(cute.make_layout((nv, nm, nn), stride=(nn, 0, 1)), cutlass.Float32)
            for i in cutlass.range_constexpr(nv):
                for n in cutlass.range_constexpr(nn):
                    r[i, 0, n] = tc[i, 0, n].to(cutlass.Float32)
            return r
        rBias = [bias_frag(mB) for mB in (mBias_i, mBias_f, mBias_g, mBias_o)]

        # double-buffered activation smem (two [bm_reg,Kc] slots inside sStage)
        sA_bufs = [cute.local_tile(sStage, (bmr, Kc), (0, 0)),
                   cute.local_tile(sStage, (bmr, Kc), (1, 0))]

        flag_ptr = mFlags.iterator + gy
        scratch_group = GY * cutlass.Int32(self.GX) * cutlass.Int32(bMg) * cutlass.Int32(K_hh)
        gy_base = gy * cutlass.Int32(self.GX) * cutlass.Int32(bMg) * cutlass.Int32(K_hh)
        slice_stride = cutlass.Int32(bMg) * cutlass.Int32(K_hh)
        hh_gy_base = gy * cutlass.Int32(bMg) * cutlass.Int32(K_hh)

        def clamp(x, lo, hi):
            return 0.0 - cute.arch.fmax(0.0 - cute.arch.fmax(x, lo), 0.0 - hi)

        for t in cutlass.range(self.T):
            if cutlass.const_expr(self.reverse):
                tt = self.T - 1 - t; wslot = tt
            else:
                tt = t; wslot = t + 1
            buf = t % 2
            base_scratch = buf * scratch_group + (gy * cutlass.Int32(self.GX) + gx) * slice_stride

            # ===== Phase 1: partial down-proj (resident W_dn) -> scratch =====
            for sub in cutlass.range_constexpr(self.nsub):
                sc_off = base_scratch + cutlass.Int32(sub * bmr) * cutlass.Int32(K_hh)
                gPart = cute.make_tensor(mScratch.iterator + sc_off,
                                         cute.make_layout((bmr, K_hh), stride=(K_hh, 1)))
                tCpart = thr_mma_i8.partition_C(gPart)
                acc_dp = tiled_mma_i8.make_fragment_C(tCpart)
                acc_dp.fill(0)
                if t != 0:
                    sHid_sub = cute.local_tile(sHid, (bmr, HX), (sub, 0))
                    rA = tiled_mma_i8.make_fragment_A(thr_mma_i8.partition_A(sHid_sub))
                    cute.copy(tcA_i8, thrA_i8.partition_S(sHid_sub), thrA_i8.retile(rA))
                    for kb in cutlass.range_constexpr(nkb_dp):
                        cute.gemm(tiled_mma_i8, acc_dp, rA[None, None, kb], rW_dn[None, None, kb], acc_dp)
                cute.autovec_copy(acc_dp, tCpart)

            # ===== Phase 2: row-split all-reduce (2 barriers, one monotonic flag) =====
            if cutlass.const_expr(self.do_allreduce):
                cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                if tidx == 0:
                    cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                    need1 = cutlass.Int32(self.GX) * (2 * t + 1)
                    got = cutlass.Int32(0)
                    while got < need1:
                        got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                nred = (self.rpc * K_hh) // nt
                red_row0 = cutlass.Int32(gx * self.rpc)
                for j in cutlass.range_constexpr(nred):
                    idx = tidx + j * nt
                    rr = idx // K_hh; kk = idx % K_hh
                    grow = red_row0 + cutlass.Int32(rr)
                    pbase = buf * scratch_group + gy_base + grow * cutlass.Int32(K_hh) + kk
                    acc = cutlass.Int32(0)
                    for g in cutlass.range_constexpr(self.GX):
                        acc = acc + mScratch[pbase + cutlass.Int32(g) * slice_stride]
                    mScratchHH[hh_gy_base + grow * cutlass.Int32(K_hh) + kk] = (
                        acc.to(cutlass.Float32) * mCombScale[kk, 0].to(cutlass.Float32)).to(cutlass.Float16)
                cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                if tidx == 0:
                    cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                    need2 = cutlass.Int32(self.GX) * (2 * t + 2)
                    got = cutlass.Int32(0)
                    while got < need2:
                        got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                cute.arch.sync_threads()

            # ===== Phase 3: f16 gate GEMM + epilogue, software-pipelined over subtiles =====
            # epilogue(sub-1) [ALU/SFU/smem] overlaps MMA(sub) [tensor pipe]; the resident
            # gate B needs no per-step LDSM, only A is loaded per subtile.
            def build(sub, b):
                sAb = sA_bufs[b]
                hh_sub0 = hh_gy_base + cutlass.Int32(sub * bmr) * cutlass.Int32(K_hh)
                for j in cutlass.range_constexpr((bmr * K_hh) // nt):
                    idx = tidx + j * nt
                    rr = idx // K_hh; kk = idx % K_hh
                    sAb[rr, kk] = mScratchHH[hh_sub0 + cutlass.Int32(rr) * cutlass.Int32(K_hh) + kk]
                rb = gy * bMg + sub * bmr
                gX = cute.make_tensor(
                    mX.iterator + (tt * N + cutlass.Int32(rb)) * cutlass.Int32(self.R),
                    cute.make_layout((bmr, self.R), stride=(self.R, 1)))
                gX = cute.make_tensor(gX.iterator.align(16), gX.layout)
                sAx = cute.local_tile(sStage, (bmr, self.R), (b, 1))
                cute.copy(copy_x, thr_cx.partition_S(gX), thr_cx.partition_D(sAx))
                cute.arch.cp_async_commit_group()

            def mma(b):
                sAb = sA_bufs[b]
                rA_g = g_mma.make_fragment_A(g_thr.partition_A(sAb))
                cute.copy(g_tcA, g_thrA.partition_S(sAb), g_thrA.retile(rA_g))
                a = [g_mma.make_fragment_C(tCg0) for _ in range(4)]
                for x in a:
                    x.fill(0.0)
                for kb in cutlass.range_constexpr(nkb_g):
                    for gi in cutlass.range_constexpr(4):
                        cute.gemm(g_mma, a[gi], rA_g[None, None, kb], rB_gate[gi][None, None, kb], a[gi])
                return a

            def epi(accs, sub):
                rb = gy * bMg + sub * bmr
                sC_tl = cute.local_tile(sCell, (bmr, HX), (sub, 0))
                tCgC = g_thr.partition_C(sC_tl)
                ring_off = cutlass.Int64(wslot) * cutlass.Int64(N) * cutlass.Int64(H) \
                    + cutlass.Int64(rb) * cutlass.Int64(H) + cutlass.Int64(gx * HX)
                gRing = cute.make_tensor(mHH_all.iterator + ring_off,
                                         cute.make_layout((bmr, HX), stride=(H, 1)))
                r_h = cute.make_fragment_like(accs[0], self.out_dtype)
                r_c = cute.make_fragment_like(accs[0], cdt)
                for i in cutlass.range_constexpr(nv):
                    for mm in cutlass.range_constexpr(nm):
                        for cc in cutlass.range_constexpr(nn):
                            if cutlass.const_expr(self.do_epilogue):
                                raw_i = accs[0][i, mm, cc].to(cutlass.Float32) + rBias[0][i, 0, cc]
                                raw_f = accs[1][i, mm, cc].to(cutlass.Float32) + rBias[1][i, 0, cc]
                                raw_g = accs[2][i, mm, cc].to(cutlass.Float32) + rBias[2][i, 0, cc]
                                raw_o = accs[3][i, mm, cc].to(cutlass.Float32) + rBias[3][i, 0, cc]
                                i_a = clamp(raw_i * 0.2 + 0.5, 0.0, 1.0)
                                f_a = clamp(raw_f * 0.2 + 0.5, 0.0, 1.0)
                                o_a = clamp(raw_o * 0.2 + 0.5, 0.0, 1.0)
                                g_a = clamp(raw_g, -1.0, 1.0)
                                c_old = tCgC[i, mm, cc].to(cutlass.Float32)
                                c_new = f_a * c_old + i_a * g_a
                                r_c[i, mm, cc] = c_new.to(cdt)
                                h_new = o_a * cute.math.tanh(c_new)
                                hs = h_new * 127.0
                                half = clamp(hs * 1e30, -0.5, 0.5)
                                r_h[i, mm, cc] = (hs + half).to(self.out_dtype)
                            else:
                                # cheap write: sum all 4 accs (keeps all 4 MMAs live), no clamp/tanh
                                v = (accs[0][i, mm, cc] + accs[1][i, mm, cc]
                                     + accs[2][i, mm, cc] + accs[3][i, mm, cc]).to(cutlass.Float32)
                                r_c[i, mm, cc] = v.to(cdt)
                                r_h[i, mm, cc] = v.to(self.out_dtype)
                cute.autovec_copy(r_c, tCgC)
                cute.autovec_copy(r_h, g_thr.partition_C(cute.local_tile(sHid, (bmr, HX), (sub, 0))))
                cute.autovec_copy(r_h, g_thr.partition_C(gRing))

            # prologue
            build(0, 0)
            cute.arch.cp_async_wait_group(0); cute.arch.sync_threads()
            accA = mma(0)
            for sub in cutlass.range_constexpr(1, self.nsub):
                b = sub % 2
                build(sub, b)
                cute.arch.cp_async_wait_group(0); cute.arch.sync_threads()
                accB = mma(b)              # tensor pipe
                epi(accA, sub - 1)         # ALU/SFU/smem — overlaps the MMA above
                accA = accB
            epi(accA, self.nsub - 1)

            cute.arch.sync_threads()

        for sub in cutlass.range_constexpr(self.nsub):
            row_base = gy * bMg + sub * bmr
            cell_off = cutlass.Int64(row_base) * cutlass.Int64(H) + cutlass.Int64(gx * HX)
            gCell = cute.make_tensor(mCell.iterator + cell_off,
                                     cute.make_layout((bmr, HX), stride=(H, 1)))
            sC_tl = cute.local_tile(sCell, (bmr, HX), (sub, 0))
            ncc = (bmr * HX) // nt
            for j in cutlass.range_constexpr(ncc):
                idx = tidx + j * nt
                gCell[idx // HX, idx % HX] = sC_tl[idx // HX, idx % HX].to(cutlass.Float32)
        return


__all__ = ["TensorOpFactoredLstmPersistentI8", "PERSIST_BM", "OUT_SCALE"]
