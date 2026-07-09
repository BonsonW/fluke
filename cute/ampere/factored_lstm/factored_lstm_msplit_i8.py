"""M-split (row-per-warp) persistent factorised-LSTM recurrence for A100 (sm80).

atom_layout = (8,1,1): the 8 warps split batch ROWS (each warp owns 16 rows / one
m16 MMA tile), NOT output columns.  Each warp computes its OWN hh_down (down-proj
C-fragment over all K_hh) and consumes it as its OWN NON-REPLICATED gate A operand,
so the hh_down(C-frag) -> gate A-operand conversion is SAME-WARP -> an intra-warp
`shuffle_sync` relayout with NO smem for A (proven bit-exact in test_ca_shuffle.py).
This removes the N-split kernel's `short_scoreboard` smem-staging wall.

Grid = (GX, GY): GX splits hidden H into GX slices (register-resident weight slice),
GY = batch tiles.  Each CTA owns bM_group rows (8 warps * nsub * 16).

variant:
  core     - just the 2 int8 MMAs/step + intra-warp C->A shuffle + resident int8
             weights.  NO all-reduce (warp's own partial as hh_down), cheap epilogue.
             GO/NO-GO microbench.  RESULT (N=2048,T=2048,GX=32,bMg=1024): 25.8us/step
             (was 83us int8 N-split); ncu: short_scoreboard 12->0.17, mio_throttle
             ~9->0.09, tensor-util 10%->51%.  The M-split shuffle-C->A eliminates the
             smem-staging wall == the structural win.
  core_ar  - core + real cross-CTA row-split all-reduce of hh_down (single-buffered gmem
             scratch, 2 monotonic-flag barriers) + resident gate B + coalesced
             A-buffer->smem->ldmatrix reload + real x load.  Cheap epilogue.
             BEST: 105us/step @ GX=32, bMg=128, min_blocks=1 (down from 198).
             DECOMPOSITION (bMg=128): core 19 + barrier/write +30 + read +12 + reload/gate +44.
             LEVER SWEEP (all data-driven, N=T=2048):
               A) bMg DOWN = the win.  ar3 read: bMg 1024->128 = 128->61us; DRAM read
                  collapses 32MB->0.39MB/step (partials go L2-resident).  full core_ar
                  198->105us.  The "44us DRAM floor" WAS a bMg artifact (working set >
                  40MB L2 at bMg=1024).  bMg=128 is the minimum (8 warps*16).
               B) fp16 partials: NO help (107us) -- the strided reduce read is 32B-sector
                  bound, halving dtype doesn't cut sectors.
               C) smem-staged coalesced write: WORSE (115us) -- smem round-trip + syncs
                  cost more than the scatter saved.
               GX: GX=32 optimal (GX=16=160us: HX doubles -> more compute; GX=8 OOMs smem).
               min_blocks=2 (2 CTA/SM): WORSE (178us) -- 128-reg cap -> spill explodes.
             Remaining wall: reload/gate (+44) + write (+30), L2-traffic + 255-reg-spill
             bound at 1 CTA/SM.  Still > 64us two-kernel baseline.  A gave ~2x; B/C/GX/occ
             did not cross baseline.
  full     - core_ar + LSTM epilogue (cell/tanh, int8 h) + ring write.  TODO (needs the
             all-reduce made competitive first; also proper gate-weight per-channel
             scales in the harness for <0.10 correctness).
"""
import math
from typing import Tuple, Type

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda_driver

import gemm_i8_quant  # noqa: F401  installs MmaI8Op shim

OUT_SCALE = 1.0 / 127.0


def _shuffle_c_to_a(rC, rA, lane):
    """Relayout down-proj C-frag (MMA=4,1,K_hh/8) int8 -> gate A-frag hh half
    (MMA=16,1,K_hh/32) int8, purely intra-warp (32 shuffles/lane, no smem).
    Proven bit-exact in cute/test_ca_shuffle.py."""
    n_sub = cute.size(rC, mode=[2])          # K_hh / 8
    n_kt = n_sub // 4                         # K_hh / 32
    word = []
    for jt in range(n_sub):
        c0 = rC[0, 0, jt].to(cutlass.Int32) & 0xFF
        c1 = rC[1, 0, jt].to(cutlass.Int32) & 0xFF
        c2 = rC[2, 0, jt].to(cutlass.Int32) & 0xFF
        c3 = rC[3, 0, jt].to(cutlass.Int32) & 0xFF
        word.append(c0 | (c1 << 8) | (c2 << 16) | (c3 << 24))
    t = lane % 4
    group_base = (lane // 4) * 4
    src_lo = group_base + (t % 2) * 2
    src_hi = src_lo + 1
    useB = t >= 2

    def byte(w, p):
        v = (w >> (p * 8)) & 0xFF
        return cutlass.Int8(cutlass.Int32(v - ((v & 0x80) << 1)))

    for kt in range(n_kt):
        for band in range(2):
            jtA = (kt * 32 + band * 16) // 8
            jtB = jtA + 1
            wA_lo = cute.arch.shuffle_sync(word[jtA], src_lo)
            wA_hi = cute.arch.shuffle_sync(word[jtA], src_hi)
            wB_lo = cute.arch.shuffle_sync(word[jtB], src_lo)
            wB_hi = cute.arch.shuffle_sync(word[jtB], src_hi)
            w_lo = cutlass.Int32(useB) * (wB_lo - wA_lo) + wA_lo
            w_hi = cutlass.Int32(useB) * (wB_hi - wA_hi) + wA_hi
            for rowhalf in range(2):
                e = rowhalf * 4 + band * 8
                rA[e + 0, 0, kt] = byte(w_lo, rowhalf * 2 + 0)
                rA[e + 1, 0, kt] = byte(w_lo, rowhalf * 2 + 1)
                rA[e + 2, 0, kt] = byte(w_hi, rowhalf * 2 + 0)
                rA[e + 3, 0, kt] = byte(w_hi, rowhalf * 2 + 1)


class TensorOpFactoredLstmMsplitI8:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        H: int, K_hh: int, R: int, T: int,
        GX: int = 16,
        bM_group: int = 512,
        reverse: bool = False,
        variant: str = "core",
        cell_dtype: Type[cutlass.Numeric] = cutlass.Float16,
        **_ignored,
    ):
        self.ab_dtype = ab_dtype
        self.out_dtype = out_dtype
        self.acc_dtype = acc_dtype
        self.cell_dtype = cell_dtype
        self.H = H; self.K_hh = K_hh; self.R = R; self.Kc = K_hh + R; self.T = T
        self.GX = GX; self.HX = H // GX
        self.bM_group = bM_group
        self.reverse = reverse
        self.variant = variant
        self.num_threads = 256
        self.min_blocks = int(_ignored.get("min_blocks", 1))
        self.i8_inst = (16, 8, 32)
        assert H % GX == 0
        assert bM_group % (8 * 16) == 0
        self.nsub = bM_group // (8 * 16)          # 16-row subtiles per warp
        assert self.HX % 32 == 0
        assert self.K_hh % 32 == 0
        assert self.Kc % 32 == 0

    def _make_g2s(self, dtype, rows, cols):
        ce = 128 // dtype.width
        thr_k = cols // ce
        thr_m = self.num_threads // thr_k
        assert thr_m * thr_k == self.num_threads, f"g2s {rows}x{cols}: thr_k={thr_k}"
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
        HX = self.HX; K_hh = self.K_hh; Kc = self.Kc; cdt = self.cell_dtype
        bMg = self.bM_group

        sHid_layout = cute.make_layout((bMg, HX), stride=(HX, 1))
        # weight-staging smem: big enough for [HX, Kc] and [K_hh, HX]
        stg_rows = max(HX, K_hh, 128)
        sStg_layout = cute.make_layout((stg_rows, Kc), stride=(Kc, 1))
        # persistent smem for the 4 gate weights (streamed per-gate via ldmatrix -> frees
        # the 64 resident-B registers that were causing the AR-path spill).
        sWg_layout = cute.make_layout((4 * HX, Kc), stride=(Kc, 1))
        smem_bytes = (cute.size_in_bytes(cutlass.Int8, sHid_layout)
                      + cute.size_in_bytes(cutlass.Int8, sStg_layout)
                      + cute.size_in_bytes(cutlass.Int8, sWg_layout))

        copy_wg = self._make_g2s(cutlass.Int8, HX, Kc)
        copy_wdn = self._make_g2s(cutlass.Int8, K_hh, HX)
        copy_a = self._make_g2s(cutlass.Int8, 128, Kc)     # coalesced A-buffer -> smem

        i8_op = cute.nvgpu.warp.MmaI8Op(cutlass.Int8, cutlass.Int8, cutlass.Int32, self.i8_inst)
        # down-proj: M=8*16 (8 warps), N=K_hh, K=32  (full-N, used by CORE shuffle)
        mma_dp = cute.make_tiled_mma(i8_op, cute.make_layout((8, 1, 1)),
                                     permutation_mnk=(8 * 16, K_hh, 32))
        # chunked down-proj (N=32) for the AR path -> small C-frag (16 int32) so the
        # resident gate B fits WITHOUT spill.
        mma_dp_ar = cute.make_tiled_mma(i8_op, cute.make_layout((8, 1, 1)),
                                        permutation_mnk=(8 * 16, 32, 32))
        # gate: M=8*16, N=HX, K=32
        mma_g = cute.make_tiled_mma(i8_op, cute.make_layout((8, 1, 1)),
                                    permutation_mnk=(8 * 16, HX, 32))
        sWdn_layout = cute.make_layout((K_hh, HX), stride=(HX, 1))
        smem_bytes = smem_bytes + cute.size_in_bytes(cutlass.Int8, sWdn_layout)

        N = mCell.shape[0]
        GY = cute.ceil_div(N, bMg)
        grid = (self.GX, GY, 1)
        launcher = self.kernel(
            mX, mW_dn, mCombScale, mW_i, mW_f, mW_g, mW_o, mHH_all, mCell,
            mScratch, mFlags, mScratchHH,
            sHid_layout, sStg_layout, sWg_layout, sWdn_layout,
            copy_wg, copy_wdn, copy_a, mma_dp, mma_dp_ar, mma_g, GY)
        launch_kwargs = dict(grid=grid, block=[self.num_threads, 1, 1], smem=smem_bytes,
                             min_blocks_per_mp=self.min_blocks)
        if cutlass.const_expr(stream is not None):
            launch_kwargs["stream"] = stream
        launcher.launch(**launch_kwargs)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor, mW_dn: cute.Tensor, mCombScale: cute.Tensor,
        mW_i: cute.Tensor, mW_f: cute.Tensor, mW_g: cute.Tensor, mW_o: cute.Tensor,
        mHH_all: cute.Tensor, mCell: cute.Tensor,
        mScratch: cute.Tensor, mFlags: cute.Tensor, mScratchHH: cute.Tensor,
        sHid_layout: cute.Layout, sStg_layout: cute.Layout, sWg_layout: cute.Layout,
        sWdn_layout: cute.Layout,
        copy_wg: cute.TiledCopy, copy_wdn: cute.TiledCopy, copy_a: cute.TiledCopy,
        mma_dp: cute.TiledMma, mma_dp_ar: cute.TiledMma, mma_g: cute.TiledMma, GY: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        gx, gy, _ = cute.arch.block_idx()
        N = mCell.shape[0]
        HX = self.HX; K_hh = self.K_hh; Kc = self.Kc; H = self.H
        bMg = self.bM_group; nsub = self.nsub; nt = self.num_threads
        lane = tidx % 32
        warp = tidx // 32
        do_ar = cutlass.const_expr(self.variant in ("core_ar", "full"))
        # decomposition micro-variants (CORE compute + ONE all-reduce piece at a time):
        v_bar = cutlass.const_expr(self.variant in ("ar1", "ar2", "ar3"))    # +2 barriers
        v_write = cutlass.const_expr(self.variant in ("ar2", "ar3"))         # +partial write
        v_read = cutlass.const_expr(self.variant in ("ar3"))                 # +row-split reduce read
        GXc = cutlass.Int32(self.GX)

        smem = cutlass.utils.SmemAllocator()
        sHid = smem.allocate_tensor(cutlass.Int8, sHid_layout, 16)
        sStg = smem.allocate_tensor(cutlass.Int8, sStg_layout, 16)
        sWg = smem.allocate_tensor(cutlass.Int8, sWg_layout, 16)
        sWdn = smem.allocate_tensor(cutlass.Int8, sWdn_layout, 16)   # W_dn persistent (AR chunks)

        # zero sHid (initial hidden = 0)
        sHid_flat = cute.make_tensor(sHid.iterator, cute.make_layout(bMg * HX))
        for j in cutlass.range_constexpr((bMg * HX) // nt):
            sHid_flat[tidx + j * nt] = cutlass.Int8(0)

        thr_dp = mma_dp.get_slice(tidx)
        thr_dp_ar = mma_dp_ar.get_slice(tidx)
        thr_g = mma_g.get_slice(tidx)
        atom_ldm = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Int8)
        tcA = cute.make_tiled_copy_A(atom_ldm, mma_dp)
        tcB = cute.make_tiled_copy_B(atom_ldm, mma_dp)
        tcA_ar = cute.make_tiled_copy_A(atom_ldm, mma_dp_ar)
        tcB_ar = cute.make_tiled_copy_B(atom_ldm, mma_dp_ar)
        tcBg = cute.make_tiled_copy_B(atom_ldm, mma_g)
        tcAg = cute.make_tiled_copy_A(atom_ldm, mma_g)
        thrA = tcA.get_slice(tidx)
        thrB = tcB.get_slice(tidx)
        thrA_ar = tcA_ar.get_slice(tidx)
        thrB_ar = tcB_ar.get_slice(tidx)
        thrBg = tcBg.get_slice(tidx)
        thrAg = tcAg.get_slice(tidx)
        thr_cwg = copy_wg.get_slice(tidx)
        thr_cwdn = copy_wdn.get_slice(tidx)
        thr_ca = copy_a.get_slice(tidx)

        # ---- weights (loaded once) ----
        # gate weights: 4x [HX, Kc] int8 kept RESIDENT IN SMEM (sWg), ldmatrix'd per-gate
        # in the mainloop.  W_dn stays register-resident (only 8 regs).
        gWg = [cute.local_tile(mW, (HX, Kc), (gx, 0)) for mW in (mW_i, mW_f, mW_g, mW_o)]
        for g in cutlass.range_constexpr(4):
            sWg_g = cute.local_tile(sWg, (HX, Kc), (g, 0))
            cute.copy(copy_wg, thr_cwg.partition_S(gWg[g]), thr_cwg.partition_D(sWg_g))
        cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # down-proj weight: [K_hh, HX] int8 kept PERSISTENT in smem (sWdn); CORE gets a
        # resident full B-frag, AR re-ldmatrix's [32,HX] chunks per output-N chunk.
        gWdn = cute.local_tile(mW_dn, (K_hh, HX), (0, gx))
        cute.copy(copy_wdn, thr_cwdn.partition_S(gWdn), thr_cwdn.partition_D(sWdn))
        cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        rW_dn = cute.make_fragment_like(thr_dp.partition_B(sWdn), cutlass.Int8)
        cute.copy(tcB, thrB.partition_S(sWdn), thrB.retile(rW_dn))
        cute.arch.sync_threads()

        nkb_dp = cute.size(rW_dn, mode=[2])       # HX / 32
        # gate weights ldmatrix'd from smem into REGISTERS ONCE (resident across all T):
        # no per-step ldmatrix (MIO) and no hoist-inside-T (spill).
        rB_gate = []
        for g in cutlass.range_constexpr(4):
            sWg_g = cute.local_tile(sWg, (HX, Kc), (g, 0))
            rBg = cute.make_fragment_like(thr_g.partition_B(sWg_g), cutlass.Int8)
            cute.copy(tcBg, thrBg.partition_S(sWg_g), thrBg.retile(rBg))
            rB_gate.append(rBg)
        nkb_g = cute.size(rB_gate[0], mode=[2])   # Kc / 32
        nn_g = cute.size(rB_gate[0], mode=[1])    # HX / 8

        # shape-reference views (real memrefs) for make_fragment_{A,C}.
        # atom_layout (8,1,1): each MMA/tile spans 128 rows (8 warps x 16); the MMA
        # partition hands each warp its own 16-row block, so all tiles are [128, *].
        dp_cref = cute.make_tensor(mCell.iterator, cute.make_layout((128, K_hh), stride=(K_hh, 1)))
        a_ref = cute.local_tile(sStg, (128, Kc), (0, 0))
        sHid_t0 = cute.local_tile(sHid, (128, HX), (0, 0))
        nv = cute.size(mma_g.make_fragment_C(thr_g.partition_C(sHid_t0)), mode=[0])

        # all-reduce geometry (row-split over the GX CTAs sharing gy). scratch layout
        # [buf, gy, gx, row, k]. NOTE: the cross-CTA all-reduce is the dominant FULL cost
        # (~140us/step) regardless of layout -- the remaining bottleneck, not the compute.
        slice_stride = cutlass.Int32(bMg) * cutlass.Int32(K_hh)
        gy_base = gy * GXc * slice_stride
        scratch_group = GY * GXc * slice_stride
        gy_ab = gy * cutlass.Int32(bMg) * cutlass.Int32(Kc)     # A-buffer base
        flag_ptr = mFlags.iterator + gy
        rpc = bMg // self.GX

        for t in cutlass.range(self.T):
            if cutlass.const_expr(self.reverse):
                tt = self.T - 1 - t
            else:
                tt = t
            if cutlass.const_expr(do_ar):
                # SINGLE-buffered scratch (fits L2: 2x double-buffer was 64MB > 40MB L2 ->
                # every partial missed to HBM). barrier-2 already serializes reduce before
                # the next step's overwrite, so one buffer is safe.
                base_scratch = (gy * GXc + gx) * slice_stride
                # phase 1: down-proj partials -> gmem scratch (128-row subtiles).
                # Output N (=K_hh) computed in NCH chunks so the C-frag acc is small
                # (K_hh/NCH*16/8 int32/lane), leaving room for the resident gate B ->
                # NO spill.  rA_dp (the h operand) ldmatrix'd once, reused across chunks.
                NCH = cutlass.const_expr(K_hh // 32)              # output-N chunks of 32
                cch = cutlass.const_expr(32)
                for s in cutlass.range(nsub):
                    sHid_sub = cute.local_tile(sHid, (128, HX), (s, 0))
                    rA_dp = cute.make_fragment_like(thr_dp_ar.partition_A(sHid_sub), cutlass.Int8)
                    cute.copy(tcA_ar, thrA_ar.partition_S(sHid_sub), thrA_ar.retile(rA_dp))
                    sc_off = base_scratch + s * cutlass.Int32(128) * cutlass.Int32(K_hh)
                    for nc in cutlass.range_constexpr(NCH):
                        sWdn_c = cute.local_tile(sWdn, (cch, HX), (nc, 0))
                        rW_c = cute.make_fragment_like(thr_dp_ar.partition_B(sWdn_c), cutlass.Int8)
                        cute.copy(tcB_ar, thrB_ar.partition_S(sWdn_c), thrB_ar.retile(rW_c))
                        cref_c = cute.make_tensor(mCell.iterator + nc * cch,
                                                  cute.make_layout((128, cch), stride=(K_hh, 1)))
                        acc_c = mma_dp_ar.make_fragment_C(thr_dp_ar.partition_C(cref_c))
                        acc_c.fill(0)
                        for kb in cutlass.range_constexpr(nkb_dp):
                            cute.gemm(mma_dp_ar, acc_c, rA_dp[None, None, kb], rW_c[None, None, kb], acc_c)
                        gPart = cute.make_tensor(mScratch.iterator + sc_off + nc * cch,
                                                 cute.make_layout((128, cch), stride=(K_hh, 1)))
                        cute.autovec_copy(acc_c, thr_dp_ar.partition_C(gPart))
                # barrier 1
                cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                if tidx == 0:
                    cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                    need = GXc * (2 * t + 1)
                    got = cutlass.Int32(0)
                    while got < need:
                        got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                # phase 2: row-split reduce -> A-buffer hh half + x half
                nred = (rpc * K_hh) // nt
                for j in cutlass.range_constexpr(nred):
                    idx = tidx + j * nt
                    rr = idx // K_hh; kk = idx % K_hh
                    grow = cutlass.Int32(gx * rpc) + rr
                    pbase = gy_base + grow * cutlass.Int32(K_hh) + kk
                    # running sum (2 regs) -- L2-resident at small bMg so latency is cheap;
                    # avoids the 32-load ILP list that fed register spill.
                    acc = cutlass.Int32(0)
                    for g in cutlass.range_constexpr(self.GX):
                        acc = acc + mScratch[pbase + cutlass.Int32(g) * slice_stride]
                    hh_f = acc.to(cutlass.Float32) * mCombScale[kk, 0].to(cutlass.Float32) * 127.0
                    mScratchHH[gy_ab + grow * cutlass.Int32(Kc) + kk] = cutlass.Int8(cutlass.Int32(hh_f))
                nredx = (rpc * self.R) // nt
                for j in cutlass.range_constexpr(nredx):
                    idx = tidx + j * nt
                    rr = idx // self.R; rc = idx % self.R
                    grow = cutlass.Int32(gx * rpc) + rr
                    xoff = (cutlass.Int32(tt) * N + gy * cutlass.Int32(bMg) + grow) * cutlass.Int32(self.R) + rc
                    mScratchHH[gy_ab + grow * cutlass.Int32(Kc) + cutlass.Int32(K_hh) + rc] = mX[xoff]
                # barrier 2
                cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                if tidx == 0:
                    cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                    need = GXc * (2 * t + 2)
                    got = cutlass.Int32(0)
                    while got < need:
                        got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                cute.arch.sync_threads()
                # phase 3: gate + epilogue.  A-buffer -> smem (coalesced cp.async) ->
                # ldmatrix per-warp.  Gate B ldmatrix'd per-gate INSIDE the loop (16 regs,
                # reused) -- hoisting all 4 (64 regs live) was the introduced spill.
                for s in cutlass.range(nsub):
                    sHid_sub = cute.local_tile(sHid, (128, HX), (s, 0))
                    ab_off = gy_ab + s * cutlass.Int32(128) * cutlass.Int32(Kc)
                    gA = cute.make_tensor(mScratchHH.iterator + ab_off,
                                          cute.make_layout((128, Kc), stride=(Kc, 1)))
                    gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
                    cute.arch.sync_threads()
                    cute.copy(copy_a, thr_ca.partition_S(gA), thr_ca.partition_D(a_ref))
                    cute.arch.cp_async_commit_group(); cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_threads()
                    rA_g = cute.make_fragment_like(thr_g.partition_A(a_ref), cutlass.Int8)
                    cute.copy(tcAg, thrAg.partition_S(a_ref), thrAg.retile(rA_g))
                    tC_hid = thr_g.partition_C(sHid_sub)
                    acc_sum = mma_g.make_fragment_C(tC_hid); acc_sum.fill(0)
                    for g in cutlass.range_constexpr(4):
                        cg = mma_g.make_fragment_C(tC_hid); cg.fill(0)
                        for kb in cutlass.range_constexpr(nkb_g):
                            cute.gemm(mma_g, cg, rA_g[None, None, kb], rB_gate[g][None, None, kb], cg)
                        for i in cutlass.range_constexpr(cute.size(acc_sum)):
                            acc_sum[i] = acc_sum[i] + cg[i]
                    r_h = cute.make_fragment_like(acc_sum, cutlass.Int8)
                    for i in cutlass.range_constexpr(cute.size(acc_sum)):
                        r_h[i] = acc_sum[i].to(cutlass.Int8)
                    cute.autovec_copy(r_h, tC_hid)
                cute.arch.sync_threads()
            else:
                # CORE compute (+ optional barrier/write/read micro-variants for the
                # all-reduce cost decomposition).
                base_scratch = (gy * GXc + gx) * slice_stride
                for s in cutlass.range(nsub):
                    sHid_sub = cute.local_tile(sHid, (128, HX), (s, 0))
                    acc_dp = mma_dp.make_fragment_C(thr_dp.partition_C(dp_cref))
                    acc_dp.fill(0)
                    rA_dp = cute.make_fragment_like(thr_dp.partition_A(sHid_sub), cutlass.Int8)
                    cute.copy(tcA, thrA.partition_S(sHid_sub), thrA.retile(rA_dp))
                    for kb in cutlass.range_constexpr(nkb_dp):
                        cute.gemm(mma_dp, acc_dp, rA_dp[None, None, kb], rW_dn[None, None, kb], acc_dp)
                    if cutlass.const_expr(v_write):
                        sc_off = base_scratch + s * cutlass.Int32(128) * cutlass.Int32(K_hh)
                        gPart = cute.make_tensor(mScratch.iterator + sc_off,
                                                 cute.make_layout((128, K_hh), stride=(K_hh, 1)))
                        cute.autovec_copy(acc_dp, thr_dp.partition_C(gPart))
                    rC_i8 = cute.make_rmem_tensor(
                        cute.make_layout((4, 1, K_hh // 8), stride=(1, 4, 4)), cutlass.Int8)
                    for i in cutlass.range_constexpr(4):
                        for jt in cutlass.range_constexpr(K_hh // 8):
                            rC_i8[i, 0, jt] = acc_dp[i, 0, jt].to(cutlass.Int8)
                    rA_g = cute.make_fragment_like(thr_g.partition_A(a_ref), cutlass.Int8)
                    _shuffle_c_to_a(rC_i8, rA_g, lane)
                    for kt in cutlass.range_constexpr(K_hh // 32, Kc // 32):
                        for e in cutlass.range_constexpr(16):
                            rA_g[e, 0, kt] = cutlass.Int8(1)
                    tC_hid = thr_g.partition_C(sHid_sub)
                    acc_sum = mma_g.make_fragment_C(tC_hid); acc_sum.fill(0)
                    for g in cutlass.range_constexpr(4):
                        cg = mma_g.make_fragment_C(tC_hid); cg.fill(0)
                        for kb in cutlass.range_constexpr(nkb_g):
                            cute.gemm(mma_g, cg, rA_g[None, None, kb], rB_gate[g][None, None, kb], cg)
                        for i in cutlass.range_constexpr(cute.size(acc_sum)):
                            acc_sum[i] = acc_sum[i] + cg[i]
                    r_h = cute.make_fragment_like(acc_sum, cutlass.Int8)
                    for i in cutlass.range_constexpr(cute.size(acc_sum)):
                        r_h[i] = acc_sum[i].to(cutlass.Int8)
                    cute.autovec_copy(r_h, tC_hid)
                # ---- optional all-reduce pieces (isolation micro-variants) ----
                if cutlass.const_expr(v_bar):
                    cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                    if tidx == 0:
                        cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                        need = GXc * (2 * t + 1)
                        got = cutlass.Int32(0)
                        while got < need:
                            got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                    cute.arch.sync_threads()
                if cutlass.const_expr(v_read):
                    # row-split reduce: each CTA sums GX partials for its own rpc rows
                    # (O(GX*bMg*K_hh) total, linear in GX), writes result, reads it back.
                    sink = cutlass.Int32(0)
                    nred = (rpc * K_hh) // nt
                    for j in cutlass.range_constexpr(nred):
                        idx = tidx + j * nt
                        rr = idx // K_hh; kk = idx % K_hh
                        grow = cutlass.Int32(gx * rpc) + rr
                        pbase = gy_base + grow * cutlass.Int32(K_hh) + kk
                        # constexpr-unrolled -> GX INDEPENDENT loads (ILP overlaps latency);
                        # dynamic loop was a serial-dependent chain -> latency stacked.
                        parts = [mScratch[pbase + cutlass.Int32(g) * slice_stride]
                                 for g in range(self.GX)]
                        acc = cutlass.Int32(0)
                        for g in cutlass.range_constexpr(self.GX):
                            acc = acc + parts[g]
                        mScratchHH[gy_ab + grow * cutlass.Int32(Kc) + kk] = cutlass.Int8(acc)
                        sink = sink + mScratchHH[gy_ab + grow * cutlass.Int32(Kc) + kk].to(cutlass.Int32)
                    if tidx == 999999:      # keep `sink` live (never true) -> no DCE, no real write
                        mScratchHH[0] = cutlass.Int8(sink)
                if cutlass.const_expr(v_bar):
                    cute.arch.fence_acq_rel_gpu(); cute.arch.sync_threads()
                    if tidx == 0:
                        cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(1), sem="release", scope="gpu")
                        need = GXc * (2 * t + 2)
                        got = cutlass.Int32(0)
                        while got < need:
                            got = cute.arch.atomic_add(ptr=flag_ptr, val=cutlass.Int32(0), sem="acquire", scope="gpu")
                    cute.arch.sync_threads()
                cute.arch.sync_threads()
        return


__all__ = ["TensorOpFactoredLstmMsplitI8", "OUT_SCALE"]
